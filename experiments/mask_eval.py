from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import pickle
import re
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from passportfl.innovation import InnovationAuditor
from passportfl.mask_attacks import build_registry, training_saving
from protocol_eval import (ATTACKS as CLASSIC, flag_scores, fit_threshold,
                           load_archive, rate_record)

ARCHIVE_DIR = os.path.join(_ROOT, "results", "archives")
MAD_FLOOR = 1e-2
TARGETS = (0.05, 0.01)
J_GRID = (100, 50, 20, 10, 5, 3, 2, 1)

CONFIGS: Dict[str, dict] = {
    "paper":     dict(mask_policy="declared"),
    "carry":     dict(mask_policy="declared", carry_check=True),
    "support":   dict(mask_policy="declared", monitor_support=True, monitor_churn=True,
                      always_score_mask_free=True),
    "full":      dict(mask_policy="full"),
    "derived":   dict(mask_policy="derived", always_score_mask_free=True),
    "hardened":  dict(mask_policy="full", carry_check=True, monitor_support=True,
                      monitor_churn=True, always_score_mask_free=True),
    "temporal":  dict(mask_policy="full", carry_check=True, monitor_support=True,
                      monitor_churn=True, monitor_support_drop=True,
                      always_score_mask_free=True),
    "size_only": dict(mask_policy="full", carry_check=True, monitor_support=True,
                      always_score_mask_free=True),
    "temporal_only": dict(mask_policy="full", carry_check=True,
                          monitor_support_drop=True, always_score_mask_free=True),
}

POOLS: Dict[str, dict] = {
    "cifar100_d05": dict(prefix="arch_cifar100_s", cal=[0, 1], test=[3, 4], alpha=0.5),
    "cifar100_d01": dict(prefix="arch_cifar100_d01_s", cal=[0], test=[1, 2], alpha=0.1),
    "maskmix":      dict(prefix="arch_maskmix_", cal=[0], test=[1], alpha=None),
    "maskshard10":  dict(prefix="arch_maskshard10_", cal=[0], test=[0], alpha=None),
    "shard05":      dict(prefix="arch_shard05_", cal=[0], test=[1], alpha=None),
    "shard10":      dict(prefix="arch_shard10_", cal=[0], test=[1], alpha=None),
    "shard20":      dict(prefix="arch_shard20_", cal=[0], test=[1], alpha=None),
    "shard40":      dict(prefix="arch_shard40_", cal=[0], test=[1], alpha=None),
}

_REG_CACHE: Dict[Tuple[tuple, tuple], dict] = {}
_CNT_CACHE: Dict[Tuple[float, int], np.ndarray] = {}


def registry_for(j_grid: tuple, crafts: tuple) -> dict:
    key = (tuple(j_grid), tuple(crafts))
    if key not in _REG_CACHE:
        _REG_CACHE[key] = build_registry(j_grid=key[0], crafts=key[1])
    return _REG_CACHE[key]


def find_archives(prefix: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if not os.path.isdir(ARCHIVE_DIR):
        return out
    for fn in sorted(os.listdir(ARCHIVE_DIR)):
        if not (fn.startswith(prefix) and fn.endswith(".npz")):
            continue
        m = re.search(r"_s(\d+)\.npz$", fn)
        if m:
            out[int(m.group(1))] = os.path.join(ARCHIVE_DIR, fn)
    return out


def cifar100_counts(alpha: float, seed: int, n_clients: int = 10, K: int = 100) -> np.ndarray:
    key = (alpha, seed)
    if key in _CNT_CACHE:
        return _CNT_CACHE[key]
    with open(os.path.join(_ROOT, "data", "cifar-100-python", "train"), "rb") as f:
        d = pickle.load(f, encoding="bytes")
    y = np.asarray(d[b"fine_labels"])
    rng = np.random.default_rng(seed)
    while True:
        idx_per_client = [[] for _ in range(n_clients)]
        for k in range(K):
            idx_k = np.where(y == k)[0]
            rng.shuffle(idx_k)
            p = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(p) * len(idx_k)).astype(int)[:-1]
            for c, part in enumerate(np.split(idx_k, cuts)):
                idx_per_client[c].extend(part.tolist())
        if min(len(i) for i in idx_per_client) >= 10:
            break
    cnt = np.zeros((n_clients, K), dtype=np.int64)
    for c, idx in enumerate(idx_per_client):
        u, n = np.unique(y[np.asarray(sorted(idx))], return_counts=True)
        cnt[c, u] = n
    _CNT_CACHE[key] = cnt
    return cnt


_JIT_CACHE: Dict[Tuple[str, float], dict] = {}


def jitter_archive(arch0: dict, counts: np.ndarray, p: float, seed: int) -> dict:
    key = (id(arch0), p, seed)
    _ = key
    K = counts.shape[1]
    clients = len(arch0["C"])
    rounds = max(max(v) for v in arch0["C"].values())
    pdrop = np.power(1.0 - p, counts.astype(np.float64))
    U = np.random.default_rng([int(seed), 987654321]).random((rounds + 2, clients, K))
    C: Dict[int, Dict[int, np.ndarray]] = {i: {} for i in range(clients)}
    P: Dict[int, Dict[int, np.ndarray]] = {i: {} for i in range(clients)}
    for r in range(1, rounds + 1):
        cb = arch0["cbar"].get(r - 1)
        for i in range(clients):
            tru = np.asarray(arch0["present"][i][r], dtype=bool)
            if cb is None:
                C[i][r] = arch0["C"][i][r]
                P[i][r] = tru
                continue
            keep = tru & (U[r, i] >= pdrop[i])
            row = arch0["C"][i][r].copy()
            row[~keep] = cb[~keep]
            C[i][r] = row
            P[i][r] = keep
    return {"C": C, "present": P, "cbar": arch0["cbar"]}


def score_one_archive(task: dict) -> dict:
    arch0 = load_archive(task["path"])
    jit = float(task.get("jitter") or 0.0)
    if jit > 0.0:
        ck = (task["path"], jit)
        if ck not in _JIT_CACHE:
            _JIT_CACHE.clear()
            _JIT_CACHE[ck] = jitter_archive(arch0, cifar100_counts(task["alpha"], task["seed"]),
                                            jit, task["seed"])
        arch0 = _JIT_CACHE[ck]
    clients = len(arch0["C"])
    rounds = max(max(v) for v in arch0["C"].values())
    arch = dict(arch0)
    arch["published"] = {i: {} for i in range(clients)}

    aud = InnovationAuditor(mad_floor=MAD_FLOOR, **CONFIGS[task["config"]])
    attack, victim, kind = task.get("attack"), task.get("victim"), task.get("kind")
    astart, scope = task["attack_start"], task["scope_start"]
    rng = np.random.default_rng([int(task["seed"]), int(victim or 0) + 1,
                                 int(zlib.crc32((attack or "").encode()))])
    spec = registry_for(task["j_grid"], task["crafts"])[attack] if kind == "mask" else None
    counts = (cifar100_counts(task["alpha"], task["seed"])
              if (task.get("alpha") is not None and victim is not None) else None)
    state: dict = {"published": arch["published"].get(victim, {})}

    rec_rounds: List[int] = []
    sus: List[List[float]] = []
    supp_all: List[List[float]] = []
    churn_all: List[List[float]] = []
    declared_sz, derived_sz, nT, saving = [], [], [], []

    for r in range(1, rounds + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        payloads = {}
        for cid in range(clients):
            C, M = arch["C"][cid][r], arch["present"][cid][r]
            if victim is not None and cid == victim and r >= astart:
                if kind == "mask":
                    C, M = spec.payload(arch, cid, r, rng, state)
                else:
                    C = CLASSIC[attack](arch, cid, r, rng)
            arch["published"][cid][r] = C
            payloads[cid] = (C, M)
        rs = aud.observe_round(r, payloads)
        if r < scope:
            continue
        rec_rounds.append(r)
        row = [float("nan")] * clients
        srow = [float("nan")] * clients
        crow = [float("nan")] * clients
        for cid, v in rs.per_client.items():
            row[cid] = float(v.get("suspicion", float("nan")))
            srow[cid] = float(v.get("supp", float("nan")))
            crow[cid] = float(v.get("churn", float("nan")))
        sus.append(row)
        supp_all.append(srow)
        churn_all.append(crow)
        if victim is not None:
            vv = rs.per_client.get(victim, {})
            M = np.asarray(payloads[victim][1], dtype=bool)
            declared_sz.append(float(M.sum()))
            derived_sz.append(float(vv.get("supp", float("nan"))))
            nT.append(float(vv.get("n_T", float("nan"))))
            if counts is not None and r >= astart:
                craft = spec.craft if spec is not None else craft_classic_payload(attack)
                frac = spec.shrink_frac if spec is not None else 0.1
                saving.append(training_saving(counts[victim], M, craft, frac))

    return {"key": task["key"], "rounds": rec_rounds,
            "sus": np.asarray(sus, dtype=np.float64),
            "supp_all": np.asarray(supp_all, dtype=np.float64),
            "churn_all": np.asarray(churn_all, dtype=np.float64),
            "declared": np.asarray(declared_sz, dtype=np.float64),
            "derived": np.asarray(derived_sz, dtype=np.float64),
            "n_T": np.asarray(nT, dtype=np.float64),
            "saving": np.asarray(saving, dtype=np.float64)}


def craft_classic_payload(name: str) -> str:
    return {"global_echo": "echo", "peer_copy": "echo", "lagged_self": "echo",
            "ar2_replay": "echo", "noise": "echo", "shrink": "shrink"}.get(name, "echo")


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if not pos.size or not neg.size:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt))
    np.add.at(s, inv, ranks)
    ranks = (s / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def roc_auc_strict(pos: np.ndarray, neg: np.ndarray) -> float:
    negf = neg[np.isfinite(neg)]
    if not pos.size or not negf.size:
        return float("nan")
    posf = pos[np.isfinite(pos)]
    lo = min(float(posf.min()) if posf.size else float(negf.min()), float(negf.min())) - 1.0
    return roc_auc(np.where(np.isfinite(pos), pos, lo), negf)


def evaluate(passes: List[dict], honest_fpr: dict, thr: Dict[str, float],
             astart: int) -> dict:
    pos, neg = [], []
    tp_k = tp_n = run_k = run_n = sil_k = sil_n = 0
    sav, nT, der, dec = [], [], [], []
    for p in passes:
        v = p["key"]["victim"]
        sel = np.asarray(p["rounds"]) >= astart
        if not sel.any():
            continue
        m = p["sus"][sel]
        col = m[:, v]
        pos.append(col)
        neg.append(np.delete(m, v, axis=1).ravel())
        f = flag_scores(col, thr["0.01"])
        tp_k += int(f.sum()); tp_n += int(f.size)
        run_n += 1; run_k += int(f.any())
        sil_k += int((~np.isfinite(col)).sum()); sil_n += int(col.size)
        if p["saving"].size:
            sav.append(p["saving"])
        nT.append(p["n_T"][sel]); der.append(p["derived"][sel]); dec.append(p["declared"][sel])
    pv = np.concatenate(pos) if pos else np.zeros(0)
    nv = np.concatenate(neg) if neg else np.zeros(0)

    def mean(parts):
        if not parts:
            return None
        v = np.concatenate(parts)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else None

    return {"auc": roc_auc(pv, nv), "auc_strict": roc_auc_strict(pv, nv),
            "tpr_client_round": rate_record(tp_k, tp_n),
            "tpr_client_run": rate_record(run_k, run_n),
            "silence_rate": (sil_k / sil_n) if sil_n else float("nan"),
            "honest_fpr_client_round": honest_fpr,
            "saving_mean": mean(sav), "mean_n_T": mean(nT),
            "mean_derived_support": mean(der), "mean_declared_support": mean(dec),
            "n_pos": int(pv.size), "n_neg": int(nv.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="cifar100_d05", choices=list(POOLS))
    ap.add_argument("--attack-start", type=int, default=12)
    ap.add_argument("--scope-start", type=int, default=12)
    ap.add_argument("--workers", type=int, default=min(12, (os.cpu_count() or 4)))
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--out", default="")
    ap.add_argument("--quick", action="store_true", help="one craft, short j grid")
    ap.add_argument("--honest-only", action="store_true",
                    help="skip every attack; just characterise the honest population")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="model every client as training on a fresh p-fraction sample of its "
                         "local data each round, so its honest support legitimately moves. "
                         "0 disables. Requires a pool with a known partition (alpha set).")
    a = ap.parse_args()
    if a.jitter > 0 and POOLS[a.pool]["alpha"] is None:
        print("--jitter needs the per-class counts, which are only reproducible for the "
              "Dirichlet pools (alpha set in POOLS)")
        return 1

    pool = POOLS[a.pool]
    paths = find_archives(pool["prefix"])
    cal = [s for s in pool["cal"] if s in paths]
    test = [s for s in pool["test"] if s in paths]
    if not cal or not test:
        print("pool %s: need CAL %s and TEST %s, have %s"
              % (a.pool, pool["cal"], pool["test"], sorted(paths)))
        return 1
    configs = [c for c in a.configs.split(",") if c in CONFIGS]

    arch0 = load_archive(paths[cal[0]])
    clients = len(arch0["C"])
    rounds = max(max(v) for v in arch0["C"].values())
    supports = sorted(int(arch0["present"][c][1].sum()) for c in range(clients))
    print("=" * 118)
    print("MASK-MANIPULATION EVALUATION   pool=%s  clients=%d  rounds=%d" % (a.pool, clients, rounds))
    print("  CAL (honest only, fits thresholds): seeds %s" % cal)
    print("  TEST (frozen, reported)           : seeds %s%s"
          % (test, "   [SAME AS CAL -- in-sample, flagged as such]" if test == cal else ""))
    print("  honest support sizes on CAL seed %d: %s" % (cal[0], supports))
    print("  MAD floor %.0e (the value protocol_eval.py selected on CAL)" % MAD_FLOOR)
    print("  configurations: %s" % ", ".join(configs))
    print("=" * 118)

    crafts = ("honest",) if a.quick else ("echo", "shrink", "honest", "scaled")
    j_grid = (5, 2, 1) if a.quick else J_GRID
    registry = registry_for(j_grid, crafts)
    specs = ([] if a.honest_only else
             [("mask", n) for n in registry] + [("classic", n) for n in CLASSIC])
    print("  %d attack specs (%d mask-manipulation, %d classic) x %d victims x %d TEST seeds"
          % (len(specs), len(registry) if specs else 0, len(CLASSIC) if specs else 0,
             clients, len(test)))

    common = dict(attack_start=a.attack_start, scope_start=a.scope_start,
                  j_grid=tuple(j_grid), crafts=tuple(crafts), alpha=pool["alpha"],
                  jitter=a.jitter)
    tasks = []
    for cname in configs:
        for p_name, seeds in (("CAL", cal), ("TEST", test)):
            for s in seeds:
                tasks.append(dict(key=dict(config=cname, pool=p_name, seed=s, attack=None,
                                           victim=None),
                                  path=paths[s], config=cname, seed=s, attack=None,
                                  victim=None, kind=None, **common))
        for s in test:
            for kind, name in specs:
                for v in range(clients):
                    tasks.append(dict(key=dict(config=cname, pool="TEST", seed=s, attack=name,
                                               victim=v, kind=kind),
                                      path=paths[s], config=cname, seed=s, attack=name,
                                      victim=v, kind=kind, **common))

    print("  %d replay passes total" % len(tasks))
    t0 = time.time()
    tasks.sort(key=lambda t: (t["path"], t["config"]))
    if a.workers <= 1:
        res = [score_one_archive(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(score_one_archive, tasks,
                              chunksize=max(1, len(tasks) // (a.workers * 8))))
    print("  replay done in %.1f s" % (time.time() - t0))

    by: Dict[Tuple[str, str, Optional[str]], List[dict]] = {}
    for r in res:
        by.setdefault((r["key"]["config"], r["key"]["pool"], r["key"]["attack"]), []).append(r)

    out = {"pool": a.pool, "cal": cal, "test": test, "in_sample": test == cal,
           "mad_floor": MAD_FLOOR, "attack_start": a.attack_start, "clients": clients,
           "rounds": rounds, "honest_supports": supports, "jitter": a.jitter, "configs": {}}

    hh = by.get((configs[0], "TEST", None), [])
    if hh:
        sa = np.concatenate([p["supp_all"].ravel() for p in hh])
        ca = np.concatenate([p["churn_all"].ravel() for p in hh])
        sa, ca = sa[np.isfinite(sa)], ca[np.isfinite(ca)]
        out["honest_null"] = {
            "support_mean": float(sa.mean()) if sa.size else None,
            "support_min": float(sa.min()) if sa.size else None,
            "support_max": float(sa.max()) if sa.size else None,
            "support_mad": float(np.median(np.abs(sa - np.median(sa)))) if sa.size else None,
            "churn_mean": float(ca.mean()) if ca.size else None,
            "churn_max": float(ca.max()) if ca.size else None,
            "churn_frac_nonzero": float((ca > 0).mean()) if ca.size else None,
        }
        hn_ = out["honest_null"]
        print("\n  HONEST NULL of the support statistics on TEST (jitter=%.2f):" % a.jitter)
        print("    |Shat| mean %.1f  range [%.0f, %.0f]  cohort MAD %.2f"
              % (hn_["support_mean"], hn_["support_min"], hn_["support_max"], hn_["support_mad"]))
        print("    churn  mean %.2f  max %.0f  nonzero in %.1f%% of client-rounds"
              % (hn_["churn_mean"], hn_["churn_max"], 100 * hn_["churn_frac_nonzero"]))
        if hn_["churn_frac_nonzero"] == 0.0:
            print("    churn is IDENTICALLY ZERO for every honest client-round, so its robust z")
            print("    divides by the MAD floor alone and any churn scores in the thousands. The")
            print("    power of the churn statistic here is a property of the simulation (every")
            print("    client trains on its whole shard every round), not of the method. Re-run")
            print("    with --jitter to score it against a moving honest support.")

    for cname in configs:
        cal_h = by.get((cname, "CAL", None), [])
        test_h = by.get((cname, "TEST", None), [])
        vals = np.concatenate([p["sus"].ravel() for p in cal_h]) if cal_h else np.zeros(0)
        thr = {f"{t:g}": fit_threshold(vals, t) for t in TARGETS}
        hk = hn = 0
        for p in test_h:
            f = flag_scores(p["sus"][np.asarray(p["rounds"]) >= a.attack_start], thr["0.01"])
            hk += int(f.sum()); hn += int(f.size)
        hfpr = rate_record(hk, hn)
        fin = vals[np.isfinite(vals)]
        cout = {"thresholds": thr, "cal_n": int(vals.size),
                "cal_frac_finite": float(fin.size / vals.size) if vals.size else None,
                "cal_q50": float(np.quantile(fin, 0.5)) if fin.size else None,
                "cal_q99": float(np.quantile(fin, 0.99)) if fin.size else None,
                "honest_test_fpr": hfpr, "attacks": {}}
        for kind, name in specs:
            ps = by.get((cname, "TEST", name), [])
            if ps:
                cout["attacks"][name] = dict(evaluate(ps, hfpr, thr, a.attack_start), kind=kind)
        out["configs"][cname] = cout

    outp = a.out or os.path.join(_ROOT, "results", f"mask_eval_{a.pool}.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("  wrote %s" % outp)
    report(out, configs, j_grid, crafts)
    return 0


def report(out: dict, configs: Sequence[str], j_grid, crafts) -> None:
    def get(c, name):
        return out["configs"].get(c, {}).get("attacks", {}).get(name)

    def num(c, name, field, scale=1.0, fmt="%10.3f", width=10):
        v = get(c, name)
        if not v:
            return ("%" + str(width) + "s") % "-"
        x = v[field]
        if isinstance(x, dict):
            x = x["point"]
        return fmt % (scale * x) if x is not None and np.isfinite(x) else \
            ("%" + str(width) + "s") % "nan"

    print("\n" + "=" * 118)
    print("HONEST COST: realised false-alarm rate on TEST at the CAL-frozen 1% threshold")
    print("=" * 118)
    print("  %-12s %10s %28s %10s %10s %10s"
          % ("config", "tau@1%", "honest FPR (client-round)", "CAL n", "finite", "CAL q99"))
    for c in configs:
        cc = out["configs"][c]
        print("  %-12s %10.4f %28s %10d %9.1f%% %10.3f"
              % (c, cc["thresholds"]["0.01"], cc["honest_test_fpr"]["reported"], cc["cal_n"],
                 100 * (cc["cal_frac_finite"] or 0.0), cc["cal_q99"] or float("nan")))
    if out.get("in_sample"):
        print("  NOTE: this pool has one archive, so TEST == CAL. The FPR column is IN-SAMPLE")
        print("        and is a lower bound on what a held-out seed would give.")

    for field, title, scale, fmt in (
            ("auc_strict", "detection AUC (STRICT: an undefined victim score counts as a miss)",
             1.0, "%10.3f"),
            ("auc", "detection AUC as innovation_eval computes it (undefined victim rounds DROPPED)",
             1.0, "%10.3f"),
            ("tpr_client_round", "TPR at the CAL-frozen 1% threshold, client-round", 100.0, "%9.1f%%"),
            ("silence_rate", "SILENCE: fraction of victim client-rounds with no score at all",
             100.0, "%9.1f%%")):
        print("\n" + "=" * 118)
        print("SUPPORT SHRINKING / CHURN -- %s" % title)
        print("=" * 118)
        for craft in crafts:
            print("\n  underlying fraud = %s" % craft)
            print("  %-12s" % "config" + "".join("%10s" % ("j=%d" % j) for j in j_grid)
                  + "%10s%10s%10s%10s%10s"
                  % ("churn2", "churn3", "ramp12j5", "ramp20j5", "ramp20j2"))
            for c in configs:
                row = "  %-12s" % c
                for j in j_grid:
                    row += num(c, f"shrink_{craft}_j{j}", field, scale, fmt)
                for nm in (f"churn_{craft}", f"churn3_{craft}",
                           f"ramp12_{craft}_j5", f"ramp20_{craft}_j5", f"ramp20_{craft}_j2"):
                    row += num(c, nm, field, scale, fmt)
                print(row)

    print("\n" + "=" * 118)
    print("ATTACKER'S SAVING (fraction of local training samples never touched) and what the")
    print("auditor actually had to work with (|T|, the comparable set)")
    print("=" * 118)
    for craft in crafts:
        print("  fraud=%-8s" % craft + "".join("%10s" % ("j=%d" % j) for j in j_grid))
        row = "    saving   "
        row2 = "    |T| paper"
        for j in j_grid:
            row += num("paper", f"shrink_{craft}_j{j}", "saving_mean", 100.0, "%9.1f%%")
            row2 += num("paper", f"shrink_{craft}_j{j}", "mean_n_T", 1.0, "%10.1f")
        print(row)
        print(row2)

    print("\n" + "=" * 118)
    print("CLASSIC ATTACK SUITE under each defence (AUC strict): cost of the full-row policy")
    print("=" * 118)
    print("  %-12s" % "config" + "".join("%14s" % k[:13] for k in CLASSIC) + "%12s" % "worst")
    for c in configs:
        vals = []
        for k in CLASSIC:
            v = get(c, k)
            vals.append(v["auc_strict"] if v else float("nan"))
        print("  %-12s" % c + "".join("%14.3f" % v for v in vals)
              + "%12.3f" % (np.nanmin(vals) if len(vals) else float("nan")))


if __name__ == "__main__":
    sys.exit(main())
