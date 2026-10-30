from __future__ import annotations

import argparse, itertools, json, math, os, re, sys, time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

from passportfl.innovation import EPS, MAD_SCALE, SCORE_STRUCTURAL, InnovationAuditor
from passportfl.baselines import BASELINES, score_round
from passportfl.adaptive_attack import FIXED_POINT_SCALE
from passportfl.combined_attack import (BASE_STATS, COMB_STATS, RULES, CombinedAttackConfig,
                                        CombinedAttacker, apply_norm, cohort_z_score, combine_one,
                                        schedule_factor)

ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "archives")
RULE_STATS = ["combined_" + r for r in RULES]
ALL_STATS = RULE_STATS + COMB_STATS
_CACHE: Dict[str, dict] = {}


def load_archive(path: str) -> dict:
    if path not in _CACHE:
        z = np.load(path, allow_pickle=True)
        _CACHE.clear()
        _CACHE[path] = {"C": z["C"].item(), "present": z["present"].item(), "cbar": z["cbar"].item()}
    return _CACHE[path]


def roc_auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    pos = np.asarray([p for p in pos if np.isfinite(p)], float)
    neg = np.asarray([n for n in neg if np.isfinite(n)], float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt)); np.add.at(s, inv, ranks); ranks = (s / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def raw_round_stats(rs, payloads: Dict[int, tuple], Cbar_prev: Optional[np.ndarray],
                    clients: int, mad_floor: float) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {
        "suspicion": {c: float(rs.per_client[c].get("suspicion", float("nan")))
                      for c in range(clients)}}
    for b in BASE_STATS:
        out[b] = {c: float("nan") for c in range(clients)}
    if Cbar_prev is not None and len(payloads) >= 3:
        innov = {cid: (payloads[cid][0] - Cbar_prev) for cid in payloads}
        base = score_round(innov)
        for b in BASE_STATS:
            out[b] = cohort_z_score(base.get(b, {}), mad_floor)
    return out


def combined_from_raw(raw: Dict[str, Dict[int, float]], norm: Dict[str, Sequence[float]],
                      clients: int) -> Dict[str, Dict[int, float]]:
    z = {c: {k: apply_norm(raw[k].get(c, float("nan")), norm[k]) for k in COMB_STATS}
         for c in range(clients)}
    out: Dict[str, Dict[int, float]] = {k: {c: z[c][k] for c in range(clients)}
                                        for k in COMB_STATS}
    for r in RULES:
        out["combined_" + r] = {c: combine_one(z[c], r) for c in range(clients)}
    return out


def score_one_archive(task: dict) -> dict:
    torch.set_num_threads(1)
    arch = load_archive(task["path"])
    clients = len(arch["C"])
    rounds = max(max(v) for v in arch["C"].values())
    attack_start = task["attack_start"]
    scope_start = task["scope_start"]
    victim = task["victim"]
    mad_floor = task["mad_floor"]
    honest = bool(task.get("honest"))
    norm = task.get("norm")
    n_attacked = rounds - attack_start + 1

    atk = None
    if not honest:
        cfg = CombinedAttackConfig(
            phi=task["phi"], beta=task["beta"], rule=task["rule"],
            knowledge=task["knowledge"], aim=task.get("aim", "mimic"),
            basis_steps=task["basis_steps"], full_steps=task["full_steps"],
            restarts=task["restarts"], warm_families=bool(task.get("warm", True)),
            mad_floor=mad_floor, schedule=task.get("schedule", "flat"), norm=norm)
        atk = CombinedAttacker(cfg, seed=1000 * task["seed"] + victim)

    aud = InnovationAuditor(mad_floor=mad_floor)
    raw_hist: List[Dict[str, Dict[int, float]]] = []
    rec_rounds: List[int] = []
    mats: Dict[str, List[List[float]]] = {k: [] for k in ALL_STATS}
    pred, real = [], []
    fid, rel_err, budget_used, innov_ratio, agg_dmg = [], [], [], [], []
    statuses: Dict[str, int] = {}
    prev_peer_med = float("nan")
    prev_peer_vals: List[float] = []
    rng = np.random.default_rng(7919 * task["seed"] + victim)

    for r in range(1, rounds + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        present_r = {cid: arch["present"][cid][r] for cid in range(clients)}
        payloads = {cid: (arch["C"][cid][r], present_r[cid]) for cid in range(clients)}
        info = None
        if (not honest) and r >= attack_start:
            peer_C = ({j: arch["C"][j][r] for j in range(clients) if j != victim}
                      if task["knowledge"] == "peer_current" else None)
            aim = 0.0
            mode = task.get("aim", "mimic")
            if mode == "mimic" and np.isfinite(prev_peer_med):
                aim = float(prev_peer_med)
            elif mode == "mimicj" and prev_peer_vals:
                aim = float(rng.choice(prev_peer_vals))
            Cb = aud.cbar.get(r - 1)
            if task["beta"] > 0:
                mult = schedule_factor(task.get("schedule", "flat"), r - attack_start, n_attacked)
                Cp, info = atk.craft(aud, r, victim, arch["C"][victim][r], present_r, peer_C,
                                     aim=aim, beta_mult=mult)
            else:
                Cp = (arch["C"][victim][r].copy() if Cb is None
                      else atk._cheap_objective(Cb, arch["C"][victim][r]))
                Cp = np.round(Cp * FIXED_POINT_SCALE) / FIXED_POINT_SCALE
                info = {"status": "cheap_only", "predicted_combined": float("nan")}
            payloads[victim] = (Cp, present_r[victim])
            statuses[info["status"]] = statuses.get(info["status"], 0) + 1
            if Cb is not None:
                T = (np.asarray(present_r[victim], bool) & aud.hist[victim][-1].present
                     if aud.hist.get(victim) else np.asarray(present_r[victim], bool))
                Ih = (arch["C"][victim][r] - Cb)[T]
                Ip = (Cp - Cb)[T]
                nh, npb = float(np.linalg.norm(Ih)), float(np.linalg.norm(Ip))
                if nh > EPS and npb > EPS:
                    fid.append(float((Ih * Ip).sum() / (nh * npb)))
                    innov_ratio.append(npb / nh)
                    rel_err.append(float(np.linalg.norm(Cp - arch["C"][victim][r])) / nh)
                if r in arch["cbar"]:
                    step = float(np.linalg.norm(arch["cbar"][r] - Cb))
                    if step > EPS:
                        agg_dmg.append(float(np.linalg.norm(Cp - arch["C"][victim][r]))
                                       / clients / step)
                if np.isfinite(info.get("budget", float("nan"))) and info.get("budget", 0) > 0:
                    Cc = atk._cheap_objective(Cb, arch["C"][victim][r])
                    budget_used.append(float(np.linalg.norm(Cp - Cc)) / info["budget"])

        rs = aud.observe_round(r, payloads)
        raw = raw_round_stats(rs, payloads, aud.cbar.get(r - 1), clients, mad_floor)
        raw_hist.append(raw)

        if honest:
            if r >= scope_start:
                rec_rounds.append(r)
            continue

        comb = combined_from_raw(raw, norm, clients)
        key = "combined_" + task["rule"]
        vals = [comb[key][c] for c in range(clients)
                if c != victim and np.isfinite(comb[key][c])]
        prev_peer_med = float(np.median(vals)) if vals else float("nan")
        prev_peer_vals = list(vals)
        if r < scope_start:
            continue
        rec_rounds.append(r)
        for k in ALL_STATS:
            mats[k].append([comb[k][c] for c in range(clients)])
        if r >= attack_start and info is not None:
            pred.append(info.get("predicted_combined", float("nan")))
            real.append(float(comb[key][victim]))

    out = {"key": {k: task.get(k) for k in ("seed", "victim", "phi", "beta", "rule",
                                            "knowledge", "aim", "schedule", "honest")},
           "rounds": rec_rounds, "victim": victim}
    if honest:
        out["raw"] = {k: np.asarray([[raw_hist[r - 1][k][c] for c in range(clients)]
                                     for r in rec_rounds], float) for k in COMB_STATS}
        return out
    pr = np.asarray(pred, float); re_ = np.asarray(real, float)
    ok = np.isfinite(pr) & np.isfinite(re_)
    out["mat"] = {k: np.asarray(v, float) for k, v in mats.items()}
    out["diag"] = {
        "pred_vs_real_max_abs": float(np.max(np.abs(pr[ok] - re_[ok]))) if ok.any() else None,
        "pred_vs_real_mean_abs": float(np.mean(np.abs(pr[ok] - re_[ok]))) if ok.any() else None,
        "fidelity_mean": float(np.mean(fid)) if fid else None,
        "innov_norm_ratio_mean": float(np.mean(innov_ratio)) if innov_ratio else None,
        "payload_err_rel_mean": float(np.mean(rel_err)) if rel_err else None,
        "budget_used_mean": float(np.mean(budget_used)) if budget_used else None,
        "agg_damage_mean": float(np.mean(agg_dmg)) if agg_dmg else None,
        "statuses": statuses}
    return out


def fit_threshold(vals: np.ndarray, target: float) -> float:
    v = np.where(np.isfinite(vals), vals, -np.inf).ravel()
    n = v.size
    if n == 0:
        return float("inf")
    m = int(math.floor(target * n))
    s = np.sort(v)
    idx = n - 1 - m
    return float(s[idx]) if idx >= 0 else float("-inf")


def summarise_scores(passes: List[dict], attack_start: int, clients: int,
              taus: Dict[str, float], test_seeds: Sequence[int]) -> dict:
    diag_keys = ["fidelity_mean", "innov_norm_ratio_mean", "payload_err_rel_mean",
                 "budget_used_mean", "agg_damage_mean", "pred_vs_real_max_abs",
                 "pred_vs_real_mean_abs"]
    acc = {k: [] for k in diag_keys}
    per_stat: Dict[str, dict] = {}
    for stat in ALL_STATS:
        pos, neg = [], []
        mpos, mneg, spos, sneg = [], [], [], []
        tp_k = tp_n = run_k = run_n = 0
        for p in passes:
            rounds = np.asarray(p["rounds"])
            sel = rounds >= attack_start
            if not sel.any():
                continue
            mat = p["mat"][stat][sel]
            v = p["victim"]
            pos.extend(mat[:, v].tolist())
            mask = np.ones(clients, bool); mask[v] = False
            neg.extend(mat[:, mask].ravel().tolist())
            with np.errstate(all="ignore"):
                mpos.append(float(np.nanmean(mat[:, v]))); spos.append(float(np.nanstd(mat[:, v])))
                for c in range(clients):
                    if c == v:
                        continue
                    mneg.append(float(np.nanmean(mat[:, c]))); sneg.append(float(np.nanstd(mat[:, c])))
            if p["key"]["seed"] in test_seeds:
                f = np.where(np.isfinite(mat[:, v]), mat[:, v], -np.inf) > taus[stat]
                tp_k += int(f.sum()); tp_n += int(f.size)
                run_n += 1; run_k += int(bool(f.any()))
        per_stat[stat] = {"auc": roc_auc(pos, neg), "n_pos": len(pos), "n_neg": len(neg),
                          "auc_run_mean": roc_auc(mpos, mneg),
                          "auc_run_std_low": roc_auc([-s for s in spos], [-s for s in sneg]),
                          "pos_median": float(np.nanmedian(pos)) if pos else None,
                          "neg_median": float(np.nanmedian(neg)) if neg else None,
                          "tpr_client_round_TEST": (tp_k / tp_n) if tp_n else None,
                          "tpr_client_run_TEST": (run_k / run_n) if run_n else None,
                          "n_test_cells": tp_n, "n_test_runs": run_n}
    for p in passes:
        for k in diag_keys:
            if p["diag"].get(k) is not None:
                acc[k].append(p["diag"][k])
    return {"stats": per_stat, **{k: (float(np.mean(v)) if v else None) for k, v in acc.items()}}


def run_all(tasks: List[dict], workers: int, t0: float, label: str) -> List[dict]:
    results: List[dict] = []
    if workers <= 1:
        for i, t in enumerate(tasks):
            results.append(score_one_archive(t))
            print("  %s %d/%d  %.1fs" % (label, i + 1, len(tasks), time.time() - t0), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(score_one_archive, tasks, chunksize=1)):
                results.append(r)
                if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                    print("  %s %d/%d  %.1fs" % (label, i + 1, len(tasks), time.time() - t0),
                          flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", default=ARCHIVE_DIR)
    ap.add_argument("--prefix", default="arch_cifar100_s")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--cal-seeds", default="0,1")
    ap.add_argument("--test-seeds", default="3,4")
    ap.add_argument("--phis", default="0,0.1,0.25,0.5,1.0")
    ap.add_argument("--betas", default="0,0.25,0.5,1.0,2.0")
    ap.add_argument("--cells", default="", help="explicit phi:beta pairs, comma separated, "
                                                "overrides the phis x betas product")
    ap.add_argument("--rules", default="max")
    ap.add_argument("--knowledge", default="peer_current")
    ap.add_argument("--aims", default="mimic")
    ap.add_argument("--schedules", default="flat")
    ap.add_argument("--attack-start", type=int, default=12)
    ap.add_argument("--scope-start", type=int, default=12)
    ap.add_argument("--mad-floor", type=float, default=1e-2)
    ap.add_argument("--fpr-target", type=float, default=0.01)
    ap.add_argument("--basis-steps", type=int, default=120)
    ap.add_argument("--full-steps", type=int, default=250)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--victims", default="")
    ap.add_argument("--workers", type=int, default=min(30, (os.cpu_count() or 4)))
    ap.add_argument("--norm-in", default="", help="reuse CAL constants and thresholds from a "
                                                 "previous run instead of refitting")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "results", "combined_eval.json"))
    a = ap.parse_args()

    ints = lambda s: [int(x) for x in s.split(",") if x.strip()]
    fls = lambda s: [float(x) for x in s.split(",") if x.strip()]
    seeds, cal_seeds, test_seeds = ints(a.seeds), ints(a.cal_seeds), ints(a.test_seeds)
    knows = [k.strip() for k in a.knowledge.split(",") if k.strip()]
    rules = [k.strip() for k in a.rules.split(",") if k.strip()]
    aims = [k.strip() for k in a.aims.split(",") if k.strip()]
    scheds = [k.strip() for k in a.schedules.split(",") if k.strip()]

    paths = {}
    for fn in sorted(os.listdir(a.archives)):
        if fn.startswith(a.prefix) and fn.endswith(".npz"):
            m = re.search(r"_s(\d+)\.npz$", fn)
            if m:
                paths[int(m.group(1))] = os.path.join(a.archives, fn)
    seeds = [s for s in seeds if s in paths]
    if not seeds:
        print("no archives found"); return 1
    clients = len(load_archive(paths[seeds[0]])["C"])
    victims = ints(a.victims) if a.victims else list(range(clients))
    common = {"attack_start": a.attack_start, "scope_start": a.scope_start,
              "mad_floor": a.mad_floor, "basis_steps": a.basis_steps,
              "full_steps": a.full_steps, "restarts": a.restarts,
              "warm": not a.no_warm}
    t0 = time.time()

    if a.norm_in:
        with open(a.norm_in) as f:
            prev = json.load(f)
        norm = {k: tuple(v) for k, v in prev["norm"].items()}
        taus = dict(prev["thresholds"])
        fpr = prev.get("honest_fpr_TEST", {})
        print("reused CAL constants from %s" % a.norm_in)
    else:
        htasks = [dict(common, path=paths[s], seed=s, victim=0, phi=1.0, beta=0.0,
                       rule=rules[0], knowledge=knows[0], aim="mimic", schedule="flat",
                       honest=True) for s in seeds]
        hres = run_all(htasks, min(a.workers, len(htasks)), t0, "honest")
        cal_h = [r for r in hres if r["key"]["seed"] in cal_seeds]
        test_h = [r for r in hres if r["key"]["seed"] in test_seeds]
        norm = {}
        for k in COMB_STATS:
            v = np.concatenate([r["raw"][k].ravel() for r in cal_h])
            v = v[np.isfinite(v)]
            med = float(np.median(v))
            sc = MAD_SCALE * float(np.median(np.abs(v - med))) + 1e-9
            norm[k] = (med, sc)
        def comb_mat(res):
            out = {k: [] for k in ALL_STATS}
            for p in res:
                nr = p["raw"]["suspicion"].shape[0]
                for i in range(nr):
                    raw = {k: {c: float(p["raw"][k][i, c]) for c in range(clients)}
                           for k in COMB_STATS}
                    cm = combined_from_raw(raw, norm, clients)
                    for k in ALL_STATS:
                        out[k].append([cm[k][c] for c in range(clients)])
            return {k: np.asarray(v, float) for k, v in out.items()}
        calM, tstM = comb_mat(cal_h), comb_mat(test_h)
        taus, fpr = {}, {}
        for k in ALL_STATS:
            taus[k] = fit_threshold(calM[k], a.fpr_target)
            v = tstM[k]
            f = np.where(np.isfinite(v), v, -np.inf) > taus[k]
            fpr[k] = {"realised": float(f.mean()), "n": int(f.size)}
        print("\nCAL constants (median, robust scale) from honest CAL client-rounds "
              "(seeds %s, n=%d):" % (cal_seeds, calM["combined_max"].size))
        for k in COMB_STATS:
            print("   %-18s center %9.4f  scale %9.4f" % (k, norm[k][0], norm[k][1]))
        print("frozen thresholds at nominal %g%% and the realised honest FPR on TEST %s:"
              % (100 * a.fpr_target, test_seeds))
        for k in ALL_STATS:
            print("   %-18s tau %9.4f   FPR %.3f%%  (n=%d)"
                  % (k, taus[k], 100 * fpr[k]["realised"], fpr[k]["n"]))

    if a.cells:
        cellsq = []
        for tok in a.cells.split(","):
            p, b = tok.split(":")
            cellsq.append((float(p), float(b)))
    else:
        cellsq = [(p, b) for p in fls(a.phis) for b in fls(a.betas)]
    tasks = []
    for rule, aim, sch, know, (phi, beta), s, v in itertools.product(
            rules, aims, scheds, knows, cellsq, seeds, victims):
        if beta == 0.0 and (know != knows[0] or aim != aims[0] or sch != scheds[0]
                            or rule != rules[0]):
            continue
        tasks.append(dict(common, path=paths[s], seed=s, victim=v, phi=phi, beta=beta,
                          rule=rule, knowledge=know, aim=aim, schedule=sch, honest=False,
                          norm=norm))
    tasks.sort(key=lambda t: (t["path"],))
    print("\n%d attack passes over %d cells x %d seeds x %d victims"
          % (len(tasks), max(1, len(tasks) // max(1, len(seeds) * len(victims))),
             len(seeds), len(victims)))
    results = run_all(tasks, a.workers, t0, "attack")

    cells: Dict[str, List[dict]] = {}
    for r in results:
        k = "%s|%s|%s|%s|phi=%g|beta=%g" % (r["key"]["rule"], r["key"]["aim"],
                                            r["key"]["schedule"], r["key"]["knowledge"],
                                            r["key"]["phi"], r["key"]["beta"])
        cells.setdefault(k, []).append(r)
    out = {"config": vars(a), "clients": clients, "seeds": seeds, "cal_seeds": cal_seeds,
           "test_seeds": test_seeds, "norm": {k: list(v) for k, v in norm.items()},
           "thresholds": taus, "honest_fpr_TEST": fpr, "cells": {}}
    for k, ps in sorted(cells.items()):
        out["cells"][k] = summarise_scores(ps, a.attack_start, clients, taus, test_seeds)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(os.path.abspath(a.out), "w") as f:
        json.dump(out, f, indent=1, default=float)
    report(out)
    print("\nwrote %s   (%.1fs)" % (os.path.abspath(a.out), time.time() - t0))
    return 0


def report(out: dict) -> None:
    f = lambda x, s="%8.3f": (s % x) if x is not None and np.isfinite(x) else "       -"
    pc = lambda x: ("%7.1f%%" % (100 * x)) if x is not None and np.isfinite(x) else "       -"
    print("\n" + "=" * 150)
    print("COMBINED-SCORE WHITE-BOX ATTACK. Thresholds and normalisation frozen on CAL "
          "honest client-rounds.")
    print("=" * 150)
    hdr = ("%-34s | %6s %8s %8s | %6s %6s | %7s %8s %8s %6s %7s | %8s"
           % ("rule|aim|sched|know|phi|beta", "AUCmax", "TPR c-r", "TPR run", "AUCsum",
              "AUCtwo", "fidelity", "|I|/|Ih|", "err/|Ih|", "Bused", "aggdmg", "pred-real"))
    print(hdr); print("-" * len(hdr))
    for k, v in out["cells"].items():
        s = v["stats"]
        print("%-34s | %6.3f %s %s | %6.3f %6.3f | %7s %8s %8s %6s %7s | %8s"
              % (k, s["combined_max"]["auc"], pc(s["combined_max"]["tpr_client_round_TEST"]),
                 pc(s["combined_max"]["tpr_client_run_TEST"]),
                 s["combined_sum"]["auc"], s["combined_two"]["auc"],
                 f(v["fidelity_mean"], "%7.3f"), f(v["innov_norm_ratio_mean"]),
                 f(v["payload_err_rel_mean"]), f(v["budget_used_mean"], "%6.2f"),
                 f(v["agg_damage_mean"], "%7.3f"), f(v["pred_vs_real_mean_abs"], "%8.1e")))
    print("\nPer-family AUC on the SAME crafted payloads (CAL-normalised, larger = "
          "more suspicious):")
    bh = "%-34s |" % "rule|aim|sched|know|phi|beta" + "".join("%13s" % b[:12] for b in COMB_STATS)
    print(bh); print("-" * len(bh))
    for k, v in out["cells"].items():
        print("%-34s |" % k + "".join("%13.3f" % v["stats"][b]["auc"] for b in COMB_STATS))


if __name__ == "__main__":
    sys.exit(main())
