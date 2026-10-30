from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from passportfl.baselines import BASELINES, score_round
from passportfl.innovation import InnovationAuditor, SCORE_STRUCTURAL
from passportfl.reference_baselines import FoolsGold
from passportfl.temporal_baselines import (
    TemporalBaselineAuditor, MODES, BASE_STATS, family_of,
    stat_names as tb_stat_names,
)
import experiments.protocol_eval as P


ORIENT = {"u1": None, "u2": None, "g": -1.0, "x": +1.0, "rep": +1.0}

ABLATIONS: Dict[str, Tuple[Tuple[str, ...], bool]] = {
    "abl_full":                (("u1", "u2", "g", "x", "rep"), True),
    "abl_full_nostruct":       (("u1", "u2", "g", "x", "rep"), False),
    "abl_u1g":                 (("u1", "g"), True),
    "abl_u1g_nostruct":        (("u1", "g"), False),
    "abl_u1":                  (("u1",), False),
    "abl_u2":                  (("u2",), False),
    "abl_g":                   (("g",), False),
    "abl_x":                   (("x",), False),
    "abl_rep":                 (("rep",), False),
    "abl_u1u2":                (("u1", "u2"), False),
    "abl_u1u2g":               (("u1", "u2", "g"), True),
    "abl_u1u2gx":              (("u1", "u2", "g", "x"), True),
    "abl_x_rep":               (("x", "rep"), True),
    "abl_x_rep_nostruct":      (("x", "rep"), False),
    "abl_no_u1":               (("u2", "g", "x", "rep"), True),
    "abl_no_u2":               (("u1", "g", "x", "rep"), True),
    "abl_no_g":                (("u1", "u2", "x", "rep"), True),
    "abl_no_x":                (("u1", "u2", "g", "rep"), True),
    "abl_no_rep":              (("u1", "u2", "g", "x"), True),
    "abl_memoryless":          (("g",), True),
    "abl_ownhist":             (("u1", "u2", "rep"), True),
    "abl_crosshist":           (("x",), True),
    "abl_struct_only":         ((), True),
}
ABL_NAMES = list(ABLATIONS)


def combine_scores(z: Dict[str, float], keys: Sequence[str], structural: bool) -> float:
    if z.get("undefined"):
        return float("nan")
    if structural and z.get("degenerate"):
        return SCORE_STRUCTURAL
    parts = []
    for k in keys:
        zz = z.get(f"z_{k}", float("nan"))
        if np.isfinite(zz):
            o = ORIENT[k]
            parts.append(abs(zz) if o is None else o * zz)
    if parts:
        return max(parts)
    return 0.0 if structural and not keys else float("nan")


def temporal_raw_names() -> List[str]:
    return tb_stat_names()


def all_stat_names() -> List[str]:
    ours = ["suspicion"] + ABL_NAMES
    byz = list(BASELINES) + [f"{b}_z" for b in BASELINES]
    tb = temporal_raw_names()
    return ours + byz + tb + [f"{t}_z" for t in tb]


FG_VARIANTS = {"fg_cum": 0.0, "fg_cum_imp10": 0.10}


def peer_raw_names() -> List[str]:
    return list(FG_VARIANTS) + [f"{k}_z" for k in FG_VARIANTS]


def score_one_archive_temporal(task: dict) -> dict:
    path = task["path"]
    attack = task.get("attack")
    victim = task.get("victim")
    mad_floor = task["mad_floor"]
    scope_start = task["scope_start"]
    attack_start = task["attack_start"]
    seed = task["seed"]

    base_arch = P.load_archive(path)
    clients, rounds = P.archive_shape(base_arch)
    arch = dict(base_arch)
    arch["published"] = {i: {} for i in range(clients)}

    fn = P.ATTACKS[attack] if attack else None
    rng = np.random.default_rng([int(seed), int(victim) + 1 if victim is not None else 0,
                                 P.ATTACK_ORDER.index(attack) + 1 if attack else 0])

    aud = InnovationAuditor(mad_floor=mad_floor,
                            mask_policy=task.get("mask_policy", "declared"))
    tb = TemporalBaselineAuditor(burnin_end=attack_start, seed=int(seed))
    peer = bool(task.get("peer_aware", False))
    fg = {k: FoolsGold(importance=imp) for k, imp in FG_VARIANTS.items()} if peer else {}

    names = all_stat_names() + (peer_raw_names() if peer else [])
    tb_names = temporal_raw_names()
    rec_rounds: List[int] = []
    mats = {k: [] for k in names}

    for r in range(1, rounds + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
            tb.set_aggregate(r - 1, arch["cbar"][r - 1])
        payloads = {}
        for cid in range(clients):
            C = arch["C"][cid][r]
            if fn is not None and cid == victim and r >= attack_start:
                C = fn(arch, cid, r, rng)
            arch["published"][cid][r] = C
            payloads[cid] = (C, arch["present"][cid][r])

        rs = aud.observe_round(r, payloads)
        tbs = tb.observe_round(r, payloads)

        base: Dict[str, Dict[int, float]] = {}
        fgs: Dict[str, Dict[int, float]] = {}
        Cbar_prev = arch["cbar"].get(r - 1)
        if Cbar_prev is not None:
            innov = {cid: (payloads[cid][0] - Cbar_prev) for cid in payloads}
            if len(innov) >= 3:
                base = score_round(innov)
            for k, f in fg.items():
                fgs[k] = f.update(innov)

        if r < scope_start:
            continue
        rec_rounds.append(r)
        row = {k: [float("nan")] * clients for k in names}
        for cid, v in rs.per_client.items():
            row["suspicion"][cid] = float(v.get("suspicion", float("nan")))
            for nm in ABL_NAMES:
                keys, st = ABLATIONS[nm]
                row[nm][cid] = combine_scores(v, keys, st)
        for bname, vals in base.items():
            for cid, v in vals.items():
                row[bname][cid] = float(v) if np.isfinite(v) else float("nan")
            zz = P.robust_z_score({c: vals.get(c, float("nan")) for c in range(clients)}, mad_floor)
            for cid, v in zz.items():
                row[f"{bname}_z"][cid] = float(v)
        for nm in tb_names:
            vals = {cid: tbs[cid].get(nm, float("nan")) for cid in range(clients)}
            for cid, v in vals.items():
                row[nm][cid] = float(v)
            zz = P.robust_z_score(vals, mad_floor)
            for cid, v in zz.items():
                row[f"{nm}_z"][cid] = float(v)
        for k, vals in fgs.items():
            for cid, v in vals.items():
                row[k][cid] = float(v) if np.isfinite(v) else float("nan")
            zz = P.robust_z_score({c: vals.get(c, float("nan")) for c in range(clients)}, mad_floor)
            for cid, v in zz.items():
                row[f"{k}_z"][cid] = float(v)
        for k in names:
            mats[k].append(row[k])

    return {"key": task["key"], "rounds": rec_rounds,
            "mats": {k: np.asarray(v, dtype=np.float64) for k, v in mats.items()}}


def run_tasks(tasks: List[dict], workers: int) -> List[dict]:
    if not tasks:
        return []
    tasks = sorted(tasks, key=lambda t: t["path"])
    if workers <= 1 or len(tasks) <= 2:
        return [score_one_archive_temporal(t) for t in tasks]
    chunk = max(1, len(tasks) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(score_one_archive_temporal, tasks, chunksize=chunk))


def add_absolute_z(res: List[dict]) -> List[str]:
    added = []
    for t in temporal_raw_names():
        added.append(f"{t}_zabs")
    for p in res:
        for t in temporal_raw_names():
            p["mats"][f"{t}_zabs"] = np.abs(p["mats"][f"{t}_z"])
    return added


ENSEMBLES = {
    "ens_ar": "forecast residuals only (AR1, AR2, EWMA on both sequences, both regimes)",
    "ens_stateful_causal": "every stateful baseline, CAUSAL fits only (no oracle window)",
    "ens_stateful_burnin": "every stateful baseline, BURN-IN fits only (oracle clean window)",
    "ens_stateful": "every stateful temporal baseline, both fit regimes",
    "ens_stateless": "every memoryless Byzantine rule",
    "ens_all": "stateful + stateless, everything computable from this archive",
}
AR_FAMILY = ("ar1", "ar1_std", "ar1_two", "ar2", "ar2_std", "ar2_two",
             "ewma", "ewma_std", "ewma_two", "par1", "par1_std", "par1_two",
             "par2", "par2_std", "par2_two", "pewma", "pewma_std", "pewma_two")


def add_ensembles(res: List[dict]) -> List[str]:
    tb_cols = [f"{t}_zabs" for t in temporal_raw_names()]
    byz_cols = [f"{b}_z" for b in BASELINES]
    members = {
        "ens_ar": [c for c in tb_cols
                   if any(c.startswith(f"tb_{f}_") for f in AR_FAMILY)],
        "ens_stateful_causal": [c for c in tb_cols if "_causal_" in c],
        "ens_stateful_burnin": [c for c in tb_cols if "_burnin_" in c],
        "ens_stateful": tb_cols,
        "ens_stateless": byz_cols,
        "ens_all": tb_cols + byz_cols,
    }
    for p in res:
        for name, cols in members.items():
            stk = np.stack([np.abs(p["mats"][c]) for c in cols])
            ok = np.isfinite(stk)
            stk = np.where(ok, stk, -np.inf)
            v = stk.max(axis=0)
            p["mats"][name] = np.where(ok.any(axis=0), v, np.nan)
    return list(members)


PEER_ENSEMBLES = ("ens_stateful_x", "ens_stateful_fg", "ens_stateful_x_rep", "ens_all_fg")


def add_peer_ensembles(res: List[dict]) -> List[str]:
    tb_cols = [(f"{t}_zabs", True) for t in temporal_raw_names()]
    byz_cols = [(f"{b}_z", True) for b in BASELINES]
    fg_cols = [(f"{k}_z", True) for k in FG_VARIANTS]
    members = {
        "ens_stateful_x": tb_cols + [("abl_x", False)],
        "ens_stateful_fg": tb_cols + fg_cols,
        "ens_stateful_x_rep": tb_cols + [("abl_x", False), ("abl_rep", False)],
        "ens_all_fg": tb_cols + byz_cols + fg_cols,
    }
    assert list(members) == list(PEER_ENSEMBLES)
    for p in res:
        for name, cols in members.items():
            stk = np.stack([np.abs(p["mats"][c]) if ab else p["mats"][c] for c, ab in cols])
            ok = np.isfinite(stk)
            stk = np.where(ok, stk, -np.inf)
            v = stk.max(axis=0)
            p["mats"][name] = np.where(ok.any(axis=0), v, np.nan)
    return list(members)


def portable_path(v):
    if isinstance(v, str) and os.path.isabs(v):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            return os.path.relpath(v, root).replace(os.sep, "/")
        except ValueError:
            return v
    return v


def save_score_matrices(path: str, res: List[dict], meta: dict) -> dict:
    order = sorted(res, key=lambda p: (p["key"]["seed"],
                                       -1 if p["key"]["attack"] is None
                                       else P.ATTACK_ORDER.index(p["key"]["attack"]),
                                       -1 if p["key"]["victim"] is None else p["key"]["victim"]))
    rounds = list(order[0]["rounds"])
    cols = sorted(order[0]["mats"])
    for p in order:
        assert list(p["rounds"]) == rounds, "passes differ in scored rounds"
        assert sorted(p["mats"]) == cols, "passes differ in columns"
    arrays = {
        "columns": np.asarray(cols),
        "seed": np.asarray([p["key"]["seed"] for p in order], dtype=np.int64),
        "victim": np.asarray([-1 if p["key"]["victim"] is None else p["key"]["victim"]
                             for p in order], dtype=np.int64),
        "attack": np.asarray([-1 if p["key"]["attack"] is None
                              else P.ATTACK_ORDER.index(p["key"]["attack"])
                              for p in order], dtype=np.int64),
        "attack_names": np.asarray(P.ATTACK_ORDER),
        "pool": np.asarray([p["key"]["pool"] for p in order]),
        "rounds": np.asarray(rounds, dtype=np.int64),
        "clients": np.arange(order[0]["mats"][cols[0]].shape[1], dtype=np.int64),
        "meta_json": np.asarray(json.dumps(meta, sort_keys=True, default=float)),
    }
    h = hashlib.sha256()
    for k in ("columns", "seed", "victim", "attack", "pool", "rounds"):
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    for c in cols:
        a = np.stack([p["mats"][c] for p in order]).astype(np.float64)
        arrays[f"s__{c}"] = a
        h.update(c.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **arrays)
    fh = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            fh.update(chunk)
    return {"file": os.path.basename(path), "bytes": os.path.getsize(path),
            "sha256": fh.hexdigest(), "content_sha256": h.hexdigest(),
            "passes": len(order), "rounds": rounds, "columns": len(cols)}


def auc_table(res: List[dict], seeds: Sequence[int], stats: Sequence[str],
              attack_start: int, excl_accomplice: bool = False) -> dict:
    out: Dict[str, Dict[str, dict]] = {}
    for a in P.ATTACK_ORDER:
        per_seed_vals = {s: {} for s in seeds}
        for s in seeds:
            ps = [p for p in res if p["key"]["seed"] == s and p["key"]["attack"] == a]
            if not ps:
                continue
            for stat in stats:
                pos, neg = [], []
                for p in ps:
                    m = p["mats"][stat]
                    if not m.size:
                        continue
                    sel = np.asarray(p["rounds"]) >= attack_start
                    m = m[sel]
                    v = p["key"]["victim"]
                    drop = [v]
                    if excl_accomplice:
                        acc = P.accomplice(a, v, m.shape[1])
                        if acc is not None:
                            drop.append(acc)
                    pos.append(m[:, v])
                    neg.append(np.delete(m, drop, axis=1).ravel())
                if pos:
                    per_seed_vals[s][stat] = P.roc_auc(np.concatenate(pos),
                                                        np.concatenate(neg))
        row = {}
        for stat in stats:
            vals = [per_seed_vals[s][stat] for s in seeds
                    if stat in per_seed_vals[s] and np.isfinite(per_seed_vals[s][stat])]
            row[stat] = {"mean": float(np.mean(vals)) if vals else float("nan"),
                         "std": float(np.std(vals)) if vals else float("nan"),
                         "n_seeds": len(vals),
                         "per_seed": {str(s): per_seed_vals[s].get(stat) for s in seeds}}
        out[a] = row
    return out


def worst_case(auc: dict, stat: str) -> Tuple[float, str, int]:
    vals = {a: auc[a][stat]["mean"] for a in auc if stat in auc[a]}
    fin = {a: v for a, v in vals.items() if np.isfinite(v)}
    undef = len(vals) - len(fin)
    if not fin:
        return float("nan"), "", undef
    a = min(fin, key=lambda k: fin[k])
    return fin[a], a, undef


def horizontal_rule(c: str = "=", n: int = 118) -> str:
    return c * n


def print_auc_block(title: str, auc: dict, stats: Sequence[str], labels: Dict[str, str]) -> None:
    print("\n" + horizontal_rule())
    print(title)
    print(horizontal_rule())
    w = max(len(labels.get(s, s)) for s in stats) + 2
    head = "%-*s" % (w, "detector") + "".join("%14s" % a[:13] for a in P.ATTACK_ORDER) \
        + "%14s%10s" % ("WORST", "undef")
    print(head)
    print("-" * len(head))
    for s in stats:
        wc, wa, ud = worst_case(auc, s)
        line = "%-*s" % (w, labels.get(s, s))
        for a in P.ATTACK_ORDER:
            v = auc[a][s]["mean"] if s in auc[a] else float("nan")
            line += ("%14.3f" % v) if np.isfinite(v) else "%14s" % "n/a"
        line += ("%14.3f" % wc) if np.isfinite(wc) else "%14s" % "n/a"
        line += "%10d" % ud
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", default=P.ARCHIVE_DIR)
    ap.add_argument("--archive-prefix", default="arch_cifar100_s")
    ap.add_argument("--cal", default="0,1")
    ap.add_argument("--dev", default="2")
    ap.add_argument("--test", default="3,4")
    ap.add_argument("--attack-start", type=int, default=12)
    ap.add_argument("--eval-from", type=int, default=12)
    ap.add_argument("--mask-policy", default="declared", choices=["declared", "full", "derived"],
                    help="which rows the auditor scores: the client's declared mask (what "
                         "the default), every row, or the "
                         "support derived from the payload")
    ap.add_argument("--workers", type=int, default=min(12, (os.cpu_count() or 4)))
    ap.add_argument("--mad-floor", type=float, default=None,
                    help="skip stage 0 and use this floor (default: refit on CAL honest)")
    ap.add_argument("--out", default=os.path.abspath(os.path.join(
        os.path.dirname(P.ARCHIVE_DIR), "temporal_eval.json")))
    ap.add_argument("--peer-aware", action="store_true",
                    help="also replay FoolsGold's cumulative score and add the peer-aware "
                         "ensembles (stateful + x, + FoolsGold, + x and rep; all + FoolsGold)")
    ap.add_argument("--save-scores", default=None, metavar="PATH",
                    help="write every per-client-round score matrix to this .npz file")
    a = ap.parse_args()

    def ints(s):
        return [int(x) for x in s.split(",") if x.strip() != ""]

    cal, dev, test = ints(a.cal), ints(a.dev), ints(a.test)
    paths = P.discover_archives(a.archives, a.archive_prefix)
    if not paths:
        print(f"no archives in {a.archives} with prefix {a.archive_prefix}")
        return 1
    seeds = sorted(paths)
    clients, rounds = P.archive_shape(P.load_archive(paths[seeds[0]]))
    scope_start = max(a.eval_from, a.attack_start)
    t_all = time.time()

    print(horizontal_rule())
    print("STATEFUL TEMPORAL BASELINES: head-to-head and ablation")
    print(horizontal_rule())
    print("  archives      : %s" % ", ".join(os.path.basename(paths[s]) for s in seeds))
    print("  cohort        : %d clients, %d rounds, scope r >= %d, attack onset r = %d"
          % (clients, rounds, scope_start, a.attack_start))
    chk = P.check_attack_definitions()
    print("  attack defs   : %s" % chk["detail"])
    print("  pools         : CAL %s | DEV %s | TEST %s" % (cal, dev, test))

    if a.mad_floor is not None:
        mad_floor = a.mad_floor
        print("  mad floor     : %.0e (supplied)" % mad_floor)
    else:
        grid_tasks = []
        for f in P.MAD_FLOOR_GRID:
            tt = P.build_tasks(paths, cal, f, scope_start, a.attack_start,
                               {s: clients for s in cal}, False, "CAL")
            for t in tt:
                t["key"] = dict(t["key"], floor=f)
            grid_tasks += tt
        gres = P.run_tasks(grid_tasks, a.workers)
        cal_honest = {f: [] for f in P.MAD_FLOOR_GRID}
        for r in gres:
            cal_honest[r["key"]["floor"]].append(r)
        sel = P.select_mad_floor(cal_honest, P.TARGETS)
        mad_floor = sel["selected"]
        print("  mad floor     : %.0e (selected on CAL honest only, %s)"
              % (mad_floor, sel["criterion"]))

    tasks = []
    for pool, ss in (("CAL", cal), ("DEV", dev), ("TEST", test)):
        tasks += P.build_tasks(paths, [s for s in ss if s in paths], mad_floor, scope_start,
                               a.attack_start, {s: clients for s in ss}, True, pool)
    for t in tasks:
        t["mask_policy"] = a.mask_policy
        if a.peer_aware:
            t["peer_aware"] = True
    print("  replay passes : %d (%d workers), mask policy %s"
          % (len(tasks), a.workers, a.mask_policy))
    t0 = time.time()
    res = run_tasks(tasks, a.workers)
    print("  replay done   : %.1fs" % (time.time() - t0))
    zabs = add_absolute_z(res)
    ens = add_ensembles(res)
    peer_cols: List[str] = []
    if a.peer_aware:
        peer_cols = peer_raw_names()
        ens = ens + add_peer_ensembles(res)
        print("  peer-aware    : FoolsGold %s; ensembles %s"
              % (", ".join(FG_VARIANTS), ", ".join(PEER_ENSEMBLES)))

    bad = 0
    tot = 0
    for p in res:
        x, y = p["mats"]["suspicion"], p["mats"]["abl_full"]
        m = np.isfinite(x) | np.isfinite(y)
        tot += int(m.sum())
        bad += int((np.nan_to_num(x, nan=-1.0) != np.nan_to_num(y, nan=-1.0)).sum())
    print("  integrity     : abl_full vs suspicion, %d/%d cells differ (must be 0)" % (bad, tot))
    if bad:
        print("  ABORT: the ablation does not reproduce the detector it ablates.")
        return 2

    tb_raw = temporal_raw_names()
    tb_all = tb_raw + [f"{t}_z" for t in tb_raw] + zabs
    byz_all = list(BASELINES) + [f"{b}_z" for b in BASELINES]
    stats = ["suspicion"] + ABL_NAMES + byz_all + tb_all + ens + peer_cols

    auc = auc_table(res, seeds, stats, a.attack_start)
    auc_ex = auc_table(res, seeds, stats, a.attack_start, excl_accomplice=True)

    fam_best: Dict[str, Tuple[str, float]] = {}
    for fam in BASE_STATS:
        cands = [c for c in tb_all
                 if family_of(c[:-5] if c.endswith("_zabs")
                              else (c[:-2] if c.endswith("_z") else c)) == fam]
        best, bv = None, -1.0
        for c in cands:
            wc, _, _ = worst_case(auc, c)
            if np.isfinite(wc) and wc > bv:
                best, bv = c, wc
        if best:
            fam_best[fam] = (best, bv)

    labels = {"suspicion": "OURS (suspicion)"}
    labels.update({b: f"stateless {b}" for b in BASELINES})
    labels.update({f"{b}_z": f"stateless {b} (z)" for b in BASELINES})

    rows = ["suspicion"] + byz_all
    print_auc_block(
        "1. REFERENCE: the payload auditor vs the memoryless baselines "
        "(CIFAR-100 d=0.5, %d seeds)" % len(seeds), auc, rows, labels)

    tb_rows = ["suspicion"] + [fam_best[f][0] for f in BASE_STATS if f in fam_best] + ens
    tb_labels = dict(labels)
    for f, (c, _) in fam_best.items():
        tb_labels[c] = "stateful %-11s [%s]" % (f, c.replace(f"tb_{f}_", ""))
    tb_labels.update({e: "ENSEMBLE %s" % e[4:] for e in ens})
    print_auc_block(
        "2. HEAD-TO-HEAD: the payload auditor vs stateful temporal baselines on the same archive.\n"
        "   Each baseline row is the BEST of its 6 variants (causal/burnin x raw/z/|z|),\n"
        "   selected on the same data it is scored on. That is an oracle and an upper bound.\n"
        "   The ENSEMBLE rows take the max of |z| over whole families, which is the same\n"
        "   combination rule `suspicion` uses, so they are the strongest generic detector\n"
        "   that can be built from this archive without the auditor's statistics.",
        auc, tb_rows, tb_labels)

    abl_labels = {
        "suspicion": "OURS (suspicion) = abl_full",
        "abl_u1g": "(d) self-cos + magnitude: u1,g",
        "abl_u1g_nostruct": "    same, no structural rule",
        "abl_memoryless": "memoryless part: g only",
        "abl_ownhist": "own history: u1,u2,rep",
        "abl_crosshist": "cross history: x only",
        "abl_struct_only": "structural rule alone",
        "abl_full_nostruct": "all five z, no structural",
        "abl_no_u1": "leave out u1", "abl_no_u2": "leave out u2",
        "abl_no_g": "leave out g", "abl_no_x": "leave out x",
        "abl_no_rep": "leave out rep",
        "abl_u1": "u1 alone", "abl_u2": "u2 alone", "abl_g": "g alone",
        "abl_x": "x alone", "abl_rep": "rep alone",
    }
    abl_labels.update({"abl_u1u2": "u1,u2", "abl_u1u2g": "u1,u2,g",
                       "abl_u1u2gx": "u1,u2,g,x", "abl_x_rep": "x,rep only",
                       "abl_x_rep_nostruct": "x,rep, no structural rule"})
    abl_rows = ["suspicion", "abl_u1g", "abl_u1g_nostruct", "abl_memoryless", "abl_ownhist",
                "abl_crosshist", "abl_struct_only", "abl_full_nostruct",
                "abl_u1u2", "abl_u1u2g", "abl_u1u2gx", "abl_x_rep", "abl_x_rep_nostruct",
                "abl_no_u1", "abl_no_u2", "abl_no_g", "abl_no_x", "abl_no_rep",
                "abl_u1", "abl_u2", "abl_g", "abl_x", "abl_rep"]
    print_auc_block("3. ABLATION of the auditor's statistic by component.",
                    auc, abl_rows, abl_labels)

    print("\n" + horizontal_rule())
    print("4. FULL VARIANT GRID for the stateful baselines (worst-case AUC over the 6 attacks)")
    print(horizontal_rule())
    print("  %-14s" % "family" + "".join("%13s" % f"{m}/{o}" for m in MODES
                                         for o in ("raw", "z", "|z|")))
    for fam in BASE_STATS:
        line = "  %-14s" % fam
        for m in MODES:
            for suf in ("", "_z", "_zabs"):
                c = f"tb_{fam}_{m}{suf}"
                wc, _, _ = worst_case(auc, c) if c in stats else (float("nan"), "", 0)
                line += ("%13.3f" % wc) if np.isfinite(wc) else "%13s" % "n/a"
        print(line)

    print_auc_block(
        "4b. SAME TABLE with peer_copy's copy SOURCE excluded from the negative class.\n"
        "    The source publishes the copier's own innovation, so flagging it is a correct\n"
        "    identification of the fraud, not a false alarm; leaving it in caps peer_copy AUC\n"
        "    near 0.944 for EVERY copy-aware detector. This row set is the fair comparison.",
        auc_ex, ["suspicion"] + ens + [fam_best[f][0] for f in BASE_STATS if f in fam_best]
        + [f"{b}_z" for b in BASELINES] + ["abl_u1g", "abl_ownhist"],
        {**tb_labels, **abl_labels})

    cal_res = [p for p in res if p["key"]["pool"] == "CAL" and p["key"]["attack"] is None]
    cal_mats = {s: P.stack_scores(cal_res, s) for s in stats}
    thresholds = P.fit_thresholds(cal_mats, P.TARGETS)
    rows5 = (["suspicion", "abl_no_u1", "abl_no_u2", "abl_no_g", "abl_no_x", "abl_no_rep",
              "abl_full_nostruct", "abl_u1g", "abl_memoryless", "abl_ownhist",
              "abl_crosshist", "abl_x_rep", "abl_struct_only"]
             + [f"{b}_z" for b in BASELINES]
             + [fam_best[f][0] for f in BASE_STATS if f in fam_best] + ens)
    if a.peer_aware:
        rows5 += ["abl_x", "abl_rep"] + [f"{k}_z" for k in FG_VARIANTS]
    thr_eval = {s: thresholds[s] for s in rows5}
    out = {
        "protocol": {"pools": {"CAL": cal, "DEV": dev, "TEST": test},
                     "archives": {str(s): os.path.basename(paths[s]) for s in seeds},
                     "clients": clients, "rounds": rounds, "scope_start": scope_start,
                     "attack_start": a.attack_start, "mad_floor": mad_floor,
                     "targets": list(P.TARGETS),
                     "attack_def_check": chk},
        "auc": auc,
        "auc_excl_copy_source": auc_ex,
        "family_best_variant": {f: {"column": c, "worst_auc": v} for f, (c, v) in fam_best.items()},
        "thresholds": thresholds,
        "results": {},
    }

    for pool_name, ss in (("DEV", dev), ("TEST", test)):
        ps = [p for p in res if p["key"]["pool"] == pool_name]
        if not ps:
            out["results"][pool_name] = None
            continue
        honest, atk = P.split_pool(ps)
        ev = P.evaluate_pool(honest, atk, thr_eval, P.TARGETS, clients, a.attack_start)
        out["results"][pool_name] = ev

    head = "TEST" if out["results"].get("TEST") else "DEV"
    ev = out["results"][head]
    if ev:
        for target in (0.01,):
            print("\n" + horizontal_rule())
            print("5. FROZEN-THRESHOLD %s POOL at nominal %.1f%% per-client-round FPR."
                  % (head, 100 * target))
            print("   Threshold fitted on CAL HONEST client-rounds only, then applied once.")
            print(horizontal_rule())
            hdr = ("  %-34s %10s %11s %11s" % ("detector", "realised", "worst TPR", "mean TPR")
                   + "".join("%13s" % x[:12] for x in P.ATTACK_ORDER))
            print(hdr)
            print("  %-34s %10s %11s %11s" % ("", "FPR", "over suite", "over suite"))
            print("-" * len(hdr))
            for s in rows5:
                e = ev[s][f"{target:g}"]
                tprs = [e["attacks"][at]["tpr_client_round"]["point"] for at in P.ATTACK_ORDER
                        if at in e["attacks"]]
                lab = tb_labels.get(s, abl_labels.get(s, labels.get(s, s)))
                line = "  %-34s %9.2f%% %10.1f%% %10.1f%%" % (
                    lab[:34], 100 * e["honest"]["client_round"]["point"],
                    100 * min(tprs), 100 * float(np.mean(tprs)))
                for at in P.ATTACK_ORDER:
                    line += "%12.1f%%" % (100 * e["attacks"][at]["tpr_client_round"]["point"])
                print(line)
            print("  Realised FPR is on the same held-out pool; a detector whose realised FPR")
            print("  exceeds the nominal 1%% has not transferred and its TPR is not comparable.")

    ours_wc, ours_wa, _ = worst_case(auc, "suspicion")
    best_tb = max(fam_best.items(), key=lambda kv: kv[1][1]) if fam_best else None
    ens_wc = {e: worst_case(auc, e)[0] for e in ens}
    best_byz = max(((b, worst_case(auc, b)[0]) for b in byz_all
                    if np.isfinite(worst_case(auc, b)[0])), key=lambda kv: kv[1], default=None)
    print("\n" + horizontal_rule())
    print("VERDICT")
    print(horizontal_rule())
    print("  ours, worst-case AUC over the six attacks      : %.3f  (worst attack: %s)"
          % (ours_wc, ours_wa))
    if best_tb:
        print("  best STATEFUL temporal baseline (oracle variant): %.3f  (%s / %s)"
              % (best_tb[1][1], best_tb[0], best_tb[1][0]))
    if best_byz:
        print("  best STATELESS baseline                         : %.3f  (%s)"
              % (best_byz[1], best_byz[0]))
    for e in ens:
        print("  ENSEMBLE %-38s: %.3f   (copy source excluded: %.3f)"
              % (e[4:], ens_wc[e], worst_case(auc_ex, e)[0]))
    print("  ours, copy source excluded from negatives       : %.3f"
          % worst_case(auc_ex, "suspicion")[0])
    wc_u1g, _, _ = worst_case(auc, "abl_u1g")
    wc_mem, _, _ = worst_case(auc, "abl_memoryless")
    wc_own, _, _ = worst_case(auc, "abl_ownhist")
    print("  u1 + g only (self-cosine plus magnitude)           : %.3f" % wc_u1g)
    print("  memoryless part of ours (g + structural)        : %.3f" % wc_mem)
    print("  own-history part of ours (u1,u2,rep + struct)   : %.3f" % wc_own)
    out["verdict"] = {
        "ours_worst_auc": ours_wc, "ours_worst_attack": ours_wa,
        "best_stateful_baseline": ({"family": best_tb[0], "column": best_tb[1][0],
                                    "worst_auc": best_tb[1][1]} if best_tb else None),
        "best_stateless_baseline": ({"column": best_byz[0], "worst_auc": best_byz[1]}
                                    if best_byz else None),
        "ensembles_worst_auc": ens_wc,
        "ensembles_worst_auc_excl_copy_source": {e: worst_case(auc_ex, e)[0] for e in ens},
        "ours_worst_auc_excl_copy_source": worst_case(auc_ex, "suspicion")[0],
        "abl_u1g_worst_auc": wc_u1g, "abl_memoryless_worst_auc": wc_mem,
        "abl_ownhist_worst_auc": wc_own,
        "abl_x_rep_worst_auc": worst_case(auc, "abl_x_rep")[0],
    }

    if a.peer_aware or a.save_scores:
        from passportfl.reproducibility import environment
        out["config"] = {k: portable_path(v) for k, v in vars(a).items()}
        out["command"] = " ".join(sys.argv)
        out["environment"] = environment()
    if a.peer_aware:
        out["peer_aware"] = {
            "foolsgold_columns": {k: {"importance": imp} for k, imp in FG_VARIANTS.items()}
        }
    if a.save_scores:
        meta = {"pools": {"CAL": cal, "DEV": dev, "TEST": test}, "mad_floor": mad_floor,
                "attack_start": a.attack_start, "scope_start": scope_start,
                "archive_prefix": a.archive_prefix, "mask_policy": a.mask_policy,
                "peer_aware": bool(a.peer_aware), "targets": list(P.TARGETS)}
        t1 = time.time()
        out["scores_file"] = save_score_matrices(a.save_scores, res, meta)
        print("  scores        : %s (%.1f MB, sha256 %s, %.1fs)"
              % (a.save_scores, out["scores_file"]["bytes"] / 2 ** 20,
                 out["scores_file"]["sha256"][:16], time.time() - t1))

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwrote %s   (total %.1fs)" % (a.out, time.time() - t_all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
