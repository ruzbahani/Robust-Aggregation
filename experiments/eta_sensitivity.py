from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from typing import Dict, List, Sequence

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import experiments.protocol_eval as P
import experiments.temporal_eval as T
from passportfl.baselines import BASELINES

RESULTS = os.path.join(HERE, "results")
CONFIGS = {
    "cifar100": {"prefix": "arch_cifar100_s",
                 "pools": {"CAL": [0, 1], "DEV": [2], "TEST": [3, 4], "FRESH": [5, 6, 7]},
                 "grid": list(P.MAD_FLOOR_GRID),
                 "auc_seed_sets": {"seeds_0_4": [0, 1, 2, 3, 4],
                                   "held_out_3_7": [3, 4, 5, 6, 7]},
                 "reference": "temporal_eval.json", "reference_floor": 1e-2},
    "cifar10": {"prefix": "arch_cifar10_s",
                "pools": {"CAL": [0, 1], "DEV": [2]},
                "grid": list(P.MAD_FLOOR_GRID),
                "auc_seed_sets": {"seeds_0_2": [0, 1, 2]},
                "reference": "temporal_eval_cifar10.json", "reference_floor": 3e-1},
}
STATS = ["suspicion"] + [f"{b}_z" for b in BASELINES]


def compact(ev: dict, stat: str) -> dict:
    out = {}
    for t, e in ev[stat].items():
        per, uk, un = {}, 0, 0
        for a, r in e["attacks"].items():
            fp = r["fpr_uninvolved_client_round"]
            uk += fp["k"]
            un += fp["n"]
            per[a] = {"tpr": r["tpr_client_round"]["point"], "k": r["tpr_client_round"]["k"],
                      "n": r["tpr_client_round"]["n"],
                      "trajectories_detected": r["tpr_client_run"]["k"],
                      "trajectories": r["tpr_client_run"]["n"],
                      "fpr_uninvolved": fp["point"]}
        worst = min(per, key=lambda a: per[a]["tpr"])
        h = e["honest"]["client_round"]
        out[t] = {"threshold": e["threshold"], "clean_fpr": h["point"], "clean_k": h["k"],
                  "clean_n": h["n"], "clean_trajectories_flagged": e["honest"]["client_run"]["k"],
                  "clean_trajectories": e["honest"]["client_run"]["n"],
                  "uninvolved_fpr": uk / un if un else None, "uninvolved_k": uk,
                  "uninvolved_n": un, "worst_tpr": per[worst]["tpr"], "worst_attack": worst,
                  "per_attack": per}
    return out


def mean_auc(auc: dict, stat: str, seeds: Sequence[int]) -> dict:
    per = {}
    for a in P.ATTACK_ORDER:
        v = [auc[a][stat]["per_seed"].get(str(s)) for s in seeds]
        v = [x for x in v if x is not None and np.isfinite(x)]
        per[a] = float(np.mean(v)) if v else None
    fin = {a: v for a, v in per.items() if v is not None}
    w = min(fin, key=lambda a: fin[a]) if fin else None
    return {"per_attack": per, "worst": fin[w] if w else None, "worst_attack": w}


def run_config(name: str, cfg: dict, workers: int, archives: str) -> dict:
    paths = P.discover_archives(archives, cfg["prefix"])
    pools = {k: [s for s in v if s in paths] for k, v in cfg["pools"].items()}
    seeds = sorted(s for v in pools.values() for s in v)
    clients, rounds = P.archive_shape(P.load_archive(paths[seeds[0]]))
    attack_start = scope_start = 12
    print("\n%s: %s, %d clients, %d rounds, pools %s"
          % (name, ", ".join(os.path.basename(paths[s]) for s in seeds), clients, rounds, pools))

    t0 = time.time()
    sel_tasks = []
    for f in P.MAD_FLOOR_GRID:
        tt = P.build_tasks(paths, pools["CAL"], f, scope_start, attack_start,
                           {s: clients for s in pools["CAL"]}, False, "CAL")
        for t in tt:
            t["key"] = dict(t["key"], floor=f)
        sel_tasks += tt
    sres = P.run_tasks(sel_tasks, workers)
    cal_honest = {f: [r for r in sres if r["key"]["floor"] == f] for f in P.MAD_FLOOR_GRID}
    sel = P.select_mad_floor(cal_honest, P.TARGETS)
    print("  CAL selection : %.0e (%s) [%.1fs]" % (sel["selected"], sel["criterion"],
                                                  time.time() - t0))

    t0 = time.time()
    tasks = []
    for f in cfg["grid"]:
        for pool, ss in pools.items():
            tt = P.build_tasks(paths, ss, f, scope_start, attack_start,
                               {s: clients for s in ss}, True, pool)
            for t in tt:
                t["key"] = dict(t["key"], floor=f)
            tasks += tt
    res = P.run_tasks(tasks, workers)
    print("  replay        : %d passes in %.1fs" % (len(tasks), time.time() - t0))

    grid = {}
    for f in cfg["grid"]:
        rf = [r for r in res if r["key"]["floor"] == f]
        cal = [r for r in rf if r["key"]["pool"] == "CAL" and r["key"]["attack"] is None]
        thr = P.fit_thresholds({s: P.stack_scores(cal, s) for s in STATS}, P.TARGETS)
        auc = T.auc_table(rf, seeds, ["suspicion"], attack_start)
        auc_ex = T.auc_table(rf, seeds, ["suspicion"], attack_start, excl_accomplice=True)
        row = {"thresholds": thr["suspicion"], "pools": {},
               "auc": {k: mean_auc(auc, "suspicion", v) for k, v in cfg["auc_seed_sets"].items()},
               "auc_excl_copy_source": {k: mean_auc(auc_ex, "suspicion", v)
                                        for k, v in cfg["auc_seed_sets"].items()},
               "auc_per_seed": {a: auc[a]["suspicion"]["per_seed"] for a in P.ATTACK_ORDER},
               "memoryless_rules_worst_tpr_at_1pct": {}}
        evaluated = {}
        for pool in [p for p in pools if p != "CAL"] + (["HELDOUT"] if "FRESH" in pools else []):
            ss = pools["TEST"] + pools["FRESH"] if pool == "HELDOUT" else pools[pool]
            ps = [r for r in rf if r["key"]["seed"] in ss]
            honest, atk = P.split_pool(ps)
            ev = P.evaluate_pool(honest, atk, thr, P.TARGETS, clients, attack_start)
            evaluated[pool] = ev
            row["pools"][pool] = {"seeds": ss, "auditor": compact(ev, "suspicion")}
            row["memoryless_rules_worst_tpr_at_1pct"][pool] = {
                s: compact(ev, s)["0.01"]["worst_tpr"] for s in STATS[1:]}
        grid["%g" % f] = row
        hd = "HELDOUT" if "HELDOUT" in row["pools"] else "DEV"
        o = row["pools"][hd]["auditor"]["0.01"]
        aw = next(iter(row["auc"].values()))
        print("  eta %-6g thr@1%% %7.3f | %s: worst TPR %5.1f%% (%s) clean FPR %5.2f%% "
              "uninvolved %5.2f%% | worst AUC %.4f (%s)"
              % (f, thr["suspicion"]["0.01"], hd, 100 * o["worst_tpr"], o["worst_attack"],
                 100 * o["clean_fpr"], 100 * o["uninvolved_fpr"], aw["worst"], aw["worst_attack"]))
    return {"archives": {str(s): os.path.basename(paths[s]) for s in seeds},
            "pools": pools, "clients": clients, "rounds": rounds,
            "attack_start": attack_start, "scope_start": scope_start,
            "cal_selection": sel, "grid": grid}


def check_reference(name: str, cfg: dict, out: dict, checks: List[dict]) -> None:
    p = os.path.join(RESULTS, cfg["reference"])
    if not os.path.exists(p):
        checks.append({"check": "%s reference %s" % (name, cfg["reference"]), "ok": False,
                       "detail": "reference file absent"})
        return
    ref = json.load(open(p, encoding="utf-8"))
    row = out["grid"]["%g" % cfg["reference_floor"]]
    ok_floor = abs(ref["protocol"]["mad_floor"] - cfg["reference_floor"]) < 1e-15
    checks.append({"check": "%s: reference floor is %g" % (name, cfg["reference_floor"]),
                   "ok": ok_floor, "detail": "reference %g" % ref["protocol"]["mad_floor"]})
    ok_thr = row["thresholds"] == ref["thresholds"]["suspicion"]
    checks.append({"check": "%s: thresholds equal %s" % (name, cfg["reference"]), "ok": ok_thr,
                   "detail": ""})
    bad = []
    for a in P.ATTACK_ORDER:
        for s, v in ref["auc"][a]["suspicion"]["per_seed"].items():
            if v is not None and row["auc_per_seed"][a].get(s) != v:
                bad.append((a, s))
    checks.append({"check": "%s: per-seed AUCs equal %s" % (name, cfg["reference"]),
                   "ok": not bad, "detail": str(bad[:5])})
    bad = []
    for pool in ("DEV", "TEST"):
        r = (ref.get("results") or {}).get(pool)
        if not r or pool not in row["pools"]:
            continue
        for t, e in r["suspicion"].items():
            mine = row["pools"][pool]["auditor"][t]
            if mine["clean_k"] != e["honest"]["client_round"]["k"]:
                bad.append((pool, t, "clean"))
            for a, x in e["attacks"].items():
                if mine["per_attack"][a]["k"] != x["tpr_client_round"]["k"]:
                    bad.append((pool, t, a))
    checks.append({"check": "%s: frozen-threshold counts equal %s" % (name, cfg["reference"]),
                   "ok": not bad, "detail": str(bad[:5])})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="cifar100,cifar10")
    ap.add_argument("--archives", default=P.ARCHIVE_DIR)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(RESULTS, "eta_sensitivity.json"))
    a = ap.parse_args()
    t_all = time.time()

    out = {"configs": {}}
    checks: List[dict] = []
    for name in [c.strip() for c in a.configs.split(",") if c.strip()]:
        cfg = CONFIGS[name]
        out["configs"][name] = run_config(name, cfg, a.workers, a.archives)
        out["configs"][name]["grid_floors"] = cfg["grid"]
        check_reference(name, cfg, out["configs"][name], checks)
    for c in checks:
        print("  [%s] %s %s" % ("ok" if c["ok"] else "FAIL", c["check"], c["detail"]))

    from passportfl.reproducibility import environment
    out["consistency_checks"] = checks
    out["all_checks_ok"] = all(c["ok"] for c in checks)
    out["config"] = {k: T.portable_path(v) for k, v in vars(a).items()}
    out["command"] = " ".join(sys.argv)
    out["environment"] = environment()
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\nwrote %s   (total %.1fs)" % (a.out, time.time() - t_all))
    return 0 if out["all_checks_ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
