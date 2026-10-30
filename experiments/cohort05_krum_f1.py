from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import hashlib
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

import innovation_replay as IR
import krum_margin
import protocol_eval as P
from aggregate_impact import eta
from passportfl.baselines import (krum_scores, median_deviation, norm_score,
                                  trimmed_mean_deviation)

ARCHIVES = IR.ARCHIVES
RESULTS = IR.RESULTS
PATTERN = "arch_cohort05_s*_c5_r35*.npz"
CLIENTS = 5
SEEDS = (0, 1, 2)
F_VALUES = (1, 2)
SCHEMA = "cohort05-krum-f1-v1"


def scorers() -> dict:
    out = {"krum_f%d" % f: (lambda innov, f=f: krum_scores(innov, n_byzantine=f))
           for f in F_VALUES}
    out.update({"median_dev": median_deviation, "trimmed_mean_dev": trimmed_mean_deviation,
                "norm_bound": norm_score})
    return out


def selection(mat: np.ndarray) -> dict:
    n = mat.shape[0]
    am = np.argmin(mat, axis=2)
    own = mat[np.arange(n), :, np.arange(n)]
    others = np.where(np.eye(n, dtype=bool)[:, None, :], np.inf, mat).min(axis=2)
    sel = am == np.arange(n)[:, None]
    strict = own < others
    tied = own == others
    return {"selected": sel, "strict": strict, "tied_exact": tied & ~strict}


def rate(x: np.ndarray) -> float:
    return float(np.mean(x))


def measure_krum_margin(f: int) -> dict:
    old = krum_margin.F_BYZANTINE
    krum_margin.F_BYZANTINE = f
    try:
        return krum_margin.measure(PATTERN, clients=CLIENTS)
    finally:
        krum_margin.F_BYZANTINE = old


def guarantee(paths) -> dict:
    ratio = []
    for p in paths:
        arch = P.load_archive(p)
        C, cbar = arch["C"], arch["cbar"]
        for r in sorted(cbar):
            if r < IR.EVAL_FROM or (r - 1) not in cbar:
                continue
            V = np.stack([(C[i][r] - cbar[r - 1]).ravel() for i in range(CLIENTS)])
            ratio.append(krum_margin.pair_stats(V)["ratio"])
    ratio = np.asarray(ratio)
    out = {"rounds": int(ratio.size),
           "ratio_sigma_over_mu": {"median": float(np.median(ratio)), "min": float(ratio.min()),
                                   "max": float(ratio.max())}}
    for f in F_VALUES:
        e = eta(CLIENTS, f)
        key = "f%d" % f
        if e is None:
            out[key] = {"n_gt_2f_plus_2": False, "eta": None}
            continue
        er = e * ratio
        q25, med, q75 = np.percentile(er, [25, 50, 75])
        out[key] = {"n_gt_2f_plus_2": True, "eta": e,
                    "eta_times_ratio": {"median": float(med), "q25": float(q25),
                                        "q75": float(q75), "min": float(er.min()),
                                        "max": float(er.max())},
                    "frac_rounds_precondition_holds": float(np.mean(er < 1.0)),
                    "ratio_threshold_1_over_eta": 1.0 / e}
    return out


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for f in (os.path.abspath(__file__), IR.__file__, P.__file__, krum_margin.__file__,
              os.path.join(HERE, "passportfl", "baselines.py")):
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "cohort05_f1.json"))
    a = ap.parse_args()
    t0 = time.time()

    paths = {int(p.rsplit("_s", 1)[-1].split(".")[0]): p
             for p in sorted(glob.glob(os.path.join(ARCHIVES, PATTERN)))}
    missing = [s for s in SEEDS if s not in paths]
    if missing:
        raise SystemExit("missing five-client archives for seeds %s" % missing)

    sc = scorers()
    per_seed, pooled_sel, repro = {}, {}, {}
    for s in SEEDS:
        arch = P.load_archive(paths[s])
        ref = json.load(open(os.path.join(RESULTS, "innovation_cohort05_s%d.json" % s),
                             encoding="utf-8"))["results"]
        per_seed[s] = {}
        for att in P.ATTACK_ORDER:
            m = IR.replay_matrices(arch, att, s, sc)
            ent = {"auc": {k: IR.auc_of(v)["auc"] for k, v in m["scores"].items()}}
            for f in F_VALUES:
                sel = selection(m["scores"]["krum_f%d" % f])
                ent["krum_f%d" % f] = {k: rate(v) for k, v in sel.items()}
                ent["krum_f%d" % f]["selected_by_seat"] = [rate(sel["selected"][v])
                                                           for v in range(CLIENTS)]
                pooled_sel.setdefault((att, f), []).append(sel)
            per_seed[s][att] = ent
            for k, rk in (("krum_f2", "krum"), ("median_dev", "median_dev"),
                          ("trimmed_mean_dev", "trimmed_mean_dev"), ("norm_bound", "norm_bound")):
                got = IR.auc_of(m["scores"][k])
                want = ref[att][rk]
                d = repro.setdefault(rk, {"max_abs_diff": 0.0, "counts_match": True})
                d["max_abs_diff"] = max(d["max_abs_diff"], abs(got["auc"] - want["auc"]))
                d["counts_match"] &= (got["n_pos"] == want["n_pos"]
                                      and got["n_neg"] == want["n_neg"])
        print("seed %d replayed (%.0fs)" % (s, time.time() - t0), flush=True)

    summary = {}
    for att in P.ATTACK_ORDER:
        summary[att] = {}
        for k in sc:
            v = [per_seed[s][att]["auc"][k] for s in SEEDS]
            summary[att][k] = {"auc_mean": float(np.mean(v)), "auc_sd_ddof0": float(np.std(v)),
                               "auc_sd_ddof1": float(np.std(v, ddof=1)), "per_seed": v}
        for f in F_VALUES:
            sels = pooled_sel[(att, f)]
            cat = {k: np.concatenate([x[k] for x in sels], axis=1) for k in sels[0]}
            summary[att]["krum_f%d_selection" % f] = {
                "seat_rounds": int(cat["selected"].size),
                "selected": rate(cat["selected"]), "strict": rate(cat["strict"]),
                "tied_exact": rate(cat["tied_exact"]),
                "selected_by_seat": [rate(cat["selected"][v]) for v in range(CLIENTS)],
                "strict_by_seat": [rate(cat["strict"][v]) for v in range(CLIENTS)]}

    km = {"f%d" % f: measure_krum_margin(f) for f in F_VALUES}
    km_ref = json.load(open(os.path.join(RESULTS, "krum_margin.json"),
                            encoding="utf-8"))["measured"]["cohort05"]
    km_exact = ("margin_min_positive_frac", "margin_positive_frac", "selected_frac",
                "tied_best_frac", "argmin_selected_all_seats", "argmin_selected_by_seat",
                "krum_k", "rounds")
    km_check = {k: km["f2"][k] == km_ref[k] for k in km_exact}
    for k in ("ratio_median", "near_over_pair_median"):
        km_check[k] = bool(abs(km["f2"][k] - km_ref[k]) <= 1e-12 * abs(km_ref[k]))

    g = guarantee([paths[s] for s in SEEDS])

    from passportfl.reproducibility import environment
    payload = {
        "meta": {
            "schema": SCHEMA,
            "command": "python experiments/cohort05_krum_f1.py",
            "code_sha256": code_fingerprint(),
            "archives": [os.path.basename(paths[s]) for s in SEEDS],
            "krum_k": {"f1": CLIENTS - 1 - 2, "f2": max(1, CLIENTS - 2 - 2)},
            "environment": environment(),
            "wall_time_s": round(time.time() - t0, 1),
        },
        "summary": summary,
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "krum_margin_measure": km,
        "krum_guarantee": g,
        "checks": {"innovation_eval_reproduction": repro,
                   "krum_margin_f2_matches_release": km_check,
                   "attack_definitions": P.check_attack_definitions()},
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    print("\nKrum on five clients (seeds 0-2; AUC mean over seeds, selection pooled over "
          "360 seat-rounds)")
    print("%-12s %9s %9s %10s %10s %10s %10s" % ("attack", "AUC f=2", "AUC f=1", "sel f=2",
                                                  "sel f=1", "strict f=1", "tie f=1"))
    for att in P.ATTACK_ORDER:
        s_ = summary[att]
        print("%-12s %9.4f %9.4f %10.3f %10.3f %10.3f %10.3f"
              % (att, s_["krum_f2"]["auc_mean"], s_["krum_f1"]["auc_mean"],
                 s_["krum_f2_selection"]["selected"], s_["krum_f1_selection"]["selected"],
                 s_["krum_f1_selection"]["strict"], s_["krum_f1_selection"]["tied_exact"]))
    print("global echo per seed: f=2 %s, f=1 %s; f=1 selection by seat %s"
          % (["%.4f" % x for x in summary["global_echo"]["krum_f2"]["per_seed"]],
             ["%.4f" % x for x in summary["global_echo"]["krum_f1"]["per_seed"]],
             ["%.3f" % x for x in summary["global_echo"]["krum_f1_selection"]["selected_by_seat"]]))
    print("krum_margin.measure f=1: margin_min>0 %.3f, margin_med>0 %.3f, selected %.3f, "
          "argmin all seats %.3f, k=%d"
          % (km["f1"]["margin_min_positive_frac"], km["f1"]["margin_positive_frac"],
             km["f1"]["selected_frac"], km["f1"]["argmin_selected_all_seats"], km["f1"]["krum_k"]))
    print("eta(5,1)=%.4f; eta*sigma/||mu|| median %.3f, min %.3f; precondition holds in %.3f "
          "of %d rounds" % (g["f1"]["eta"], g["f1"]["eta_times_ratio"]["median"],
                            g["f1"]["eta_times_ratio"]["min"],
                            g["f1"]["frac_rounds_precondition_holds"], g["rounds"]))
    print("checks: replay vs innovation_eval %s; krum_margin f=2 vs release %s"
          % ({k: (v["max_abs_diff"], v["counts_match"]) for k, v in repro.items()},
             all(km_check.values())))
    print("wrote %s (%.0fs)" % (a.out, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
