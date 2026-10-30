from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")

METRIC = "tail_acc_prototype_honest"
DDOF = 1
NULL = "honest_rep"
RELEASED = {0: "utility_impact_gpu_s0.json", 1: "utility_impact_s1.json",
            2: "utility_impact_s2.json", 3: "utility_impact_s3.json",
            4: "utility_impact_s4.json"}
ORDER = (NULL, "fused_b1", "fused_b1_lag", "fused_b2", "fused_b2_lag", "fused_b1_x3",
         "fused_b1_x3_lag", "fused_b1_x3_lag_nocoord")
SHOWN = ("suspicion_raw", "suspicion", "krum", "median_dev", "trimmed_mean_dev", "norm_bound",
         "cosine_cohort", "foolsgold_maxcos", "combined_sum", "combined_two")
KEYS = ("rounds", "attack_start", "eval_from", "local_epochs", "shard_cap", "victim", "rule",
        "tail_rounds", "dirichlet", "clients", "basis_steps", "full_steps", "restarts",
        "warm", "eval_n", "full_test_final", "width", "embed_dim", "dataset")


def load_runs(pattern: str) -> dict:
    out = {}
    for p in sorted(glob.glob(pattern)):
        m = re.search(r"_s(\d+)\.json$", os.path.basename(p))
        if m:
            out[int(m.group(1))] = (p, json.load(open(p, encoding="utf-8")))
    return out


def tail(run: dict, name: str) -> float:
    v = run["summary"][name]["utility"].get(METRIC)
    return float(v) if v is not None else float("nan")


def spread(x) -> dict:
    d = np.array([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(d):
        return {"n": 0}
    return {"n": int(len(d)), "mean": float(d.mean()),
            "sd": float(d.std(ddof=DDOF)) if len(d) > DDOF else None,
            "min": float(d.min()), "max": float(d.max()),
            "consistent_sign": bool(np.all(d > 0) or np.all(d < 0))}


def mean_or_none(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def scored_rows(run: dict, name: str) -> list:
    return [x for x in run["scenarios"][name]["rows"] if x["round"] >= run["config"]["eval_from"]]


def prediction_error(run: dict, name: str) -> list:
    rule = "combined_" + run["config"]["rule"]
    out = []
    for row in scored_rows(run, name):
        for c, info in row.get("attack_info", {}).items():
            p, s = info.get("predicted_combined"), row[rule].get(str(c), row[rule].get(c))
            if p is not None and s is not None and np.isfinite(p) and np.isfinite(s):
                out.append(abs(float(p) - float(s)))
    return out


def budget(run: dict, name: str) -> dict:
    util = {u["round"]: u for u in run["scenarios"][name]["utility"]}
    B, unit, ratio = [], [], []
    for row in scored_rows(run, name):
        real = util.get(row["round"], {}).get("cohort_innov_norm_median")
        for info in row.get("attack_info", {}).values():
            b, s = info.get("budget"), info.get("cohort_scale")
            if b is None or s is None or not (np.isfinite(b) and np.isfinite(s)):
                continue
            B.append(float(b))
            unit.append(float(s))
            if real is not None and np.isfinite(real) and real > 0:
                ratio.append(float(s) / float(real))
    return {"budget": float(np.mean(B)) if B else None,
            "unit": float(np.mean(unit)) if unit else None,
            "unit_over_realised": float(np.mean(ratio)) if ratio else None,
            "attacker_innov_over_cohort":
                run["summary"][name]["utility"].get("attacker_innov_over_cohort")}


def detection(run: dict, name: str) -> dict:
    det = run["summary"][name]["detection"] or {}
    cm, sr = det.get("combined_max", {}), det.get("suspicion_raw", {})
    fam = {k: {f: v.get(f) for f in ("tau", "auc", "tpr_client_round", "tpr_client_run",
                                      "fpr_client_round")}
           for k, v in det.items() if isinstance(v, dict)}
    return {"flag_rate": cm.get("tpr_client_round"), "run_flagged": cm.get("tpr_client_run"),
            "auc_combined_max": cm.get("auc"), "auditor_flag_rate": sr.get("tpr_client_round"),
            "honest_fpr": cm.get("fpr_client_round"),
            "structural_flag_rate": det.get("structural_flag_rate"), "families": fam}


def rushing_of(name: str) -> str:
    return name.split("_lag")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=os.path.join(RESULTS, "utility_impact_lagged_s*.json"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "utility_lagged_summary.json"))
    a = ap.parse_args()

    runs = load_runs(a.pattern)
    if not runs:
        print("no results match %s" % a.pattern)
        return 1
    seeds = sorted(runs)

    cfgs = {s: {k: runs[s][1]["config"].get(k) for k in KEYS} for s in seeds}
    ref = cfgs[seeds[0]]
    bad = {s: {k: v for k, v in c.items() if ref[k] != v} for s, c in cfgs.items() if c != ref}
    if bad:
        print("Not pooled: these seeds were run at different settings")
        for s, c in bad.items():
            print("  seed %d: %s" % (s, c))
        print("  reference (seed %d): %s" % (seeds[0], ref))
        return 1

    have = {s: set(runs[s][1]["summary"]) for s in seeds}
    for s in seeds:
        if "honest" not in have[s]:
            print("seed %d has no honest control" % s)
            return 1
    every = set.intersection(*have.values())
    union = set.union(*have.values())
    for s in seeds:
        miss = sorted(union - have[s])
        if miss:
            print("seed %d does not have %s; dropped from the table" % (s, ", ".join(miss)))
    names = [n for n in ORDER if n in every] + sorted(n for n in every
                                                     if n not in ORDER and n != "honest")

    honest = [tail(runs[s][1], "honest") for s in seeds]
    print("settings: %s" % ref)
    print("seeds   : %s   metric: %s   s.d. ddof=%d\n" % (seeds, METRIC, DDOF))
    print("unattacked arm per seed: %s" % " ".join("%.4f" % v for v in honest))
    hs = spread(honest)
    hrange = float(max(honest) - min(honest))
    print("   mean %.4f  sd %s  range %.4f\n" % (hs["mean"], "%.4f" % hs["sd"] if hs.get("sd") is not None
                                                else "-", hrange))

    null = None
    if NULL in every:
        nd = [tail(runs[s][1], "honest") - tail(runs[s][1], NULL) for s in seeds]
        null = {"per_seed": [float(x) for x in nd], "summary": spread(nd),
                "max_abs": float(np.max(np.abs(nd)))}

    g = lambda v, fmt="%8.4f": (fmt % v) if isinstance(v, float) and np.isfinite(v) else "%8s" % "-"
    rows = {}
    hdr = ("%-24s %9s %8s %5s | %8s %8s %8s %8s | %8s %8s %8s  %s"
           % ("scenario", "mean drop", "sd", "sign", "flag", "anyflag", "auditor", "AUCmax",
              "pred.err", "budget", "unit/re", "per-seed drop"))
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        spec = runs[seeds[0]][1]["scenarios"][name]["spec"]
        drop = [tail(runs[s][1], "honest") - tail(runs[s][1], name) for s in seeds]
        det = [detection(runs[s][1], name) for s in seeds]
        perr = [prediction_error(runs[s][1], name) for s in seeds]
        bud = [budget(runs[s][1], name) for s in seeds]
        ds = spread(drop)
        rec = {"knowledge": spec.get("knowledge", "peer_current" if spec.get("attackers") else None),
               "coalition": spec.get("coalition"),
               "attackers": runs[seeds[0]][1]["scenarios"][name]["attackers"],
               "beta": spec.get("beta"),
               "per_seed_drop": [float(x) for x in drop], "drop": ds,
               "drop_exceeds_honest_range": bool(abs(ds.get("mean", 0.0)) > hrange),
               "drop_exceeds_null_max_abs": (bool(abs(ds.get("mean", 0.0)) > null["max_abs"])
                                             if null is not None and name != NULL else None)}
        for k in ("flag_rate", "run_flagged", "auditor_flag_rate", "auc_combined_max",
                  "honest_fpr", "structural_flag_rate"):
            vals = [d[k] for d in det]
            rec[k] = {"per_seed": vals, "mean": mean_or_none(vals)}
        fams = sorted(set().union(*[d["families"] for d in det]))
        rec["families"] = {
            f: {"tau": next((d["families"][f]["tau"] for d in det if f in d["families"]), None),
                **{m: {"per_seed": [d["families"].get(f, {}).get(m) for d in det],
                       "mean": mean_or_none([d["families"].get(f, {}).get(m) for d in det])}
                   for m in ("tpr_client_round", "tpr_client_run", "auc", "fpr_client_round")}}
            for f in fams}
        flat = [e for p in perr for e in p]
        rec["prediction_error"] = {
            "per_seed_median": [float(np.median(p)) if p else None for p in perr],
            "median": float(np.median(flat)) if flat else None,
            "p90": float(np.percentile(flat, 90)) if flat else None,
            "max": float(np.max(flat)) if flat else None,
            "n_client_rounds": len(flat)}
        rec["budget"] = {k: {"per_seed": [b[k] for b in bud], "mean": mean_or_none([b[k] for b in bud])}
                         for k in ("budget", "unit", "unit_over_realised",
                                   "attacker_innov_over_cohort")}
        rec["solver_status"] = [runs[s][1]["scenarios"][name]["statuses"] for s in seeds]
        rows[name] = rec
        print("%-24s %9.4f %8s %5s | %s %s %s %s | %s %s %s  %s"
              % (name, ds["mean"], g(ds.get("sd")),
                 "yes" if ds["consistent_sign"] else "no",
                 g(rec["flag_rate"]["mean"]), g(rec["run_flagged"]["mean"]),
                 g(rec["auditor_flag_rate"]["mean"]), g(rec["auc_combined_max"]["mean"]),
                 g(rec["prediction_error"]["median"]), g(rec["budget"]["budget"]["mean"], "%8.3f"),
                 g(rec["budget"]["unit_over_realised"]["mean"], "%8.3f"),
                 " ".join("%+.4f" % x for x in drop)))
    print("\nflag = fused-max flag rate over attacked client-rounds; anyflag = fraction of "
          "seeds in which some attacker was\nflagged at least once; auditor = auditor alone "
          "at its 1% threshold; pred.err = median |predicted - realised| fused score;\n"
          "budget = mean absolute budget; unit/re = budget unit over the realised honest "
          "median innovation norm of the same round.")
    if null is not None:
        print("noise floor (honest - %s): %s   max |.| %.4f"
              % (NULL, " ".join("%+.4f" % x for x in null["per_seed"]), null["max_abs"]))

    attacked = [n for n in names if rows[n]["attackers"]]
    if attacked:
        print("\nflag rate over attacked client-rounds by family (mean over seeds; "
              "tau is the family's frozen threshold):")
        print("%-24s " % "" + " ".join("%9s" % f[:9] for f in SHOWN))
        for n in attacked:
            fam = rows[n]["families"]
            print("%-24s " % n + " ".join(
                "%9s" % (("%.3f" % fam[f]["tpr_client_round"]["mean"])
                         if f in fam and fam[f]["tpr_client_round"]["mean"] is not None else "-")
                for f in SHOWN))

    paired = {}
    for name in names:
        rush = rushing_of(name)
        if "_lag" not in name or rush not in every:
            continue
        diff = [tail(runs[s][1], rush) - tail(runs[s][1], name) for s in seeds]
        paired[name] = {"rushing": rush, "per_seed": [float(x) for x in diff],
                        "summary": spread(diff)}
    if paired:
        print("\ndrop(lagged) - drop(rushing), same seed and prefix:")
        for name, p in paired.items():
            print("   %-24s vs %-12s %s" % (name, p["rushing"],
                                           " ".join("%+.4f" % x for x in p["per_seed"])))

    released = {}
    for s in seeds:
        p = os.path.join(RESULTS, RELEASED.get(s, ""))
        if s not in RELEASED or not os.path.exists(p):
            continue
        rel = json.load(open(p, encoding="utf-8"))
        same = all(rel["config"].get(k) == ref[k] for k in KEYS)
        released[s] = {"file": RELEASED[s], "same_settings": same}
        for name in ("honest", "fused_b1", "fused_b2", "fused_b1_x3"):
            if same and name in rel["summary"] and name in have[s]:
                released[s][name] = {"rerun": tail(runs[s][1], name), "released": tail(rel, name),
                                     "difference": tail(runs[s][1], name) - tail(rel, name)}
    if released:
        print("\nre-run minus released tail accuracy, same seed:")
        for s, v in released.items():
            print("   seed %d  %s" % (s, "  ".join("%s %+.4f" % (n, v[n]["difference"])
                                                   for n in ("honest", "fused_b1", "fused_b2",
                                                             "fused_b1_x3") if n in v)
                                      if v["same_settings"] else "settings differ; not compared"))

    payload = {"config": ref, "metric": METRIC, "sd_ddof": DDOF, "seeds": seeds,
               "files": {int(s): os.path.basename(runs[s][0]) for s in seeds},
               "environment": {int(s): runs[s][1].get("environment") for s in seeds},
               "honest_per_seed": [float(x) for x in honest], "honest": hs,
               "honest_range": hrange, "noise_floor": null, "scenarios": rows,
               "lagged_minus_rushing": paired, "rerun_vs_released": released}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
