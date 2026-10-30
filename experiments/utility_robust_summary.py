from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")

METRICS = {"proto": ("tail_acc_prototype_honest", "acc_prototype"),
           "cls": ("tail_acc_classifier_honest", "acc_classifier")}
FINAL = {"proto": "final_acc_prototype_honest", "cls": "final_acc_classifier_honest"}
DDOF = 1
NULL = "honest_rep"
ARMS = ("echo", "exclude", NULL)
REFERENCE_RULE = "mean"
RANK_STATS = ("krum", "median_dev", "trimmed_mean_dev")
FLAG_STATS = ("suspicion_raw", "combined_max") + RANK_STATS
KEYS = ("rounds", "attack_start", "eval_from", "local_epochs", "shard_cap", "victim", "rule",
        "tail_rounds", "dirichlet", "clients", "basis_steps", "full_steps", "restarts",
        "warm", "eval_n", "full_test_final", "width", "embed_dim", "dataset", "device", "mad_floor")
RELEASED = {0: "utility_impact_gpu_s0.json", 1: "utility_impact_s1.json",
            2: "utility_impact_s2.json", 3: "utility_impact_s3.json",
            4: "utility_impact_s4.json"}
NAME = re.compile(r"^utility_robust_([a-z]+)_s(\d+)\.json$")


def load_runs(directory: str) -> dict:
    out: dict = {}
    for p in sorted(glob.glob(os.path.join(directory, "utility_robust_*_s*.json"))):
        m = NAME.match(os.path.basename(p))
        if not m:
            continue
        run = json.load(open(p, encoding="utf-8"))
        recorded = {(run.get("aggregation") or {}).get("rule")} | {
            (s.get("aggregation") or {}).get("rule") for s in run["scenarios"].values()}
        recorded.discard(None)
        if recorded != {m.group(1)}:
            raise SystemExit("%s records aggregation rule(s) %s" % (p, sorted(recorded)))
        out.setdefault(m.group(1), {})[int(m.group(2))] = (p, run)
    return out


def spread(x) -> dict:
    d = np.array([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(d):
        return {"n": 0}
    return {"n": int(len(d)), "mean": float(d.mean()),
            "sd": float(d.std(ddof=DDOF)) if len(d) > DDOF else None,
            "min": float(d.min()), "max": float(d.max()),
            "consistent_sign": bool(np.all(d > 0) or np.all(d < 0)),
            "per_seed": [None if v is None else float(v) for v in x]}


def tail(run: dict, name: str, m: str) -> float:
    v = run["summary"][name]["utility"].get(METRICS[m][0])
    return float(v) if v is not None else float("nan")


def matched_tail(run: dict, name: str, m: str, clients) -> float:
    rec = run["scenarios"][name].get("per_client_utility")
    if not rec:
        return float("nan")
    cfg = run["config"]
    rows = [e for e in rec if e["round"] > cfg["rounds"] - cfg["tail_rounds"]]
    key = METRICS[m][1]
    vals = [float(np.mean([e[key][str(c)] for c in clients])) for e in rows]
    return float(np.mean(vals)) if vals else float("nan")


def seat_scores(run: dict, name: str, seat: int, taus: dict) -> dict:
    ev = run["config"]["eval_from"]
    rows = [x for x in run["scenarios"][name]["rows"] if x["round"] >= ev]
    s = str(seat)
    out = {"rounds": [x["round"] for x in rows]}
    for k in RANK_STATS:
        ranks, n_ranked, score = [], [], []
        for x in rows:
            vals = {c: float(v) for c, v in x[k].items() if v is not None and np.isfinite(v)}
            v = vals.get(s)
            ranks.append(None if v is None else 1 + sum(w < v for c, w in vals.items() if c != s))
            n_ranked.append(len(vals))
            score.append(v)
        fin = [r for r in ranks if r is not None]
        out[k] = {"rank": ranks, "n_ranked": n_ranked, "score": score,
                  "rank1_fraction": float(np.mean([r == 1 for r in fin])) if fin else None,
                  "median_rank": float(np.median(fin)) if fin else None,
                  "mean_score": float(np.mean([v for v in score if v is not None]))
                  if any(v is not None for v in score) else None}
    out["flags"] = {}
    for k in FLAG_STATS:
        tau = taus["suspicion_paper"] if k == "suspicion_raw" else taus.get(k)
        fl = [bool(v is not None and np.isfinite(v) and tau is not None and v > tau)
              for v in (x[k].get(s) for x in rows)]
        out["flags"][k] = {"tau": tau, "per_round": fl,
                           "rate": float(np.mean(fl)) if fl else None,
                           "any": bool(any(fl)) if fl else None}
    return out


def detection(run: dict, name: str) -> dict:
    det = run["summary"][name].get("detection") or {}
    return {k: {f: det[k].get(f) for f in ("auc", "tau", "tpr_client_round", "tpr_client_run",
                                           "fpr_client_round")}
            for k in FLAG_STATS if isinstance(det.get(k), dict)}


def aggregation_records(run: dict, name: str) -> dict:
    ev = run["config"]["eval_from"]
    rec = [r for r in (run["scenarios"][name].get("aggregation") or {}).get("rounds", [])
           if r["round"] >= ev]
    if not rec:
        return {}
    g = lambda k: [r[k] for r in rec if r[k] is not None and np.isfinite(r[k])]
    return {"rounds": len(rec),
            "accepted_min": int(min(r["accepted"] for r in rec)),
            "contributions_min": int(min(r["contributions"] for r in rec)),
            "classes_held_min": int(min(r["classes_held"] for r in rec)),
            "holders_min": int(min(r["holders_min"] for r in rec)),
            "holders_median_mean": float(np.mean(g("holders_median"))),
            "dev_from_mean_rel_mean": float(np.mean(g("dev_from_mean_rel"))),
            "dev_from_mean_over_step_mean": (float(np.mean(g("dev_from_mean_over_step")))
                                             if g("dev_from_mean_over_step") else None)}


def per_seed(run: dict, names) -> dict:
    S = run["scenarios"]
    victim = run["config"]["victim"]
    out = {"tail": {n: {m: tail(run, n, m) for m in METRICS} for n in names},
           "final_full_test": {n: {m: run["summary"][n]["utility"].get(FINAL[m]) for m in METRICS}
                               for n in names},
           "drift": {n: run["summary"][n]["utility"].get("mean_cbar_rel_drift") for n in names
                     if n != "honest"},
           "aggregation": {n: aggregation_records(run, n) for n in names}}
    arms = [a for a in ARMS if a in names]
    out["drop"] = {a: {m: tail(run, "honest", m) - tail(run, a, m) for m in METRICS} for a in arms}
    common = set(S["honest"]["honest_ids"])
    for n in names:
        common &= set(S[n]["honest_ids"])
    common = sorted(common)
    out["matched_clients"] = common
    out["matched_drop"] = {a: {m: matched_tail(run, "honest", m, common)
                               - matched_tail(run, a, m, common) for m in METRICS} for a in arms}
    chk = [abs(matched_tail(run, n, m, S[n]["honest_ids"]) - tail(run, n, m))
           for n in names for m in METRICS if S[n].get("per_client_utility")]
    out["per_client_check_max_abs"] = float(max(chk)) if chk else None
    if "echo" in names and "exclude" in names:
        out["echo_minus_exclude_drop"] = {m: out["drop"]["echo"][m] - out["drop"]["exclude"][m]
                                          for m in METRICS}
        out["echo_drop_exceeds_exclude_drop"] = {m: bool(out["echo_minus_exclude_drop"][m] > 0)
                                                 for m in METRICS}
    if "echo" in names:
        seat = S["echo"]["attackers"][0]
        out["free_rider_seat"] = seat
        out["free_rider"] = seat_scores(run, "echo", seat, run["taus"])
        out["seat_in_honest_arm"] = seat_scores(run, "honest", seat, run["taus"])
        out["detection"] = detection(run, "echo")
    out["victim"] = victim
    return out


def across(ps: dict, seeds) -> dict:
    first = ps[seeds[0]]
    out = {"tail": {n: {m: spread([ps[s]["tail"][n][m] for s in seeds]) for m in METRICS}
                    for n in first["tail"]},
           "drop": {a: {m: spread([ps[s]["drop"][a][m] for s in seeds]) for m in METRICS}
                    for a in first["drop"]},
           "matched_drop": {a: {m: spread([ps[s]["matched_drop"][a][m] for s in seeds])
                                for m in METRICS} for a in first["matched_drop"]},
           "drift": {n: spread([ps[s]["drift"][n] for s in seeds]) for n in first["drift"]}}
    if NULL in first["drop"]:
        out["noise_floor_max_abs"] = {m: float(np.max(np.abs([ps[s]["drop"][NULL][m]
                                                              for s in seeds]))) for m in METRICS}
    if "echo_minus_exclude_drop" in first:
        out["echo_minus_exclude_drop"] = {m: spread([ps[s]["echo_minus_exclude_drop"][m]
                                                     for s in seeds]) for m in METRICS}
        out["echo_drop_exceeds_exclude_drop"] = {
            m: {"per_seed": [ps[s]["echo_drop_exceeds_exclude_drop"][m] for s in seeds],
                "n_true": int(sum(ps[s]["echo_drop_exceeds_exclude_drop"][m] for s in seeds)),
                "n": len(seeds)} for m in METRICS}
    if "free_rider" in first:
        fr = {}
        for who in ("free_rider", "seat_in_honest_arm"):
            fr[who] = {k: {"rank1_fraction": spread([ps[s][who][k]["rank1_fraction"] for s in seeds]),
                           "median_rank_pooled": float(np.median(
                               [r for s in seeds for r in ps[s][who][k]["rank"] if r is not None])),
                           "mean_score": spread([ps[s][who][k]["mean_score"] for s in seeds])}
                       for k in RANK_STATS}
            fr[who]["flags"] = {k: {"rate": spread([ps[s][who]["flags"][k]["rate"] for s in seeds]),
                                    "any_per_seed": [ps[s][who]["flags"][k]["any"] for s in seeds]}
                                for k in FLAG_STATS}
        out["free_rider"] = fr
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=RESULTS, help="directory of utility_robust_<rule>_s<seed>.json")
    ap.add_argument("--released-dir", default=RESULTS)
    ap.add_argument("--out", default=os.path.join(RESULTS, "utility_robust_summary.json"))
    a = ap.parse_args()

    runs = load_runs(a.dir)
    if not runs:
        print("no utility_robust_<rule>_s<seed>.json in %s" % a.dir)
        return 1

    cfgs = {(r, s): {k: v[1]["config"].get(k) for k in KEYS} for r, d in runs.items()
            for s, v in d.items()}
    ref = cfgs[min(cfgs)]
    bad = {rs: {k: v for k, v in c.items() if ref[k] != v} for rs, c in cfgs.items() if c != ref}
    if bad:
        print("Not pooled: these files were run at different settings")
        for (r, s), c in bad.items():
            print("  %s seed %d: %s" % (r, s, c))
        print("  reference %s seed %d: %s" % (min(cfgs) + (ref,)))
        return 1
    print("settings: %s" % ref)
    print("metrics : %s; s.d. ddof=%d\n" % (", ".join(v[0] for v in METRICS.values()), DDOF))

    payload = {"config": ref, "metrics": {m: v[0] for m, v in METRICS.items()}, "sd_ddof": DDOF,
               "files": {}, "complete": {}, "environment": {}, "rules": {}, "cross_rule": {},
               "rerun_vs_released": {}}
    g = lambda v, fmt="%+.4f": (fmt % v) if isinstance(v, float) and np.isfinite(v) else "-"
    per_rule = {}
    for rule in sorted(runs, key=lambda r: (r != REFERENCE_RULE, r)):
        d = runs[rule]
        seeds = sorted(d)
        payload["files"][rule] = {str(s): os.path.basename(d[s][0]) for s in seeds}
        payload["complete"][rule] = {str(s): "aggregation" in d[s][1] for s in seeds}
        payload["environment"][rule] = {str(s): d[s][1].get("environment") for s in seeds}
        have = {s: set(d[s][1]["summary"]) for s in seeds}
        if not all("honest" in h for h in have.values()):
            print("%s: a seed has no honest control; skipped" % rule)
            continue
        every = set.intersection(*have.values())
        missing = {str(s): sorted(set.union(*have.values()) - have[s]) for s in seeds
                   if set.union(*have.values()) - have[s]}
        names = ["honest"] + [n for n in ("honest_rep", "echo", "exclude") if n in every] + sorted(
            n for n in every if n not in ("honest", "honest_rep", "echo", "exclude"))
        ps = {s: per_seed(d[s][1], names) for s in seeds}
        pooled = across(ps, seeds)
        per_rule[rule] = (seeds, ps)
        payload["rules"][rule] = {"seeds": seeds, "scenarios": names,
                                  "scenarios_missing": missing,
                                  "per_seed": {str(s): ps[s] for s in seeds},
                                  "across_seeds": pooled}

        print("=" * 100)
        print("aggregation rule %s; seeds %s%s" % (rule, seeds, "" if not missing else
                                                    "; missing %s" % missing))
        print("=" * 100)
        print("%-8s %-10s %9s %9s %8s  %s" % ("metric", "scenario", "tail mean", "sd", "", "per seed"))
        for m in METRICS:
            for n in names:
                t = pooled["tail"][n][m]
                print("%-8s %-10s %9.4f %9s %8s  %s" % (m, n, t["mean"], g(t.get("sd"), "%.4f"), "",
                                                        " ".join("%.4f" % v for v in t["per_seed"])))
        print("\npaired drops (honest - arm), mean +- sd over seeds [per seed]:")
        for m in METRICS:
            for kind in ("drop", "matched_drop"):
                for arm, v in pooled[kind].items():
                    x = v[m]
                    print("  %-5s %-13s %-10s %s +- %s  [%s]" % (m, kind, arm, g(x["mean"]),
                                                                 g(x.get("sd"), "%.4f"),
                                                                 " ".join(g(y) for y in x["per_seed"])))
        if "echo_minus_exclude_drop" in pooled:
            for m in METRICS:
                x = pooled["echo_minus_exclude_drop"][m]
                e = pooled["echo_drop_exceeds_exclude_drop"][m]
                print("  %-5s drop(echo) - drop(exclude): %s +- %s  [%s]; echo > exclude in %d/%d seeds"
                      % (m, g(x["mean"]), g(x.get("sd"), "%.4f"),
                         " ".join(g(y) for y in x["per_seed"]), e["n_true"], e["n"]))
        if "noise_floor_max_abs" in pooled:
            print("  noise floor max |honest - honest_rep|: %s" % pooled["noise_floor_max_abs"])
        if "free_rider" in pooled:
            print("\necho free-rider (seat %s) versus the same seat in the honest arm:"
                  % ps[seeds[0]].get("free_rider_seat"))
            for k in RANK_STATS:
                fr, hs = pooled["free_rider"]["free_rider"][k], pooled["free_rider"]["seat_in_honest_arm"][k]
                print("  %-17s ranked most central in %s of rounds (honest seat %s); pooled median "
                      "rank %s (honest seat %s)" % (k, g(fr["rank1_fraction"]["mean"], "%.3f"),
                                                    g(hs["rank1_fraction"]["mean"], "%.3f"),
                                                    fr["median_rank_pooled"], hs["median_rank_pooled"]))
            for k in FLAG_STATS:
                fr = pooled["free_rider"]["free_rider"]["flags"][k]
                print("  flagged by %-17s rate %s, any-flag per seed %s"
                      % (k, g(fr["rate"]["mean"], "%.3f"), fr["any_per_seed"]))
        for n in names:
            agg = [ps[s]["aggregation"][n] for s in seeds if ps[s]["aggregation"][n]]
            if agg:
                print("  aggregation %-10s dev from count-weighted mean %.4f (x step %s); accepted >= %d"
                      % (n, np.mean([x["dev_from_mean_rel_mean"] for x in agg]),
                         g(mean_or_none([x["dev_from_mean_over_step_mean"] for x in agg]), "%.3f"),
                         min(x["accepted_min"] for x in agg)))
        print()

    if REFERENCE_RULE in per_rule:
        rs, rp = per_rule[REFERENCE_RULE]
        for rule, (seeds, ps) in per_rule.items():
            if rule == REFERENCE_RULE:
                continue
            common = [s for s in seeds if s in rs]
            if not common:
                continue
            cr = {"seeds": common}
            for m in METRICS:
                cr[m] = {"honest_tail": spread([ps[s]["tail"]["honest"][m] - rp[s]["tail"]["honest"][m]
                                                for s in common])}
                for arm in ("echo", "exclude"):
                    if arm in ps[common[0]]["drop"] and arm in rp[common[0]]["drop"]:
                        cr[m]["drop_" + arm] = spread([ps[s]["drop"][arm][m] - rp[s]["drop"][arm][m]
                                                       for s in common])
                if "echo_minus_exclude_drop" in ps[common[0]] and "echo_minus_exclude_drop" in rp[common[0]]:
                    cr[m]["echo_minus_exclude_drop"] = spread(
                        [ps[s]["echo_minus_exclude_drop"][m] - rp[s]["echo_minus_exclude_drop"][m]
                         for s in common])
            payload["cross_rule"][rule + "_minus_" + REFERENCE_RULE] = cr
            print("%s minus %s, paired by seed %s:" % (rule, REFERENCE_RULE, common))
            for k, v in cr["proto"].items():
                print("   proto %-24s %s +- %s  [%s]" % (k, g(v["mean"]), g(v.get("sd"), "%.4f"),
                                                        " ".join(g(y) for y in v["per_seed"])))

        rel_out = {}
        for s in rs:
            p = os.path.join(a.released_dir, RELEASED.get(s, ""))
            if s not in RELEASED or not os.path.exists(p):
                continue
            rel = json.load(open(p, encoding="utf-8"))
            same = all(rel["config"].get(k) == ref[k] for k in KEYS)
            rel_out[str(s)] = {"file": RELEASED[s], "same_settings": same}
            if same:
                for n in ("honest", "echo", "exclude"):
                    if n in rel["summary"] and n in rp[s]["tail"]:
                        rel_out[str(s)][n] = {"rerun": rp[s]["tail"][n]["proto"],
                                              "released": tail(rel, n, "proto"),
                                              "difference": rp[s]["tail"][n]["proto"] - tail(rel, n, "proto")}
        payload["rerun_vs_released"] = rel_out
        if rel_out:
            print("\nmean-rule re-run minus released tail prototype accuracy, same seed:")
            for s, v in rel_out.items():
                print("   seed %s  %s" % (s, "  ".join("%s %+.4f" % (n, v[n]["difference"])
                                                       for n in ("honest", "echo", "exclude") if n in v)
                                          if v["same_settings"] else "settings differ; not compared"))

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("\nwrote %s" % a.out)
    return 0


def mean_or_none(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


if __name__ == "__main__":
    raise SystemExit(main())
