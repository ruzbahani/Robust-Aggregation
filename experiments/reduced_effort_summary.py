from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
RULES = ("suspicion", "norm_bound", "krum", "median_dev", "trimmed_mean_dev",
         "cosine_cohort", "foolsgold_maxcos")


def ms(vals):
    v = np.array([x for x in vals if np.isfinite(x)], float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v))


def summarize_runs(runs: dict) -> dict:
    seeds = sorted(runs)
    rec = {"seeds": seeds, "n": len(seeds)}
    steps = runs[seeds[0]]
    rec["victim_steps"] = steps["victim_steps_per_round"][-1]
    rec["full_steps"] = steps["victim_full_steps"]
    rec["innovation_over_cohort"] = ms([runs[s]["victim_innovation_over_cohort_median"]["median"]
                                        for s in seeds])
    rec["auc"] = {r: ms([runs[s]["auc"].get(r, float("nan")) for s in seeds]) for r in RULES}
    rec["tpr"] = ms([runs[s]["tpr_client_round"] for s in seeds])
    rec["honest_fpr"] = ms([runs[s]["honest_fpr_client_round"] for s in seeds])
    return rec


def row_text(label: str, rec: dict) -> str:
    return ("%-9s %5d %4d/%-4d %9.3f" % (label, rec["n"], rec["victim_steps"], rec["full_steps"],
                                         rec["innovation_over_cohort"][0])
            + "".join("%14s" % ("%.3f+-%.3f" % rec["auc"][r][:2]) for r in RULES)
            + "%12s" % ("%.3f+-%.3f" % rec["tpr"][:2]))


def header() -> str:
    return "%-9s %5s %9s %9s" % ("fraction", "seeds", "steps", "move") + \
        "".join("%14s" % r[:13] for r in RULES) + "%12s" % "auditor TPR"


def load_runs(results: str, pattern: str, regex: str, key=str) -> dict:
    runs = {}
    for p in sorted(glob.glob(os.path.join(results, pattern))):
        m = re.search(regex, os.path.basename(p))
        if not m:
            continue
        with open(p, encoding="utf-8") as fh:
            runs.setdefault(key(m.group(1)), {})[int(m.group(2))] = json.load(fh)
    return runs


def summarize_modes(results: str, out_path: str, steps_runs: dict) -> int:
    modes = load_runs(results, "reduced_effort_fwd*_s*.json", r"reduced_effort_(fwd[0-9.]*)_s(\d+)\.json$")
    modes.update(load_runs(results, "reduced_effort_zero_s*.json", r"reduced_effort_(zero)_s(\d+)\.json$"))
    if not modes:
        print("no forward-only or zero-computation results")
        return 1
    out = {}
    print("\n" + header())
    print("-" * len(header()))
    for frac in sorted(steps_runs):
        rec = summarize_runs(steps_runs[frac])
        rec["mode"] = "steps"
        out["steps_f%g" % frac] = rec
        print(row_text("%g" % frac, rec))
    for tag in sorted(modes):
        runs = modes[tag]
        rec = summarize_runs(runs)
        first = runs[min(runs)]
        rec["mode"] = first["mode"]
        rec["victim_forward_passes"] = first["victim_forward_passes_per_round"][-1]
        rec["attack_start"] = first["config"]["attack_start"]
        rec["published_train_loss_attacked"] = ms([float(np.mean(r["victim_published_train_loss"][r["config"]["attack_start"] - 1:]))
                                                   for r in runs.values()])
        rec["published_train_loss_last_honest"] = ms([r["victim_published_train_loss"][r["config"]["attack_start"] - 2]
                                                      for r in runs.values()])
        out[tag] = rec
        print(row_text(tag, rec))
    print("\nsteps = optimizer steps of the victim per attacked round; fwd runs %s forward passes"
          % ", ".join("%s: %d" % (t, out[t]["victim_forward_passes"]) for t in sorted(modes)))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote %s" % out_path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "reduced_effort_summary.json"))
    ap.add_argument("--modes-out", default=os.path.join(RESULTS, "reduced_effort_summary_modes.json"))
    ap.add_argument("--modes-only", action="store_true",
                    help="write only the forward-only and zero-computation summary")
    ap.add_argument("--results", default=RESULTS, help="directory of the per-seed result files")
    a = ap.parse_args()

    runs = load_runs(a.results, "reduced_effort_f*_s*.json", r"reduced_effort_f([0-9.]+)_s(\d+)\.json$", float)
    if a.modes_only:
        return summarize_modes(a.results, a.modes_out, runs)
    if not runs:
        print("no reduced_effort results")
        return 1

    out = {}
    hdr = header()
    print(hdr)
    print("-" * len(hdr))
    for frac in sorted(runs):
        rec = summarize_runs(runs[frac])
        out["%g" % frac] = rec
        print(row_text("%g" % frac, rec))
    print("\nmove = victim innovation norm over the cohort median, median over rounds, mean over seeds")
    print("the shrink attack scales the honest innovation to 0.1; auditor reads it at AUC 1.000, TPR 1.000")

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
