from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")

ATTACKS = ("global_echo", "peer_copy", "lagged_self", "ar2_replay", "noise", "shrink")
INVERTERS = ("krum", "median_dev", "trimmed_mean_dev", "norm_bound")
GROUPS = [
    ("baseline  10 clients, 35 rounds", "innovation_cifar100_s%d.json", (0, 1, 2)),
    ("late      10 clients, 70 rounds", "innovation_late70_s%d.json", (0, 1, 2)),
    ("cohort     5 clients, 35 rounds", "innovation_cohort05_s%d.json", (0, 1, 2)),
    ("cohort    20 clients, 35 rounds", "innovation_cohort20_s%d.json", (0, 1, 2)),
    ("cohort    40 clients, 35 rounds", "innovation_cohort40_s%d.json", (0, 1, 2)),
    ("no passport, plain BatchNorm   ", "innovation_nopp_s%d.json", (0, 1, 2)),
]


def load_results(pattern: str, seeds) -> list:
    out = []
    for s in seeds:
        p = os.path.join(RESULTS, pattern % s)
        if os.path.exists(p):
            out.append(json.load(open(p, encoding="utf-8")))
    return out


def seed_auc(run: dict, attack: str, det: str):
    v = run.get("results", {}).get(attack, {}).get(det, {}).get("auc")
    return float(v) if v is not None and np.isfinite(v) else float("nan")


def mean_and_std(vals):
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    if not v.size:
        return float("nan"), float("nan"), 0
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else 0.0), int(v.size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "sweep_summary.json"))
    a = ap.parse_args()

    payload = {}
    print("Free-rider detection AUC by rule. Below 0.5 means the rule ranks the free-rider")
    print("as more honest than the honest cohort; 0.000 means it does so in every round.")
    print("Every row here uses seeds 0, 1 and 2 so the rows are comparable with each other.")
    print("The robust-ranking table reports the baseline over five seeds (Krum AUC 0.011); the")
    print("baseline row below is that measurement restricted to three seeds, not a second,")
    print("disagreeing estimate.\n")
    hdr = "%-34s %6s" % ("configuration", "seeds") + "".join("%18s" % d for d in INVERTERS)
    print(hdr)
    print("-" * len(hdr))
    for label, pat, seeds in GROUPS:
        runs = load_results(pat, seeds)
        if not runs:
            print("%-34s %6s   (no archive)" % (label, "-"))
            continue
        row, cells = {}, []
        for det in INVERTERS:
            m, s_, n = mean_and_std([seed_auc(r, "global_echo", det) for r in runs])
            row[det] = {"mean": m, "sd": s_, "n": n}
            cells.append("%18s" % ("-" if not n else "%.3f+-%.3f" % (m, s_)))
        print("%-34s %6d" % (label, len(runs)) + "".join(cells))
        payload.setdefault(label, {})["inverters"] = row
        payload[label]["n_runs"] = len(runs)
        payload[label]["clients"] = runs[0]["config"]["clients"]
        payload[label]["rounds"] = runs[0]["config"]["rounds"]

    print()
    print("Payload auditor AUC per attack, and the worst case over the six.\n")
    hdr2 = "%-34s" % "configuration" + "".join("%13s" % t[:12] for t in ATTACKS) + "%11s" % "worst"
    print(hdr2)
    print("-" * len(hdr2))
    for label, pat, seeds in GROUPS:
        runs = load_results(pat, seeds)
        if not runs:
            continue
        per, cells = {}, []
        for t in ATTACKS:
            m, s_, n = mean_and_std([seed_auc(r, t, "suspicion") for r in runs])
            per[t] = {"mean": m, "sd": s_, "n": n}
            cells.append("%13s" % ("-" if not n else "%.3f" % m))
        finite = [v["mean"] for v in per.values() if np.isfinite(v["mean"])]
        worst = min(finite) if finite else float("nan")
        print("%-34s" % label + "".join(cells) + "%11.3f" % worst)
        payload[label]["auditor"] = per
        payload[label]["auditor_worst"] = worst

    base = payload.get("baseline  10 clients, 35 rounds")
    if base:
        print()
        print("Read against the baseline row. A result that holds only at ten clients and")
        print("thirty-five rounds is specific to that configuration; a result that moves")
        print("little across these rows is not.")
        for label, v in payload.items():
            if label.startswith("baseline") or "auditor_worst" not in v:
                continue
            d = v["auditor_worst"] - base["auditor_worst"]
            inv = max(v["inverters"][k]["mean"] for k in INVERTERS
                      if np.isfinite(v["inverters"][k]["mean"]))
            print("   %-34s auditor worst %+.3f vs baseline, largest inverter AUC %.3f"
                  % (label, d, inv))

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
