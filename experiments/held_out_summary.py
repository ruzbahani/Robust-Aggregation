from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "experiments"))

from protocol_eval import clopper_pearson

RESULTS = os.path.join(HERE, "results")
ATTACKS = ["global_echo", "peer_copy", "lagged_self", "ar2_replay", "noise", "shrink"]


def main() -> int:
    with open(os.path.join(RESULTS, "temporal_eval.json")) as fh:
        te = json.load(fh)
    with open(os.path.join(RESULTS, "fresh_seed_transfer.json")) as fh:
        fs = json.load(fh)

    blk = te["results"]["TEST"]["suspicion"]["0.01"]
    orig_k = blk["honest"]["client_round"]["k"]
    orig_n = blk["honest"]["client_round"]["n"]

    print("Threshold %.4f, fitted on CAL seeds 0 and 1, never refitted.\n"
          % fs["threshold"])
    print("%-24s %8s %8s %9s" % ("held-out seeds", "flags", "rounds", "FPR"))
    print("-" * 54)
    print("%-24s %8d %8d %8.2f%%" % ("3, 4 (TEST)", orig_k, orig_n,
                                     100 * orig_k / orig_n))
    for s, v in sorted(fs["honest"]["per_seed"].items(), key=lambda kv: int(kv[0])):
        print("%-24s %8d %8d %8.2f%%" % ("%s (additional)" % s, v["flags"], v["rounds"],
                                         100 * v["fpr"]))

    k = orig_k + fs["honest"]["flags"]
    n = orig_n + fs["honest"]["rounds"]
    _, lo, hi = clopper_pearson(k, n)
    print("-" * 54)
    print("%-24s %8d %8d %8.2f%%   [%.2f, %.2f]"
          % ("all five, pooled", k, n, 100 * k / n, 100 * lo, 100 * hi))

    rates = [orig_k / orig_n] + [v["fpr"] for v in fs["honest"]["per_seed"].values()]
    per_seed_rates = {"3+4": orig_k / orig_n}
    per_seed_rates.update({s: v["fpr"] for s, v in fs["honest"]["per_seed"].items()})
    print()
    print("per-seed spread: %.2f%% to %.2f%%, a factor of %.1f"
          % (100 * min(rates), 100 * max(rates), max(rates) / max(min(rates), 1e-9)))

    print()
    print("%-16s %10s %10s %9s" % ("attack", "detected", "rounds", "TPR"))
    print("-" * 50)
    tpr = {}
    for a in ATTACKS:
        ok = blk["attacks"][a]["tpr_client_round"]
        fk, fn = fs["tpr"][a]["k"], fs["tpr"][a]["n"]
        kk, nn = ok["k"] + fk, ok["n"] + fn
        _, tlo, thi = clopper_pearson(kk, nn)
        tpr[a] = {"k": kk, "n": nn, "tpr": kk / nn, "ci": [tlo, thi]}
        print("%-16s %10d %10d %8.1f%%   [%.1f, %.1f]"
              % (a, kk, nn, 100 * kk / nn, 100 * tlo, 100 * thi))

    worst = min(v["tpr"] for v in tpr.values())
    print()
    print("Across five held-out seeds: worst-case detection %.1f%%, realized false alarms"
          % (100 * worst))
    print("%.2f%% against a nominal 1.00%%, per-seed %.2f%% to %.2f%%."
          % (100 * k / n, 100 * min(rates), 100 * max(rates)))

    out = {
        "threshold": fs["threshold"], "fitted_on": [0, 1],
        "held_out_seeds": [3, 4] + fs["fresh_seeds"],
        "honest": {"flags": k, "rounds": n, "fpr": k / n, "ci": [lo, hi],
                   "per_seed": per_seed_rates,
                   "spread": [min(rates), max(rates)]},
        "tpr": tpr, "worst_tpr": worst,
    }
    with open(os.path.join(RESULTS, "held_out_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote results/held_out_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
