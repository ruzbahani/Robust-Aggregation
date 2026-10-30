from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")

SEED_FILES = {0: "utility_impact_gpu_s0.json"}
METRIC = "tail_acc_prototype_honest"


def load_results() -> dict:
    out = {}
    for seed, name in SEED_FILES.items():
        p = os.path.join(RESULTS, name)
        if os.path.exists(p):
            out[seed] = json.load(open(p, encoding="utf-8"))
    for p in sorted(glob.glob(os.path.join(RESULTS, "utility_impact_s*.json"))):
        base = os.path.basename(p)
        try:
            seed = int(base.split("_s")[-1].split(".")[0])
        except ValueError:
            continue
        if seed not in out:
            out[seed] = json.load(open(p, encoding="utf-8"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "utility_seed_spread.json"))
    a = ap.parse_args()

    runs = load_results()
    if not runs:
        print("no utility_impact results found")
        return 1

    keys = ("rounds", "attack_start", "local_epochs", "shard_cap", "victim", "rule",
            "tail_rounds", "dirichlet", "clients")
    cfgs = {s: {k: r["config"].get(k) for k in keys} for s, r in runs.items()}
    ref = cfgs[min(cfgs)]
    bad = {s: c for s, c in cfgs.items() if c != ref}
    if bad:
        print("Not pooled: these seeds were run at different settings")
        for s, c in bad.items():
            print("  seed %d: %s" % (s, {k: v for k, v in c.items() if ref[k] != v}))
        print("  reference (seed %d): %s" % (min(cfgs), ref))
        return 1

    seeds = sorted(runs)
    have = {s: set(runs[s]["summary"]) for s in seeds}
    full = max(have.values(), key=len)
    partial = {s: sorted(full - have[s]) for s in seeds if full - have[s]}
    if partial:
        print("Seeds that do not carry every scenario. Either the run is incomplete or the")
        print("scenario was not requested for that seed:")
        for s, miss in partial.items():
            print("   seed %d does not have %s" % (s, ", ".join(miss)))
        print("   The table below covers only the scenarios present in every seed, so a")
        print("   scenario missing from one seed is dropped rather than averaged over a")
        print("   smaller set than the others.\n")
    scen = [s for s in sorted(full) if s != "honest"]
    scen = [s for s in scen if all(s in have[x] for x in seeds)]

    def acc(seed, name):
        v = runs[seed]["summary"][name]["utility"].get(METRIC)
        return float(v) if v is not None else float("nan")

    honest = np.array([acc(s, "honest") for s in seeds])
    print("settings: %s" % ref)
    print("metric  : %s\n" % METRIC)
    print("unattacked arm, per seed")
    for s, v in zip(seeds, honest):
        print("   seed %d  %.4f" % (s, v))
    hs = float(honest.max() - honest.min())
    print("   mean %.4f  sd %.4f  range %.4f  (n=%d)"
          % (honest.mean(), honest.std(ddof=1) if len(honest) > 1 else 0.0, hs, len(honest)))
    print()

    rows = {}
    hdr = "%-14s %9s %9s %9s %9s  %s" % ("scenario", "mean drop", "sd", "min", "max",
                                         "per-seed")
    print(hdr)
    print("-" * len(hdr))
    for name in scen:
        att = np.array([acc(s, name) for s in seeds])
        drop = honest - att
        ok = np.isfinite(drop)
        if not ok.any():
            continue
        d = drop[ok]
        same_sign = bool(np.all(d > 0) or np.all(d < 0))
        rows[name] = {
            "seeds": [int(s) for s, k in zip(seeds, ok) if k],
            "per_seed_drop": [float(x) for x in d],
            "mean_drop": float(d.mean()),
            "sd_drop": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "min_drop": float(d.min()), "max_drop": float(d.max()),
            "consistent_sign": same_sign,
            "honest_range": hs,
            "drop_exceeds_honest_range": bool(abs(d.mean()) > hs),
        }
        print("%-14s %9.4f %9.4f %9.4f %9.4f  %s"
              % (name, d.mean(), rows[name]["sd_drop"], d.min(), d.max(),
                 " ".join("%+.4f" % x for x in d)))

    print()
    print("A drop is inside the run-to-run spread only if it is small against the")
    print("unattacked range (%.4f) AND inconsistent in sign across seeds." % hs)
    if len(seeds) < 2:
        print("   (one seed only: there is no run-to-run spread yet, so nothing below")
        print("    is a verdict. Run more seeds.)")
    for name, r in rows.items():
        d = np.array(r["per_seed_drop"])
        if np.allclose(d, 0.0):
            verdict = "no measurable change in any seed"
        elif not r["consistent_sign"]:
            verdict = "sign flips across seeds, so it is inside the spread"
        elif abs(r["mean_drop"]) > hs:
            verdict = "same sign in every seed and larger than the honest range"
        else:
            verdict = ("same sign in every seed but smaller than the honest range, "
                       "so repeatable and small rather than absent")
        print("   %-14s %s" % (name, verdict))

    payload = {"config": ref, "metric": METRIC, "seeds": [int(s) for s in seeds],
               "honest_per_seed": [float(x) for x in honest],
               "honest_mean": float(honest.mean()),
               "honest_sd": float(honest.std(ddof=1)) if len(honest) > 1 else 0.0,
               "honest_range": hs, "scenarios": rows}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
