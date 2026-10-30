from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
SEEDS = (0, 1, 2)
TAIL = 5


def load_results(tag: str):
    p = os.path.join(RESULTS, "fedproto_%s.json" % tag)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def tail_accuracy(run: dict, key: str) -> float:
    vals = [r[key] for r in run["history"] if r.get(key) is not None]
    return float(np.mean(vals[-TAIL:])) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "passport_ablation.json"))
    a = ap.parse_args()

    rows, missing = [], []
    for s in SEEDS:
        pp, nopp = load_results("abl_pp_s%d" % s), load_results("abl_nopp_s%d" % s)
        if pp is None or nopp is None:
            missing.append(s)
            continue
        for name, r in (("passport", pp), ("no-passport", nopp)):
            acc = [x.get("accepted") for x in r["history"] if x.get("accepted") is not None]
            if acc and min(acc) == 0:
                print("seed %d %s: a round accepted 0 contributions; the behavioural checks "
                      "were enforced and the prototypes froze. Rerun with --warmup above "
                      "--rounds." % (s, name))
                return 1
        rows.append({
            "seed": s,
            "rounds": len(pp["history"]),
            "passport_cls": tail_accuracy(pp, "acc_classifier"),
            "nopassport_cls": tail_accuracy(nopp, "acc_classifier"),
            "passport_proto": tail_accuracy(pp, "acc_prototype"),
            "nopassport_proto": tail_accuracy(nopp, "acc_prototype"),
        })
    if missing:
        print("missing seeds: %s" % missing)
    if not rows:
        print("no paired runs found")
        return 1

    for r in rows:
        r["d_cls"] = r["nopassport_cls"] - r["passport_cls"]
        r["d_proto"] = r["nopassport_proto"] - r["passport_proto"]

    hdr = "%-6s %10s %12s %9s %10s %12s %9s" % (
        "seed", "pp cls", "no-pp cls", "diff", "pp proto", "no-pp proto", "diff")
    print("tail accuracy, mean of the last %d rounds\n" % TAIL)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-6d %10.4f %12.4f %+9.4f %10.4f %12.4f %+9.4f"
              % (r["seed"], r["passport_cls"], r["nopassport_cls"], r["d_cls"],
                 r["passport_proto"], r["nopassport_proto"], r["d_proto"]))

    def agg(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)

    dc_m, dc_s = agg("d_cls")
    dp_m, dp_s = agg("d_proto")
    print("-" * len(hdr))
    print("%-6s %10s %12s %+9.4f %10s %12s %+9.4f" % ("mean", "", "", dc_m, "", "", dp_m))
    print("%-6s %10s %12s %9.4f %10s %12s %9.4f" % ("sd", "", "", dc_s, "", "", dp_s))
    print()
    print("A positive difference means the plain-BatchNorm arm scores higher, so the")
    print("passport layer costs that much accuracy. The sign is what matters first:")
    for label, m, s_ in (("classifier", dc_m, dc_s), ("prototype", dp_m, dp_s)):
        same = all(np.sign(r["d_cls" if label == "classifier" else "d_proto"])
                   == np.sign(m) for r in rows)
        print("   %-11s mean %+.4f, sd %.4f, same sign in every seed: %s"
              % (label, m, s_, same))

    payload = {"seeds": [r["seed"] for r in rows], "tail_rounds": TAIL, "per_seed": rows,
               "mean_d_classifier": dc_m, "sd_d_classifier": dc_s,
               "mean_d_prototype": dp_m, "sd_d_prototype": dp_s}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
