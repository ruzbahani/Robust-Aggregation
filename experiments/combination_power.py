from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from innovation_eval import ATTACKS
from passportfl.innovation import InnovationAuditor

ARCHIVES = os.path.join(HERE, "results", "archives")
CAL, TEST = [0, 1], [3, 4]
STRUCT = 1e3
MAD_FLOOR = 0.01

ORIENT = (("u1", None), ("u2", None), ("g", -1.0), ("x", +1.0), ("rep", +1.0))
STATS = [k for k, _ in ORIENT]


def load_archive(seed):
    for name in ("arch_cifar100_s%d_c10_r35_s%d.npz" % (seed, seed),
                 "arch_cifar100_s%d_c10_r35_d0.5_s%d.npz" % (seed, seed)):
        p = os.path.join(ARCHIVES, name)
        if os.path.exists(p):
            return p
    return None


def score_archive(seed, attack, victim, clients=10, attack_start=12, scope_start=12):
    z = np.load(load_archive(seed), allow_pickle=True)
    arch = {"C": z["C"].item(), "present": z["present"].item(), "cbar": z["cbar"].item()}
    arch["published"] = {i: {} for i in range(clients)}
    rng = np.random.default_rng(1000 * seed + (victim or 0))
    aud = InnovationAuditor(mad_floor=MAD_FLOOR)
    rows = []
    for r in sorted(arch["C"][0]):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        payloads = {}
        for cid in range(clients):
            C = arch["C"][cid][r]
            if attack and cid == victim and r >= attack_start:
                C = ATTACKS[attack](arch, cid, r, rng)
            arch["published"][cid][r] = C
            payloads[cid] = (C, arch["present"][cid][r])
        rs = aud.observe_round(r, payloads)
        if r < max(scope_start, attack_start if attack else scope_start):
            continue
        for cid, v in rs.per_client.items():
            raw = [v.get(k, np.nan) for k in STATS]
            zz = [v.get("z_" + k, np.nan) for k in STATS]
            rows.append({
                "cid": cid, "round": r,
                "raw": raw, "z": zz,
                "susp": v.get("suspicion", np.nan),
                "is_attacker": bool(attack and cid == victim),
            })
    return rows


def oriented_score(rows, key):
    a = np.array([r[key] for r in rows], dtype=float)
    out = np.empty_like(a)
    for i, (_, sign) in enumerate(ORIENT):
        out[:, i] = np.abs(a[:, i]) if sign is None else sign * a[:, i]
    return out


def aggregate_over_seeds(rows, key):
    return np.nanmax(oriented_score(rows, key), axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal", type=float, default=0.01)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "combination_power.json"))
    args = ap.parse_args()

    cal = []
    for s in CAL:
        if load_archive(s):
            cal += score_archive(s, None, None)
    if not cal:
        print("missing archives")
        return 1

    cal_raw_mx = aggregate_over_seeds(cal, "raw")
    cal_z_mx = aggregate_over_seeds(cal, "z")
    per = args.nominal / len(STATS)
    thr = {
        "max_raw": float(np.nanquantile(cal_raw_mx, 1 - args.nominal)),
        "max_z": float(np.nanquantile(cal_z_mx, 1 - args.nominal)),
    }
    cal_or = oriented_score(cal, "raw")
    thr_any = []
    for i in range(len(STATS)):
        c = cal_or[:, i]
        c = c[np.isfinite(c)]
        thr_any.append(float(np.quantile(c, 1 - per)) if c.size >= 50 else np.inf)

    def flags(rows):
        rmx, zmx = aggregate_over_seeds(rows, "raw"), aggregate_over_seeds(rows, "z")
        R = oriented_score(rows, "raw")
        anyf = np.zeros(len(rows), dtype=bool)
        for i in range(len(STATS)):
            anyf |= np.nan_to_num(R[:, i], nan=-np.inf) > thr_any[i]
        struct = np.array([r["susp"] >= STRUCT for r in rows])
        return {
            "max_raw": (rmx > thr["max_raw"]) | struct,
            "any_raw": anyf | struct,
            "max_z": (zmx > thr["max_z"]) | struct,
        }

    honest = []
    for s in TEST:
        if load_archive(s):
            honest += score_archive(s, None, None)
    hf = flags(honest)

    print("Fitted on CAL seeds %s, applied unchanged to TEST seeds %s.\n" % (CAL, TEST))
    print("%-10s %12s %s" % ("attack", "", "  ".join("%9s" % k for k in ("max_raw", "any_raw", "max_z"))))
    print("-" * 62)
    print("%-10s %12s %s" % ("honest FPR", "",
                             "  ".join("%8.2f%%" % (100 * hf[k].mean())
                                       for k in ("max_raw", "any_raw", "max_z"))))
    print()

    out = {"nominal": args.nominal, "thresholds": thr,
           "per_stat_thresholds": dict(zip(STATS, thr_any)),
           "fpr": {k: float(v.mean()) for k, v in hf.items()}, "tpr": {}}

    for attack in ATTACKS:
        tp = {k: [] for k in ("max_raw", "any_raw", "max_z")}
        for s in TEST:
            if not load_archive(s):
                continue
            for victim in range(10):
                rows = score_archive(s, attack, victim)
                f = flags(rows)
                idx = np.array([r["is_attacker"] for r in rows])
                for k in tp:
                    tp[k].append(f[k][idx].mean())
        res = {k: float(np.mean(v)) for k, v in tp.items()}
        out["tpr"][attack] = res
        print("%-10s %12s %s" % (attack, "",
                                 "  ".join("%8.1f%%" % (100 * res[k])
                                           for k in ("max_raw", "any_raw", "max_z"))))

    worst = {k: min(v[k] for v in out["tpr"].values()) for k in ("max_raw", "any_raw", "max_z")}
    out["worst_tpr"] = worst
    print("-" * 62)
    print("%-10s %12s %s" % ("worst", "",
                             "  ".join("%8.1f%%" % (100 * worst[k])
                                       for k in ("max_raw", "any_raw", "max_z"))))

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
