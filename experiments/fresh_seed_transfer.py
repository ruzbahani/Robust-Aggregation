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

FROZEN_THRESHOLD = 3.7859094579543044
MAD_FLOOR = 0.01
SCOPE_START = ATTACK_START = 12
STRUCT = 1e3


def load_archive(seed: int) -> str | None:
    for name in ("arch_cifar100_s%d_c10_r35_d0.5_s%d.npz" % (seed, seed),
                 "arch_cifar100_s%d_c10_r35_s%d.npz" % (seed, seed)):
        p = os.path.join(ARCHIVES, name)
        if os.path.exists(p):
            return p
    return None


def score_run(seed: int, attack: str | None, victim: int | None, clients: int = 10):
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
            if attack and cid == victim and r >= ATTACK_START:
                C = ATTACKS[attack](arch, cid, r, rng)
            arch["published"][cid][r] = C
            payloads[cid] = (C, arch["present"][cid][r])
        rs = aud.observe_round(r, payloads)
        if r < SCOPE_START:
            continue
        for cid, v in rs.per_client.items():
            s = v.get("suspicion", float("nan"))
            rows.append({"cid": cid, "round": r, "s": s,
                         "attacker": bool(attack and cid == victim)})
    return rows


def clopper_pearson(k: int, n: int, conf: float = 0.95):
    from protocol_eval import clopper_pearson as cp
    _, lo, hi = cp(k, n, conf)
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[5, 6, 7])
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "fresh_seed_transfer.json"))
    args = ap.parse_args()

    seeds = [s for s in args.seeds if load_archive(s)]
    for s in args.seeds:
        if s not in seeds:
            print("seed %d: no archive, skipped" % s)
    if not seeds:
        return 1

    print("Threshold %.4f, fitted on CAL seeds 0 and 1, applied unchanged." % FROZEN_THRESHOLD)
    print("Fresh seeds: %s. Nothing below is refitted.\n" % ", ".join(map(str, seeds)))

    print("%-8s %10s %10s %10s" % ("seed", "flags", "rounds", "FPR"))
    print("-" * 42)
    hk = hn = 0
    per_seed = {}
    for s in seeds:
        rows = score_run(s, None, None, args.clients)
        f = [r for r in rows if np.isfinite(r["s"]) and r["s"] > FROZEN_THRESHOLD]
        n = len([r for r in rows if np.isfinite(r["s"])])
        hk += len(f)
        hn += n
        per_seed[s] = {"flags": len(f), "rounds": n, "fpr": len(f) / max(n, 1)}
        print("%-8d %10d %10d %9.2f%%" % (s, len(f), n, 100 * len(f) / max(n, 1)))
    lo, hi = clopper_pearson(hk, hn)
    print("%-8s %10d %10d %9.2f%%   [%.2f, %.2f]"
          % ("pooled", hk, hn, 100 * hk / hn, 100 * lo, 100 * hi))

    print()
    print("%-14s %10s %10s %10s" % ("attack", "detected", "rounds", "TPR"))
    print("-" * 48)
    tpr = {}
    for attack in ATTACKS:
        k = n = 0
        for s in seeds:
            for victim in range(args.clients):
                rows = score_run(s, attack, victim, args.clients)
                a = [r for r in rows if r["attacker"] and r["round"] >= ATTACK_START]
                k += sum(1 for r in a if (not np.isfinite(r["s"])) is False
                         and (r["s"] > FROZEN_THRESHOLD or r["s"] >= STRUCT))
                n += len(a)
        tpr[attack] = {"k": k, "n": n, "tpr": k / max(n, 1)}
        lo2, hi2 = clopper_pearson(k, n)
        print("%-14s %10d %10d %9.1f%%   [%.1f, %.1f]"
              % (attack, k, n, 100 * k / max(n, 1), 100 * lo2, 100 * hi2))

    worst = min(v["tpr"] for v in tpr.values())
    print()
    print("worst-case detection on unseen seeds: %.1f%%" % (100 * worst))
    print("realized false alarms on unseen seeds: %.2f%% against a nominal 1.00%%"
          % (100 * hk / hn))

    out = {"threshold": FROZEN_THRESHOLD, "fitted_on": [0, 1], "fresh_seeds": seeds,
           "mad_floor": MAD_FLOOR, "scope_start": SCOPE_START,
           "honest": {"flags": hk, "rounds": hn, "fpr": hk / hn, "ci": [lo, hi],
                      "per_seed": per_seed},
           "tpr": tpr, "worst_tpr": worst}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
