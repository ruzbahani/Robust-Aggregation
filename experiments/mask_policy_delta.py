from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))
import protocol_eval as P
from passportfl.innovation import SCORE_STRUCTURAL, InnovationAuditor

ARCH = os.path.join(HERE, "results", "archives")
TEST = {3: "arch_cifar100_s3_c10_r35_d0.5_s3.npz", 4: "arch_cifar100_s4_c10_r35_d0.5_s4.npz"}
TAU = 3.7859094579543044
MAD = 0.01
START = 12


def auditor_scores(path, seed, attack, victim, policy):
    base = P.load_archive(path)
    clients, rounds = P.archive_shape(base)
    arch = dict(base)
    arch["published"] = {i: {} for i in range(clients)}
    fn = P.ATTACKS[attack] if attack else None
    rng = np.random.default_rng([int(seed), (victim + 1) if victim is not None else 0,
                                 (P.ATTACK_ORDER.index(attack) + 1) if attack else 0])
    aud = InnovationAuditor(mad_floor=MAD, mask_policy=policy)
    out = {}
    for r in range(1, rounds + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        payloads = {}
        for cid in range(clients):
            Cc = arch["C"][cid][r]
            if fn is not None and cid == victim and r >= START:
                Cc = fn(arch, cid, r, rng)
            arch["published"][cid][r] = Cc
            payloads[cid] = (Cc, arch["present"][cid][r])
        rs = aud.observe_round(r, payloads)
        if r >= START:
            for cid, v in rs.per_client.items():
                out[(r, cid)] = float(v.get("suspicion", float("nan")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "mask_policy_delta.json"))
    a = ap.parse_args()

    deltas = []
    n = 0
    flips = {"honest_replay": 0, "victim": 0, "bystander": 0}
    stat_max = hon_max = 0.0
    for seed, fn_ in TEST.items():
        path = os.path.join(ARCH, fn_)
        tasks = [(None, None)] + [(atk, v) for atk in P.ATTACK_ORDER for v in range(10)]
        for attack, victim in tasks:
            sa = auditor_scores(path, seed, attack, victim, "declared")
            sb = auditor_scores(path, seed, attack, victim, "full")
            for key in sa:
                x, y = sa[key], sb[key]
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                n += 1
                deltas.append(abs(x - y))
                if (x > TAU) != (y > TAU):
                    if attack is None:
                        flips["honest_replay"] += 1
                    elif key[1] == victim:
                        flips["victim"] += 1
                    else:
                        flips["bystander"] += 1
                for val in (x, y):
                    if val < SCORE_STRUCTURAL / 4:
                        stat_max = max(stat_max, val)
                if attack is None:
                    hon_max = max(hon_max, x)
        print("seed %d done" % seed, flush=True)

    d = np.array(deltas)
    out = {"scored_client_rounds": n, "tau": TAU,
           "abs_change_max": float(d.max()), "abs_change_mean": float(d.mean()),
           "share_changed": float(np.mean(d > 1e-9)),
           "decision_flips": flips, "decision_flips_total": int(sum(flips.values())),
           "largest_statistical_suspicion": float(stat_max),
           "largest_honest_suspicion_declared": float(hon_max),
           "clip": 250.0}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
