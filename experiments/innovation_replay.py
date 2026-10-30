from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
from typing import Callable, Dict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

import protocol_eval as P
from passportfl.baselines import BASELINES

ARCHIVES = os.path.join(HERE, "results", "archives")
RESULTS = os.path.join(HERE, "results")
ATTACK_START = 12
EVAL_FROM = 12

Scorer = Callable[[Dict[int, np.ndarray]], Dict[int, float]]


def replay_matrices(arch: dict, attack: str, seed: int, scorers: Dict[str, Scorer],
                    attack_start: int = ATTACK_START, eval_from: int = EVAL_FROM) -> dict:
    clients, rounds = P.archive_shape(arch)
    work = dict(arch)
    rng = np.random.default_rng(seed)
    fn = P.ATTACKS[attack]
    first = max(eval_from, attack_start)
    scored = [r for r in range(first, rounds + 1) if (r - 1) in arch["cbar"]]
    out = {k: np.full((clients, len(scored), clients), np.nan) for k in scorers}
    for victim in range(clients):
        work["published"] = {i: {} for i in range(clients)}
        t = 0
        for r in range(1, rounds + 1):
            payloads = {}
            for cid in range(clients):
                C = arch["C"][cid][r]
                if cid == victim and r >= attack_start:
                    C = fn(work, cid, r, rng)
                work["published"][cid][r] = C
                payloads[cid] = C
            Cbar_prev = arch["cbar"].get(r - 1)
            if Cbar_prev is None or r < first:
                continue
            innov = {cid: payloads[cid] - Cbar_prev for cid in payloads}
            for name, sfn in scorers.items():
                try:
                    vals = sfn(innov)
                except Exception:
                    vals = {cid: float("nan") for cid in innov}
                for cid in range(clients):
                    out[name][victim, t, cid] = float(vals.get(cid, float("nan")))
            t += 1
    return {"rounds": scored, "scores": out}


def pos_neg(mat: np.ndarray):
    clients = mat.shape[0]
    eye = np.eye(clients, dtype=bool)[:, None, :]
    pos = mat[np.broadcast_to(eye, mat.shape)]
    neg = mat[np.broadcast_to(~eye, mat.shape)]
    return pos[np.isfinite(pos)], neg[np.isfinite(neg)]


def auc_of(mat: np.ndarray) -> dict:
    pos, neg = pos_neg(mat)
    return {"auc": P.roc_auc(pos, neg), "n_pos": int(pos.size), "n_neg": int(neg.size)}


def reproduce_released(arch: dict, seed: int, tag: str, attacks=None, rules=None) -> dict:
    path = os.path.join(RESULTS, "innovation_%s.json" % tag)
    ref = json.load(open(path, encoding="utf-8"))["results"]
    rules = list(rules or ("krum", "median_dev", "trimmed_mean_dev", "norm_bound"))
    attacks = list(attacks or P.ATTACK_ORDER)
    out = {}
    for a in attacks:
        m = replay_matrices(arch, a, seed, {k: BASELINES[k] for k in rules})
        out[a] = {}
        for k in rules:
            got = auc_of(m["scores"][k])
            want = ref[a][k]
            out[a][k] = {"auc": got["auc"], "released": want["auc"],
                         "abs_diff": abs(got["auc"] - want["auc"]),
                         "counts_match": got["n_pos"] == want["n_pos"]
                         and got["n_neg"] == want["n_neg"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="arch_cifar100_s0_c10_r35_s0.npz")
    ap.add_argument("--tag", default="cifar100_s0", help="innovation_<tag>.json to compare with")
    ap.add_argument("--seed", type=int, default=None, help="default: the archive's seed")
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else int(a.archive.rsplit("_s", 1)[-1].split(".")[0])
    arch = P.load_archive(os.path.join(ARCHIVES, a.archive))
    rep = reproduce_released(arch, seed, a.tag)
    worst = max(v["abs_diff"] for r in rep.values() for v in r.values())
    counts = all(v["counts_match"] for r in rep.values() for v in r.values())
    for att, r in rep.items():
        print("%-12s " % att + "  ".join("%s %.4f (released %.4f)" % (k, v["auc"], v["released"])
                                         for k, v in r.items()))
    print("max |AUC - released| = %.3g; counts match: %s" % (worst, counts))
    return 0 if worst == 0.0 and counts else 1


if __name__ == "__main__":
    raise SystemExit(main())
