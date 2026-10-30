from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from passportfl.reference_baselines import fltrust_scores
from faithful_baselines_eval import innovations
from innovation_eval import ATTACKS, roc_auc

ARCHIVES = os.path.join(HERE, "results", "archives")
OUT = os.path.join(HERE, "results", "fltrust_reference_eval.json")
REFS = ("root", "random", "cohort_mean", "cohort_median", "drift")
FPRS = (0.01, 0.05, 0.10)
CAL_SEEDS = (0, 1)


def tpr_at_fpr(pos, neg, fpr: float) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    best = 0.0
    for tau in np.unique(np.concatenate([neg, pos])):
        if (neg >= tau).mean() <= fpr:
            best = max(best, float((pos >= tau).mean()))
    return best


def threshold_at_fpr(neg, fpr: float) -> float:
    neg = np.asarray(neg, float)
    cands = np.unique(neg)
    ok = [t for t in cands if (neg >= t).mean() <= fpr]
    return float(min(ok)) if ok else float("inf")


def references(I: dict, root: np.ndarray, cbar: dict, r: int, rng: np.random.Generator) -> dict:
    stack = np.stack([np.asarray(I[c], np.float64) for c in sorted(I)])
    return {"root": root,
            "random": rng.standard_normal(root.shape),
            "cohort_mean": stack.mean(0),
            "cohort_median": np.median(stack, 0),
            "drift": np.asarray(cbar[r - 1], np.float64) - np.asarray(cbar[r - 2], np.float64)}


def evaluate_seed(arch_path: str, g0: np.ndarray, attack_start: int, eval_from: int,
                  clients: int, seed: int) -> dict:
    z = np.load(arch_path, allow_pickle=True)
    arch = {"C": z["C"].item(), "present": z["present"].item(), "cbar": z["cbar"].item()}
    rounds = [r for r in sorted(arch["cbar"]) if (r - 1) in arch["cbar"]]
    gmap = {r: g0[i] for i, r in enumerate(rounds[:len(g0)])}
    scores = {a: {ref: {"pos": [], "neg": []} for ref in REFS} for a in ATTACKS}
    for attack in ATTACKS:
        for victim in range(clients):
            rng = np.random.default_rng(1000 * seed + victim)
            rng_ref = np.random.default_rng(7000 + 1000 * seed + victim)
            arch["published"] = {i: {} for i in range(clients)}
            for r in rounds:
                attacked = r >= attack_start
                I = innovations(arch, victim, attack, r, rng, clients, attacked)
                if r not in gmap or r < eval_from or (r - 2) not in arch["cbar"]:
                    continue
                refs = references(I, gmap[r], arch["cbar"], r, rng_ref)
                for ref, g in refs.items():
                    s = fltrust_scores(I, g)
                    for cid in range(clients):
                        v = s.get(cid)
                        if not np.isfinite(v):
                            continue
                        if cid == victim and attacked:
                            scores[attack][ref]["pos"].append(float(v))
                        elif cid != victim:
                            scores[attack][ref]["neg"].append(float(v))
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--attack-start", type=int, default=12)
    ap.add_argument("--eval-from", type=int, default=12)
    ap.add_argument("--root-size", type=int, default=100)
    a = ap.parse_args()

    pats = ["arch_cifar100_s%d_c10_r35_s%d.npz", "arch_cifar100_s%d_c10_r35_d0.5_s%d.npz"]
    per_seed = {}
    for s in a.seeds:
        arch_path = next((os.path.join(ARCHIVES, p % (s, s)) for p in pats
                          if os.path.exists(os.path.join(ARCHIVES, p % (s, s)))), None)
        cache = os.path.join(ARCHIVES, "fltrust_root_cifar100_d05_r%d_s%d.npz" % (a.root_size, s))
        if arch_path is None or not os.path.exists(cache):
            print("seed %d: archive or root cache missing, skipped" % s)
            continue
        g0 = np.load(cache)["g0"]
        print("seed %d: %s" % (s, os.path.basename(arch_path)), flush=True)
        per_seed[s] = evaluate_seed(arch_path, g0, a.attack_start, a.eval_from, a.clients, s)

    summary = {}
    for attack in ATTACKS:
        summary[attack] = {}
        for ref in REFS:
            aucs = [roc_auc(per_seed[s][attack][ref]["pos"], per_seed[s][attack][ref]["neg"]) for s in per_seed]
            pos = np.concatenate([per_seed[s][attack][ref]["pos"] for s in per_seed])
            neg = np.concatenate([per_seed[s][attack][ref]["neg"] for s in per_seed])
            p0_neg = float((neg >= 0.0).mean())
            p0_pos = float((pos >= 0.0).mean())
            cal = [s for s in per_seed if s in CAL_SEEDS]
            tst = [s for s in per_seed if s not in CAL_SEEDS]
            split = {}
            if cal and tst:
                neg_cal = np.concatenate([per_seed[s][attack][ref]["neg"] for s in cal])
                pos_t = np.concatenate([per_seed[s][attack][ref]["pos"] for s in tst])
                neg_t = np.concatenate([per_seed[s][attack][ref]["neg"] for s in tst])
                for f in FPRS:
                    tau = threshold_at_fpr(neg_cal, f)
                    split["fpr%g" % f] = {"tau": tau, "tpr": float((pos_t >= tau).mean()),
                                          "fpr_test": float((neg_t >= tau).mean())}
            summary[attack][ref] = {
                "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                "auc_per_seed": [float(x) for x in aucs],
                "p0_honest": p0_neg, "p0_attacker": p0_pos,
                "auc_if_attacker_at_ceiling": 1.0 - p0_neg / 2.0,
                "n_pos": int(len(pos)), "n_neg": int(len(neg)),
                "tpr_at_fpr_pooled": {"fpr%g" % f: tpr_at_fpr(pos, neg, f) for f in FPRS},
                "split_cal01_test_rest": split,
            }
    out = {"seeds": list(per_seed), "attack_start": a.attack_start, "eval_from": a.eval_from,
           "references": list(REFS), "cal_seeds": list(CAL_SEEDS), "summary": summary}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("\nFLTrust under five references, %d seeds. AUC is the per-seed mean; p0 is the share of"
          " honest client-rounds at trust zero; TPR is pooled at the stated false-alarm rate." % len(per_seed))
    print("%-13s %-14s %7s %7s %9s %8s %8s %8s" % ("attack", "reference", "AUC", "p0", "1-p0/2", "TPR@1%", "TPR@5%", "TPR@10%"))
    for attack in ATTACKS:
        for ref in REFS:
            v = summary[attack][ref]
            t = v["tpr_at_fpr_pooled"]
            print("%-13s %-14s %7.3f %7.3f %9.3f %8.3f %8.3f %8.3f" % (
                attack, ref, v["auc_mean"], v["p0_honest"], v["auc_if_attacker_at_ceiling"],
                t["fpr0.01"], t["fpr0.05"], t["fpr0.1"]))
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
