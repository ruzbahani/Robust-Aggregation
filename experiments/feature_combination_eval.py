from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from passportfl.innovation import SCORE_STAT_MAX, SCORE_STRUCTURAL, InnovationAuditor
from passportfl.temporal_baselines import isolation_forest_fit, isolation_forest_score
from protocol_eval import (ATTACKS, ATTACK_ORDER, roc_auc, archive_shape,
                           discover_archives, fit_threshold, load_archive)

ARCHIVE_DIR = os.path.join(HERE, "results", "archives")
OUT = os.path.join(HERE, "results", "feature_combination.json")
KEYS = ["u1", "u2", "g", "x", "rep"]
ORIENT = {"u1": None, "u2": None, "g": -1.0, "x": +1.0, "rep": +1.0}
MAD_FLOOR = 0.01
MIN_COHORT = 5
SCOPE_START = 12
ATTACK_START = 12
CAL = (0, 1)
TEST = (3, 4)
TARGET = 0.01
COMBINERS = ["max_z", "sum_z", "mean_z", "max_zsd", "logreg_z", "mahal_raw", "iforest_raw"]


def replay(path: str, attack, victim, seed: int) -> list:
    arch = dict(load_archive(path))
    clients, rounds = archive_shape(arch)
    arch["published"] = {i: {} for i in range(clients)}
    fn = ATTACKS[attack] if attack else None
    rng = np.random.default_rng([int(seed), int(victim) + 1 if victim is not None else 0,
                                 ATTACK_ORDER.index(attack) + 1 if attack else 0])
    aud = InnovationAuditor(mad_floor=MAD_FLOOR)
    cells = []
    for r in range(1, rounds + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        payloads = {}
        for cid in range(clients):
            C = arch["C"][cid][r]
            if fn is not None and cid == victim and r >= ATTACK_START:
                C = fn(arch, cid, r, rng)
            arch["published"][cid][r] = C
            payloads[cid] = (C, arch["present"][cid][r])
        rs = aud.observe_round(r, payloads)
        if r < SCOPE_START:
            continue
        raw = np.full((clients, len(KEYS)), np.nan)
        for cid in range(clients):
            v = rs.per_client[cid]
            raw[cid] = [v.get(k, float("nan")) for k in KEYS]
        zsd = np.full_like(raw, np.nan)
        for j in range(len(KEYS)):
            col = raw[:, j]
            fin = np.isfinite(col)
            if fin.sum() >= MIN_COHORT:
                mu, sd = float(col[fin].mean()), float(col[fin].std())
                zsd[fin, j] = (col[fin] - mu) / (sd + MAD_FLOOR)
        for cid in range(clients):
            v = rs.per_client[cid]
            cells.append({
                "seed": seed, "attack": attack, "victim": victim, "round": r, "cid": cid,
                "raw": raw[cid].tolist(),
                "z": [float(v.get("z_" + k, float("nan"))) for k in KEYS],
                "zsd": zsd[cid].tolist(),
                "undefined": bool(v.get("undefined")),
                "structural": bool(v.get("degenerate")) or (aud.carry_check and float(v.get("carry_viol", 0.0)) > 0.0),
                "suspicion": float(v.get("suspicion", float("nan"))),
            })
    return cells


def oriented_score(z: list) -> np.ndarray:
    out = []
    for k, zz in zip(KEYS, z):
        o = ORIENT[k]
        out.append(abs(zz) if o is None else o * zz)
    return np.asarray(out, float)


def logreg_features(cell) -> np.ndarray:
    z = oriented_score(cell["z"])
    fin = np.isfinite(z)
    return np.concatenate([np.where(fin, z, 0.0), fin.astype(float), [1.0]])


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 50) -> np.ndarray:
    n_pos, n_neg = float(y.sum()), float((1 - y).sum())
    w = np.where(y > 0, 0.5 / max(n_pos, 1), 0.5 / max(n_neg, 1)) * len(y)
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ beta))
        g = X.T @ (w * (p - y)) + l2 * beta
        H = (X * (w * p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        beta = beta - step
        if float(np.abs(step).max()) < 1e-8:
            break
    return beta


class Combiners:
    def __init__(self, cal_cells: list, seed: int = 0):
        honest = [c for c in cal_cells if c["attack"] is None and not c["undefined"] and not c["structural"]]
        R = np.asarray([c["raw"] for c in honest], float)
        self.raw_median = np.nanmedian(R, axis=0)
        Rf = np.where(np.isfinite(R), R, self.raw_median)
        self.mu = Rf.mean(0)
        cov = np.cov(Rf.T) + 1e-6 * np.eye(Rf.shape[1])
        self.icov = np.linalg.inv(cov)
        self.forest = isolation_forest_fit(Rf, n_trees=64, seed=seed)
        atk = [c for c in cal_cells if not c["undefined"] and not c["structural"]]
        X = np.stack([logreg_features(c) for c in atk])
        y = np.asarray([1.0 if (c["attack"] is not None and c["cid"] == c["victim"]) else 0.0 for c in atk])
        self.beta = fit_logreg(X, y)
        self.n_fit = {"honest_cells": len(honest), "logreg_cells": len(atk), "logreg_positives": int(y.sum())}

    def score(self, cell, name: str) -> float:
        if cell["undefined"]:
            return float("nan")
        if cell["structural"]:
            return SCORE_STRUCTURAL
        if name in ("max_z", "sum_z", "mean_z"):
            parts = oriented_score(cell["z"])
            parts = parts[np.isfinite(parts)]
            if not parts.size:
                return float("nan")
            if name == "max_z":
                return float(min(parts.max(), SCORE_STAT_MAX))
            return float(parts.sum()) if name == "sum_z" else float(parts.mean())
        if name == "max_zsd":
            parts = oriented_score(cell["zsd"])
            parts = parts[np.isfinite(parts)]
            return float(parts.max()) if parts.size else float("nan")
        if name == "logreg_z":
            return float(logreg_features(cell) @ self.beta)
        r = np.asarray(cell["raw"], float)
        r = np.where(np.isfinite(r), r, self.raw_median)
        if name == "mahal_raw":
            d = r - self.mu
            return float(np.sqrt(d @ self.icov @ d))
        if name == "iforest_raw":
            return isolation_forest_score(self.forest, r)
        raise ValueError(name)


def flag_rate(vals: np.ndarray, thr: float) -> np.ndarray:
    return np.where(np.isfinite(vals), vals, -np.inf) > thr


def evaluate(comb: Combiners, cal_cells: list, test_cells: list) -> dict:
    out = {}
    for name in COMBINERS:
        cal_h = np.asarray([comb.score(c, name) for c in cal_cells if c["attack"] is None], float)
        thr = fit_threshold(cal_h, TARGET)
        test_h = np.asarray([comb.score(c, name) for c in test_cells if c["attack"] is None], float)
        clean = float(flag_rate(test_h, thr).mean())
        per = {}
        for a in ATTACK_ORDER:
            cells = [c for c in test_cells if c["attack"] == a]
            pos = np.asarray([comb.score(c, name) for c in cells if c["cid"] == c["victim"]], float)
            neg = np.asarray([comb.score(c, name) for c in cells if c["cid"] != c["victim"]], float)
            per[a] = {"tpr": float(flag_rate(pos, thr).mean()), "auc": roc_auc(pos, neg),
                      "bystander_fpr": float(flag_rate(neg, thr).mean()),
                      "n_pos": int(len(pos)), "n_neg": int(len(neg))}
        out[name] = {"threshold": thr, "clean_fpr": clean, "n_clean": int(len(test_h)),
                     "per_attack": per,
                     "worst_tpr": min(v["tpr"] for v in per.values()),
                     "worst_auc": min(v["auc"] for v in per.values()),
                     "mean_auc": float(np.mean([v["auc"] for v in per.values()]))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-prefix", default="arch_cifar100_s")
    ap.add_argument("--cal", default="0,1")
    ap.add_argument("--test", default="3,4")
    a = ap.parse_args()
    cal = tuple(int(x) for x in a.cal.split(","))
    test = tuple(int(x) for x in a.test.split(","))
    paths = discover_archives(ARCHIVE_DIR, a.archive_prefix)
    t0 = time.time()
    pools = {}
    for label, seeds in (("CAL", cal), ("TEST", test)):
        cells = []
        for s in seeds:
            clients, _ = archive_shape(load_archive(paths[s]))
            cells += replay(paths[s], None, None, s)
            for atk in ATTACK_ORDER:
                for v in range(clients):
                    cells += replay(paths[s], atk, v, s)
            print("%s seed %d replayed, %d cells so far, %.0fs" % (label, s, len(cells), time.time() - t0), flush=True)
        pools[label] = cells

    comb = Combiners(pools["CAL"], seed=0)
    dev = max(abs(comb.score(c, "max_z") - c["suspicion"]) for c in pools["TEST"]
              if np.isfinite(c["suspicion"]) and np.isfinite(comb.score(c, "max_z")))
    res = evaluate(comb, pools["CAL"], pools["TEST"])
    out = {"cal_seeds": list(cal), "test_seeds": list(test), "target_fpr": TARGET,
           "scope_start": SCOPE_START, "attack_start": ATTACK_START, "mad_floor": MAD_FLOOR,
           "fit_sizes": comb.n_fit, "max_z_vs_auditor_max_abs_dev": float(dev),
           "logreg_coefficients": {"z_" + k: float(b) for k, b in zip(KEYS, comb.beta[:5])} |
                                  {"present_" + k: float(b) for k, b in zip(KEYS, comb.beta[5:10])} |
                                  {"intercept": float(comb.beta[10])},
           "results": res}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("\nsame five statistics, seven combiners; thresholds fitted on CAL honest at 1%%, "
          "evaluated on TEST seeds %s. max_z reproduces the auditor to %.1e." % (list(test), dev))
    print("%-12s %8s %8s %8s" % ("combiner", "clean", "worstTPR", "worstAUC")
          + "".join("%12s" % a[:11] for a in ATTACK_ORDER))
    for name in COMBINERS:
        v = res[name]
        print("%-12s %7.2f%% %7.1f%% %8.3f" % (name, 100 * v["clean_fpr"], 100 * v["worst_tpr"], v["worst_auc"])
              + "".join("%6.1f%%/%.2f" % (100 * v["per_attack"][a]["tpr"], v["per_attack"][a]["auc"]) for a in ATTACK_ORDER))
    print("cells per attack: TPR at the frozen threshold / AUC")
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
