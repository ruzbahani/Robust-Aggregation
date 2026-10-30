from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import hashlib
import json
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

import krum_margin
import protocol_eval as P
from krum_margin import pair_stats
from passportfl import baselines
from passportfl.baselines import (bulyan_center, krum_scores, median_deviation,
                                  trimmed_mean_deviation)

ARCHIVES = os.path.join(HERE, "results", "archives")
RESULTS = os.path.join(HERE, "results")

CONFIGS = {
    "cifar100_d05": ("arch_cifar100_s[0-4]_c10_r35*.npz", 10),
    "cifar100_d01": ("arch_cifar100_d01_s*_c10_r35*.npz", 10),
    "cifar10": ("arch_cifar10_s*_c10_r35*.npz", 10),
    "late70": ("arch_late70_s*_c10_r70*.npz", 10),
    "cohort05": ("arch_cohort05_s*_c5_r35*.npz", 5),
    "cohort20": ("arch_cohort20_s*_c20_r35*.npz", 20),
    "cohort40": ("arch_cohort40_s*_c40_r35*.npz", 40),
    "nopassport": ("arch_nopp_s*_c10_r35*.npz", 10),
}
INNOVATION_FILES = {
    "cifar100_d05": "innovation_cifar100_s%d.json",
    "cifar100_d01": "innovation_cifar100_d01_s%d.json",
    "cifar10": "innovation_cifar10_s%d.json",
    "late70": "innovation_late70_s%d.json",
    "cohort05": "innovation_cohort05_s%d.json",
    "cohort20": "innovation_cohort20_s%d.json",
    "cohort40": "innovation_cohort40_s%d.json",
    "nopassport": "innovation_nopp_s%d.json",
}
EXTRA_CONFIGS = {
    "hetero_fedproto": ("arch_hetero_s*_c10_r35*.npz", 10),
    "hetero_fedtgp_lam1": ("arch_fedtgp_lam1_s*_c10_r35*.npz", 10),
    "hetero_fedtgp_lam10": ("arch_fedtgp_lam10_s*_c10_r35*.npz", 10),
    "hetero_fedtgp_lam01": ("arch_fedtgp_lam01_s*_c10_r35*.npz", 10),
}
EXTRA_INNOVATION_FILES = {
    "hetero_fedproto": "innovation_hetero_s%d.json",
    "hetero_fedtgp_lam1": "innovation_fedtgp_lam1_s%d.json",
    "hetero_fedtgp_lam10": "innovation_fedtgp_lam10_s%d.json",
    "hetero_fedtgp_lam01": "innovation_fedtgp_lam01_s%d.json",
}
NO_LIVE_RULE = ("hetero_fedtgp_lam1", "hetero_fedtgp_lam10", "hetero_fedtgp_lam01")
EXTRA_ARCHIVES = os.path.join(RESULTS, "archives_rev2")

F_BYZANTINE = 2
TRIM = 0.2
ATTACK_START = 12
SCOPE_START = 12
BYZ_SCALE = -3.0
ATTACKS = ("global_echo", "peer_copy", "lagged_self", "ar2_replay")
LOO_COHORTS = ("leave_one_out", "leave_one_out_fixed")
COHORTS = ATTACKS + LOO_COHORTS + ("byzantine_ref",)
RULES = ("fedproto", "mean", "krum", "median", "trimmed_mean", "bulyan")
N_DEPENDENT = ("krum", "trimmed_mean", "bulyan")
METRICS = ("shift", "progress", "cos", "shift_own", "progress_own")
ZERO_TOL = 1e-12
SCHEMA = "aggregate-impact-v2"


def pairwise_sq(V: np.ndarray) -> np.ndarray:
    return np.stack([((V - V[i]) ** 2).sum(axis=1) for i in range(len(V))])


def krum_from_distances(D: np.ndarray, k: int) -> np.ndarray:
    return np.array([np.sum(np.sort(np.delete(D[i], i))[:k]) for i in range(len(D))])


def bulyan_from_distances(V: np.ndarray, D: np.ndarray, n_byzantine: int = F_BYZANTINE,
                          n_param: int | None = None):
    n = len(V) if n_param is None else n_param
    f = min(n_byzantine, max(0, (n - 3) // 4))
    pool = list(range(len(V)))
    selected = []
    target = max(1, n - 2 * f)
    while pool and len(selected) < target:
        if len(pool) < 3:
            if n_param is None:
                selected.extend(pool)
                break
            selected.extend(pool[:target - len(selected)])
            break
        fk = min(f, max(0, len(pool) - 3))
        s = krum_from_distances(D[np.ix_(pool, pool)], max(1, len(pool) - fk - 2))
        best = pool[int(np.argmin(s))]
        selected.append(best)
        pool.remove(best)
    S = V[selected]
    beta = max(1, len(selected) - 2 * f)
    med = np.median(S, axis=0)
    order = np.argsort(np.abs(S - med), axis=0)[:beta]
    return np.take_along_axis(S, order, axis=0).mean(axis=0), selected


def bulyan_defined(n: int) -> bool:
    return min(F_BYZANTINE, max(0, (n - 3) // 4)) >= 1


def trimmed_centre(V: np.ndarray, n_param: int | None = None) -> np.ndarray:
    n = len(V)
    c = int(np.floor(TRIM * (n if n_param is None else n_param)))
    S = np.sort(V, axis=0)
    core = S[c:n - c] if n - 2 * c > 0 else S
    return core.mean(axis=0)


def rule_outputs(V: np.ndarray, D: np.ndarray, W: np.ndarray, K: int, d: int,
                 n_param: int | None = None, rules=RULES) -> dict:
    m = len(V)
    n = m if n_param is None else n_param
    scores = krum_from_distances(D, max(1, n - F_BYZANTINE - 2))
    sel = int(np.argmin(scores))
    out = {}
    if "fedproto" in rules:
        out["fedproto"] = np.einsum("ik,ikd->kd", W, V.reshape(m, K, d)).ravel()
    if "mean" in rules:
        out["mean"] = V.mean(axis=0)
    if "krum" in rules:
        out["krum"] = V[sel]
    if "median" in rules:
        out["median"] = np.median(V, axis=0)
    if "trimmed_mean" in rules:
        out["trimmed_mean"] = trimmed_centre(V, n_param)
    bsel = None
    if "bulyan" in rules and bulyan_defined(n):
        out["bulyan"], bsel = bulyan_from_distances(V, D, n_param=n_param)
    return {"out": out, "krum_scores": scores, "krum_sel": sel, "bulyan_sel": bsel}


def recover_weights(arch: dict, clients: int, rounds: int):
    C, cbar, present = arch["C"], arch["cbar"], arch["present"]
    K = cbar[1].shape[0]
    pres = np.stack([present[i][1] for i in range(clients)])
    constant = all(np.array_equal(present[i][r], pres[i])
                   for i in range(clients) for r in range(1, rounds + 1))
    if not constant:
        raise RuntimeError("present masks change across rounds; fixed weights do not apply")
    W = np.zeros((clients, K))
    rs = list(range(1, rounds + 1))
    worst_abs = worst_rel = worst_sum = 0.0
    carried = 0.0
    for k in range(K):
        H = np.flatnonzero(pres[:, k])
        b = np.concatenate([cbar[r][k] for r in rs])
        if not len(H):
            carried = max(carried, max(float(np.abs(cbar[r][k] - cbar[r - 1][k]).max())
                                       for r in rs[1:]))
            continue
        A = np.stack([np.concatenate([C[i][r][k] for r in rs]) for i in H], axis=1)
        w = np.linalg.lstsq(A, b, rcond=None)[0]
        res = A @ w - b
        worst_abs = max(worst_abs, float(np.abs(res).max()))
        worst_rel = max(worst_rel, float(np.abs(res).max() / max(np.abs(b).max(), 1e-300)))
        worst_sum = max(worst_sum, abs(float(w.sum()) - 1.0))
        W[H, k] = w
    info = {"present_constant": constant,
            "max_abs_residual": worst_abs, "max_rel_residual": worst_rel,
            "max_abs_weight_sum_minus_one": worst_sum,
            "min_weight": float(W[pres].min()),
            "holders_min": int(pres.sum(0).min()), "holders_max": int(pres.sum(0).max()),
            "classes_without_holder": int((pres.sum(0) == 0).sum()),
            "carried_class_max_change": carried}
    return W, pres, info


def renormalise(W: np.ndarray) -> np.ndarray:
    s = W.sum(axis=0)
    return np.where(s > 0, W / np.where(s > 0, s, 1.0), 0.0)


def attacked_innovations(base: dict, clients: int, rounds: int, seed: int, attack: str,
                         scored: list) -> np.ndarray:
    fn = P.ATTACKS[attack]
    idx = {r: t for t, r in enumerate(scored)}
    dim = base["cbar"][1].size
    out = np.empty((len(scored), clients, dim))
    for victim in range(clients):
        arch = dict(base)
        arch["published"] = {i: {} for i in range(clients)}
        rng = np.random.default_rng([int(seed), int(victim) + 1,
                                     P.ATTACK_ORDER.index(attack) + 1])
        for r in range(1, rounds + 1):
            for cid in range(clients):
                C = arch["C"][cid][r]
                if cid == victim and r >= ATTACK_START:
                    C = fn(arch, cid, r, rng)
                arch["published"][cid][r] = C
            if r in idx:
                out[idx[r], victim] = (arch["published"][victim][r]
                                       - base["cbar"][r - 1]).ravel()
    return out


def verify_cohort(V: np.ndarray, res: dict) -> dict:
    innov = {i: V[i] for i in range(len(V))}
    ref = krum_scores(innov, n_byzantine=F_BYZANTINE)
    ref_arr = np.array([ref[i] for i in range(len(V))])
    ref_sel = min(ref, key=lambda i: ref[i])
    out = {"krum_scores_identical": bool(np.array_equal(ref_arr, res["krum_scores"])),
           "krum_selection_identical": bool(ref_sel == res["krum_sel"]),
           "krum_tied_minimum": bool(np.sum(ref_arr == ref_arr.min()) > 1)}
    md = median_deviation(innov)
    tm = trimmed_mean_deviation(innov, trim=TRIM)
    o = res["out"]
    out["median_dev_max_abs_diff"] = float(max(
        abs(md[i] - float(np.linalg.norm(V[i] - o["median"]))) for i in innov))
    out["trimmed_dev_max_abs_diff"] = float(max(
        abs(tm[i] - float(np.linalg.norm(V[i] - o["trimmed_mean"]))) for i in innov))
    if "bulyan" in o:
        b = bulyan_center(innov, n_byzantine=F_BYZANTINE)
        out["bulyan_identical"] = bool(np.array_equal(b, o["bulyan"]))
        out["bulyan_max_abs_diff"] = float(np.abs(b - o["bulyan"]).max())
    return out


def verify_fixed(V: np.ndarray, res: dict, n_param: int) -> dict:
    innov = {i: V[i] for i in range(len(V))}
    ref = krum_scores(innov, n_byzantine=F_BYZANTINE - 1)
    ref_arr = np.array([ref[i] for i in range(len(V))])
    c = int(np.floor(TRIM * n_param))
    tm = trimmed_mean_deviation(innov, trim=(c + 0.5) / len(V))
    out = {"krum_scores_identical": bool(np.array_equal(ref_arr, res["krum_scores"])),
           "krum_selection_identical": bool(min(ref, key=lambda i: ref[i]) == res["krum_sel"]),
           "trimmed_dev_max_abs_diff": float(max(
               abs(tm[i] - float(np.linalg.norm(V[i] - res["out"]["trimmed_mean"])))
               for i in innov))}
    if res["bulyan_sel"] is not None:
        f = min(F_BYZANTINE, max(0, (n_param - 3) // 4))
        pool, selected = list(innov), []
        target = max(1, n_param - 2 * f)
        while pool and len(selected) < target:
            if len(pool) < 3:
                selected.extend(pool[:target - len(selected)])
                break
            s = krum_scores({i: innov[i] for i in pool}, min(f, max(0, len(pool) - 3)))
            best = min(s, key=lambda k: s[k])
            selected.append(best)
            pool.remove(best)
        S = np.stack([V[i] for i in selected])
        med = np.median(S, axis=0)
        order = np.argsort(np.abs(S - med), axis=0)[:max(1, len(selected) - 2 * f)]
        b = np.take_along_axis(S, order, axis=0).mean(axis=0)
        out["bulyan_identical"] = bool(np.array_equal(b, res["out"]["bulyan"]))
        out["bulyan_set_size"] = len(selected)
    return out


def run_archive(task: dict) -> dict:
    t0 = time.time()
    path, seed = task["path"], task["seed"]
    base = P.load_archive(path)
    clients, rounds = P.archive_shape(base)
    C, cbar = base["C"], base["cbar"]
    K, d = cbar[1].shape
    scored = [r for r in range(SCOPE_START, rounds + 1) if (r - 1) in cbar]
    T = len(scored)
    rules = tuple(task.get("rules", RULES))

    if "fedproto" in rules:
        W, pres, winfo = recover_weights(base, clients, rounds)
    else:
        W, winfo = None, None
    att = {a: attacked_innovations(base, clients, rounds, seed, a, scored) for a in ATTACKS}

    met = {c: {ru: {m: np.full((T, clients), np.nan) for m in METRICS + ("zero",)}
               for ru in rules} for c in COHORTS}
    clean_met = {ru: {m: np.full(T, np.nan) for m in ("progress", "cos", "zero", "norm_over_mu")}
                 for ru in rules}
    gap_over_mu = np.full(T, np.nan)
    krum_sel = {c: np.full((T, clients), -1, dtype=int) for c in COHORTS}
    krum_payload = {c: np.zeros((T, clients), dtype=bool) for c in ATTACKS + ("byzantine_ref",)}
    bulyan_in = {c: np.zeros((T, clients), dtype=bool) for c in ATTACKS + ("byzantine_ref",)}
    ratio = np.full(T, np.nan)
    ratio_direct = np.full(T, np.nan)
    clean_krum_sel = np.full(T, -1, dtype=int)
    fedproto_identity = 0.0
    echo_zero = 0.0
    auc_lists = {a: {s: ([], []) for s in ("krum", "median_dev", "trimmed_mean_dev")}
                 for a in ATTACKS}
    checks, checks_fixed = [], []
    vrounds = {scored[0], scored[-1]}
    vseats = sorted({0, clients - 1})

    for t, r in enumerate(scored):
        V = np.stack([(C[i][r] - cbar[r - 1]).ravel() for i in range(clients)])
        mu = V.mean(axis=0)
        mu2 = float(mu @ mu)
        mun = np.sqrt(mu2)
        ratio[t] = pair_stats(V)["ratio"]
        ratio_direct[t] = np.sqrt(np.mean(np.sum((V - mu) ** 2, axis=1))) / mun
        D = pairwise_sq(V)
        clean = rule_outputs(V, D, W, K, d, rules=rules)
        clean_krum_sel[t] = clean["krum_sel"]
        if "fedproto" in rules:
            fedproto_identity = max(fedproto_identity, float(np.abs(
                clean["out"]["fedproto"] - (cbar[r] - cbar[r - 1]).ravel()).max()))
            gap_over_mu[t] = float(np.linalg.norm(mu - clean["out"]["fedproto"])) / mun
        own = {}
        for ru, R in clean["out"].items():
            nr = float(np.linalg.norm(R))
            own[ru] = (R, float(R @ R))
            clean_met[ru]["progress"][t] = float(R @ mu) / mu2
            clean_met[ru]["zero"][t] = nr <= ZERO_TOL * mun
            clean_met[ru]["norm_over_mu"][t] = nr / mun
            if nr > ZERO_TOL * mun:
                clean_met[ru]["cos"][t] = float(R @ mu) / (nr * mun)
        if r in vrounds:
            checks.append({"round": r, "cohort": "clean", "seat": None,
                           **verify_cohort(V, clean)})

        for c in COHORTS:
            fixed = c == "leave_one_out_fixed"
            for v in range(clients):
                if c in LOO_COHORTS:
                    keep = [i for i in range(clients) if i != v]
                    Vc, Dc = V[keep], D[np.ix_(keep, keep)]
                    Wc = renormalise(W[keep]) if W is not None else None
                    x = None
                else:
                    if c == "byzantine_ref":
                        x = BYZ_SCALE * np.delete(V, v, axis=0).mean(axis=0)
                    else:
                        x = att[c][t, v]
                    Vc = V.copy()
                    Vc[v] = x
                    dv = ((Vc - x) ** 2).sum(axis=1)
                    Dc = D.copy()
                    Dc[v, :] = dv
                    Dc[:, v] = dv
                    Wc = W
                if fixed:
                    res = rule_outputs(Vc, Dc, Wc, K, d, n_param=clients, rules=N_DEPENDENT)
                else:
                    res = rule_outputs(Vc, Dc, Wc, K, d, rules=rules)
                for ru, R in res["out"].items():
                    nr = float(np.linalg.norm(R))
                    Rc, rc2 = own[ru]
                    dist = float(np.linalg.norm(R - Rc))
                    mm = met[c][ru]
                    mm["shift"][t, v] = dist / mun
                    mm["progress"][t, v] = float(R @ mu) / mu2
                    mm["zero"][t, v] = nr <= ZERO_TOL * mun
                    if nr > ZERO_TOL * mun:
                        mm["cos"][t, v] = float(R @ mu) / (nr * mun)
                    if rc2 > (ZERO_TOL * mun) ** 2:
                        mm["shift_own"][t, v] = dist / np.sqrt(rc2)
                        mm["progress_own"][t, v] = float(R @ Rc) / rc2
                krum_sel[c][t, v] = res["krum_sel"] if c not in LOO_COHORTS else (
                    keep[res["krum_sel"]])
                if x is not None:
                    krum_payload[c][t, v] = bool(np.array_equal(res["out"]["krum"], x))
                    if res["bulyan_sel"] is not None:
                        bulyan_in[c][t, v] = v in res["bulyan_sel"]
                if c in ATTACKS:
                    if c == "global_echo":
                        echo_zero = max(echo_zero, float(np.abs(x).max()))
                    sc = {"krum": res["krum_scores"],
                          "median_dev": np.linalg.norm(Vc - res["out"]["median"], axis=1),
                          "trimmed_mean_dev": np.linalg.norm(
                              Vc - res["out"]["trimmed_mean"], axis=1)}
                    for s, arr in sc.items():
                        auc_lists[c][s][0].append(float(arr[v]))
                        auc_lists[c][s][1].extend(float(arr[j]) for j in range(clients)
                                                  if j != v)
                if r in vrounds and v in vseats:
                    if fixed:
                        checks_fixed.append({"round": r, "seat": v,
                                             **verify_fixed(Vc, res, clients)})
                    else:
                        checks.append({"round": r, "cohort": c, "seat": v,
                                       **verify_cohort(Vc, res)})

    auc = {a: {s: P.roc_auc(np.asarray(p), np.asarray(n)) for s, (p, n) in v.items()}
           for a, v in auc_lists.items()}
    return {"schema": SCHEMA, "config": task["config"], "seed": seed, "rules": list(rules),
            "archive": os.path.basename(path), "clients": clients, "rounds": rounds,
            "scored_rounds": scored, "K": int(K), "d": int(d),
            "weights": winfo, "fedproto_identity_max_abs": fedproto_identity,
            "global_echo_innovation_max_abs": echo_zero,
            "metrics": met, "clean": clean_met, "krum_sel": krum_sel,
            "krum_payload_output": krum_payload, "bulyan_attacker_in_set": bulyan_in,
            "clean_krum_sel": clean_krum_sel, "ratio": ratio, "ratio_direct": ratio_direct,
            "gap_over_mu": gap_over_mu,
            "auc": auc, "checks": checks, "checks_fixed": checks_fixed,
            "elapsed_s": time.time() - t0}


def _r(x, sig=6):
    if x is None or not np.isfinite(x):
        return None
    return float(f"{float(x):.{sig}g}")


def quart(x) -> dict:
    x = np.asarray(x, dtype=float).ravel()
    f = x[np.isfinite(x)]
    if not f.size:
        return {"median": None, "q25": None, "q75": None, "n": 0}
    q25, med, q75 = np.percentile(f, [25, 50, 75])
    return {"median": _r(med), "q25": _r(q25), "q75": _r(q75), "n": int(f.size)}


def eta(n: int, f: int):
    if n - 2 * f - 2 <= 0:
        return None
    return float(np.sqrt(2.0 * (n - f + (f * (n - f - 2) + f ** 2 * (n - f - 1))
                                / (n - 2 * f - 2))))


def summarise(cfg: str, parts: list, krum_margin_ref: dict) -> dict:
    n = parts[0]["clients"]
    live = parts[0].get("rules", RULES)
    rules = [ru for ru in live if not (ru == "bulyan" and not bulyan_defined(n))]
    seat = np.arange(n)

    def pool(c, ru, m):
        return np.concatenate([p["metrics"][c][ru][m].ravel() for p in parts])

    def cohorts_of(ru):
        return [c for c in COHORTS if c != "leave_one_out_fixed" or ru in N_DEPENDENT]

    def over(sh, ref):
        ok = ref > ZERO_TOL
        q = quart(np.where(ok, sh / np.where(ok, ref, 1.0), np.nan))
        q["loo_zero_frac"] = _r(float(1.0 - ok.mean()))
        return q

    summ = {}
    for ru in rules:
        summ[ru] = {}
        loo = pool("leave_one_out", ru, "shift")
        for c in cohorts_of(ru):
            ent = {m: quart(pool(c, ru, m)) for m in METRICS}
            ent["zero_output_frac"] = _r(float(np.nanmean(pool(c, ru, "zero"))))
            ent["negative_progress_frac"] = _r(float(np.mean(pool(c, ru, "progress") < 0)))
            if c not in LOO_COHORTS:
                sh = pool(c, ru, "shift")
                ent["shift_over_loo"] = over(sh, loo)
                if ru in N_DEPENDENT:
                    ent["shift_over_loo_fixed"] = over(
                        sh, pool("leave_one_out_fixed", ru, "shift"))
            if ru == "krum" and c not in LOO_COHORTS:
                sel = np.concatenate([(p["krum_sel"][c] == seat[None, :]).ravel()
                                      for p in parts])
                ent["krum_attacker_selected"] = _r(float(sel.mean()))
                ent["krum_attacker_payload_output"] = _r(float(np.concatenate(
                    [p["krum_payload_output"][c].ravel() for p in parts]).mean()))
            if ru == "bulyan" and c not in LOO_COHORTS:
                ent["bulyan_attacker_in_selection"] = _r(float(np.concatenate(
                    [p["bulyan_attacker_in_set"][c].ravel() for p in parts]).mean()))
            summ[ru][c] = ent

    clean = {ru: {"progress": quart(np.concatenate([p["clean"][ru]["progress"] for p in parts])),
                  "cos": quart(np.concatenate([p["clean"][ru]["cos"] for p in parts])),
                  "norm_over_mu": quart(np.concatenate(
                      [p["clean"][ru]["norm_over_mu"] for p in parts])),
                  "zero_output_frac": _r(float(np.mean(np.concatenate(
                      [p["clean"][ru]["zero"] for p in parts]))))}
             for ru in rules}
    if "fedproto" in rules:
        clean["uniform_minus_weighted_over_mu"] = quart(np.concatenate(
            [p["gap_over_mu"] for p in parts]))

    def nanmed(x):
        f = np.asarray(x, dtype=float)
        f = f[np.isfinite(f)]
        return _r(float(np.median(f))) if f.size else None

    per_seed = {}
    for p in parts:
        ps = {"metric_medians": {}, "krum_attacker_selected": {},
              "clean_progress_median": {}}
        for ru in rules:
            ps["metric_medians"][ru] = {
                c: [nanmed(p["metrics"][c][ru][m]) for m in METRICS] for c in cohorts_of(ru)}
            ps["clean_progress_median"][ru] = nanmed(p["clean"][ru]["progress"])
        for c in COHORTS:
            if c not in LOO_COHORTS:
                ps["krum_attacker_selected"][c] = _r(float(
                    (p["krum_sel"][c] == seat[None, :]).mean()))
        per_seed[str(p["seed"])] = ps

    ratio = np.concatenate([p["ratio"] for p in parts])
    rdir = np.concatenate([p["ratio_direct"] for p in parts])
    e = eta(n, F_BYZANTINE)
    km = krum_margin_ref.get(cfg, {})
    km_set = sorted(km.get("source_archives", []))
    guarantee = {
        "n": n, "f": F_BYZANTINE, "n_gt_2f_plus_2": bool(n > 2 * F_BYZANTINE + 2),
        "eta": _r(e), "ratio_threshold_1_over_eta": _r(1.0 / e) if e else None,
        "ratio": quart(ratio),
        "eta_times_ratio": quart(e * ratio) if e else None,
        "frac_rounds_eta_times_ratio_below_1": _r(float(np.mean(e * ratio < 1.0))) if e else None,
        "eta_times_ratio_min": _r(float(np.min(e * ratio))) if e else None,
        "ratio_pair_stats_vs_direct_max_rel_diff": _r(float(np.max(np.abs(ratio - rdir) / rdir))),
        "krum_margin_json_ratio_median": _r(km.get("ratio_median")),
        "krum_margin_same_archive_set": km_set == sorted(p["archive"] for p in parts),
        "rounds": int(ratio.size),
    }
    fb = min(F_BYZANTINE, max(0, (n - 3) // 4))
    bul = {"f_used": fb, "n_ge_4f_plus_3_with_f_used": bool(n >= 4 * fb + 3) if fb else None,
           "n_ge_4f_plus_3_with_f_2": bool(n >= 4 * F_BYZANTINE + 3),
           "evaluated": bulyan_defined(n)}

    chk = [c for p in parts for c in p["checks"]]
    wchecks = {
        "weights_max_abs_residual": _r(max(p["weights"]["max_abs_residual"] for p in parts), 3),
        "weights_max_rel_residual": _r(max(p["weights"]["max_rel_residual"] for p in parts), 3),
        "weights_max_abs_sum_minus_one": _r(max(p["weights"]["max_abs_weight_sum_minus_one"]
                                                for p in parts), 3),
        "weights_min": _r(min(p["weights"]["min_weight"] for p in parts)),
        "classes_without_holder": max(p["weights"]["classes_without_holder"] for p in parts),
        "fedproto_clean_vs_archived_step_max_abs": _r(max(p["fedproto_identity_max_abs"]
                                                          for p in parts), 3),
    } if "fedproto" in rules else {}
    checks = {
        **wchecks,
        "global_echo_innovation_max_abs": _r(max(p["global_echo_innovation_max_abs"]
                                                 for p in parts), 3),
        "sampled_cohorts": len(chk),
        "krum_scores_identical": sum(c["krum_scores_identical"] for c in chk),
        "krum_selection_identical": sum(c["krum_selection_identical"] for c in chk),
        "krum_samples_with_tied_minimum": sum(c["krum_tied_minimum"] for c in chk),
        "median_dev_max_abs_diff": _r(max(c["median_dev_max_abs_diff"] for c in chk), 3),
        "trimmed_dev_max_abs_diff": _r(max(c["trimmed_dev_max_abs_diff"] for c in chk), 3),
    }
    bchk = [c for c in chk if "bulyan_identical" in c]
    if bchk:
        checks["bulyan_identical"] = sum(c["bulyan_identical"] for c in bchk)
        checks["bulyan_sampled"] = len(bchk)
        checks["bulyan_max_abs_diff"] = _r(max(c["bulyan_max_abs_diff"] for c in bchk), 3)
    fchk = [c for p in parts for c in p["checks_fixed"]]
    fixed = {"sampled_cohorts": len(fchk),
             "krum_scores_identical": sum(c["krum_scores_identical"] for c in fchk),
             "krum_selection_identical": sum(c["krum_selection_identical"] for c in fchk),
             "trimmed_dev_max_abs_diff": _r(max(c["trimmed_dev_max_abs_diff"] for c in fchk), 3)}
    bf = [c for c in fchk if "bulyan_identical" in c]
    if bf:
        fixed["bulyan_identical"] = sum(c["bulyan_identical"] for c in bf)
        fixed["bulyan_set_sizes"] = sorted({c["bulyan_set_size"] for c in bf})
    checks["leave_one_out_fixed"] = fixed

    repro = {}
    for a in ATTACKS:
        repro[a] = {}
        for s in ("krum", "median_dev", "trimmed_mean_dev"):
            mine, ref = [], []
            for p in parts:
                fp = os.path.join(RESULTS, {**INNOVATION_FILES, **EXTRA_INNOVATION_FILES}[cfg]
                                  % p["seed"])
                mine.append(p["auc"][a][s])
                if os.path.exists(fp):
                    ref.append(json.load(open(fp, encoding="utf-8"))["results"][a][s]["auc"])
                else:
                    ref.append(None)
            diffs = [abs(x - y) for x, y in zip(mine, ref) if y is not None]
            repro[a][s] = {"this_replay": [_r(x, 4) for x in mine],
                           "innovation_eval": [_r(x, 4) for x in ref],
                           "max_abs_diff": _r(max(diffs), 3) if diffs else None}

    return {"clients": n, "seeds": [p["seed"] for p in parts],
            "archives": [p["archive"] for p in parts],
            "scored_rounds_per_seed": len(parts[0]["scored_rounds"]),
            "seat_rounds": int(sum(len(p["scored_rounds"]) * n for p in parts)),
            "rules": summ, "clean": clean, "per_seed": per_seed,
            "krum_guarantee": guarantee, "bulyan_precondition": bul,
            "checks": checks, "detection_auc_reproduction": repro}


def closed_loop_reference() -> dict | None:
    p = os.path.join(RESULTS, "utility_seed_spread.json")
    if not os.path.exists(p):
        return None
    u = json.load(open(p, encoding="utf-8"))
    e = u["scenarios"].get("echo")
    return {"source": "results/utility_seed_spread.json",
            "config": u["config"], "metric": u["metric"], "seeds": u["seeds"],
            "honest_per_seed": u["honest_per_seed"], "honest_mean": u["honest_mean"],
            "honest_sd_ddof1": u["honest_sd"], "honest_range": u["honest_range"],
            "echo": e, "sd_ddof": 1}


def _f3(x) -> str:
    return "-" if x is None else "%.3f" % x


def print_tables(summary: dict) -> None:
    for cfg, s in summary.items():
        rules = list(s["rules"])
        print("\n%s  (N=%d, seeds %s, %d seat-rounds)" % (cfg, s["clients"], s["seeds"],
                                                          s["seat_rounds"]))
        for title, pm, sm in (("relative to ||mu||: median progress (median shift)",
                               "progress", "shift"),
                              ("relative to the rule's clean output: median progress_own "
                               "(median shift_own)", "progress_own", "shift_own")):
            print("  " + title)
            print("  %-14s" % "cohort" + "".join("%20s" % ru for ru in rules))
            if pm == "progress":
                print("  %-14s" % "clean" + "".join(
                    "%20s" % _f3(s["clean"][ru]["progress"]["median"]) for ru in rules))
            for c in COHORTS:
                print("  %-14s" % c[:14] + "".join(
                    "%20s" % ("%s (%s)" % (_f3(s["rules"][ru][c][pm]["median"]),
                                           _f3(s["rules"][ru][c][sm]["median"]))
                              if c in s["rules"][ru] else "-")
                    for ru in rules))
        print("  Krum selects the attacker seat: " + ", ".join(
            "%s %.3f" % (c, s["rules"]["krum"][c]["krum_attacker_selected"])
            for c in COHORTS if c not in LOO_COHORTS))
        gap = s["clean"].get("uniform_minus_weighted_over_mu", {}).get("median")
        print("  clean ||R||/||mu||: " + ", ".join(
            "%s %s" % (ru, _f3(s["clean"][ru]["norm_over_mu"]["median"])) for ru in rules)
              + ";  ||mu - fedproto step||/||mu|| %s" % _f3(gap))
        g = s["krum_guarantee"]
        if g["eta"] is not None:
            print("  Krum Prop. 1: eta=%.3f, median ratio %.3f, median eta*ratio %.2f, "
                  "rounds with eta*ratio<1: %.3f"
                  % (g["eta"], g["ratio"]["median"], g["eta_times_ratio"]["median"],
                     g["frac_rounds_eta_times_ratio_below_1"]))
        else:
            print("  Krum Prop. 1: n > 2f+2 fails (n=%d, f=%d); eta undefined"
                  % (g["n"], g["f"]))


def compact_json(obj) -> str:
    s = json.dumps(obj, indent=1)
    for pat, o, c in ((r"\[\s+([^\[\]{}]*?)\s+\]", "[", "]"), (r"\{\s+([^\[\]{}]*?)\s+\}", "{", "}")):
        s = re.sub(pat, lambda m: o + re.sub(r"\s*\n\s*", " ", m.group(1)) + c, s)
    if json.loads(s) != json.loads(json.dumps(obj)):
        raise RuntimeError("compact serialisation changed the payload")
    return s


def part_path(parts_dir: str, cfg: str, seed: int) -> str:
    return os.path.join(parts_dir, "aggregate_impact_%s_s%d.pkl" % (cfg, seed))


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for f in (os.path.abspath(__file__), P.__file__, krum_margin.__file__, baselines.__file__):
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help="comma-separated subset of: " + ", ".join(CONFIGS)
                         + "; revision configurations, read from --extra-archives: "
                         + ", ".join(EXTRA_CONFIGS))
    ap.add_argument("--extra-archives", default=EXTRA_ARCHIVES)
    ap.add_argument("--seeds", default="", help="optional comma-separated seed filter")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--parts", default="",
                    help="directory caching one result per (configuration, seed); cached "
                         "results are reused, so a long run can be split into chunks")
    ap.add_argument("--no-write", action="store_true",
                    help="compute (and cache) the selected archives without writing the JSON")
    ap.add_argument("--out", default=os.path.join(RESULTS, "aggregate_impact.json"))
    a = ap.parse_args()
    t0 = time.time()

    cfgs = [c.strip() for c in a.configs.split(",") if c.strip()]
    unknown = [c for c in cfgs if c not in CONFIGS and c not in EXTRA_CONFIGS]
    if unknown:
        raise SystemExit("unknown configuration(s): %s" % unknown)
    if set(cfgs) & set(EXTRA_CONFIGS) and not a.no_write and os.path.abspath(
            a.out) == os.path.abspath(ap.get_default("out")):
        raise SystemExit("revision configurations go to a separate --out, not %s"
                         % ap.get_default("out"))
    seed_filter = {int(s) for s in a.seeds.split(",") if s.strip()}

    tasks = []
    for cfg in cfgs:
        pat, n = CONFIGS[cfg] if cfg in CONFIGS else EXTRA_CONFIGS[cfg]
        directory = ARCHIVES if cfg in CONFIGS else a.extra_archives
        for path in sorted(glob.glob(os.path.join(directory, pat))):
            seed = int(re.search(r"_s(\d+)\.npz$", os.path.basename(path)).group(1))
            if seed_filter and seed not in seed_filter:
                continue
            tk = {"config": cfg, "seed": seed, "path": path, "clients": n}
            if cfg in NO_LIVE_RULE:
                tk["rules"] = tuple(ru for ru in RULES if ru != "fedproto")
            tasks.append(tk)
    if not tasks:
        raise SystemExit("no archives selected")

    code = code_fingerprint()
    results, todo = {}, []
    for tk in tasks:
        pp = part_path(a.parts, tk["config"], tk["seed"]) if a.parts else None
        if pp and os.path.exists(pp):
            with open(pp, "rb") as fh:
                got = pickle.load(fh)
            if (got.get("schema") == SCHEMA and got.get("code_sha256") == code
                    and got.get("archive") == os.path.basename(tk["path"])):
                results[(tk["config"], tk["seed"])] = got
                continue
        todo.append(tk)
    n_cached = len(tasks) - len(todo)
    print("%d archives selected, %d cached, %d to compute" % (len(tasks), n_cached, len(todo)),
          flush=True)

    def store(res):
        res["code_sha256"] = code
        results[(res["config"], res["seed"])] = res
        print("  %-13s seed %d  %4d rounds x %2d seats  %.0fs"
              % (res["config"], res["seed"], len(res["scored_rounds"]), res["clients"],
                 res["elapsed_s"]), flush=True)
        if a.parts:
            os.makedirs(a.parts, exist_ok=True)
            with open(part_path(a.parts, res["config"], res["seed"]), "wb") as fh:
                pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)

    todo.sort(key=lambda tk: -tk["clients"])
    if a.workers <= 1 or len(todo) <= 1:
        for tk in todo:
            store(run_archive(tk))
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for res in ex.map(run_archive, todo):
                store(res)

    if a.no_write:
        print("computed; JSON not written (--no-write). %.0fs" % (time.time() - t0))
        return 0

    km_path = os.path.join(RESULTS, "krum_margin.json")
    km = json.load(open(km_path, encoding="utf-8"))["measured"] if os.path.exists(km_path) else {}
    summary = {}
    for cfg in cfgs:
        parts = sorted((v for (c, _), v in results.items() if c == cfg), key=lambda p: p["seed"])
        if parts:
            summary[cfg] = summarise(cfg, parts, km)

    print_tables(summary)

    from passportfl.reproducibility import environment
    payload = {
        "meta": {
            "schema": SCHEMA,
            "command": "python experiments/aggregate_impact.py"
                       + ("" if cfgs == list(CONFIGS) else " --configs " + ",".join(cfgs))
                       + (" --seeds " + a.seeds if seed_filter else ""),
            "code_sha256": code,
            "parts_merged": n_cached,
            "compute_s_total": round(sum(v["elapsed_s"] for v in results.values()), 1),
            "f_byzantine": F_BYZANTINE, "trim": TRIM, "attack_start": ATTACK_START,
            "scope_start": SCOPE_START, "byzantine_ref": "I_v = %g * mean_{j != v} I_j" % BYZ_SCALE,
            "seating": "replacement, every seat",
            "per_seed_fields": list(METRICS),
            "open_loop": True,
            "environment": environment(),
            "wall_time_s": round(time.time() - t0, 1),
        },
        "configs": summary,
        "closed_loop_reference": closed_loop_reference(),
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(compact_json(payload))
    print("\nwrote %s (%.0fs)" % (a.out, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
