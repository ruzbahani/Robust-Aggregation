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

import aggregate_impact as AI
import protocol_eval as P
from passportfl.baselines import (bulyan_center, krum_scores, median_deviation,
                                  trimmed_mean_deviation)

ARCHIVES = os.path.join(HERE, "results", "archives")
RESULTS = os.path.join(HERE, "results")
SCHEMA = "inversion-condition-v1"

CONFIGS = {
    "cifar100_d05": ("arch_cifar100_s*_c10_r35*.npz", "cifar100", 0.5),
    "cifar100_d01": ("arch_cifar100_d01_s*_c10_r35*.npz", "cifar100", 0.1),
    "cifar10": ("arch_cifar10_s*_c10_r35*.npz", "cifar10", 0.5),
    "late70": ("arch_late70_s*_c10_r70*.npz", "cifar100", 0.5),
    "cohort05": ("arch_cohort05_s*_c5_r35*.npz", "cifar100", 0.5),
    "cohort20": ("arch_cohort20_s*_c20_r35*.npz", "cifar100", 0.5),
    "cohort40": ("arch_cohort40_s*_c40_r35*.npz", "cifar100", 0.5),
    "nopassport": ("arch_nopp_s*_c10_r35*.npz", "cifar100", 0.5),
}
SCOPE_START = 12
CENTRE_RULES = ("median", "trimmed_mean", "bulyan")
WINDOWS = {"scored": (SCOPE_START, 10 ** 9), "early": (2, SCOPE_START - 1), "all": (2, 10 ** 9)}


def krum_fs(n: int):
    if n > 2 * 2 + 2:
        return (2,), 2
    return (1, 2), 1


def cifar100_labels():
    path = os.path.join(HERE, "data", "cifar-100-python", "train")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        entry = pickle.load(fh, encoding="latin1")
    return np.asarray(entry["fine_labels"])


def count_weights(labels, clients: int, alpha: float, seed: int):
    import torch
    torch.set_num_threads(1)
    from passportfl.data import dirichlet_partition

    class _Labels:
        targets = labels

    K = int(labels.max()) + 1
    parts = dirichlet_partition(_Labels(), clients, alpha, K, seed)
    n = np.stack([np.bincount(labels[np.asarray(p.indices, dtype=np.int64)], minlength=K)
                  for p in parts]).astype(np.float64)
    tot = n.sum(axis=0)
    return np.where(tot > 0, n / np.where(tot > 0, tot, 1.0), 0.0), n > 0


def _q(x) -> float:
    return float(x)


def analyse_archive(task: dict) -> dict:
    t0 = time.time()
    path, seed = task["path"], task["seed"]
    arch = P.load_archive(path)
    N, R = P.archive_shape(arch)
    C, cbar, present = arch["C"], arch["cbar"], arch["present"]
    K, d = cbar[1].shape
    W_ls, pres, winfo = AI.recover_weights(arch, N, R)
    W_cnt = task.get("w_counts")
    fs, f0 = krum_fs(N)
    rules = [ru for ru in CENTRE_RULES if ru != "bulyan" or AI.bulyan_defined(N)]
    rounds = [r for r in range(2, R + 1) if (r - 1) in cbar and r in cbar]
    T = len(rounds)
    eye = np.eye(N, dtype=bool)

    scal = {k: np.full(T, np.nan) for k in (
        "identity_ls", "identity_counts", "identity_own_movement", "nonheld_max_abs",
        "step", "mu_norm", "sigma", "mu_over_sigma", "step_over_sigma", "gap_over_mu",
        "offset_min", "offset_median", "step_over_offset_min", "own_move_median",
        "prev_offset_median", "innovation_median", "cos_own_prev_median",
        "step_over_own_move_median")}
    seat = {}
    for ru in rules:
        for m in ("M_min", "M_med", "centre_norm", "dist_min", "dist_med", "sc1", "sc2",
                  "coff_over_amin"):
            seat["%s_%s" % (ru, m)] = np.full((T, N), np.nan)
    for f in fs:
        for m in ("D_min", "D_med", "zero", "hon_min", "hon_med", "argmin", "tight", "loose"):
            seat["krum_f%d_%s" % (f, m)] = np.full((T, N), np.nan)
    seat["H"] = np.full((T, N), np.nan)
    checks = []
    vrounds = {rounds[0], rounds[-1]}
    vseats = sorted({0, N - 1})

    for t, r in enumerate(rounds):
        Cr = np.stack([C[i][r] for i in range(N)])
        Cp = np.stack([C[i][r - 1] for i in range(N)])
        I = Cr - cbar[r - 1]
        step = cbar[r] - cbar[r - 1]
        scal["identity_ls"][t] = np.abs(np.einsum("ik,ikd->kd", W_ls, I) - step).max()
        Wref = W_ls
        if W_cnt is not None:
            scal["identity_counts"][t] = np.abs(np.einsum("ik,ikd->kd", W_cnt, I) - step).max()
            Wref = W_cnt
        own = Cr - Cp
        scal["identity_own_movement"][t] = np.abs(np.einsum("ik,ikd->kd", Wref, own)
                                                  - step).max()
        scal["nonheld_max_abs"][t] = max((float(np.abs(I[i][~pres[i]]).max())
                                          if (~pres[i]).any() else 0.0) for i in range(N))

        V = I.reshape(N, -1)
        s = step.ravel()
        ns = float(np.linalg.norm(s))
        mu = V.mean(axis=0)
        mun = float(np.linalg.norm(mu))
        sig = float(np.sqrt(np.mean(np.sum((V - mu) ** 2, axis=1))))
        a = np.linalg.norm(V - s, axis=1)
        mv = np.linalg.norm(own.reshape(N, -1), axis=1)
        po = np.linalg.norm((Cp - cbar[r - 1]).reshape(N, -1), axis=1)
        cosv = np.array([float(own[i].ravel() @ (Cp[i] - cbar[r - 1]).ravel())
                         / max(mv[i] * po[i], 1e-300) for i in range(N)])
        scal["step"][t] = ns
        scal["mu_norm"][t] = mun
        scal["sigma"][t] = sig
        scal["mu_over_sigma"][t] = mun / sig
        scal["step_over_sigma"][t] = ns / sig
        scal["gap_over_mu"][t] = float(np.linalg.norm(mu - s)) / mun
        scal["offset_min"][t] = a.min()
        scal["offset_median"][t] = np.median(a)
        scal["step_over_offset_min"][t] = ns / a.min()
        scal["own_move_median"][t] = np.median(mv)
        scal["prev_offset_median"][t] = np.median(po)
        scal["innovation_median"][t] = np.median(np.linalg.norm(V, axis=1))
        scal["cos_own_prev_median"][t] = np.median(cosv)
        scal["step_over_own_move_median"][t] = ns / np.median(mv)

        D = AI.pairwise_sq(V)
        for v in range(N):
            H = np.flatnonzero(np.arange(N) != v)
            Vc = V.copy()
            Vc[v] = 0.0
            dv = ((Vc - Vc[v]) ** 2).sum(axis=1)
            Dc = D.copy()
            Dc[v, :] = dv
            Dc[:, v] = dv
            aH = a[H]
            amin = float(aH.min())
            seat["H"][t, v] = ns < amin

            cents = {"median": np.median(Vc, axis=0), "trimmed_mean": AI.trimmed_centre(Vc)}
            if "bulyan" in rules:
                cents["bulyan"] = AI.bulyan_from_distances(Vc, Dc)[0]
            for ru, c in cents.items():
                nc = float(np.linalg.norm(c))
                dh = np.linalg.norm(V[H] - c, axis=1)
                coff = float(np.linalg.norm(c - s))
                seat[ru + "_M_min"][t, v] = dh.min() - nc
                seat[ru + "_M_med"][t, v] = np.median(dh) - nc
                seat[ru + "_centre_norm"][t, v] = nc
                seat[ru + "_dist_min"][t, v] = dh.min()
                seat[ru + "_dist_med"][t, v] = np.median(dh)
                seat[ru + "_sc1"][t, v] = ns + 2.0 * coff < amin
                seat[ru + "_sc2"][t, v] = nc + coff < amin
                seat[ru + "_coff_over_amin"][t, v] = coff / amin

            DH = D[np.ix_(H, H)]
            DHo = np.where(eye[:N - 1, :N - 1], np.inf, DH)
            low = np.maximum(aH - ns, 0.0) ** 2
            for f in fs:
                k = max(1, N - f - 2)
                sc = AI.krum_from_distances(Dc, k)
                hon = sc[H]
                key = "krum_f%d_" % f
                seat[key + "D_min"][t, v] = hon.min() - sc[v]
                seat[key + "D_med"][t, v] = np.median(hon) - sc[v]
                seat[key + "zero"][t, v] = sc[v]
                seat[key + "hon_min"][t, v] = hon.min()
                seat[key + "hon_med"][t, v] = np.median(hon)
                seat[key + "argmin"][t, v] = int(np.argmin(sc)) == v
                zero_ub = np.sort((ns + aH) ** 2)[:k].sum()
                hon_lb = np.sort(np.hstack([DHo, low[:, None]]), axis=1)[:, :k].sum(axis=1)
                seat[key + "tight"][t, v] = zero_ub < hon_lb.min()
                a_k = np.sort(aH)[k - 1]
                seat[key + "loose"][t, v] = (k * (ns + a_k) ** 2
                                             < (k * np.minimum(DHo.min(axis=1), low)).min())

                if r in vrounds and v in vseats and f == f0:
                    cohort = {i: Vc[i] for i in range(N)}
                    ref = krum_scores(cohort, n_byzantine=f)
                    md = median_deviation(cohort)
                    td = trimmed_mean_deviation(cohort)
                    ent = {"round": r, "seat": v, "f": f,
                           "krum_scores_identical": bool(np.array_equal(
                               sc, np.array([ref[i] for i in range(N)]))),
                           "median_dev_max_abs_diff": float(max(
                               abs(md[i] - float(np.linalg.norm(Vc[i] - cents["median"])))
                               for i in range(N))),
                           "trimmed_dev_max_abs_diff": float(max(
                               abs(td[i] - float(np.linalg.norm(Vc[i] - cents["trimmed_mean"])))
                               for i in range(N)))}
                    if "bulyan" in cents:
                        ent["bulyan_identical"] = bool(np.array_equal(
                            cents["bulyan"], bulyan_center(cohort)))
                    checks.append(ent)

    mask_ok = None
    if task.get("present_counts") is not None:
        mask_ok = bool(np.array_equal(task["present_counts"], pres))
    winfo = dict(winfo)
    if W_cnt is not None:
        winfo["ls_vs_counts_max_abs_weight_diff"] = float(np.abs(W_ls - W_cnt).max())
        winfo["partition_present_mask_matches_archive"] = mask_ok
    return {"config": task["config"], "seed": seed, "archive": os.path.basename(path),
            "clients": N, "rounds_total": R, "K": int(K), "d": int(d), "rounds": rounds,
            "krum_f": list(fs), "krum_f_primary": f0, "rules": rules,
            "weights": winfo, "counts_available": W_cnt is not None,
            "scalars": scal, "seat": seat, "checks": checks,
            "elapsed_s": time.time() - t0}


def _r(x, sig=6):
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    return float(f"{x:.{sig}g}")


def exact_ok(res: dict) -> np.ndarray:
    ok = np.ones_like(res["seat"]["H"], dtype=bool)
    for ru in res["rules"]:
        ok &= res["seat"][ru + "_M_min"] > 0
    ok &= res["seat"]["krum_f%d_D_min" % res["krum_f_primary"]] > 0
    return ok


def failures(res: dict) -> list:
    out = []
    for t, r in enumerate(res["rounds"]):
        for ru in res["rules"]:
            bad = np.flatnonzero(~(res["seat"][ru + "_M_min"][t] > 0))
            if bad.size:
                out.append({"round": r, "rule": ru, "seats": bad.tolist(),
                            "M_min_worst": _r(res["seat"][ru + "_M_min"][t].min())})
        f = res["krum_f_primary"]
        bad = np.flatnonzero(~(res["seat"]["krum_f%d_D_min" % f][t] > 0))
        if bad.size:
            out.append({"round": r, "rule": "krum_f%d" % f, "seats": bad.tolist(),
                        "argmin_selects_zero_seats": int(
                            res["seat"]["krum_f%d_argmin" % f][t].sum()),
                        "D_min_worst": _r(res["seat"]["krum_f%d_D_min" % f][t].min())})
    for f in res["krum_f"]:
        if f == res["krum_f_primary"]:
            continue
        D = res["seat"]["krum_f%d_D_min" % f]
        out.append({"rule": "krum_f%d" % f, "secondary": True,
                    "rounds_with_strict_failure": int((~(D > 0)).any(axis=1).sum()),
                    "rounds": len(res["rounds"]),
                    "seat_rounds_tied_D_min_zero": int((D == 0).sum()),
                    "seat_rounds_argmin_selects_zero": int(
                        res["seat"]["krum_f%d_argmin" % f].sum()),
                    "seat_rounds": int(D.size)})
    return out


def per_round_table(res: dict) -> dict:
    tab = {"round": list(res["rounds"])}
    for k, v in res["scalars"].items():
        tab[k] = [_r(x, 3 if k.startswith(("identity", "nonheld")) else 6) for x in v]
    S = res["seat"]
    for ru in res["rules"]:
        tab[ru + "_exact_frac"] = [_r(x) for x in (S[ru + "_M_min"] > 0).mean(axis=1)]
        tab[ru + "_M_min_worst"] = [_r(x) for x in S[ru + "_M_min"].min(axis=1)]
        tab[ru + "_M_med_worst"] = [_r(x) for x in S[ru + "_M_med"].min(axis=1)]
        tab[ru + "_centre_over_dist_min_worst"] = [_r(x) for x in (
            S[ru + "_centre_norm"] / S[ru + "_dist_min"]).max(axis=1)]
        tab[ru + "_sc1_frac"] = [_r(x) for x in S[ru + "_sc1"].mean(axis=1)]
        tab[ru + "_sc2_frac"] = [_r(x) for x in S[ru + "_sc2"].mean(axis=1)]
        tab[ru + "_coff_over_amin_median"] = [_r(x) for x in np.median(
            S[ru + "_coff_over_amin"], axis=1)]
    for f in res["krum_f"]:
        key = "krum_f%d_" % f
        tab[key + "strict_frac"] = [_r(x) for x in (S[key + "D_min"] > 0).mean(axis=1)]
        tab[key + "argmin_frac"] = [_r(x) for x in S[key + "argmin"].mean(axis=1)]
        tab[key + "D_min_worst"] = [_r(x) for x in S[key + "D_min"].min(axis=1)]
        tab[key + "zero_over_hon_min_worst"] = [_r(x) for x in (
            S[key + "zero"] / S[key + "hon_min"]).max(axis=1)]
        tab[key + "tight_frac"] = [_r(x) for x in S[key + "tight"].mean(axis=1)]
        tab[key + "loose_frac"] = [_r(x) for x in S[key + "loose"].mean(axis=1)]
    tab["H_frac"] = [_r(x) for x in S["H"].mean(axis=1)]
    tab["all_rules_exact_frac"] = [_r(x) for x in exact_ok(res).mean(axis=1)]
    return tab


def first_round_holding(res: dict, key: str, lo: int = 2):
    ok = res["seat"][key].astype(bool).all(axis=1) if key in res["seat"] else exact_ok(res).all(axis=1)
    rounds = np.asarray(res["rounds"])
    sel = rounds >= lo
    ok, rounds = ok[sel], rounds[sel]
    if not ok[-1]:
        return None
    bad = np.flatnonzero(~ok)
    return int(rounds[bad[-1] + 1]) if bad.size else int(rounds[0])


def rng3(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"min": None, "median": None, "max": None}
    return {"min": _r(x.min()), "median": _r(np.median(x)), "max": _r(x.max())}


def summarise(cfg: str, parts: list) -> dict:
    n = parts[0]["clients"]
    rules = parts[0]["rules"]
    fs, f0 = parts[0]["krum_f"], parts[0]["krum_f_primary"]
    out = {"clients": n, "seeds": [p["seed"] for p in parts],
           "archives": [p["archive"] for p in parts], "rules": rules,
           "krum_f": fs, "krum_f_primary": f0}

    def mask(p, lo, hi):
        r = np.asarray(p["rounds"])
        return (r >= lo) & (r <= hi)

    for w, (lo, hi) in WINDOWS.items():
        ent = {}
        ms = [mask(p, lo, hi) for p in parts]
        ent["archive_rounds"] = int(sum(m.sum() for m in ms))
        ent["seat_rounds"] = int(sum(m.sum() * p["clients"] for p, m in zip(parts, ms)))

        def cover(key, test=None):
            vals = [p["seat"][key][m] for p, m in zip(parts, ms)]
            v = np.concatenate([x.ravel() for x in vals])
            v = (v > 0) if test == "pos" else v.astype(bool)
            return _r(v.mean())

        cov = {}
        for ru in rules:
            cov[ru] = {"exact_M_min": cover(ru + "_M_min", "pos"),
                       "exact_M_med": cover(ru + "_M_med", "pos"),
                       "SC1": cover(ru + "_sc1"), "SC2": cover(ru + "_sc2")}
        for f in fs:
            key = "krum_f%d_" % f
            cov["krum_f%d" % f] = {"exact_D_min": cover(key + "D_min", "pos"),
                                   "exact_D_med": cover(key + "D_med", "pos"),
                                   "argmin_selects_zero": cover(key + "argmin"),
                                   "krum_tight": cover(key + "tight"),
                                   "krum_loose": cover(key + "loose")}
        cov["H_step_below_min_offset"] = cover("H")
        allok = np.concatenate([exact_ok(p)[m].ravel() for p, m in zip(parts, ms)])
        cov["all_rules_exact"] = _r(allok.mean())
        ent["coverage_seat_rounds"] = cov

        fail_rounds = []
        for p, m in zip(parts, ms):
            ok = exact_ok(p)
            for t in np.flatnonzero(m):
                if not ok[t].all():
                    r = p["rounds"][t]
                    bad = [ru for ru in rules if not (p["seat"][ru + "_M_min"][t] > 0).all()]
                    if not (p["seat"]["krum_f%d_D_min" % f0][t] > 0).all():
                        bad.append("krum_f%d" % f0)
                    fail_rounds.append({"seed": p["seed"], "round": int(r), "rules": bad,
                                        "seats_failing_any": int((~ok[t]).sum())})
        ent["rounds_with_exact_failure"] = len(fail_rounds)
        ent["exact_failures"] = fail_rounds

        def sc(key):
            return np.concatenate([p["scalars"][key][m] for p, m in zip(parts, ms)])

        ent["ranges"] = {k: rng3(sc(k)) for k in (
            "mu_over_sigma", "step_over_sigma", "step_over_offset_min", "gap_over_mu",
            "step", "offset_min", "offset_median", "own_move_median", "prev_offset_median",
            "cos_own_prev_median", "step_over_own_move_median")}
        ent["ranges"]["coff_over_amin"] = {
            ru: rng3(np.concatenate([p["seat"][ru + "_coff_over_amin"][m].ravel()
                                     for p, m in zip(parts, ms)])) for ru in rules}
        ent["ranges"]["centre_over_dist_min_worst_seat"] = {
            ru: _r(max(float((p["seat"][ru + "_centre_norm"][m] / p["seat"][ru + "_dist_min"][m]).max())
                       for p, m in zip(parts, ms) if m.any())) for ru in rules}
        ent["ranges"]["krum_zero_over_hon_min_worst_seat"] = {
            "f%d" % f: _r(max(float((p["seat"]["krum_f%d_zero" % f][m]
                                     / p["seat"]["krum_f%d_hon_min" % f][m]).max())
                              for p, m in zip(parts, ms) if m.any())) for f in fs}
        out[w] = ent

    held, failed = [], []
    for p in parts:
        ok = exact_ok(p).all(axis=1)
        x = p["scalars"]["step_over_offset_min"]
        held += list(x[ok])
        failed += list(x[~ok])
    out["step_over_offset_min_threshold"] = {
        "max_where_every_seat_inverts": _r(max(held)) if held else None,
        "min_where_some_seat_fails": _r(min(failed)) if failed else None}

    dec = {}
    for lab, pick in (("round_2", lambda p: 2), ("round_12", lambda p: 12),
                      ("last_round", lambda p: p["rounds"][-1])):
        rows = [(p, p["rounds"].index(pick(p))) for p in parts if pick(p) in p["rounds"]]
        dec[lab] = {k: _r(np.median([p["scalars"][k][t] for p, t in rows]))
                    for k in ("own_move_median", "prev_offset_median", "offset_median",
                              "innovation_median", "step", "cos_own_prev_median",
                              "step_over_own_move_median", "mu_norm", "sigma",
                              "mu_over_sigma", "step_over_sigma", "step_over_offset_min")}
    out["decomposition_at"] = dec

    out["first_round_from_which_all_seats_hold"] = {
        p["archive"]: {"exact_all_rules": first_round_holding(p, "__exact__"),
                       **{"%s_SC1" % ru: first_round_holding(p, ru + "_sc1") for ru in rules},
                       **{"%s_SC2" % ru: first_round_holding(p, ru + "_sc2") for ru in rules},
                       "krum_f%d_tight" % f0: first_round_holding(p, "krum_f%d_tight" % f0)}
        for p in parts}

    ident = {"identity_ls_max_abs": _r(max(np.nanmax(p["scalars"]["identity_ls"]) for p in parts), 3),
             "identity_own_movement_max_abs": _r(max(np.nanmax(p["scalars"]["identity_own_movement"])
                                                     for p in parts), 3),
             "nonheld_innovation_max_abs": _r(max(np.nanmax(p["scalars"]["nonheld_max_abs"])
                                                  for p in parts), 3),
             "counts_available": all(p["counts_available"] for p in parts),
             "weights_ls_fit_max_abs_residual": _r(max(p["weights"]["max_abs_residual"]
                                                       for p in parts), 3),
             "weights_ls_max_abs_sum_minus_one": _r(max(p["weights"]["max_abs_weight_sum_minus_one"]
                                                        for p in parts), 3),
             "classes_without_holder": max(p["weights"]["classes_without_holder"] for p in parts)}
    if ident["counts_available"]:
        ident["identity_counts_max_abs"] = _r(max(np.nanmax(p["scalars"]["identity_counts"])
                                                  for p in parts), 3)
        ident["ls_vs_counts_max_abs_weight_diff"] = _r(max(
            p["weights"]["ls_vs_counts_max_abs_weight_diff"] for p in parts), 3)
        ident["partition_present_mask_matches_archive"] = all(
            p["weights"]["partition_present_mask_matches_archive"] for p in parts)
    out["identity"] = ident

    chk = [c for p in parts for c in p["checks"]]
    out["implementation_checks"] = {
        "sampled": len(chk),
        "krum_scores_identical": sum(c["krum_scores_identical"] for c in chk),
        "median_dev_max_abs_diff": _r(max(c["median_dev_max_abs_diff"] for c in chk), 3),
        "trimmed_dev_max_abs_diff": _r(max(c["trimmed_dev_max_abs_diff"] for c in chk), 3)}
    bc = [c for c in chk if "bulyan_identical" in c]
    if bc:
        out["implementation_checks"]["bulyan_identical"] = sum(c["bulyan_identical"] for c in bc)
    return out


def krum_margin_crosscheck(cfg: str, parts: list, km: dict) -> dict | None:
    ref = km.get(cfg)
    if not ref or sorted(ref.get("source_archives", [])) != sorted(p["archive"] for p in parts):
        return None
    if 2 not in parts[0]["krum_f"]:
        return None
    out = {"same_archive_set": True, "rules": {}}
    worst = 0.0
    for ru in parts[0]["rules"]:
        def col(m):
            return np.concatenate([p["seat"][ru + "_" + m][np.asarray(p["rounds"]) >= SCOPE_START, 0]
                                   for p in parts])
        mine = {"centre_norm": float(np.median(col("centre_norm"))),
                "honest_to_centre_median": float(np.median(col("dist_med"))),
                "honest_to_centre_min": float(np.median(col("dist_min"))),
                "centre_inverts_all": float(np.mean(col("M_min") > 0)),
                "centre_inverts": float(np.mean(col("M_med") > 0))}
        r_ = ref["centres_by_rule"][ru]
        diffs = {k: abs(mine[k] - r_[k]) / max(abs(r_[k]), 1e-12) for k in mine}
        worst = max(worst, max(diffs.values()))
        out["rules"][ru] = {"this": {k: _r(v) for k, v in mine.items()},
                            "krum_margin_json": {k: _r(r_[k]) for k in mine}}

    def kcol(m, seat_all=False):
        return np.concatenate([(p["seat"]["krum_f2_" + m][np.asarray(p["rounds"]) >= SCOPE_START]
                                if seat_all else
                                p["seat"]["krum_f2_" + m][np.asarray(p["rounds"]) >= SCOPE_START, 0])
                               for p in parts])
    mine = {"margin_min_positive_frac": float(np.mean(kcol("D_min") > 0)),
            "margin_positive_frac": float(np.mean(kcol("D_med") > 0)),
            "argmin_selected_all_seats": float(np.mean(kcol("argmin", True)))}
    out["krum_f2"] = {"this": mine, "krum_margin_json": {k: ref[k] for k in mine}}
    out["max_rel_diff_centre_medians"] = _r(worst, 3)
    out["krum_rates_max_abs_diff"] = _r(max(abs(mine[k] - ref[k]) for k in mine), 3)
    return out


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for f in (os.path.abspath(__file__), AI.__file__, P.__file__,
              os.path.join(HERE, "passportfl", "baselines.py")):
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def print_summary(summary: dict) -> None:
    print("\n%-13s %4s %9s | %-34s | %-22s | %s" % (
        "config", "N", "exact12+", "SC1 med/tm/bul (12+)", "SC2 med/tm/bul", "Krum tight/loose (12+)"))
    for cfg, s in summary.items():
        sc = s["scored"]["coverage_seat_rounds"]
        f0 = s["krum_f_primary"]

        def g(ru, k):
            return "%.2f" % sc[ru][k] if ru in sc else " n/a"
        print("%-13s %4d %9.3f | %s / %s / %s                 | %s / %s / %s     | %.2f / %.2f (f=%d)" % (
            cfg, s["clients"], sc["all_rules_exact"],
            g("median", "SC1"), g("trimmed_mean", "SC1"), g("bulyan", "SC1"),
            g("median", "SC2"), g("trimmed_mean", "SC2"), g("bulyan", "SC2"),
            sc["krum_f%d" % f0]["krum_tight"], sc["krum_f%d" % f0]["krum_loose"], f0))
    print("\n%-13s %16s %16s %18s %14s  %s" % ("config", "mu/sigma (12+)", "s/sigma (12+)",
                                               "s/min a (12+)", "fail 2-11", "failing rounds"))
    for cfg, s in summary.items():
        rg = s["scored"]["ranges"]
        e = s["early"]
        print("%-13s %7.2f-%-8.2f %7.2f-%-8.2f %8.3f-%-9.3f %5d of %-5d  %s" % (
            cfg, rg["mu_over_sigma"]["min"], rg["mu_over_sigma"]["max"],
            rg["step_over_sigma"]["min"], rg["step_over_sigma"]["max"],
            rg["step_over_offset_min"]["min"], rg["step_over_offset_min"]["max"],
            e["rounds_with_exact_failure"], e["archive_rounds"],
            sorted({(x["seed"], x["round"]) for x in s["all"]["exact_failures"]})))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help="comma-separated subset of: " + ", ".join(CONFIGS))
    ap.add_argument("--seeds", default="", help="optional comma-separated seed filter")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-counts", action="store_true",
                    help="skip the partition-count weights (least-squares weights only)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "inversion_condition.json"))
    a = ap.parse_args()
    t0 = time.time()

    cfgs = [c.strip() for c in a.configs.split(",") if c.strip()]
    unknown = [c for c in cfgs if c not in CONFIGS]
    if unknown:
        raise SystemExit("unknown configuration(s): %s" % unknown)
    seed_filter = {int(s) for s in a.seeds.split(",") if s.strip()}

    labels = None if a.no_counts else cifar100_labels()
    tasks = []
    for cfg in cfgs:
        pat, dataset, alpha = CONFIGS[cfg]
        for path in sorted(glob.glob(os.path.join(ARCHIVES, pat))):
            seed = int(re.search(r"_s(\d+)\.npz$", os.path.basename(path)).group(1))
            if seed_filter and seed not in seed_filter:
                continue
            tk = {"config": cfg, "seed": seed, "path": path, "w_counts": None,
                  "present_counts": None}
            if dataset == "cifar100" and labels is not None:
                n = int(re.search(r"_c(\d+)_r", os.path.basename(path)).group(1))
                tk["w_counts"], tk["present_counts"] = count_weights(labels, n, alpha, seed)
            tasks.append(tk)
    if not tasks:
        raise SystemExit("no archives selected")
    print("%d archives; partition counts for %d" % (len(tasks),
          sum(t["w_counts"] is not None for t in tasks)), flush=True)

    tasks.sort(key=lambda tk: -int(re.search(r"_c(\d+)_r", os.path.basename(tk["path"])).group(1)))
    results = []

    def store(res):
        results.append(res)
        print("  %-13s seed %d  N=%2d  %2d rounds  %.0fs" % (res["config"], res["seed"],
              res["clients"], len(res["rounds"]), res["elapsed_s"]), flush=True)

    if a.workers <= 1 or len(tasks) <= 1:
        for tk in tasks:
            store(analyse_archive(tk))
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for res in ex.map(analyse_archive, tasks):
                store(res)

    km_path = os.path.join(RESULTS, "krum_margin.json")
    km = json.load(open(km_path, encoding="utf-8"))["measured"] if os.path.exists(km_path) else {}
    summary, archives, cross = {}, {}, {}
    for cfg in cfgs:
        parts = sorted((v for v in results if v["config"] == cfg), key=lambda p: p["seed"])
        if not parts:
            continue
        summary[cfg] = summarise(cfg, parts)
        cc = krum_margin_crosscheck(cfg, parts, km)
        if cc is not None:
            cross[cfg] = cc
        for p in parts:
            archives[p["archive"]] = {
                "config": cfg, "seed": p["seed"], "clients": p["clients"],
                "krum_f": p["krum_f"], "krum_f_primary": p["krum_f_primary"],
                "weights": {k: (_r(v, 3) if isinstance(v, float) else v)
                            for k, v in p["weights"].items()},
                "failures": failures(p), "per_round": per_round_table(p),
                "elapsed_s": round(p["elapsed_s"], 1)}

    print_summary(summary)

    from passportfl.reproducibility import environment
    payload = {
        "meta": {
            "schema": SCHEMA,
            "command": "python experiments/inversion_condition.py"
                       + ("" if cfgs == list(CONFIGS) else " --configs " + ",".join(cfgs))
                       + (" --seeds " + a.seeds if seed_filter else "")
                       + (" --no-counts" if a.no_counts else ""),
            "code_sha256": code_fingerprint(),
            "rounds": "every round r >= 2 (innovation needs Cbar^{r-1}); windows: scored "
                      "r >= %d, early 2..%d" % (SCOPE_START, SCOPE_START - 1),
            "environment": environment(),
            "wall_time_s": round(time.time() - t0, 1),
            "compute_s_total": round(sum(p["elapsed_s"] for p in results), 1),
        },
        "configs": summary,
        "krum_margin_crosscheck": cross,
        "archives": archives,
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(AI.compact_json(payload))
    print("\nwrote %s (%.0fs)" % (a.out, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
