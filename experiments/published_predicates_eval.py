from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from passportfl.reference_baselines import EPS, _cosine
from passportfl.reproducibility import environment
from protocol_eval import (ATTACKS, check_attack_definitions, clopper_pearson,
                           discover_archives, fit_threshold, load_archive)

ARCHIVES = os.path.join(HERE, "results", "archives")
OUT = os.path.join(HERE, "results", "published_predicates.json")
ROOT_CACHE = "fltrust_root_cifar100_d05_r100_s%d.npz"
FLTRUST_REF = os.path.join(HERE, "results", "fltrust_reference_eval.json")

PREDICATES = {
    "l2_bound":         {"quantity": "l2", "side": "upper", "root": False, "role": "primary"},
    "linf_bound":       {"quantity": "linf", "side": "upper", "root": False, "role": "primary"},
    "median_l2_bound":  {"quantity": "med_ratio", "side": "upper", "root": False, "role": "primary"},
    "norm_ball":        {"quantity": "ball", "side": "upper", "root": True, "role": "primary"},
    "zeno":             {"quantity": "zeno", "side": "lower", "root": True, "role": "primary"},
    "cosine_last_update": {"quantity": "cos_drift", "side": "lower", "root": False, "role": "primary"},
    "range_submission": {"quantity": "range_C", "side": "upper", "root": False, "role": "primary"},
    "cosine_last_update_literal": {"quantity": "cos_drift", "side": "upper", "root": False,
                                   "role": "sensitivity"},
    "median_l2_bound_r1.5": {"quantity": "med_ratio", "side": "upper", "root": False,
                             "role": "sensitivity", "fixed": 1.5},
    "l2_lower_bound":   {"quantity": "l2", "side": "lower", "root": False, "role": "sensitivity"},
}

PROTOCOLS = {
    "EIFFeL": ["l2_bound", "norm_ball", "zeno", "cosine_last_update"],
    "RoFL": ["l2_bound", "linf_bound", "median_l2_bound"],
    "ACORN": ["l2_bound", "linf_bound"],
    "Prio": ["range_submission"],
    "two-sided l2 (sensitivity)": ["l2_bound", "l2_lower_bound"],
}


def make_drift_attack(pre_rounds, sign: float = 1.0):
    def attack_drift_extrapolation(arch, cid, r, rng):
        cbar = arch["cbar"]
        target = arch.setdefault("drift_target", {})
        if cid not in target:
            target[cid] = float(np.median([np.linalg.norm(arch["C"][cid][k] - cbar[k - 1])
                                           for k in pre_rounds]))
        step = cbar[r - 1] - cbar[r - 2]
        ns = float(np.linalg.norm(step))
        if ns <= EPS:
            return cbar[r - 1].copy()
        return cbar[r - 1] + (sign * target[cid] / ns) * step
    return attack_drift_extrapolation


def attack_ar2_replay_published(arch, cid, r, rng):
    pub = arch["published"][cid]
    rounds = sorted(k for k in pub if k < r)
    if len(rounds) < 3:
        return arch["C"][cid][r].copy()
    c1, c2, c3 = pub[rounds[-1]], pub[rounds[-2]], pub[rounds[-3]]
    return c1 + (c1 - c2) + 0.25 * ((c1 - c2) - (c2 - c3))


def features(I: np.ndarray, C: np.ndarray, med_norm: float, g0, drift: np.ndarray) -> dict:
    flat = np.asarray(I, np.float64).ravel()
    n = float(np.linalg.norm(flat))
    f = {"l2": n,
         "linf": float(np.abs(flat).max()),
         "med_ratio": n / med_norm if med_norm > EPS else (math.inf if n > EPS else 0.0),
         "range_C": float(np.abs(C).max()),
         "cos_drift": _cosine(flat, np.asarray(drift, np.float64).ravel())}
    if g0 is not None:
        f["ball"] = float(np.linalg.norm(flat - g0))
        f["cos_root"] = _cosine(flat, g0)
        u = flat * (float(np.linalg.norm(g0)) / n) if n > EPS else np.zeros_like(flat)
        f["zeno"] = float(g0 @ u)
    return f


def stack(recs: list) -> dict:
    return {k: np.asarray([x[k] for x in recs], np.float64) for k in recs[0]}


def replay_seed(arch: dict, g0map, attack_fns: dict, attack_start: int, eval_from: int,
                clients: int, seed: int) -> dict:
    cbar = arch["cbar"]
    last = max(cbar)
    scored = [r for r in range(eval_from, last + 1) if (r - 1) in cbar and (r - 2) in cbar]
    drift = {r: cbar[r - 1] - cbar[r - 2] for r in scored}
    hon_I = {r: {c: arch["C"][c][r] - cbar[r - 1] for c in range(clients)} for r in scored}
    hon_n = {r: {c: float(np.linalg.norm(hon_I[r][c])) for c in range(clients)} for r in scored}
    g = (lambda r: g0map[r]) if g0map is not None else (lambda r: None)

    clean = stack([features(hon_I[r][c], arch["C"][c][r], float(np.median(list(hon_n[r].values()))),
                            g(r), drift[r]) for r in scored for c in range(clients)])
    attacked = {}
    for name, fn in attack_fns.items():
        recs = []
        for victim in range(clients):
            rng = np.random.default_rng(1000 * seed + victim)
            work = dict(arch)
            work["published"] = {i: {} for i in range(clients)}
            for r in range(1, last + 1):
                for cid in range(clients):
                    C = arch["C"][cid][r]
                    if cid == victim and r >= attack_start:
                        C = fn(work, cid, r, rng)
                    work["published"][cid][r] = C
                if r not in drift or r < attack_start:
                    continue
                Cv = np.asarray(work["published"][victim][r], np.float64)
                Iv = Cv - cbar[r - 1]
                norms = [hon_n[r][c] for c in range(clients) if c != victim]
                norms.append(float(np.linalg.norm(Iv)))
                recs.append(features(Iv, Cv, float(np.median(norms)), g(r), drift[r]))
        attacked[name] = stack(recs)
    return {"clean": clean, "attacked": attacked, "rounds": scored}


def signed(raw: np.ndarray, side: str) -> np.ndarray:
    return raw if side == "upper" else -raw


def calibrate(cal: np.ndarray, pred: dict, rate: float) -> float:
    if "fixed" in pred:
        return float(pred["fixed"]) if pred["side"] == "upper" else -float(pred["fixed"])
    return fit_threshold(signed(cal, pred["side"]), rate)


def joint_thresholds(scores: list, rate: float):
    n = len(scores[0])
    allowed = int(math.floor(rate * n))
    for m in range(allowed, -1, -1):
        thr = [float(np.sort(s)[n - 1 - m]) for s in scores]
        rej = np.zeros(n, bool)
        for s, t in zip(scores, thr):
            rej |= s > t
        if rej.sum() <= allowed:
            return thr, m, int(rej.sum())
    raise RuntimeError("unreachable: m = 0 rejects nothing")


def accepted(feats: dict, names: list, thr: dict) -> np.ndarray:
    ok = None
    for p in names:
        s = signed(feats[PREDICATES[p]["quantity"]], PREDICATES[p]["side"]) <= thr[p]
        ok = s if ok is None else (ok & s)
    return ok


def rate_rec(ok: np.ndarray) -> dict:
    k, n = int(ok.sum()), int(ok.size)
    p, lo, hi = clopper_pearson(k, n)
    return {"k": k, "n": n, "rate": p, "ci95": [lo, hi]}


def evaluate(names: list, thr: dict, data: dict, seeds: list, attacks: list) -> dict:
    out = {"test_seeds": seeds,
           "honest": rate_rec(np.concatenate([accepted(data[s]["clean"], names, thr) for s in seeds]))}
    out["attacks"] = {}
    for a in attacks:
        per = [accepted(data[s]["attacked"][a], names, thr) for s in seeds]
        rec = rate_rec(np.concatenate(per))
        rec["per_seed_k"] = [int(x.sum()) for x in per]
        out["attacks"][a] = rec
    return out


def bound_value(t: float, side: str) -> float:
    return t if side == "upper" else -t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--test", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    ap.add_argument("--check-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="seeds of results/fltrust_reference_eval.json, for the consistency check")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--attack-start", type=int, default=12)
    ap.add_argument("--eval-from", type=int, default=12)
    ap.add_argument("--rate", type=float, default=0.01, help="nominal CAL rejection rate")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    t0 = time.time()

    pre_rounds = list(range(2, a.attack_start))
    attack_fns = {
        "global_echo": ATTACKS["global_echo"],
        "drift_extrapolation": make_drift_attack(pre_rounds, +1.0),
        "lagged_self": ATTACKS["lagged_self"],
        "ar2_replay": ATTACKS["ar2_replay"],
        "peer_copy": ATTACKS["peer_copy"],
        "ar2_replay_published": attack_ar2_replay_published,
        "drift_reversal": make_drift_attack(pre_rounds, -1.0),
    }
    primary_attacks = list(attack_fns)[:5]

    paths = discover_archives(ARCHIVES, "arch_cifar100_s")
    seeds = sorted(set(a.cal) | set(a.test) | set(a.check_seeds))
    data, roots = {}, {}
    for s in seeds:
        if s not in paths:
            raise SystemExit("no CIFAR-100 d0.5 archive for seed %d" % s)
        arch = load_archive(paths[s])
        g0map = None
        cache = os.path.join(ARCHIVES, ROOT_CACHE % s)
        if os.path.exists(cache):
            z = np.load(cache)
            rs = [r for r in sorted(arch["cbar"]) if (r - 1) in arch["cbar"]]
            assert list(z["rounds"]) == rs[:len(z["g0"])], "root cache rounds misaligned"
            g0map = {int(r): np.asarray(v, np.float64).ravel() for r, v in zip(z["rounds"], z["g0"])}
            roots[s] = os.path.basename(cache)
        fns = attack_fns if s in a.cal or s in a.test else {}
        data[s] = replay_seed(arch, g0map, fns, a.attack_start, a.eval_from, a.clients, s)
        print("seed %d: %s%s  (%.1fs)" % (s, os.path.basename(paths[s]),
              ", root g0" if g0map else "", time.time() - t0), flush=True)

    checks = {"attack_definitions": check_attack_definitions()}
    if os.path.exists(FLTRUST_REF):
        with open(FLTRUST_REF, encoding="utf-8") as fh:
            ref = json.load(fh)["summary"]["global_echo"]
        cs = [s for s in a.check_seeds if s in roots]
        cos_root = np.concatenate([data[s]["clean"]["cos_root"] for s in cs])
        cos_drift = np.concatenate([data[s]["clean"]["cos_drift"] for s in cs])
        checks["fltrust_p0"] = {
            "seeds": cs,
            "root": {"here": float((cos_root <= 0).mean()), "released": ref["root"]["p0_honest"]},
            "drift": {"here": float((cos_drift <= 0).mean()), "released": ref["drift"]["p0_honest"]}}
    ge = np.concatenate([data[s]["attacked"]["global_echo"]["l2"] for s in a.test])
    checks["global_echo_max_innovation_norm"] = float(ge.max())
    err = 0.0
    for s in a.test:
        arch = load_archive(paths[s])
        cb = arch["cbar"]
        tgt = [float(np.median([np.linalg.norm(arch["C"][v][k] - cb[k - 1]) for k in pre_rounds]))
               for v in range(a.clients)]
        l2 = data[s]["attacked"]["drift_extrapolation"]["l2"].reshape(a.clients, -1)
        err = max(err, float(np.max(np.abs(l2 - np.asarray(tgt)[:, None]) / np.asarray(tgt)[:, None])))
    checks["drift_norm_max_rel_error"] = err

    root_cal = all(s in roots for s in a.cal)
    root_test = [s for s in a.test if s in roots]
    bounds, thr = {}, {}
    for p, d in PREDICATES.items():
        if d["root"] and not root_cal:
            continue
        cal = np.concatenate([data[s]["clean"][d["quantity"]] for s in a.cal])
        t = calibrate(cal, d, a.rate)
        thr[p] = t
        rej = signed(cal, d["side"]) > t
        bounds[p] = {"role": d["role"], "side": d["side"],
                     "bound": bound_value(t, d["side"]), "fitted": "fixed" not in d,
                     "cal_rejected": int(rej.sum()), "cal_n": int(rej.size),
                     "cal_median": float(np.median(cal))}
    if "zeno" in bounds:
        bounds["zeno"]["eps"] = -bounds["zeno"]["bound"]

    results = {}
    for p in thr:
        seeds_p = root_test if PREDICATES[p]["root"] else list(a.test)
        results[p] = evaluate([p], thr, data, seeds_p, list(attack_fns))
        results[p]["attacker_median_quantity"] = {
            k: float(np.median(np.concatenate([data[s]["attacked"][k][PREDICATES[p]["quantity"]]
                                                for s in seeds_p]))) for k in attack_fns}

    combined = {}
    for proto, names in PROTOCOLS.items():
        if not all(n in thr for n in names):
            continue
        root = any(PREDICATES[n]["root"] for n in names)
        seeds_p = root_test if root else list(a.test)
        cal_scores = [signed(np.concatenate([data[s]["clean"][PREDICATES[n]["quantity"]] for s in a.cal]),
                             PREDICATES[n]["side"]) for n in names]
        indiv_rej = np.zeros(len(cal_scores[0]), bool)
        for n, sc in zip(names, cal_scores):
            indiv_rej |= sc > thr[n]
        jt, m, jrej = joint_thresholds(cal_scores, a.rate)
        jthr = dict(zip(names, jt))
        combined[proto] = {
            "components": names,
            "individual_bounds": {"cal_rejected": int(indiv_rej.sum()), "cal_n": int(indiv_rej.size),
                                  **evaluate(names, thr, data, seeds_p, list(attack_fns))},
            "joint_1pct": {"per_component_exceedances": m, "cal_rejected": jrej,
                           "cal_n": int(indiv_rej.size),
                           "bounds": {n: bound_value(jthr[n], PREDICATES[n]["side"]) for n in names},
                           **evaluate(names, jthr, data, seeds_p, list(attack_fns))},
        }

    payload = {
        "config": {"dataset": "cifar100", "dirichlet": 0.5, "clients": a.clients,
                   "cal_seeds": a.cal, "test_seeds": a.test, "root_test_seeds": root_test,
                   "attack_start": a.attack_start, "eval_from": a.eval_from,
                   "nominal_cal_rejection": a.rate, "drift_norm_rounds": [pre_rounds[0], pre_rounds[-1]],
                   "attacks_primary": primary_attacks,
                   "attacks_sensitivity": [k for k in attack_fns if k not in primary_attacks],
                   "archives": {str(s): os.path.basename(paths[s]) for s in seeds},
                   "root_caches": {str(s): v for s, v in roots.items()},
                   "zeno_a": 0.0},
        "bounds": bounds,
        "results": results,
        "combined": combined,
        "meta": {"environment": environment(), "checks": checks,
                 "wall_s": round(time.time() - t0, 1)},
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    short = {"global_echo": "echo", "drift_extrapolation": "drift", "lagged_self": "lag",
             "ar2_replay": "ar2", "peer_copy": "peer", "ar2_replay_published": "ar2pub",
             "drift_reversal": "drift-"}
    print("\nAcceptance (%%) on TEST; bounds fitted on CAL seeds %s at %.0f%% honest rejection."
          % (a.cal, 100 * a.rate))
    head = "%-28s %10s %7s" % ("predicate", "bound", "honest") + "".join("%8s" % short[k] for k in attack_fns)
    print(head)
    print("-" * len(head))
    for p, res in results.items():
        print("%-28s %10.4g %7.1f" % (p, bounds[p]["bound"], 100 * res["honest"]["rate"])
              + "".join("%8.1f" % (100 * res["attacks"][k]["rate"]) for k in attack_fns))
    for proto, c in combined.items():
        for mode in ("individual_bounds", "joint_1pct"):
            r = c[mode]
            print("%-28s %10s %7.1f" % ("%s all, %s" % (proto, "indiv" if mode[0] == "i" else "joint1%"),
                                        "", 100 * r["honest"]["rate"])
                  + "".join("%8.1f" % (100 * r["attacks"][k]["rate"]) for k in attack_fns))
    print("\nchecks: %s" % json.dumps(checks))
    print("wrote %s (%.1fs)" % (a.out, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
