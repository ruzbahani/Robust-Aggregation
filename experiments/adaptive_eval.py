from __future__ import annotations

import argparse, itertools, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

from passportfl.innovation import EPS, SCORE_STRUCTURAL, InnovationAuditor
from passportfl.baselines import BASELINES, cosine_to_cohort, score_round
from passportfl.reference_baselines import FoolsGold
from passportfl.adaptive_attack import AttackConfig, WhiteBoxAttacker, FIXED_POINT_SCALE
from passportfl.innovation import MAD_SCALE

FG_OBSERVERS = {"foolsgold_history": 0.0, "foolsgold_history_imp10": 0.10}

STATS = ["suspicion", "suspicion_g2s", "suspicion_cos"] + list(BASELINES) + list(FG_OBSERVERS)
MIN_COHORT = 5


def robust_z_score(vals: Dict[int, float], mad_floor: float) -> Dict[int, float]:
    fin = [v for v in vals.values() if np.isfinite(v)]
    if len(fin) < MIN_COHORT:
        return {c: float("nan") for c in vals}
    arr = np.asarray(fin, float)
    med = float(np.median(arr))
    scale = MAD_SCALE * float(np.median(np.abs(arr - med))) + mad_floor
    return {c: ((v - med) / scale if np.isfinite(v) and scale > 0 else float("nan"))
            for c, v in vals.items()}


def detector_variant(v: dict, g_two_sided: bool = False, z_cos: float = float("nan")) -> float:
    if v.get("undefined"):
        return float("nan")
    if v.get("degenerate"):
        return SCORE_STRUCTURAL
    parts = []
    for key, oriented in (("u1", None), ("u2", None),
                          ("g", None if g_two_sided else -1.0),
                          ("x", +1.0), ("rep", +1.0)):
        zz = v.get("z_" + key, float("nan"))
        if np.isfinite(zz):
            parts.append(abs(zz) if oriented is None else oriented * zz)
    if np.isfinite(z_cos):
        parts.append(z_cos)
    return max(parts) if parts else float("nan")

ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "archives")
_CACHE: Dict[str, dict] = {}


def load_archive(path: str) -> dict:
    if path not in _CACHE:
        z = np.load(path, allow_pickle=True)
        _CACHE.clear()
        _CACHE[path] = {"C": z["C"].item(), "present": z["present"].item(), "cbar": z["cbar"].item()}
    return _CACHE[path]


def roc_auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    pos = np.asarray([p for p in pos if np.isfinite(p)], float)
    neg = np.asarray([n for n in neg if np.isfinite(n)], float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt)); np.add.at(s, inv, ranks); ranks = (s / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def score_one_archive(task: dict) -> dict:
    torch.set_num_threads(1)
    arch = load_archive(task["path"])
    clients = len(arch["C"])
    rounds = max(max(v) for v in arch["C"].values())
    attack_start = task["attack_start"]
    scope_start = task["scope_start"]
    victim = task["victim"]
    mad_floor = task["mad_floor"]

    cfg = AttackConfig(phi=task["phi"], beta=task["beta"], knowledge=task["knowledge"],
                       basis_steps=task["basis_steps"], full_steps=task["full_steps"],
                       restarts=task["restarts"], mad_floor=mad_floor,
                       g_two_sided=bool(task.get("target") == "suspicion_g2s"),
                       use_cos=bool(task.get("target") == "suspicion_cos"),
                       aim=task.get("aim", "min"))
    atk = WhiteBoxAttacker(cfg, seed=1000 * task["seed"] + victim)

    aud = InnovationAuditor(mad_floor=mad_floor)
    fg = {name: FoolsGold(importance=imp) for name, imp in FG_OBSERVERS.items()}
    mats: Dict[str, List[List[float]]] = {k: [] for k in STATS}
    rec_rounds: List[int] = []
    pred, real = [], []
    fid, rel_err, budget_used, innov_ratio = [], [], [], []
    statuses: Dict[str, int] = {}

    for r in range(1, rounds + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        present_r = {cid: arch["present"][cid][r] for cid in range(clients)}
        payloads = {}
        info = None
        for cid in range(clients):
            payloads[cid] = (arch["C"][cid][r], present_r[cid])
        if r >= attack_start and not task.get("honest"):
            peer_C = ({j: arch["C"][j][r] for j in range(clients) if j != victim}
                      if cfg.knowledge == "peer_current" else None)
            if cfg.beta > 0:
                Cp, info = atk.craft(aud, r, victim, arch["C"][victim][r], present_r, peer_C)
            else:
                Cbar_prev = aud.cbar.get(r - 1)
                Cp = (arch["C"][victim][r].copy() if Cbar_prev is None
                      else atk._cheap_objective(Cbar_prev, arch["C"][victim][r]))
                Cp = np.round(Cp * FIXED_POINT_SCALE) / FIXED_POINT_SCALE
                info = {"status": "cheap_only", "predicted_suspicion": float("nan")}
            payloads[victim] = (Cp, present_r[victim])
            statuses[info["status"]] = statuses.get(info["status"], 0) + 1
            Cb = aud.cbar.get(r - 1)
            if Cb is not None:
                T = np.asarray(present_r[victim], bool) & aud.hist[victim][-1].present \
                    if aud.hist.get(victim) else np.asarray(present_r[victim], bool)
                Ih = (arch["C"][victim][r] - Cb)[T]
                Ip = (Cp - Cb)[T]
                nh, npb = float(np.linalg.norm(Ih)), float(np.linalg.norm(Ip))
                if nh > EPS and npb > EPS:
                    fid.append(float((Ih * Ip).sum() / (nh * npb)))
                    innov_ratio.append(npb / nh)
                    rel_err.append(float(np.linalg.norm(Cp - arch["C"][victim][r])) / nh)
                if np.isfinite(info.get("budget", float("nan"))) and info.get("budget", 0) > 0:
                    Cc = atk._cheap_objective(Cb, arch["C"][victim][r])
                    budget_used.append(float(np.linalg.norm(Cp - Cc)) / info["budget"])

        rs = aud.observe_round(r, payloads)

        base: Dict[str, Dict[int, float]] = {}
        zcos: Dict[int, float] = {c: float("nan") for c in range(clients)}
        Cb = aud.cbar.get(r - 1)
        if Cb is not None:
            innov = {cid: (payloads[cid][0] - Cb) for cid in payloads}
            if len(innov) >= 3:
                base = score_round(innov)
                zcos = robust_z_score(cosine_to_cohort(innov), mad_floor)
                for name, f in fg.items():
                    base[name] = f.update(innov)

        if r < scope_start:
            continue
        rec_rounds.append(r)
        mats["suspicion"].append([float(rs.per_client[c].get("suspicion", float("nan")))
                                  for c in range(clients)])
        mats["suspicion_g2s"].append([detector_variant(rs.per_client[c], g_two_sided=True)
                                      for c in range(clients)])
        mats["suspicion_cos"].append([detector_variant(rs.per_client[c], z_cos=zcos.get(c, float("nan")))
                                      for c in range(clients)])
        for b in list(BASELINES) + list(FG_OBSERVERS):
            vals = base.get(b, {})
            mats[b].append([float(vals.get(c, float("nan"))) for c in range(clients)])
        if r >= attack_start and info is not None:
            tgt = task.get("target") or "suspicion"
            pred.append(info.get("predicted_suspicion", float("nan")))
            real.append(float(mats[tgt][-1][victim]))

    M = {k: np.asarray(v, dtype=np.float64) for k, v in mats.items()}
    pr = np.asarray(pred, float); re_ = np.asarray(real, float)
    ok = np.isfinite(pr) & np.isfinite(re_)
    return {
        "key": {k: task.get(k) for k in ("seed", "victim", "phi", "beta", "knowledge",
                                        "target", "aim", "honest")},
        "rounds": rec_rounds,
        "mat": M,
        "victim": victim,
        "diag": {
            "pred_vs_real_max_abs": float(np.max(np.abs(pr[ok] - re_[ok]))) if ok.any() else None,
            "pred_vs_real_mean_abs": float(np.mean(np.abs(pr[ok] - re_[ok]))) if ok.any() else None,
            "pred_mean": float(np.mean(pr[ok])) if ok.any() else None,
            "real_mean": float(np.mean(re_[ok])) if ok.any() else None,
            "fidelity_mean": float(np.mean(fid)) if fid else None,
            "innov_norm_ratio_mean": float(np.mean(innov_ratio)) if innov_ratio else None,
            "payload_err_rel_mean": float(np.mean(rel_err)) if rel_err else None,
            "budget_used_mean": float(np.mean(budget_used)) if budget_used else None,
            "statuses": statuses,
        },
    }


def summarise_scores(passes: List[dict], attack_start: int, clients: int, taus: Dict[str, float],
              test_seeds: Sequence[int]) -> dict:
    diag_keys = ["fidelity_mean", "innov_norm_ratio_mean", "payload_err_rel_mean",
                 "budget_used_mean", "pred_vs_real_max_abs", "pred_vs_real_mean_abs",
                 "pred_mean", "real_mean"]
    acc = {k: [] for k in diag_keys}
    per_stat: Dict[str, dict] = {}
    for stat in STATS:
        pos, neg = [], []
        tp_k = tp_n = run_k = run_n = 0
        for p in passes:
            rounds = np.asarray(p["rounds"])
            sel = rounds >= attack_start
            if not sel.any():
                continue
            mat = p["mat"][stat][sel]
            v = p["victim"]
            pos.extend(mat[:, v].tolist())
            mask = np.ones(clients, bool); mask[v] = False
            neg.extend(mat[:, mask].ravel().tolist())
            if stat in taus and p["key"]["seed"] in test_seeds:
                f = np.where(np.isfinite(mat[:, v]), mat[:, v], -np.inf) > taus[stat]
                tp_k += int(f.sum()); tp_n += int(f.size)
                run_n += 1
                run_k += int(bool(f.any()))
        per_stat[stat] = {
            "auc": roc_auc(pos, neg),
            "n_pos": len(pos), "n_neg": len(neg),
            "pos_median": float(np.nanmedian(pos)) if pos else None,
            "neg_median": float(np.nanmedian(neg)) if neg else None,
            "tpr_client_round_TEST": (tp_k / tp_n) if tp_n else None,
            "tpr_client_run_TEST": (run_k / run_n) if run_n else None,
            "n_test_cells": tp_n, "n_test_runs": run_n,
        }
    for p in passes:
        for k in diag_keys:
            if p["diag"].get(k) is not None:
                acc[k].append(p["diag"][k])
    return {"stats": per_stat,
            **{k: (float(np.mean(v)) if v else None) for k, v in acc.items()}}


def fit_threshold(vals: np.ndarray, target: float) -> float:
    v = np.where(np.isfinite(vals), vals, -np.inf).ravel()
    n = v.size
    if n == 0:
        return float("inf")
    m = int(math.floor(target * n))
    s = np.sort(v)
    idx = n - 1 - m
    return float(s[idx]) if idx >= 0 else float("-inf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", default=ARCHIVE_DIR)
    ap.add_argument("--prefix", default="arch_cifar100_s")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--cal-seeds", default="0,1")
    ap.add_argument("--test-seeds", default="3,4")
    ap.add_argument("--phis", default="0,0.1,0.25,0.5,1.0")
    ap.add_argument("--betas", default="0,0.5,1.0,2.0,4.0")
    ap.add_argument("--knowledge", default="peer_current,peer_lagged")
    ap.add_argument("--aims", default="min",
                    help="min = drive the score to zero; mimic = drive it to the cohort's "
                         "own typical score so the payload is not conspicuously normal")
    ap.add_argument("--targets", default="suspicion",
                    help="which score the attacker minimises: suspicion (the published one) "
                         "and/or suspicion_g2s (the hardened two-sided-magnitude variant)")
    ap.add_argument("--attack-start", type=int, default=12)
    ap.add_argument("--scope-start", type=int, default=12)
    ap.add_argument("--mad-floor", type=float, default=1e-2)
    ap.add_argument("--fpr-target", type=float, default=0.01)
    ap.add_argument("--basis-steps", type=int, default=150)
    ap.add_argument("--full-steps", type=int, default=200)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--victims", default="")
    ap.add_argument("--workers", type=int, default=min(24, (os.cpu_count() or 4)))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "results", "adaptive_eval.json"))
    a = ap.parse_args()

    ints = lambda s: [int(x) for x in s.split(",") if x.strip()]
    fls = lambda s: [float(x) for x in s.split(",") if x.strip()]
    seeds, cal_seeds, test_seeds = ints(a.seeds), ints(a.cal_seeds), ints(a.test_seeds)
    phis, betas = fls(a.phis), fls(a.betas)
    knows = [k.strip() for k in a.knowledge.split(",") if k.strip()]
    targets = [t.strip() for t in a.targets.split(",") if t.strip()]
    aims = [t.strip() for t in a.aims.split(",") if t.strip()]

    paths = {}
    import re
    for fn in sorted(os.listdir(a.archives)):
        if fn.startswith(a.prefix) and fn.endswith(".npz"):
            m = re.search(r"_s(\d+)\.npz$", fn)
            if m:
                paths[int(m.group(1))] = os.path.join(a.archives, fn)
    seeds = [s for s in seeds if s in paths]
    if not seeds:
        print("no archives found"); return 1
    clients = len(load_archive(paths[seeds[0]])["C"])
    victims = ints(a.victims) if a.victims else list(range(clients))
    common = {"attack_start": a.attack_start, "scope_start": a.scope_start,
              "mad_floor": a.mad_floor, "basis_steps": a.basis_steps,
              "full_steps": a.full_steps, "restarts": a.restarts}

    t0 = time.time()

    honest_tasks = [dict(common, path=paths[s], seed=s, victim=0, phi=1.0, beta=0.0,
                         knowledge=knows[0], target="suspicion", aim="min", honest=True)
                    for s in seeds]
    hres = run_all(honest_tasks, a.workers, t0, "honest")
    cal_h = [r for r in hres if r["key"]["seed"] in cal_seeds]
    test_h = [r for r in hres if r["key"]["seed"] in test_seeds]
    taus, fpr = {}, {}
    for stat in STATS:
        cal = np.concatenate([r["mat"][stat].ravel() for r in cal_h]) if cal_h else np.zeros(0)
        taus[stat] = fit_threshold(cal, a.fpr_target)
        if test_h:
            v = np.concatenate([r["mat"][stat].ravel() for r in test_h])
            f = np.where(np.isfinite(v), v, -np.inf) > taus[stat]
            fpr[stat] = {"realised": float(f.mean()), "n": int(f.size)}
    print("\nthresholds fitted on CAL honest client-rounds (seeds %s, n=%d) at nominal %g%%"
          % (cal_seeds, int(sum(r["mat"]["suspicion"].size for r in cal_h)), 100 * a.fpr_target))
    for stat in ("suspicion", "suspicion_g2s", "suspicion_cos"):
        print("  %-16s tau = %10.4f   realised honest FPR on TEST %s = %.3f%% (n=%d)"
              % (stat, taus[stat], test_seeds, 100 * fpr.get(stat, {}).get("realised", float("nan")),
                 fpr.get(stat, {}).get("n", 0)))

    tasks = []
    for aim, tgt, know, phi, beta, s, v in itertools.product(aims, targets, knows, phis,
                                                             betas, seeds, victims):
        if beta == 0.0 and (know != knows[0] or tgt != targets[0] or aim != aims[0]):
            continue
        tasks.append(dict(common, path=paths[s], seed=s, victim=v, phi=phi, beta=beta,
                          knowledge=know, target=tgt, aim=aim, honest=False))
    print("\n%d attack passes = %d cells x %d seeds x %d victims"
          % (len(tasks), len(tasks) // max(1, len(seeds) * len(victims)), len(seeds), len(victims)))
    results = run_all(tasks, a.workers, t0, "attack")

    cells: Dict[str, List[dict]] = {}
    for r in results:
        k = "%s/%s|%s|phi=%g|beta=%g" % (r["key"]["target"], r["key"]["aim"],
                                         r["key"]["knowledge"], r["key"]["phi"],
                                         r["key"]["beta"])
        cells.setdefault(k, []).append(r)
    out = {"config": vars(a), "clients": clients, "seeds": seeds, "cal_seeds": cal_seeds,
           "test_seeds": test_seeds, "thresholds": taus, "honest_fpr_TEST": fpr, "cells": {}}
    for k, ps in sorted(cells.items()):
        out["cells"][k] = summarise_scores(ps, a.attack_start, clients, taus, test_seeds)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(os.path.abspath(a.out), "w") as f:
        json.dump(out, f, indent=1, default=float)

    report(out, taus, fpr)
    print("\nwrote %s   (%.1fs)" % (os.path.abspath(a.out), time.time() - t0))
    return 0


def run_all(tasks: List[dict], workers: int, t0: float, label: str) -> List[dict]:
    results: List[dict] = []
    if workers <= 1:
        for i, t in enumerate(tasks):
            results.append(score_one_archive(t))
            print("  %s %d/%d  %.1fs" % (label, i + 1, len(tasks), time.time() - t0), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(score_one_archive, tasks, chunksize=1)):
                results.append(r)
                if (i + 1) % 25 == 0 or i + 1 == len(tasks):
                    print("  %s %d/%d  %.1fs" % (label, i + 1, len(tasks), time.time() - t0),
                          flush=True)
    return results


def report(out: dict, taus: Dict[str, float], fpr: dict) -> None:
    f = lambda x, s="%8.3f": (s % x) if x is not None and np.isfinite(x) else "       -"
    pc = lambda x: ("%7.1f%%" % (100 * x)) if x is not None and np.isfinite(x) else "       -"
    print("\n" + "=" * 134)
    print("WHITE-BOX DETECTOR-AWARE ATTACK -- cost curve.  Thresholds frozen on CAL honest "
          "client-rounds at nominal 1%.")
    for s in ("suspicion", "suspicion_g2s", "suspicion_cos"):
        print("  tau(%-13s) = %8.4f   realised honest FPR on TEST = %.3f%%"
              % (s, taus[s], 100 * fpr.get(s, {}).get("realised", float("nan"))))
    print("=" * 134)
    hdr = ("%-19s %-13s %5s %5s | %6s %8s %8s | %7s %7s | %7s %8s %8s %6s | %8s"
           % ("attacker aims/mode", "knowledge", "phi", "beta", "AUC", "TPR c-r", "TPR run",
              "AUC g2s", "AUC cos", "fidelity", "|I|/|Ih|", "err/|Ih|", "Bused", "pred-real"))
    print(hdr); print("-" * len(hdr))
    for k, v in out["cells"].items():
        tgt, know, phi, beta = k.split("|")
        s1, s2, s3 = (v["stats"]["suspicion"], v["stats"]["suspicion_g2s"],
                      v["stats"]["suspicion_cos"])
        print("%-19s %-13s %5s %5s | %6.3f %s %s | %7.3f %7.3f | %7s %8s %8s %6s | %8s"
              % (tgt, know, phi.split("=")[1], beta.split("=")[1],
                 s1["auc"], pc(s1["tpr_client_round_TEST"]), pc(s1["tpr_client_run_TEST"]),
                 s2["auc"], s3["auc"],
                 f(v["fidelity_mean"], "%7.3f"), f(v["innov_norm_ratio_mean"]),
                 f(v["payload_err_rel_mean"]), f(v["budget_used_mean"], "%6.2f"),
                 f(v["pred_vs_real_mean_abs"], "%8.1e")))
    print("\nAUC / TPR   = the PUBLISHED score (magnitude term one-sided, as in the paper)")
    print("AUC g2s     = same replay, magnitude term made two-sided")
    print("AUC cos     = same replay, plus a sixth term: the cohort-alignment z")
    print("fidelity  = cos(published innovation, honest innovation), mean over attacked rounds")
    print("|I|/|Ih|  = published innovation norm / honest innovation norm")
    print("err/|Ih|  = ||C_published - C_honest||_F / ||honest innovation||_F")
    print("Bused     = fraction of the evasion budget the optimiser actually spent "
          "(1.00 means the budget bound is what stopped it)")
    print("pred-real = mean |surrogate prediction - auditor's realised score|; zero for "
          "peer_current by construction, the knowledge gap for peer_lagged")
    print("\nBaseline detectors on the SAME crafted payloads (AUC):")
    cols = list(BASELINES) + list(FG_OBSERVERS)
    bh = "%-19s %-13s %5s %5s |" % ("attacker aims", "knowledge", "phi", "beta") + \
         "".join("%18s" % b for b in cols)
    print(bh); print("-" * len(bh))
    for k, v in out["cells"].items():
        tgt, know, phi, beta = k.split("|")
        print("%-19s %-13s %5s %5s |" % (tgt, know, phi.split("=")[1], beta.split("=")[1])
              + "".join("%18.3f" % v["stats"][b]["auc"] for b in cols))
    print("\nThe last %d columns are observers: they score the same crafted payloads but "
          "were not\nin the attacker's objective, so they say what leaving FoolsGold out "
          "of the fusion costs." % len(FG_OBSERVERS))


if __name__ == "__main__":
    sys.exit(main())
