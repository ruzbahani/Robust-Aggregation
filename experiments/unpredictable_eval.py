from __future__ import annotations

import argparse, importlib, itertools, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

import numpy as np

from passportfl.innovation import EPS, SCORE_STRUCTURAL, InnovationAuditor
from passportfl.unpredictable import (CODE_DEGENERATE, CODE_OK, CODE_UNDEFINED,
                                      BeaconSchedule, CommitReveal, Draw, Mechanism,
                                      STAT_NAMES, fit_per_draw_thresholds, fit_threshold,
                                      oriented_from_z, score_under_draws)

ARCHIVE_DIR = os.path.join(_HERE, "..", "results", "archives")
FIXED_POINT_SCALE = 1 << 16

_pe = importlib.import_module("protocol_eval")
ATTACKS, ATTACK_ORDER = _pe.ATTACKS, _pe.ATTACK_ORDER


def mechanisms() -> Dict[str, Mechanism]:
    return {
        "det": Mechanism("det", 5, "max"),
        "sub4": Mechanism("sub4", 4, "max"),
        "sub3": Mechanism("sub3", 3, "max"),
        "sub2": Mechanism("sub2", 2, "max"),
        "sub3w": Mechanism("sub3w", 3, "max", weight_levels=(0.5, 1.0, 2.0),
                           max_weight_combos=6),
        "sub3band": Mechanism("sub3band", 3, "max", tau_band=(-0.75, 0.0, 0.75)),
        "wsum3": Mechanism("wsum3", 3, "sum", weight_levels=(0.5, 1.0, 2.0),
                           max_weight_combos=6),
        "full": Mechanism("full", 3, "max", weight_levels=(0.5, 1.0, 2.0),
                          max_weight_combos=6, tau_band=(-0.75, 0.0, 0.75)),
    }


_CACHE: Dict[str, dict] = {}


def load_archive(path: str) -> dict:
    if path not in _CACHE:
        z = np.load(path, allow_pickle=True)
        _CACHE.clear()
        _CACHE[path] = {"C": z["C"].item(), "present": z["present"].item(),
                        "cbar": z["cbar"].item()}
    return _CACHE[path]


def score_one_archive(task: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    arch = load_archive(task["path"])
    clients = len(arch["C"])
    rounds = max(max(v) for v in arch["C"].values())
    kind = task["kind"]
    victim = task.get("victim")
    attack_start, scope_start = task["attack_start"], task["scope_start"]
    mad_floor = task["mad_floor"]

    work = dict(arch)
    work["published"] = {i: {} for i in range(clients)}
    fn = ATTACKS[task["attack"]] if kind == "nonadaptive" else None
    rng = np.random.default_rng([int(task["seed"]), int(victim or 0) + 1,
                                 (ATTACK_ORDER.index(task["attack"]) + 1)
                                 if kind == "nonadaptive" else 0])

    atk = None
    if kind == "adaptive":
        from passportfl.adaptive_attack import AttackConfig
        from passportfl.unpredictable import RandomisedAttacker
        cfg = AttackConfig(phi=task["phi"], beta=task["beta"], knowledge=task["knowledge"],
                           basis_steps=task["basis_steps"], full_steps=task["full_steps"],
                           restarts=task["restarts"], mad_floor=mad_floor)
        atk = RandomisedAttacker(cfg, task["draws"], np.asarray(task["tau"]), task["mode"],
                                 seed=1000 * task["seed"] + int(victim))

    aud = InnovationAuditor(mad_floor=mad_floor)
    rec_rounds: List[int] = []
    Trows: List[np.ndarray] = []
    Crows: List[np.ndarray] = []
    det_rows: List[np.ndarray] = []
    fid, rel_err, innov_ratio, pred_w, pred_m = [], [], [], [], []
    statuses: Dict[str, int] = {}

    for r in range(1, rounds + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        present_r = {cid: arch["present"][cid][r] for cid in range(clients)}
        payloads = {}
        for cid in range(clients):
            C = arch["C"][cid][r]
            if fn is not None and cid == victim and r >= attack_start:
                C = fn(work, cid, r, rng)
            work["published"][cid][r] = C
            payloads[cid] = (C, present_r[cid])

        if kind == "adaptive" and r >= attack_start:
            Cb = aud.cbar.get(r - 1)
            if task["beta"] > 0:
                peer_C = ({j: arch["C"][j][r] for j in range(clients) if j != victim}
                          if task["knowledge"] == "peer_current" else None)
                Cp, info = atk.craft(aud, r, victim, arch["C"][victim][r], present_r, peer_C)
            else:
                Cp = (arch["C"][victim][r].copy() if Cb is None
                      else atk._cheap_objective(Cb, arch["C"][victim][r]))
                Cp = np.round(Cp * FIXED_POINT_SCALE) / FIXED_POINT_SCALE
                info = {"status": "cheap_only"}
            payloads[victim] = (Cp, present_r[victim])
            work["published"][victim][r] = Cp
            statuses[info["status"]] = statuses.get(info["status"], 0) + 1
            if np.isfinite(info.get("predicted_worst_margin", float("nan"))):
                pred_w.append(info["predicted_worst_margin"])
                pred_m.append(info["predicted_mean_margin"])
            if Cb is not None and aud.hist.get(victim):
                T = np.asarray(present_r[victim], bool) & aud.hist[victim][-1].present
                Ih, Ip = (arch["C"][victim][r] - Cb)[T], (Cp - Cb)[T]
                nh, npb = float(np.linalg.norm(Ih)), float(np.linalg.norm(Ip))
                if nh > EPS and npb > EPS:
                    fid.append(float((Ih * Ip).sum() / (nh * npb)))
                    innov_ratio.append(npb / nh)
                    rel_err.append(float(np.linalg.norm(Cp - arch["C"][victim][r])) / nh)

        rs = aud.observe_round(r, payloads)
        if r < scope_start:
            continue
        rec_rounds.append(r)
        Trow = np.full((clients, len(STAT_NAMES)), np.nan)
        Crow = np.zeros(clients, dtype=np.int8)
        drow = np.full(clients, np.nan)
        for cid in range(clients):
            v = rs.per_client[cid]
            t, code = oriented_from_z(v)
            Trow[cid] = t
            Crow[cid] = code
            drow[cid] = float(v.get("suspicion", float("nan")))
        Trows.append(Trow)
        Crows.append(Crow)
        det_rows.append(drow)

    return {
        "key": {k: task.get(k) for k in ("seed", "kind", "attack", "victim", "phi", "beta",
                                         "knowledge", "mech", "mode")},
        "rounds": rec_rounds,
        "T": np.asarray(Trows),
        "code": np.asarray(Crows),
        "det": np.asarray(det_rows),
        "victim": victim,
        "diag": {"fidelity": float(np.mean(fid)) if fid else None,
                 "innov_ratio": float(np.mean(innov_ratio)) if innov_ratio else None,
                 "payload_err_rel": float(np.mean(rel_err)) if rel_err else None,
                 "pred_worst_margin": float(np.mean(pred_w)) if pred_w else None,
                 "pred_mean_margin": float(np.mean(pred_m)) if pred_m else None,
                 "statuses": statuses},
    }


def run_all(tasks: List[dict], workers: int, t0: float, label: str) -> List[dict]:
    out: List[dict] = []
    if not tasks:
        return out
    if workers <= 1:
        for i, t in enumerate(tasks):
            out.append(score_one_archive(t))
            print("  %s %d/%d  %.1fs" % (label, i + 1, len(tasks), time.time() - t0), flush=True)
        return out
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(score_one_archive, tasks, chunksize=1)):
            out.append(r)
            if (i + 1) % 25 == 0 or i + 1 == len(tasks):
                print("  %s %d/%d  %.1fs" % (label, i + 1, len(tasks), time.time() - t0),
                      flush=True)
    return out


def margins_drawn(p: dict, sched: BeaconSchedule, tau: np.ndarray) -> np.ndarray:
    idx = np.asarray([sched.draw_index(r) for r in p["rounds"]], dtype=int)
    out = np.full(p["T"].shape[:2], np.nan)
    for j in np.unique(idx):
        rows = np.where(idx == j)[0]
        s = score_under_draws(p["T"][rows], p["code"][rows], [sched.draws[j]])[..., 0]
        out[rows] = s - tau[j]
    return out


def raw_drawn(p: dict, sched: BeaconSchedule) -> np.ndarray:
    idx = np.asarray([sched.draw_index(r) for r in p["rounds"]], dtype=int)
    out = np.full(p["T"].shape[:2], np.nan)
    for j in np.unique(idx):
        rows = np.where(idx == j)[0]
        out[rows] = score_under_draws(p["T"][rows], p["code"][rows], [sched.draws[j]])[..., 0]
    return out


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


def flag_rate(m: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(m), m, -np.inf) > 0.0


def clopper_pearson_upper(k: int, n: int) -> float:
    return _pe.clopper_pearson(k, n)[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", default=ARCHIVE_DIR)
    ap.add_argument("--prefix", default="arch_cifar100_s")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--cal-seeds", default="0,1")
    ap.add_argument("--test-seeds", default="3,4")
    ap.add_argument("--attack-start", type=int, default=12)
    ap.add_argument("--scope-start", type=int, default=12)
    ap.add_argument("--mad-floor", type=float, default=1e-2)
    ap.add_argument("--fpr-target", type=float, default=0.01)
    ap.add_argument("--beacon-seed", type=int, default=20260914)
    ap.add_argument("--mechs", default="det,sub4,sub3,sub2,sub3w,sub3band,wsum3,full")
    ap.add_argument("--adaptive-mechs", default="det,sub3,sub3band,wsum3")
    ap.add_argument("--modes", default="expected,worst")
    ap.add_argument("--knowledge", default="peer_current,peer_lagged")
    ap.add_argument("--phis", default="0,0.1,0.25,0.5,1.0")
    ap.add_argument("--betas", default="0,0.25,0.5,1.0,2.0")
    ap.add_argument("--victims", default="0,1")
    ap.add_argument("--basis-steps", type=int, default=40)
    ap.add_argument("--full-steps", type=int, default=80)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--stage", default="all", help="all | honest | nonadaptive | adaptive")
    ap.add_argument("--workers", type=int, default=min(24, (os.cpu_count() or 4)))
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "results", "unpredictable_eval.json"))
    ap.add_argument("--txt", default=os.path.join(_HERE, "..", "results", "unpredictable_costcurve.txt"))
    a = ap.parse_args()

    ints = lambda s: [int(x) for x in s.split(",") if x.strip()]
    fls = lambda s: [float(x) for x in s.split(",") if x.strip()]
    strs = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    seeds, cal_seeds, test_seeds = ints(a.seeds), ints(a.cal_seeds), ints(a.test_seeds)
    phis, betas = fls(a.phis), fls(a.betas)
    victims = ints(a.victims)

    import re
    paths = {}
    for fn in sorted(os.listdir(a.archives)):
        if fn.startswith(a.prefix) and fn.endswith(".npz"):
            m = re.search(r"_s(\d+)\.npz$", fn)
            if m:
                paths[int(m.group(1))] = os.path.join(a.archives, fn)
    seeds = [s for s in seeds if s in paths]
    if not seeds:
        print("no archives found")
        return 1
    clients = len(load_archive(paths[seeds[0]])["C"])

    MECH = mechanisms()
    mech_names = [m for m in strs(a.mechs) if m in MECH]
    scheds = {m: BeaconSchedule(MECH[m], a.beacon_seed) for m in mech_names}
    common = {"attack_start": a.attack_start, "scope_start": a.scope_start,
              "mad_floor": a.mad_floor, "basis_steps": a.basis_steps,
              "full_steps": a.full_steps, "restarts": a.restarts}

    t0 = time.time()
    out: dict = {"config": vars(a), "clients": clients, "seeds": seeds,
                 "cal_seeds": cal_seeds, "test_seeds": test_seeds,
                 "mechanisms": {m: MECH[m].describe() for m in mech_names}}
    print("mechanisms under test:")
    for m in mech_names:
        print("  " + MECH[m].describe())

    honest = run_all([dict(common, path=paths[s], seed=s, kind="honest", victim=0)
                      for s in seeds], min(a.workers, len(seeds)), t0, "honest")
    cal_h = [r for r in honest if r["key"]["seed"] in cal_seeds]
    test_h = [r for r in honest if r["key"]["seed"] in test_seeds]

    cal_T = np.concatenate([r["T"].reshape(-1, len(STAT_NAMES)) for r in cal_h])
    cal_code = np.concatenate([r["code"].ravel() for r in cal_h])

    calib: Dict[str, dict] = {}
    for m in mech_names:
        sched = scheds[m]
        cal_scores = score_under_draws(cal_T, cal_code, sched.draws)
        tau = fit_per_draw_thresholds(cal_scores, sched.draws, a.fpr_target)
        cal_drawn = np.concatenate([raw_drawn(r, sched).ravel() for r in cal_h])
        tau_single = fit_threshold(cal_drawn, a.fpr_target)
        mats = [margins_drawn(r, sched, tau) for r in test_h]
        f = np.concatenate([flag_rate(x).ravel() for x in mats]) if mats else np.zeros(0, bool)
        run_k = sum(int(flag_rate(x).any(axis=0).sum()) for x in mats)
        run_n = len(mats) * clients
        raws = [raw_drawn(r, sched) for r in test_h]
        fs = np.concatenate([flag_rate(x - tau_single).ravel() for x in raws]) if raws else np.zeros(0, bool)
        run_ks = sum(int(flag_rate(x - tau_single).any(axis=0).sum()) for x in raws)
        calib[m] = {
            "n_draws": len(sched.draws),
            "tau_per_draw": tau.tolist(),
            "tau_min": float(tau.min()), "tau_med": float(np.median(tau)),
            "tau_max": float(tau.max()),
            "tau_single": float(tau_single),
            "fpr_client_round_TEST": float(f.mean()) if f.size else None,
            "n_client_rounds_TEST": int(f.size),
            "fpr_client_run_TEST": (run_k / run_n) if run_n else None,
            "fpr_client_run_TEST_hi95": clopper_pearson_upper(run_k, run_n) if run_n else None,
            "n_runs_TEST": run_n,
            "fpr_client_round_TEST_single_tau": float(fs.mean()) if fs.size else None,
            "fpr_client_run_TEST_single_tau": (run_ks / run_n) if run_n else None,
        }
        print("  %-9s |support|=%4d  tau %7.3f..%7.3f (med %6.3f)  honest FPR c-r TEST "
              "%6.3f%%  run %5.1f%%   [single tau %7.3f -> %6.3f%%]"
              % (m, len(sched.draws), tau.min(), tau.max(), np.median(tau),
                 100 * calib[m]["fpr_client_round_TEST"],
                 100 * calib[m]["fpr_client_run_TEST"], tau_single,
                 100 * calib[m]["fpr_client_round_TEST_single_tau"]), flush=True)
    out["calibration"] = calib

    if "det" in calib:
        det_ref = np.concatenate([r["det"].ravel() for r in cal_h])
        tau_ref = fit_threshold(det_ref, a.fpr_target)
        out["det_threshold_crosscheck"] = {
            "tau_from_auditor_suspicion": float(tau_ref),
            "tau_from_5of5_mechanism": float(calib["det"]["tau_per_draw"][0]),
            "abs_diff": abs(tau_ref - calib["det"]["tau_per_draw"][0])}
        print("  cross-check: tau from the auditor's own suspicion = %.4f, from the 5-of-5 "
              "mechanism = %.4f" % (tau_ref, calib["det"]["tau_per_draw"][0]))

    nonad: Dict[str, dict] = {}
    if a.stage in ("all", "nonadaptive"):
        tasks = [dict(common, path=paths[s], seed=s, kind="nonadaptive", attack=at, victim=v)
                 for at in ATTACK_ORDER for s in seeds for v in range(clients)]
        print("\n%d non-adaptive passes" % len(tasks))
        res = run_all(tasks, a.workers, t0, "nonadaptive")
        by_atk: Dict[str, List[dict]] = {}
        for r in res:
            by_atk.setdefault(r["key"]["attack"], []).append(r)
        for m in mech_names:
            sched, tau = scheds[m], np.asarray(calib[m]["tau_per_draw"])
            per = {}
            for at, ps in by_atk.items():
                pos, neg = [], []
                tp_k = tp_n = run_k = run_n = 0
                for p in ps:
                    mm = margins_drawn(p, sched, tau)
                    v = p["victim"]
                    msk = np.ones(clients, bool); msk[v] = False
                    pos.extend(mm[:, v].tolist()); neg.extend(mm[:, msk].ravel().tolist())
                    if p["key"]["seed"] in test_seeds:
                        f = flag_rate(mm[:, v])
                        tp_k += int(f.sum()); tp_n += int(f.size)
                        run_n += 1; run_k += int(bool(f.any()))
                per[at] = {"auc": roc_auc(pos, neg),
                           "tpr_client_round_TEST": (tp_k / tp_n) if tp_n else None,
                           "tpr_client_run_TEST": (run_k / run_n) if run_n else None}
            worst = min(v["tpr_client_round_TEST"] for v in per.values())
            worst_run = min(v["tpr_client_run_TEST"] for v in per.values())
            worst_auc = min(v["auc"] for v in per.values())
            nonad[m] = {"per_attack": per, "worst_tpr_client_round_TEST": worst,
                        "worst_tpr_client_run_TEST": worst_run, "worst_auc": worst_auc}
            print("  %-9s worst-case TPR over the 6-attack suite = %5.1f%% client-round, "
                  "%5.1f%% client-run  (worst AUC %.3f)"
                  % (m, 100 * worst, 100 * worst_run, worst_auc), flush=True)
        out["nonadaptive"] = nonad

    if a.stage in ("all", "adaptive"):
        ad_mechs = [m for m in strs(a.adaptive_mechs) if m in mech_names]
        modes, knows = strs(a.modes), strs(a.knowledge)
        tasks = []
        for phi, s, v in itertools.product(phis, seeds, victims):
            if 0.0 in betas:
                tasks.append(dict(common, path=paths[s], seed=s, kind="adaptive", victim=v,
                                  phi=phi, beta=0.0, knowledge=knows[0], mech="_shared",
                                  mode="_shared", draws=[], tau=[]))
        for m, mode, know, phi, beta, s, v in itertools.product(
                ad_mechs, modes, knows, phis, [b for b in betas if b > 0], seeds, victims):
            if m == "det" and mode != modes[0]:
                continue
            tasks.append(dict(common, path=paths[s], seed=s, kind="adaptive", victim=v,
                              phi=phi, beta=beta, knowledge=know, mech=m, mode=mode,
                              draws=scheds[m].draws, tau=calib[m]["tau_per_draw"]))
        print("\n%d adaptive passes" % len(tasks))
        res = run_all(tasks, a.workers, t0, "adaptive")

        cells: Dict[str, List[dict]] = {}
        for r in res:
            k = r["key"]
            cells.setdefault("%s|%s|%s|phi=%g|beta=%g"
                             % (k["mech"], k["mode"], k["knowledge"], k["phi"], k["beta"]),
                             []).append(r)
        adapt: Dict[str, dict] = {}
        for key, ps in sorted(cells.items()):
            mname = key.split("|")[0]
            targets = ad_mechs if mname == "_shared" else [mname]
            for m in targets:
                sched, tau = scheds[m], np.asarray(calib[m]["tau_per_draw"])
                pos, neg = [], []
                tp_k = tp_n = run_k = run_n = 0
                dg = {"fidelity": [], "innov_ratio": [], "payload_err_rel": [],
                      "pred_worst_margin": [], "pred_mean_margin": [], "real_worst": []}
                for p in ps:
                    mm = margins_drawn(p, sched, tau)
                    rows = np.asarray(p["rounds"]) >= a.attack_start
                    mm = mm[rows]
                    v = p["victim"]
                    msk = np.ones(clients, bool); msk[v] = False
                    pos.extend(mm[:, v].tolist()); neg.extend(mm[:, msk].ravel().tolist())
                    if p["key"]["seed"] in test_seeds:
                        f = flag_rate(mm[:, v])
                        tp_k += int(f.sum()); tp_n += int(f.size)
                        run_n += 1; run_k += int(bool(f.any()))
                    for kk in ("fidelity", "innov_ratio", "payload_err_rel",
                               "pred_worst_margin", "pred_mean_margin"):
                        if p["diag"].get(kk) is not None:
                            dg[kk].append(p["diag"][kk])
                    col = mm[:, v]
                    if np.isfinite(col).any():
                        dg["real_worst"].append(float(np.nanmax(col)))
                out_key = key if mname != "_shared" else key.replace("_shared|_shared", "%s|-" % m)
                adapt[out_key] = {
                    "auc": roc_auc(pos, neg),
                    "tpr_client_round_TEST": (tp_k / tp_n) if tp_n else None,
                    "tpr_client_run_TEST": (run_k / run_n) if run_n else None,
                    "n_test_cells": tp_n, "n_test_runs": run_n,
                    "victim_margin_median": float(np.nanmedian(pos)) if pos else None,
                    **{k2: (float(np.mean(v2)) if v2 else None) for k2, v2 in dg.items()},
                }
        out["adaptive"] = adapt

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(os.path.abspath(a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    report(out, os.path.abspath(a.txt))
    print("\nwrote %s and %s   (%.1fs)"
          % (os.path.abspath(a.out), os.path.abspath(a.txt), time.time() - t0))
    return 0


def report(out: dict, txt_path: str) -> None:
    L: List[str] = []
    P = L.append
    pc = lambda x: ("%6.1f%%" % (100 * x)) if x is not None and np.isfinite(x) else "     -"
    fm = lambda x, s="%7.3f": (s % x) if x is not None and np.isfinite(x) else "      -"

    P("=" * 132)
    P("UNPREDICTABLE-AT-PUBLISH-TIME SCORING: commit-then-reveal and beacon-randomised fusion")
    P("CIFAR-100 Dirichlet 0.5, 10 clients, 35 rounds, scope and attack from round 12.")
    P("Thresholds fitted on CAL honest client-rounds (seeds %s) at nominal %g%% per-client-round FPR,"
      % (out["cal_seeds"], 100 * out["config"]["fpr_target"]))
    P("frozen, and applied to TEST seeds %s. One threshold per draw, every draw calibrated on the"
      % (out["test_seeds"],))
    P("same CAL honest client-rounds (an honest client's z vector does not depend on the beacon).")
    P("=" * 132)
    P("")
    P("MECHANISMS")
    for k, v in out["mechanisms"].items():
        P("  " + v)
    cc = out.get("det_threshold_crosscheck")
    if cc:
        P("  cross-check: the 5-of-5 mechanism reproduces the auditor's own threshold to %.2e"
          % cc["abs_diff"])
    P("")
    P("WHAT RANDOMISATION COSTS HONEST CLIENTS")
    h = ("%-10s %7s %9s %9s %9s | %11s %9s | %13s %11s"
         % ("mech", "draws", "tau min", "tau med", "tau max", "FPR c-r", "FPR run",
            "single-tau c-r", "run"))
    P(h); P("-" * len(h))
    for m, c in out["calibration"].items():
        P("%-10s %7d %9.3f %9.3f %9.3f | %11s %9s | %13s %11s"
          % (m, c["n_draws"], c["tau_min"], c["tau_med"], c["tau_max"],
             pc(c["fpr_client_round_TEST"]), pc(c["fpr_client_run_TEST"]),
             pc(c["fpr_client_round_TEST_single_tau"]),
             pc(c["fpr_client_run_TEST_single_tau"])))
    P("-" * len(h))
    P("  FPR c-r / run   : realised honest false-alarm rate on TEST with the per-draw thresholds")
    P("                    frozen on CAL. n = %d client-rounds, %d client-runs."
      % (list(out["calibration"].values())[0]["n_client_rounds_TEST"],
         list(out["calibration"].values())[0]["n_runs_TEST"]))
    P("  single-tau      : an operator that fits one threshold and does not recalibrate")
    P("                    per draw. The gap between the two columns is the cost of getting the")
    P("                    calibration wrong, not a cost of randomisation itself.")

    if "nonadaptive" in out:
        P("")
        P("THE SIX NON-ADAPTIVE ATTACKS under randomised verification")
        atks = list(list(out["nonadaptive"].values())[0]["per_attack"].keys())
        h = ("%-10s" % "mech" + "".join("%14s" % a[:13] for a in atks)
             + "%12s%11s%11s" % ("worst TPR", "worst run", "worst AUC"))
        P(h); P("-" * len(h))
        for m, v in out["nonadaptive"].items():
            row = "%-10s" % m
            for a in atks:
                row += "%14s" % pc(v["per_attack"][a]["tpr_client_round_TEST"])
            row += "%12s%11s%11.3f" % (pc(v["worst_tpr_client_round_TEST"]),
                                       pc(v.get("worst_tpr_client_run_TEST")), v["worst_auc"])
            P(row)
        P("-" * len(h))
        P("  TPR at the client-round level on TEST seeds, thresholds frozen on CAL.")
        P("  worst run = worst-case over the suite at the client-RUN level (flagged at least once")
        P("              after onset). A beacon that enforces 3 of 5 catches a persistent attacker")
        P("              in the rounds where its discriminating statistic is drawn, which is why")
        P("              the per-round column falls and the per-run column does not.")

    if "adaptive" in out:
        P("")
        P("THE WHITE-BOX ADAPTIVE ATTACKER (Kerckhoffs: it knows the mechanism, the support and")
        P("the frozen per-draw thresholds; it does not know the draw)")
        h = ("%-10s %-9s %-13s %5s %5s | %7s %9s %9s | %9s %9s %8s %8s"
             % ("mech", "response", "knowledge", "phi", "beta", "AUC", "TPR c-r", "TPR run",
                "pred wmrg", "real wmrg", "fidelity", "|I|/|Ih|"))
        P(h); P("-" * len(h))
        for k, v in out["adaptive"].items():
            mech, mode, know, phi, beta = k.split("|")
            P("%-10s %-9s %-13s %5s %5s | %7.3f %9s %9s | %9s %9s %8s %8s"
              % (mech, mode, know, phi.split("=")[1], beta.split("=")[1], v["auc"],
                 pc(v["tpr_client_round_TEST"]), pc(v["tpr_client_run_TEST"]),
                 fm(v.get("pred_worst_margin"), "%9.3f"), fm(v.get("real_worst"), "%9.3f"),
                 fm(v.get("fidelity"), "%8.3f"), fm(v.get("innov_ratio"), "%8.3f")))
        P("-" * len(h))
        P("  response   = which rational reply to the draw distribution the attacker solved:")
        P("               expected = minimise E_d[score], worst = minimise max_d[score] (minimax),")
        P("               prob = minimise a smooth surrogate for the probability of being flagged.")
        P("               '-' is beta = 0, where the attacker publishes C_cheap and solves nothing.")
        P("  knowledge  = peer_current is the rushing adversary; peer_lagged is what")
        P("               commit-then-reveal leaves it (mechanism A).")
        P("  AUC / TPR  = on the MARGIN s_drawn - tau_drawn, which is the quantity that is")
        P("               comparable across rounds scored by different scorers. A flag is margin > 0.")
        P("  pred wmrg  = the worst-case margin the attacker BELIEVED it had achieved;")
        P("  real wmrg  = the worst margin the auditor actually realised on the victim. The gap")
        P("               between them is exactly what the attacker cannot see at publish time.")

    body = "\n".join(L)
    print("\n" + body)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(body + "\n")


if __name__ == "__main__":
    sys.exit(main())
