from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, hashlib, json, pickle, re, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

import numpy as np
import torch

from passportfl.adaptive_attack import AttackConfig
from passportfl.combined_attack import COMB_STATS, CombinedAttackConfig
from passportfl.innovation import EPS, MAD_SCALE, InnovationAuditor
from passportfl.limited_access import (ExactOracle, LimitedAccessAttacker, ShadowFederation,
                                       mimic_aim, parse_level, replay_shadow)

import combined_eval as ce

SCHEMA = 1
RESULTS = os.path.join(_HERE, "..", "results")
DEFAULT_LEVELS = ("current,delayed:1,delayed:2,delayed:3,query:4,query:16,query:64,query:256,"
                  "decision:4,decision:16,decision:64,feedback,pseudo,shadow:5,shadow:6")
RELEASED_TAU_P = os.path.join(RESULTS, "adaptive_costcurve.json")
RELEASED_FUSED = os.path.join(RESULTS, "combined_grid.json")


def archive_paths(archives: str, prefix: str) -> Dict[int, str]:
    out = {}
    for fn in sorted(os.listdir(archives)):
        if fn.startswith(prefix) and fn.endswith(".npz"):
            m = re.search(r"_s(\d+)\.npz$", fn)
            if m:
                out[int(m.group(1))] = os.path.join(archives, fn)
    return out


def payload_digest(C: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(C, dtype=np.float64).tobytes()).hexdigest()[:16]


def honest_task(path: str, seed: int, a) -> dict:
    return {"path": path, "seed": seed, "victim": 0, "phi": 1.0, "beta": 0.0, "rule": "max",
            "knowledge": "peer_current", "aim": "mimic", "schedule": "flat", "honest": True,
            "attack_start": a.attack_start, "scope_start": a.scope_start,
            "mad_floor": a.mad_floor, "basis_steps": 0, "full_steps": 0, "restarts": 0}


def calibrate(honest: Dict[int, dict], cal_seeds: Sequence[int], test_seeds: Sequence[int],
              clients: int, fpr_target: float) -> dict:
    cal = [honest[s] for s in cal_seeds]
    norm = {}
    for k in COMB_STATS:
        v = np.concatenate([h["raw"][k].ravel() for h in cal])
        v = v[np.isfinite(v)]
        med = float(np.median(v))
        norm[k] = (med, MAD_SCALE * float(np.median(np.abs(v - med))) + 1e-9)

    def fused(h):
        rows = []
        for i in range(h["raw"]["suspicion"].shape[0]):
            raw = {k: {c: float(h["raw"][k][i, c]) for c in range(clients)} for k in COMB_STATS}
            cm = ce.combined_from_raw(raw, norm, clients)["combined_max"]
            rows.append([cm[c] for c in range(clients)])
        return np.asarray(rows, float)

    score = {"P": {s: h["raw"]["suspicion"] for s, h in honest.items()},
             "F": {s: fused(h) for s, h in honest.items()}}
    out = {"norm": norm, "tau": {}, "cal_median": {}, "honest": {}}
    for dfn in ("P", "F"):
        calv = np.concatenate([score[dfn][s].ravel() for s in cal_seeds])
        tau = ce.fit_threshold(calv, fpr_target)
        out["tau"][dfn] = tau
        out["cal_median"][dfn] = float(np.median(calv[np.isfinite(calv)]))
        flag = {s: np.where(np.isfinite(m), m, -np.inf) > tau for s, m in score[dfn].items()}
        tst = np.concatenate([flag[s] for s in test_seeds], axis=0)
        runs = np.concatenate([flag[s].any(axis=0) for s in test_seeds])
        out["honest"][dfn] = {
            "fpr_client_round_TEST": float(tst.mean()), "n_client_rounds_TEST": int(tst.size),
            "fpr_run_TEST": float(runs.mean()), "n_runs_TEST": int(runs.size),
            "flag_matrix_TEST": {str(s): flag[s].any(axis=0).astype(int).tolist()
                                 for s in test_seeds}}
    return out


def check_against_release(calib: dict) -> dict:
    rep = {}
    cc = json.load(open(RELEASED_TAU_P, encoding="utf-8"))
    rep["tau_P"] = {"refit": calib["tau"]["P"], "released": cc["thresholds"]["suspicion"]}
    g = json.load(open(RELEASED_FUSED, encoding="utf-8"))
    rep["tau_F"] = {"refit": calib["tau"]["F"], "released": g["thresholds"]["combined_max"]}
    rep["norm_max_abs_diff"] = max(abs(a - b) for k in COMB_STATS
                                   for a, b in zip(calib["norm"][k], g["norm"][k]))
    rep["released_fused_suspicion_tau_not_used"] = g["thresholds"]["suspicion"]
    ok = (rep["tau_P"]["refit"] == rep["tau_P"]["released"]
          and rep["tau_F"]["refit"] == rep["tau_F"]["released"]
          and rep["norm_max_abs_diff"] == 0.0)
    rep["identical"] = bool(ok)
    if not ok:
        raise RuntimeError("calibration differs from the release: %r" % rep)
    return rep


def defended_scores(defense: str, rs, payloads: Dict[int, tuple],
                    Cbar_prev: Optional[np.ndarray], clients: int, mad_floor: float,
                    norm: Dict[str, tuple]) -> Dict[int, float]:
    if defense == "P":
        return {c: float(rs.per_client[c].get("suspicion", float("nan"))) for c in range(clients)}
    raw = ce.raw_round_stats(rs, payloads, Cbar_prev, clients, mad_floor)
    return ce.combined_from_raw(raw, norm, clients)["combined_max"]


def shadow_federation(path: str, seed: int, defense: str, norm: Dict[str, tuple],
                      mad_floor: float) -> ShadowFederation:
    arch = ce.load_archive(path)
    clients = len(arch["C"])
    fn = lambda rs, pay, Cb: defended_scores(defense, rs, pay, Cb, clients, mad_floor, norm)
    return replay_shadow(arch, seed, os.path.basename(path), mad_floor, fn)


def score_pass(task: dict) -> dict:
    torch.set_num_threads(1)
    t_pass = time.perf_counter()
    victim, beta, mad_floor = task["victim"], task["beta"], task["mad_floor"]
    attack_start, scope_start = task["attack_start"], task["scope_start"]
    defense, level = task["defense"], parse_level(task["level"])
    norm = {k: tuple(v) for k, v in task["norm"].items()}
    tau = task["tau"]
    if beta <= 0:
        raise ValueError("beta must be positive")
    shadow = (shadow_federation(task["shadow_path"], level.shadow_seed, defense, norm, mad_floor)
              if level.kind == "shadow" else None)
    arch = ce.load_archive(task["path"])
    clients = len(arch["C"])
    rounds = max(max(v) for v in arch["C"].values())
    know = "peer_current" if level.kind == "current" else "peer_lagged"
    bs, fs, rst = task["solver"]
    if defense == "P":
        cfg = AttackConfig(phi=task["phi"], beta=beta, knowledge=know, basis_steps=bs,
                           full_steps=fs, restarts=rst, mad_floor=mad_floor, aim="mimic")
    else:
        cfg = CombinedAttackConfig(phi=task["phi"], beta=beta, rule="max", knowledge=know,
                                   aim="mimic", basis_steps=bs, full_steps=fs, restarts=rst,
                                   warm_families=False, mad_floor=mad_floor, schedule="flat",
                                   norm=norm)
    atk = LimitedAccessAttacker(defense, level, cfg, 1000 * task["seed"] + victim, tau=tau,
                                cal_median=task["cal_median"], onset=attack_start,
                                n_pseudo=clients - 1, fb_step=task["fb_step"],
                                fb_decay=task["fb_decay"], es_sigma0=task["es_sigma0"],
                                es_tol=task["es_tol"], shadow=shadow)

    aud = InnovationAuditor(mad_floor=mad_floor)
    score_hist: Dict[int, Dict[int, float]] = {}
    mat: List[List[float]] = []
    rec_rounds: List[int] = []
    log: List[dict] = []
    for r in range(1, rounds + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        Cb = aud.cbar.get(r - 1)
        present_r = {cid: arch["present"][cid][r] for cid in range(clients)}
        payloads = {cid: (arch["C"][cid][r], present_r[cid]) for cid in range(clients)}
        entry = None
        if r >= attack_start:
            C_hon = arch["C"][victim][r]
            oracle = None
            if level.kind in ("query", "decision"):
                oracle = ExactOracle(
                    aud, r, payloads, victim,
                    lambda rs, pay: defended_scores(defense, rs, pay, Cb, clients, mad_floor,
                                                    norm)[victim],
                    level.queries, tau)
            peer_C = ({j: arch["C"][j][r] for j in range(clients) if j != victim}
                      if level.kind == "current" else None)
            t0 = time.perf_counter()
            Cp, info = atk.craft(aud, score_hist, r, victim, C_hon, present_r, peer_C, oracle)
            dt = time.perf_counter() - t0
            payloads[victim] = (Cp, present_r[victim])
            B = float(info.get("budget", float("nan")))
            dist = float(np.linalg.norm(Cp - atk.solver._cheap_objective(Cb, C_hon)))
            entry = {"round": r, "status": info.get("status"),
                     "predicted": float(info.get("predicted", float("nan"))),
                     "aim": float(info.get("aim", float("nan"))), "budget": B,
                     "unit": float(info.get("cohort_scale", float("nan"))),
                     "budget_used": dist / B if np.isfinite(B) and B > 0 else float("nan"),
                     "oracle_calls": int(info.get("oracle_calls", 0)),
                     "margin": float(info.get("margin", 0.0)),
                     "rung": info.get("rung"), "all_flagged": info.get("all_flagged"),
                     "oracle_score": info.get("oracle_score"),
                     "seconds": dt, "payload": payload_digest(Cp), "_dist": dist}

        rs = aud.observe_round(r, payloads)
        sc = defended_scores(defense, rs, payloads, Cb, clients, mad_floor, norm)
        score_hist[r] = sc
        if entry is not None:
            real = float(sc[victim])
            entry["realised"] = real
            entry["flagged"] = bool(np.isfinite(real) and real > tau)
            live = [rs.per_client[c]["norm"] for c in range(clients) if c != victim
                    and np.isfinite(rs.per_client[c]["norm"]) and rs.per_client[c]["norm"] > EPS]
            unit = float(np.median(live)) if live else float("nan")
            entry["true_unit"] = unit
            dist = entry.pop("_dist")
            entry["budget_used_true"] = dist / (beta * unit) if unit > 0 else float("nan")
            if level.kind == "shadow":
                entry["shadow_rho"] = float(info.get("shadow_rho", float("nan")))
                entry["true_aim"] = mimic_aim({c: s for c, s in sc.items() if c != victim},
                                              structural_cut=(defense == "P"))
            atk.observe_outcome(entry["flagged"])
            log.append(entry)
        if r >= scope_start:
            rec_rounds.append(r)
            mat.append([float(sc[c]) for c in range(clients)])
    return {"key": {k: task[k] for k in ("defense", "level", "beta", "seed", "victim")},
            "signature": task["signature"], "archive": os.path.basename(task["path"]),
            "rounds": rec_rounds, "mat": np.asarray(mat, float), "victim": victim,
            "log": log, "seconds": time.perf_counter() - t_pass}


def task_signature(task: dict) -> str:
    keep = {k: v for k, v in task.items() if k not in ("path", "signature", "shadow_path")}
    keep["archive"] = os.path.basename(task["path"])
    if "shadow_path" in task:
        keep["shadow_archive"] = os.path.basename(task["shadow_path"])
    keep["schema"] = SCHEMA
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=float).encode()).hexdigest()


def part_path(parts: str, task: dict) -> str:
    return os.path.join(parts, "%s_%s_b%g_s%d_v%d.pkl" % (
        task["defense"], task["level"].replace(":", "-"), task["beta"], task["seed"],
        task["victim"]))


def load_part(path: str, signature: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        got = pickle.load(fh)
    return got if got.get("signature") == signature else None


def save_part(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def run_tasks(tasks: List[dict], workers: int, parts: str, fn=None, label: str = "pass"
              ) -> List[dict]:
    fn = fn or score_pass
    out, todo = [], []
    for t in tasks:
        got = load_part(part_path(parts, t), t["signature"]) if parts else None
        (out.append(got) if got is not None else todo.append(t))
    print("%d %s tasks, %d cached, %d to run" % (len(tasks), label, len(out), len(todo)),
          flush=True)
    t0 = time.time()

    def store(t, res):
        if parts:
            save_part(part_path(parts, t), res)
        out.append(res)
        print("  %-3s %-12s beta %-4g seed %d victim %d  %6.1fs   [%d/%d, %.0fs]"
              % (t["defense"], t["level"], t["beta"], t["seed"], t["victim"], res["seconds"],
                 len(out), len(tasks), time.time() - t0), flush=True)

    if workers <= 1 or len(todo) <= 1:
        for t in todo:
            store(t, fn(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, t): t for t in todo}
            for f in as_completed(futs):
                store(futs[f], f.result())
    return out


def summarise_cell(passes: List[dict], attack_start: int, tau: float,
                   test_seeds: Sequence[int]) -> dict:
    pos, neg = [], []
    tp = tn = run_k = run_n = run_all_k = 0
    peer_k = peer_n = peer_cr_k = peer_cr_n = 0
    per_pass = {k: [] for k in ("budget_used", "budget_used_true", "pred_err", "realised",
                                "oracle_calls", "unit_ratio")}
    errs, real, secs, calls, statuses = [], [], [], [], {}
    rung0_flag = all_flag = 0
    for p in passes:
        rounds = np.asarray(p["rounds"])
        sel = rounds >= attack_start
        mat = p["mat"][sel]
        v = p["victim"]
        others = np.ones(mat.shape[1], bool); others[v] = False
        pos.extend(mat[:, v].tolist()); neg.extend(mat[:, others].ravel().tolist())
        fl = np.where(np.isfinite(mat), mat, -np.inf) > tau
        run_all_k += int(fl[:, v].any())
        if p["key"]["seed"] in test_seeds:
            tp += int(fl[:, v].sum()); tn += int(fl[:, v].size)
            run_k += int(fl[:, v].any()); run_n += 1
            peer_k += int(fl[:, others].any(axis=0).sum()); peer_n += int(others.sum())
            peer_cr_k += int(fl[:, others].sum()); peer_cr_n += int(fl[:, others].size)
        lg = p["log"]
        e = [abs(x["predicted"] - x["realised"]) for x in lg
             if np.isfinite(x["predicted"]) and np.isfinite(x["realised"])]
        errs.extend(e)
        for key, vals in (("budget_used", [x["budget_used"] for x in lg]),
                          ("budget_used_true", [x["budget_used_true"] for x in lg]),
                          ("pred_err", e), ("realised", [x["realised"] for x in lg]),
                          ("oracle_calls", [x["oracle_calls"] for x in lg]),
                          ("unit_ratio", [x["unit"] / x["true_unit"] for x in lg
                                          if x["true_unit"] > 0])):
            vals = [x for x in vals if np.isfinite(x)]
            if vals:
                per_pass[key].append(float(np.mean(vals)))
        real.extend(x["realised"] for x in lg if np.isfinite(x["realised"]))
        secs.extend(x["seconds"] for x in lg)
        calls.extend(x["oracle_calls"] for x in lg)
        for x in lg:
            statuses[x["status"]] = statuses.get(x["status"], 0) + 1
            if x.get("rung") is not None:
                rung0_flag += int(x["rung"] > 0 or bool(x.get("all_flagged")))
                all_flag += int(bool(x.get("all_flagged")))
    m = lambda k: float(np.mean(per_pass[k])) if per_pass[k] else None
    out = {"n_passes": len(passes), "auc": ce.roc_auc(pos, neg), "n_pos": len(pos),
           "n_neg": len(neg),
           "tpr_client_round_TEST": tp / tn if tn else None, "n_test_client_rounds": tn,
           "flagged_client_rounds_TEST": tp,
           "tpr_run_TEST": run_k / run_n if run_n else None, "n_test_runs": run_n,
           "tpr_run_all": run_all_k / len(passes) if passes else None,
           "peer_run_fpr_TEST": peer_k / peer_n if peer_n else None,
           "peer_client_round_fpr_TEST": peer_cr_k / peer_cr_n if peer_cr_n else None,
           "realised_mean": float(np.mean(real)) if real else None,
           "realised_median": float(np.median(real)) if real else None,
           "realised_max": float(np.max(real)) if real else None,
           "budget_used_mean": m("budget_used"), "budget_used_true_mean": m("budget_used_true"),
           "unit_over_true_unit_mean": m("unit_ratio"),
           "pred_vs_real_mean_abs": m("pred_err"),
           "pred_vs_real_max_abs": float(np.max(errs)) if errs else None,
           "oracle_calls_mean": float(np.mean(calls)) if calls else None,
           "oracle_calls_max": int(np.max(calls)) if calls else None,
           "craft_seconds_mean": float(np.mean(secs)) if secs else None,
           "craft_seconds_median": float(np.median(secs)) if secs else None,
           "statuses": statuses}
    if any(x.get("rung") is not None for p in passes for x in p["log"]):
        out["ladder_rounds_first_rung_flagged"] = rung0_flag
        out["ladder_rounds_all_flagged"] = all_flag
    if any("true_aim" in x for p in passes for x in p["log"]):
        gap, agap, rho = [], [], [x["shadow_rho"] for p in passes for x in p["log"]]
        for p in passes:
            g = [x["aim"] - x["true_aim"] for x in p["log"]
                 if np.isfinite(x["aim"]) and np.isfinite(x["true_aim"])]
            if g:
                gap.append(float(np.mean(g)))
                agap.append(float(np.mean(np.abs(g))))
        out["aim_minus_true_median_mean"] = float(np.mean(gap)) if gap else None
        out["aim_minus_true_median_abs_mean"] = float(np.mean(agap)) if agap else None
        out["shadow_rho_median"] = float(np.nanmedian(rho)) if rho else None
        out["shadow_rho_range"] = [float(np.nanmin(rho)), float(np.nanmax(rho))] if rho else None
    return out


def released_fused_check(cells: dict) -> dict:
    g = json.load(open(RELEASED_FUSED, encoding="utf-8"))
    rep = {}
    for key, c in cells.items():
        dfn, lvl, b = key.split("|")
        if dfn != "F" or lvl != "current" or c["n_passes"] != 10:
            continue
        rk = "max|mimic|flat|peer_current|phi=0|beta=%s" % b.split("=")[1]
        if rk not in g["cells"]:
            continue
        ref = g["cells"][rk]
        s = ref["stats"]["combined_max"]
        rep[key] = {"auc": [c["auc"], s["auc"]],
                    "tpr_client_round_TEST": [c["tpr_client_round_TEST"],
                                              s["tpr_client_round_TEST"]],
                    "tpr_run_TEST": [c["tpr_run_TEST"], s["tpr_client_run_TEST"]],
                    "budget_used_mean": [c["budget_used_mean"], ref["budget_used_mean"]],
                    "pred_vs_real_mean_abs": [c["pred_vs_real_mean_abs"],
                                              ref["pred_vs_real_mean_abs"]]}
        rep[key]["detection_identical"] = all(rep[key][k][0] == rep[key][k][1]
                                              for k in ("auc", "tpr_client_round_TEST",
                                                        "tpr_run_TEST"))
    return rep


def timing_table(passes: List[dict]) -> dict:
    acc: Dict[str, dict] = {}
    for p in passes:
        k = "%s|%s" % (p["key"]["defense"], p["key"]["level"])
        a = acc.setdefault(k, {"round_s": [], "pass_s": []})
        a["round_s"].extend(x["seconds"] for x in p["log"])
        a["pass_s"].append(p["seconds"])
    return {k: {"round_s_mean": float(np.mean(a["round_s"])),
                "round_s_median": float(np.median(a["round_s"])),
                "round_s_max": float(np.max(a["round_s"])),
                "pass_s_mean": float(np.mean(a["pass_s"])), "n_passes": len(a["pass_s"])}
            for k, a in sorted(acc.items())}


def report(out: dict) -> None:
    f = lambda x, s="%7.3f": (s % x) if x is not None and np.isfinite(x) else "      -"
    pc = lambda k, n: "%3d/%-3d" % (k, n)
    for dfn in ("P", "F"):
        rows = [(k, v) for k, v in out["cells"].items() if k.startswith(dfn + "|")]
        if not rows:
            continue
        print("\n" + "=" * 132)
        print("%s  tau = %.4f   honest TEST: client-round FPR %.3f%%, run FPR %d/%d"
              % ({"P": "PAYLOAD AUDITOR", "F": "FUSED MAXIMUM"}[dfn],
                 out["calibration"]["tau"][dfn],
                 100 * out["calibration"]["honest"][dfn]["fpr_client_round_TEST"],
                 round(out["calibration"]["honest"][dfn]["fpr_run_TEST"]
                       * out["calibration"]["honest"][dfn]["n_runs_TEST"]),
                 out["calibration"]["honest"][dfn]["n_runs_TEST"]))
        print("=" * 132)
        h = ("%-12s %5s | %6s %8s %7s %7s | %7s %7s | %6s %6s %6s | %7s %7s | %6s %6s"
             % ("level", "beta", "AUC", "TPR c-r", "run T", "run all", "s mean", "s max",
                "Bused", "Btrue", "unit", "|e|mean", "|e|max", "calls", "s/rnd"))
        print(h); print("-" * len(h))
        for k, c in rows:
            _, lvl, b = k.split("|")
            print("%-12s %5s | %6.3f %8s %7s %7s | %7s %7s | %6s %6s %6s | %7s %7s | %6s %6s"
                  % (lvl, b.split("=")[1], c["auc"],
                     pc(c["flagged_client_rounds_TEST"], c["n_test_client_rounds"]),
                     pc(round((c["tpr_run_TEST"] or 0) * c["n_test_runs"]), c["n_test_runs"]),
                     pc(round((c["tpr_run_all"] or 0) * c["n_passes"]), c["n_passes"]),
                     f(c["realised_mean"]), f(c["realised_max"]),
                     f(c["budget_used_mean"], "%6.3f"), f(c["budget_used_true_mean"], "%6.3f"),
                     f(c["unit_over_true_unit_mean"], "%6.3f"),
                     f(c["pred_vs_real_mean_abs"]), f(c["pred_vs_real_max_abs"]),
                     f(c["oracle_calls_mean"], "%6.1f"), f(c["craft_seconds_mean"], "%6.2f")))
    print("\nTPR c-r = flagged / attacked TEST client-rounds; run T = attacked TEST runs with at "
          "least one flag;\nrun all = the same over every pass (CAL seeds included, for "
          "information); Bused = ||C - C_cheap|| / B with\nthe attacker's own budget; Btrue = "
          "the same against beta times the true round-r unit; unit = the\nattacker's budget "
          "unit over the true one; |e| = |surrogate prediction - realised score|.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", default=os.path.join(RESULTS, "archives"))
    ap.add_argument("--prefix", default="arch_cifar100_s")
    ap.add_argument("--defenses", default="P,F")
    ap.add_argument("--levels", default=DEFAULT_LEVELS)
    ap.add_argument("--betas", default="0.5,1,2")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--victims", default="0,1")
    ap.add_argument("--cal-seeds", default="0,1")
    ap.add_argument("--test-seeds", default="3,4")
    ap.add_argument("--phi", type=float, default=0.0)
    ap.add_argument("--attack-start", type=int, default=12)
    ap.add_argument("--scope-start", type=int, default=12)
    ap.add_argument("--mad-floor", type=float, default=1e-2)
    ap.add_argument("--fpr-target", type=float, default=0.01)
    ap.add_argument("--p-solver", default="80,80,3",
                    help="basis steps, full steps, restarts of the auditor-only solver")
    ap.add_argument("--f-solver", default="40,80,1",
                    help="the same for the fused solver (family warm starts always off)")
    ap.add_argument("--fb-step", type=float, default=0.5)
    ap.add_argument("--fb-decay", type=float, default=0.8)
    ap.add_argument("--es-sigma0", type=float, default=0.1)
    ap.add_argument("--es-tol", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--parts", default=os.path.join(_HERE, "..", ".reproduction",
                                                    "limited_access_parts"),
                    help="directory caching one result per pass (git-ignored by default); "
                         "empty disables caching")
    ap.add_argument("--no-write", action="store_true",
                    help="run (and cache) the selected passes without writing the JSON")
    ap.add_argument("--out", default=os.path.join(RESULTS, "limited_access_grid.json"))
    a = ap.parse_args()
    t_start = time.time()

    ints = lambda s: [int(x) for x in s.split(",") if x.strip()]
    fls = lambda s: [float(x) for x in s.split(",") if x.strip()]
    strs = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    workers = max(1, min(20, a.workers))
    defenses, betas = strs(a.defenses), fls(a.betas)
    levels = [parse_level(x).name for x in strs(a.levels)]
    seeds, victims = ints(a.seeds), ints(a.victims)
    cal_seeds, test_seeds = ints(a.cal_seeds), ints(a.test_seeds)
    solver = {"P": tuple(ints(a.p_solver)), "F": tuple(ints(a.f_solver))}

    paths = archive_paths(a.archives, a.prefix)
    need = sorted(set(seeds) | set(cal_seeds) | set(test_seeds))
    shadow_seeds = sorted({parse_level(x).shadow_seed for x in levels
                           if parse_level(x).kind == "shadow"})
    missing = [s for s in need + shadow_seeds if s not in paths]
    if missing:
        raise SystemExit("no archive for seeds %s" % missing)
    clash = sorted(set(shadow_seeds) & set(need))
    if clash:
        raise SystemExit("shadow seeds %s are in the evaluated or calibration seeds" % clash)
    clients = len(ce.load_archive(paths[need[0]])["C"])

    htasks = []
    for s in need:
        t = honest_task(paths[s], s, a)
        t.update(defense="H", level="honest", signature="")
        t["signature"] = task_signature(t)
        htasks.append(t)
    hres = run_tasks(htasks, min(workers, len(htasks)), a.parts, fn=_honest_pass,
                     label="honest")
    honest = {h["seed"]: h for h in hres}
    calib = calibrate(honest, cal_seeds, test_seeds, clients, a.fpr_target)
    calib["release_check"] = check_against_release(calib)
    print("tau P %.10f   tau F %.10f   CAL medians P %.4f F %.4f   identical to release: %s"
          % (calib["tau"]["P"], calib["tau"]["F"], calib["cal_median"]["P"],
             calib["cal_median"]["F"], calib["release_check"]["identical"]), flush=True)

    common = {"phi": a.phi, "attack_start": a.attack_start, "scope_start": a.scope_start,
              "mad_floor": a.mad_floor, "fb_step": a.fb_step, "fb_decay": a.fb_decay,
              "es_sigma0": a.es_sigma0, "es_tol": a.es_tol,
              "norm": {k: list(v) for k, v in calib["norm"].items()}}
    tasks = []
    for dfn in defenses:
        for lvl in levels:
            for b in betas:
                for s in seeds:
                    for v in victims:
                        t = dict(common, path=paths[s], defense=dfn, level=lvl, beta=b, seed=s,
                                 victim=v, solver=list(solver[dfn]), tau=calib["tau"][dfn],
                                 cal_median=calib["cal_median"][dfn])
                        lv = parse_level(lvl)
                        if lv.kind == "shadow":
                            t["shadow_path"] = paths[lv.shadow_seed]
                        t["signature"] = task_signature(t)
                        tasks.append(t)
    cost = lambda t: ((2.0 if t["defense"] == "F" else 1.0)
                      * (1.0 + parse_level(t["level"]).queries / 64.0))
    tasks.sort(key=lambda t: -cost(t))
    results = run_tasks(tasks, workers, a.parts)
    if a.no_write:
        print("computed; JSON not written (--no-write). %.0fs" % (time.time() - t_start))
        return 0

    cells: Dict[str, List[dict]] = {}
    for p in results:
        k = p["key"]
        cells.setdefault("%s|%s|beta=%g" % (k["defense"], k["level"], k["beta"]), []).append(p)
    order = {lvl: i for i, lvl in enumerate(levels)}
    keys = sorted(cells, key=lambda k: (defenses.index(k.split("|")[0]),
                                        order[k.split("|")[1]], float(k.split("=")[1])))
    out_cells = {k: summarise_cell(cells[k], a.attack_start, calib["tau"][k[0]], test_seeds)
                 for k in keys}
    for dfn in ("P", "F"):
        flags = calib["honest"][dfn]["flag_matrix_TEST"]
        vic = [flags[str(s)][v] for s in test_seeds for v in victims]
        calib["honest"][dfn]["fpr_run_TEST_victim_ids"] = float(np.mean(vic))
        calib["honest"][dfn]["n_runs_TEST_victim_ids"] = len(vic)

    from passportfl.reproducibility import environment
    out = {
        "meta": {"schema": SCHEMA,
                 "command": "python experiments/limited_access_eval.py"
                            + "".join(" --%s %s" % (k, getattr(a, k.replace("-", "_")))
                                      for k in ("defenses", "levels", "betas", "seeds",
                                                "victims")
                                      if getattr(a, k.replace("-", "_")) != ap.get_default(
                                          k.replace("-", "_"))),
                 "environment": environment(),
                 "compute_s_total": round(sum(p["seconds"] for p in results), 1),
                 "wall_s_this_call": round(time.time() - t_start, 1)},
        "config": {"defenses": defenses, "levels": levels, "betas": betas, "seeds": seeds,
                   "victims": victims, "cal_seeds": cal_seeds, "test_seeds": test_seeds,
                   "phi": a.phi, "attack_start": a.attack_start, "scope_start": a.scope_start,
                   "mad_floor": a.mad_floor, "fpr_target": a.fpr_target,
                   "solver_basis_full_restarts": solver, "fused_warm_families": False,
                   "aim": "mimic", "feedback_step": a.fb_step, "feedback_decay": a.fb_decay,
                   "search_sigma0_in_B": a.es_sigma0, "search_tol": a.es_tol,
                   "archives": {str(s): os.path.basename(paths[s]) for s in need},
                   "shadow_archives": {str(s): os.path.basename(paths[s])
                                       for s in shadow_seeds}},
        "calibration": {"tau": calib["tau"], "cal_median": calib["cal_median"],
                        "norm": {k: list(v) for k, v in calib["norm"].items()},
                        "honest": calib["honest"], "release_check": calib["release_check"]},
        "cells": out_cells,
        "released_fused_check": released_fused_check(out_cells),
        "timing": timing_table(results)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(os.path.abspath(a.out), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    report(out)
    print("\nwrote %s   (%.0fs)" % (os.path.abspath(a.out), time.time() - t_start))
    return 0


def _honest_pass(task: dict) -> dict:
    t0 = time.perf_counter()
    res = ce.score_one_archive(task)
    return {"seed": task["seed"], "raw": res["raw"], "rounds": res["rounds"],
            "signature": task["signature"], "seconds": time.perf_counter() - t0,
            "key": {"defense": "H", "level": "honest", "beta": 0.0, "seed": task["seed"],
                    "victim": 0}}


if __name__ == "__main__":
    sys.exit(main())
