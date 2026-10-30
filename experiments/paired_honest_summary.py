from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import numpy as np

from passportfl.probe import MerkleTree, leaf_hash, prototype_drift_ratio
from passportfl.reproducibility import environment

import ghost_extended_eval as gx
from protocol_eval import clopper_pearson

RESULTS = os.path.join(REPO, "results")
LOGGED = ("deployed_w5", "deployed_w0", "two_sided_paper_w5", "two_sided_paper_w0")
EXTRA_STATS = gx.STATS + ("max_drift_rate",)
JOINT_WINDOW = (12, 35)
POST_FROM = 6
FLAGS = ("fused", "auditor", "norm_bound", "l2_two_sided")


def verdict(row: dict, warmup: int, bands: Optional[dict], band_from: int = 0) -> bool:
    if "proto_ratio" not in row:
        return False
    if row["round"] <= warmup:
        return True
    if not all(row["checks_no_warmup"].values()):
        return False
    if bands is not None and row["round"] >= band_from:
        for s in gx.STATS:
            lo, hi = bands[s]
            if row["defined"][s] and not (lo <= row[s] <= hi):
                return False
    return True


def verifiers(paper: dict, window: Optional[dict] = None,
              window6: Optional[dict] = None) -> Dict[str, Tuple[int, Optional[dict], int]]:
    v = {"deployed_w5": (5, None, 0), "deployed_w0": (0, None, 0),
         "two_sided_paper_w5": (5, paper, 0), "two_sided_paper_w0": (0, paper, 0)}
    if window is not None:
        v.update({"two_sided_window_w5": (5, window, JOINT_WINDOW[0]),
                  "two_sided_window_w0": (0, window, JOINT_WINDOW[0])})
    if window6 is not None:
        v["two_sided_window6_w5"] = (5, window6, POST_FROM)
    return v


JOINT_GHOST_NAMES = {"deployed_w5": "deployed_w5", "deployed_w0": "deployed_w0",
                     "two_sided_window_w5": "two_sided_w5", "two_sided_window_w0": "two_sided_w0",
                     "two_sided_window6_w5": "two_sided_w5_b6", "two_sided_paper_w5": "two_sided_w5_paper"}


def recompute_check(runs: dict, paper: dict) -> dict:
    out = {"client_rounds": 0, "mismatches": {n: 0 for n in LOGGED}, "logged_paper_bands_equal": True,
           "run_consistency": {}}
    for s, d in runs.items():
        for n in ("two_sided_paper_w5", "two_sided_paper_w0"):
            if {k: tuple(v) for k, v in d["receivers"][n]["bands"].items()} != paper:
                out["logged_paper_bands_equal"] = False
        for row in d["client_rounds"]:
            out["client_rounds"] += 1
            for n, spec in verifiers(paper).items():
                out["mismatches"][n] += int(verdict(row, *spec) != row["verdicts"][n]["accepted"])
        out["run_consistency"][str(s)] = {k: d["checks"][k] for k in (
            "server_rejections", "receiver_stat_mismatches", "server_stat_mismatches",
            "recompute_mismatches")}
    out["total_mismatches"] = int(sum(out["mismatches"].values()))
    return out


def rate(k: int, n: int) -> dict:
    p, lo, hi = clopper_pearson(k, n)
    return {"k": int(k), "n": int(n), "rate": p, "ci95": [lo, hi]}


def window_rows(d: dict, lo: int, hi: int) -> List[dict]:
    return [x for x in d["client_rounds"] if lo <= x["round"] <= hi]


def acceptance_block(rows: List[dict], clients: int, vers: dict) -> dict:
    out = {}
    for n, (w, b, f) in vers.items():
        ok = [verdict(x, w, b, f) for x in rows]
        per_client = [int(sum(not o for o, x in zip(ok, rows) if x["client"] == c)) for c in range(clients)]
        post = [x for x in rows if x["round"] > w and "proto_ratio" in x]
        fails = {k: int(sum(not x["checks_no_warmup"][k] for x in post)) for k in ("proto", "stability", "drift")}
        if b is not None:
            for s in gx.STATS:
                fails["band_" + s] = int(sum(x["round"] >= f and x["defined"][s]
                                             and not (b[s][0] <= x[s] <= b[s][1]) for x in post))
        out[n] = {"accepted": rate(sum(ok), len(ok)), "rejected_per_client": per_client,
                  "clients_rejected_at_least_once": int(sum(c > 0 for c in per_client)),
                  "clients_never_rejected": int(sum(c == 0 for c in per_client)),
                  "failing_checks_after_warmup": fails}
    return out


def stats_block(rows: List[dict]) -> dict:
    out = {}
    for s in EXTRA_STATS:
        key = "max_drift" if s == "max_drift_rate" else s
        v = [x[s] for x in rows if "proto_ratio" in x and x["defined"][key] and x[s] is not None]
        out[s] = gx.pct(v) if v else None
    return out


def fit_bands(rows: List[dict]) -> Tuple[dict, dict]:
    bands, n = {}, {}
    for s in EXTRA_STATS:
        key = "max_drift" if s == "max_drift_rate" else s
        v = [x[s] for x in rows if "proto_ratio" in x and x["defined"][key] and x[s] is not None]
        bands[s] = tuple(float(q) for q in np.percentile(v, gx.BAND_PCT))
        n[s] = len(v)
    return bands, n


def windows_of(d: dict) -> Dict[str, Tuple[int, int]]:
    R = int(d["config"]["rounds"])
    out = {"rounds_1_%d" % R: (1, R), "rounds_%d_%d" % (POST_FROM, R): (POST_FROM, R)}
    if R >= JOINT_WINDOW[0]:
        out["rounds_%d_%d" % (JOINT_WINDOW[0], min(JOINT_WINDOW[1], R))] = (JOINT_WINDOW[0], min(JOINT_WINDOW[1], R))
    return out


def payload_constants() -> dict:
    g = json.load(open(os.path.join(RESULTS, "combined_grid.json"), encoding="utf-8"))
    c = json.load(open(os.path.join(RESULTS, "adaptive_costcurve.json"), encoding="utf-8"))
    p = json.load(open(os.path.join(RESULTS, "published_predicates.json"), encoding="utf-8"))
    j = p["combined"]["two-sided l2 (sensitivity)"]["joint_1pct"]
    return {"norm": {k: tuple(v) for k, v in g["norm"].items()},
            "tau_fused": float(g["thresholds"]["combined_max"]),
            "tau_norm_bound": float(g["thresholds"]["norm_bound"]),
            "tau_auditor": float(c["thresholds"]["suspicion"]),
            "l2_band": (float(j["bounds"]["l2_lower_bound"]), float(j["bounds"]["l2_bound"])),
            "sources": {"fused, norm_bound": "results/combined_grid.json norm, thresholds",
                        "auditor": "results/adaptive_costcurve.json thresholds.suspicion",
                        "l2": "results/published_predicates.json combined['two-sided l2 "
                              "(sensitivity)'].joint_1pct.bounds"},
            "released_honest_TEST": {"l2_joint_accepted": j["honest"]}}


def payload_flags(path: str, seed: int, const: dict, lo: int, hi: int) -> dict:
    import combined_eval as ce
    import limited_access_eval as la
    task = la.honest_task(path, seed, SimpleNamespace(attack_start=lo, scope_start=lo, mad_floor=1e-2))
    res = ce.score_one_archive(task)
    arch = ce.load_archive(path)
    clients = len(arch["C"])
    rounds = [r for r in res["rounds"] if r <= hi]
    out = {"rounds": rounds, "clients": clients}
    fused, normb, aud, l2 = [], [], [], []
    for i, r in enumerate(res["rounds"]):
        if r > hi:
            continue
        raw = {k: {c: float(res["raw"][k][i, c]) for c in range(clients)} for k in res["raw"]}
        comb = ce.combined_from_raw(raw, const["norm"], clients)
        fused.append([comb["combined_max"][c] for c in range(clients)])
        normb.append([comb["norm_bound"][c] for c in range(clients)])
        aud.append([raw["suspicion"][c] for c in range(clients)])
        l2.append([float(np.linalg.norm(np.asarray(arch["C"][c][r], np.float64) - arch["cbar"][r - 1]))
                   for c in range(clients)])
    exceed = lambda m, t: np.where(np.isfinite(m), m, -np.inf) > t
    fused, normb, aud, l2 = (np.asarray(x, float) for x in (fused, normb, aud, l2))
    out["flags"] = {"fused": exceed(fused, const["tau_fused"]), "auditor": exceed(aud, const["tau_auditor"]),
                    "norm_bound": exceed(normb, const["tau_norm_bound"]),
                    "l2_two_sided": ~((l2 >= const["l2_band"][0]) & (l2 <= const["l2_band"][1]))}
    out["values"] = {"fused": fused, "auditor": aud, "norm_bound": normb, "l2": l2}
    return out


def released_payload_check(const: dict) -> dict:
    import limited_access_eval as la
    paths = la.archive_paths(os.path.join(RESULTS, "archives"), "arch_cifar100_s")
    lg = json.load(open(os.path.join(RESULTS, "limited_access_grid.json"), encoding="utf-8"))["calibration"]["honest"]
    flags = {s: payload_flags(paths[s], s, const, *JOINT_WINDOW)["flags"] for s in range(3, 8) if s in paths}
    out = {"archives": {str(s): os.path.basename(paths[s]) for s in flags}}
    for k, rel in (("fused", "F"), ("auditor", "P")):
        f = np.concatenate([flags[s][k] for s in (3, 4)])
        out[k] = {"recomputed_fpr_TEST_3_4": float(f.mean()), "n": int(f.size),
                  "released": lg[rel]["fpr_client_round_TEST"],
                  "identical": float(f.mean()) == lg[rel]["fpr_client_round_TEST"]}
    acc = np.concatenate([~flags[s]["l2_two_sided"] for s in range(3, 8)])
    rel_l2 = const["released_honest_TEST"]["l2_joint_accepted"]
    out["l2_two_sided"] = {"recomputed_accepted_TEST_3_7": int(acc.sum()), "n": int(acc.size),
                           "released": [rel_l2["k"], rel_l2["n"]],
                           "identical": [int(acc.sum()), int(acc.size)] == [rel_l2["k"], rel_l2["n"]]}
    nb = np.concatenate([flags[s]["norm_bound"] for s in (3, 4)])
    out["norm_bound"] = {"recomputed_fpr_TEST_3_4": float(nb.mean()), "n": int(nb.size)}
    out["all_identical"] = bool(out["fused"]["identical"] and out["auditor"]["identical"]
                                and out["l2_two_sided"]["identical"])
    return out


def replay_job(args) -> dict:
    probes_path, payload_path, seed, cid, logged, n_seq, bands_by_name, windows = args
    z = np.load(probes_path, allow_pickle=False)
    scale = float(z["scale"])
    Zq, opened, roots = z["Zq"][cid], z["opened"][cid].astype(np.int64), z["roots"][cid]
    pay = np.load(payload_path, allow_pickle=True)
    Cd = pay["C"].item()[cid]
    R = opened.shape[0]
    tr = gx.Trajectory(Z0=Zq[0] / scale, C=[np.asarray(Cd[r], np.float64) for r in range(1, R + 1)],
                       Z=[Zq[r] / scale for r in range(1, R + 1)])
    root_mismatch = int(sum(MerkleTree([leaf_hash(e) for e in (tr.Z0 if r == 0 else tr.Z[r - 1])]).root.hex()
                            != roots[r] for r in range(R + 1)))
    t0 = time.time()
    hr = gx.HistoryReplay(tr)
    st = hr.run(lambda q: opened[q - 1][None, :])
    diff = 0.0
    for row in logged:
        i = row["round"] - 1
        for s in gx.STATS:
            diff = max(diff, abs(float(st[s][0, i]) - row[s]))
        diff = max(diff, abs(float(st["max_drift_rate"][0, i]) - (row["max_drift_rate"] or 0.0)))
    def replay_verdicts(stats, spec):
        w, b, f = spec
        ok = gx.HistoryReplay.verdicts(stats, w, b)
        if b is not None and f > 1:
            ok[:, :f - 1] = gx.HistoryReplay.verdicts(stats, w, None)[:, :f - 1]
        return ok

    verdict_mm = 0
    for name, spec in bands_by_name.items():
        ok = replay_verdicts(st, spec)[0]
        verdict_mm += int(sum(bool(ok[row["round"] - 1]) != row["recomputed"][name] for row in logged))
    out = {"seed": seed, "client": cid, "root_mismatches": root_mismatch,
           "realised_replay": {"max_abs_stat_diff": diff, "verdict_mismatches": verdict_mm}}
    if n_seq > 0:
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
            b"paired-honest-mc|%d|%d" % (seed, cid)).digest()[:8], "little"))
        mc = hr.run(lambda r: np.argsort(rng.random((n_seq, gx.M)), axis=1)[:, :gx.T])
        out["mc"] = {}
        for name, spec in bands_by_name.items():
            ok = replay_verdicts(mc, spec)
            out["mc"][name] = {}
            for wname, (lo, hi) in windows.items():
                sel = ok[:, lo - 1:hi]
                out["mc"][name][wname] = {"sequences_all_accepted": int(sel.all(1).sum()),
                                          "mean_round_acceptance": float(sel.mean())}
    out["seconds"] = time.time() - t0
    return out


def legacy_bands() -> dict:
    import combined_eval as ce
    import limited_access_eval as la
    lo, hi = JOINT_WINDOW
    paths = la.archive_paths(os.path.join(RESULTS, "archives"), "arch_cifar100_s")
    ratio = []
    for s in (0, 1):
        arch = ce.load_archive(paths[s])
        ratio += [prototype_drift_ratio(np.asarray(arch["C"][c][r]), np.asarray(arch["C"][c][r - 1]))
                  for c in arch["C"] for r in range(lo, hi + 1)]
    stab, mdr = [], []
    for s in (0, 1):
        log = json.load(open(os.path.join(RESULTS, "fedproto_abl_pp_s%d.json" % s), encoding="utf-8"))["verify_log"]
        rows = [e for e in log if lo <= int(e["round"]) <= hi]
        stab += [e["stability"] for e in rows]
        mdr += [e["max_drift"] for e in rows]
    pct = lambda v: [float(q) for q in np.percentile(v, gx.BAND_PCT)]
    return {"proto_ratio": {"band": pct(ratio), "n": len(ratio),
                            "source": [os.path.basename(paths[s]) for s in (0, 1)]},
            "label_stability": {"band": pct(stab), "n": len(stab),
                                "source": ["fedproto_abl_pp_s0.json", "fedproto_abl_pp_s1.json"]},
            "max_drift": {"band": pct(mdr), "n": len(mdr),
                          "source": ["fedproto_abl_pp_s0.json", "fedproto_abl_pp_s1.json"]}}


def ghost_reference() -> Optional[dict]:
    path = os.path.join(RESULTS, "ghost_extended.json")
    if not os.path.exists(path):
        return None
    g = json.load(open(path, encoding="utf-8"))
    out = {}
    for kind in ("original", "calibrated"):
        ch = g["ghosts"][kind]["chain_head"]
        out[kind] = {"rounds": ch["rounds"], "accepted_chain_head": ch["accepted"],
                     "medians_rounds_6_40": ch["medians_rounds_gt_5"],
                     "mc_two_sided_w5_all_accepted": g["ghosts"][kind]["random_history_monte_carlo"]
                     ["two_sided_w5"]["sequences_all_rounds_accepted"],
                     "mc_sequences": g["ghosts"][kind]["random_history_monte_carlo"]["sequences"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--cal-seeds", default="0,1")
    ap.add_argument("--dev-seeds", default="2")
    ap.add_argument("--test-seeds", default="3,4")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--archive-dir", default=os.path.join(RESULTS, "paired_honest"))
    ap.add_argument("--paper-bands", default=os.path.join(RESULTS, "ghost_extended.json"))
    ap.add_argument("--mc-seqs", type=int, default=10000,
                    help="random challenge sequences per honest client (0: realised replay only)")
    ap.add_argument("--no-replay", action="store_true", help="skip step 8")
    ap.add_argument("--no-payload", action="store_true", help="skip step 7")
    ap.add_argument("--no-legacy", action="store_true", help="skip step 9")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(RESULTS, "paired_honest_summary.json"))
    a = ap.parse_args()
    t_all = time.time()
    ints = lambda s: [int(x) for x in s.split(",") if x.strip()]
    seeds, cal, dev, test = ints(a.seeds), ints(a.cal_seeds), ints(a.dev_seeds), ints(a.test_seeds)

    runs, missing = {}, []
    for s in seeds:
        p = os.path.join(a.results, "paired_honest_s%d.json" % s)
        if os.path.exists(p):
            runs[s] = json.load(open(p, encoding="utf-8"))
        else:
            missing.append(s)
    if not runs:
        print("no paired_honest_s*.json under %s" % a.results)
        return 1
    paper = gx_paper_bands(a.paper_bands)
    out = {"meta": {"script": "experiments/paired_honest_summary.py", "environment": environment(),
                    "inputs": {str(s): {"file": "paired_honest_s%d.json" % s,
                                        "rounds": int(d["config"]["rounds"]), "scheme": d["config"]["scheme"],
                                        "clients": int(d["config"]["clients"]),
                                        "final_acc_prototype": d["history"][-1]["acc_prototype"],
                                        "final_acc_classifier": d["history"][-1]["acc_classifier"]}
                               for s, d in runs.items()},
                    "missing_seeds": missing},
           "config": {"seeds": seeds, "cal_seeds": cal, "dev_seeds": dev, "test_seeds": test,
                      "joint_window": list(JOINT_WINDOW), "post_warmup_from": POST_FROM,
                      "band_percentiles": list(gx.BAND_PCT), "paper_bands": {k: list(v) for k, v in paper.items()},
                      "mc_seqs": a.mc_seqs}}

    print("[1/9] verdict recomputation ...")
    out["verdict_recomputation"] = vr = recompute_check(runs, paper)
    print("      %d client-rounds, mismatches %s" % (vr["client_rounds"], vr["mismatches"]))

    cal_ok = bool(cal) and all(s in runs for s in cal)
    cal_rows = [x for s in cal if s in runs for x in window_rows(runs[s], *JOINT_WINDOW)]
    sens = [x for s in cal if s in runs for x in window_rows(runs[s], POST_FROM, JOINT_WINDOW[1])]
    window, window_n = fit_bands(cal_rows) if cal_ok and cal_rows else (None, None)
    window6 = fit_bands(sens)[0] if cal_ok and sens else None
    wb3 = {s: window[s] for s in gx.STATS} if window else None
    wb6 = {s: window6[s] for s in gx.STATS} if window6 else None
    vers = verifiers(paper, wb3, wb6)
    out["config"]["verifiers"] = {n: {"warmup": w, "bands": ("none" if b is None else "paper" if b is paper
                                                             else "window_12_35" if b is wb3 else "window_6_35"),
                                      "bands_from_round": max(f, w + 1) if b is not None else None,
                                      "joint_ghost_eval_name": JOINT_GHOST_NAMES.get(n)}
                                  for n, (w, b, f) in vers.items()}

    print("[2-3/9] acceptance and statistics ...")
    out["per_seed"] = {}
    for s, d in runs.items():
        clients = int(d["config"]["clients"])
        out["per_seed"][str(s)] = {
            w: {"acceptance": acceptance_block(window_rows(d, lo, hi), clients, vers),
                "statistics": stats_block(window_rows(d, lo, hi))}
            for w, (lo, hi) in windows_of(d).items()}

    print("[4-5/9] window bands and false alarms ...")
    out["window_bands"] = ({"bands_p1_p99": {k: list(v) for k, v in window.items()}, "n": window_n,
                            }
                           if window else {"unavailable": "CAL seeds %s missing" % cal})
    if window:
        out["bands"] = {"window_12_35": {s: list(wb3[s]) for s in gx.STATS},
                        "window_6_35": {s: list(wb6[s]) for s in gx.STATS}}
        out["bands_undefined_rows"] = {
            k: int(sum(not all(x["defined"].values()) for x in rr))
            for k, rr in (("window_12_35", cal_rows), ("window_6_35", sens))}
    roles = {"CAL_in_sample": cal, "DEV": dev, "TEST": test}
    fa = {}
    for role, ss in roles.items():
        have = [s for s in ss if s in runs]
        if not have:
            continue
        by_seed = {s: window_rows(runs[s], *JOINT_WINDOW) for s in have}
        rows = [x for s in have for x in by_seed[s]]
        fa[role] = {"seeds": have, "client_rounds": len(rows)}
        for n, spec in vers.items():
            rej = sum(not verdict(x, *spec) for x in rows)
            fa[role][n] = rate(rej, len(rows))
            hit = [any(not verdict(x, *spec) for x in by_seed[s] if x["client"] == c)
                   for s in have for c in range(int(runs[s]["config"]["clients"]))]
            fa[role][n]["clients_rejected_at_least_once"] = int(sum(hit))
            fa[role][n]["clients"] = len(hit)
    out["false_alarms_rounds_12_35"] = fa
    out["statistics_rounds_12_35_by_role"] = {
        role: {"seeds": [s for s in ss if s in runs],
               **stats_block([x for s in ss if s in runs for x in window_rows(runs[s], *JOINT_WINDOW)])}
        for role, ss in roles.items() if any(s in runs for s in ss)}

    print("[6/9] Fig. 3a block ...")
    if 0 in runs:
        d = runs[0]
        R = int(d["config"]["rounds"])
        out["fig3a"] = {"seed": 0, "rounds": R, "honest": {}, "ghosts": ghost_reference()}
        for w, (lo, hi) in (("rounds_1_%d" % R, (1, R)), ("rounds_6_%d" % R, (6, R))):
            rows = window_rows(d, lo, hi)
            blk = {"client_rounds": len(rows)}
            for n, spec in vers.items():
                ok = [verdict(x, *spec) for x in rows]
                blk[n] = {"fraction_accepted": float(np.mean(ok)),
                          "clients_accepted_every_round": int(sum(
                              all(o for o, x in zip(ok, rows) if x["client"] == c)
                              for c in range(int(d["config"]["clients"]))))}
            blk["medians"] = {s: (v["median"] if v else None) for s, v in stats_block(rows).items()}
            out["fig3a"]["honest"][w] = blk

    if not a.no_payload:
        print("[7/9] payload layer ...")
        const = payload_constants()
        pl = {"constants": {k: (list(v) if isinstance(v, tuple) else v) for k, v in const.items() if k != "norm"},
              "regression_released_archives": released_payload_check(const), "per_seed": {}}
        print("      released-archive check identical: %s" % pl["regression_released_archives"]["all_identical"])
        joint_rows = {}
        for s, d in runs.items():
            path = os.path.join(a.archive_dir, "paired_honest_payload_s%d.npz" % s)
            if not os.path.exists(path):
                pl["per_seed"][str(s)] = {"missing": os.path.basename(path)}
                continue
            lo, hi = JOINT_WINDOW
            pf = payload_flags(path, s, const, lo, hi)
            idx = {(x["round"], x["client"]): x for x in d["client_rounds"]}
            fl = pf["flags"]
            anyf = fl["fused"] | fl["auditor"] | fl["norm_bound"] | fl["l2_two_sided"]
            blk = {"rounds": [pf["rounds"][0], pf["rounds"][-1]], "client_rounds": int(anyf.size),
                   **{k: rate(int(v.sum()), int(v.size)) for k, v in fl.items()},
                   "any_payload_flag": rate(int(anyf.sum()), int(anyf.size))}
            jr = []
            for i, r in enumerate(pf["rounds"]):
                for c in range(pf["clients"]):
                    row = idx[(r, c)]
                    jr.append({"seed": s, "round": r, "client": c,
                               **{k: bool(fl[k][i, c]) for k in fl},
                               **{n: verdict(row, *spec) for n, spec in vers.items()}})
            joint_rows[s] = jr
            for n in vers:
                ok = [x[n] and not any(x[k] for k in FLAGS) for x in jr]
                blk["joint_rejected_" + n] = rate(int(sum(not o for o in ok)), len(ok))
            pl["per_seed"][str(s)] = blk
        pooled = {}
        for role, ss in roles.items():
            jr = [x for s in ss if s in joint_rows for x in joint_rows[s]]
            if not jr:
                continue
            pooled[role] = {"seeds": [s for s in ss if s in joint_rows], "client_rounds": len(jr),
                            "any_payload_flag": rate(sum(any(x[k] for k in FLAGS) for x in jr), len(jr))}
            for n in vers:
                rej = sum(not (x[n] and not any(x[k] for k in FLAGS)) for x in jr)
                pooled[role]["joint_rejected_" + n] = rate(rej, len(jr))
                traj = []
                for s in pooled[role]["seeds"]:
                    d = runs[s]
                    for c in range(int(d["config"]["clients"])):
                        com = all(verdict(x, *vers[n]) for x in window_rows(d, POST_FROM, JOINT_WINDOW[1])
                                  if x["client"] == c)
                        pay = all(not any(x[k] for k in FLAGS) for x in joint_rows[s] if x["client"] == c)
                        traj.append(com and pay)
                pooled[role]["trajectories_fully_accepted_" + n] = rate(int(sum(traj)), len(traj))
        pl["pooled"] = pooled
        pl["client_round_table"] = {str(s): v for s, v in joint_rows.items()}
        out["payload_layer"] = pl

    if not a.no_replay:
        print("[8/9] realised and random challenge replay (%d sequences per client) ..." % a.mc_seqs)
        jobs = []
        for s, d in runs.items():
            pp = os.path.join(a.archive_dir, "paired_honest_probes_s%d.npz" % s)
            pa = os.path.join(a.archive_dir, "paired_honest_payload_s%d.npz" % s)
            if not (os.path.exists(pp) and os.path.exists(pa)):
                continue
            wins = windows_of(d)
            for c in range(int(d["config"]["clients"])):
                logged = []
                for x in d["client_rounds"]:
                    if x["client"] == c and "proto_ratio" in x:
                        logged.append({**{k: x[k] for k in EXTRA_STATS}, "round": x["round"],
                                       "recomputed": {n: verdict(x, *spec) for n, spec in vers.items()}})
                jobs.append((pp, pa, s, c, logged, a.mc_seqs, vers, wins))
        t0 = time.time()
        with ProcessPoolExecutor(max(1, min(a.workers, 4))) as ex:
            res = list(ex.map(replay_job, jobs))
        rc = {"jobs": len(res), "seconds": time.time() - t0,
              "root_mismatches": int(sum(r["root_mismatches"] for r in res)),
              "realised_max_abs_stat_diff": max((r["realised_replay"]["max_abs_stat_diff"] for r in res), default=None),
              "realised_verdict_mismatches": int(sum(r["realised_replay"]["verdict_mismatches"] for r in res))}
        if a.mc_seqs > 0:
            mc = {}
            for role, ss in roles.items():
                rr = [r for r in res if r["seed"] in ss]
                if not rr:
                    continue
                mc[role] = {"seeds": sorted({r["seed"] for r in rr}), "clients": len(rr), "per_verifier": {}}
                for n in vers:
                    mc[role]["per_verifier"][n] = {}
                    for wname in rr[0]["mc"][n]:
                        tot = sum(r["mc"][n][wname]["sequences_all_accepted"] for r in rr)
                        mc[role]["per_verifier"][n][wname] = {
                            "sequences_all_accepted": rate(tot, a.mc_seqs * len(rr)),
                            "clients_with_any_fully_accepted_sequence": int(sum(
                                r["mc"][n][wname]["sequences_all_accepted"] > 0 for r in rr)),
                            "mean_round_acceptance": float(np.mean([r["mc"][n][wname]["mean_round_acceptance"]
                                                                    for r in rr]))}
            rc["monte_carlo"] = mc
        rc["per_client"] = res
        out["random_challenges"] = rc
        print("      %d clients, root mismatches %d, realised max stat diff %s, verdict mismatches %d (%.0fs)"
              % (rc["jobs"], rc["root_mismatches"], rc["realised_max_abs_stat_diff"],
                 rc["realised_verdict_mismatches"], rc["seconds"]))

    if not a.no_legacy:
        print("[9/9] legacy window bands ...")
        out["legacy_window_bands"] = legacy_bands()

    out["meta"]["total_seconds"] = time.time() - t_all
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=jsonable)
    report(out)
    print("wrote %s (%.0fs)" % (a.out, out["meta"]["total_seconds"]))
    return 0


def gx_paper_bands(path: str) -> dict:
    g = json.load(open(path, encoding="utf-8"))
    return {s: tuple(float(x) for x in g["honest_reference"]["bands_p1_p99"][s]) for s in gx.STATS}


def jsonable(o):
    if isinstance(o, tuple):
        return list(o)
    return gx.jsonable(o)


def report(out: dict) -> None:
    for s, blk in out["per_seed"].items():
        for w, b in blk.items():
            print("seed %s %-12s " % (s, w) + "  ".join(
                "%s %d/%d" % (n, v["accepted"]["k"], v["accepted"]["n"]) for n, v in b["acceptance"].items()))
    for role, b in out.get("false_alarms_rounds_12_35", {}).items():
        print("%-14s rejected in rounds 12-35: " % role + "  ".join(
            "%s %.3f" % (n, v["rate"]) for n, v in b.items() if isinstance(v, dict)))
    for role, b in out.get("statistics_rounds_12_35_by_role", {}).items():
        print("%-14s medians in rounds 12-35: " % role + "  ".join(
            "%s %.4f" % (s, b[s]["median"]) for s in EXTRA_STATS if b.get(s)))
    if out.get("window_bands", {}).get("bands_p1_p99"):
        print("window bands (CAL, rounds 12-35): %s" % {k: [round(x, 4) for x in v]
                                                       for k, v in out["window_bands"]["bands_p1_p99"].items()})


if __name__ == "__main__":
    sys.exit(main())
