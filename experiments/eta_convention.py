from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import functools
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from passportfl.innovation import InnovationAuditor

RESULTS = os.path.join(HERE, "results")
ARCHIVES = os.path.join(RESULTS, "archives")
ATTACKS = ("global_echo", "peer_copy", "lagged_self", "ar2_replay", "noise", "shrink")
NEW_FLOOR = 1e-2
OLD_FLOOR = 1e-3
VERIFY_TAGS = ("cifar100_s0", "nopp_s0", "cohort05_s0", "cifar10_s0")
GROUPS = {
    "cifar100_d05": ("cifar100_s%d", (0, 1, 2, 3, 4)),
    "cifar100_d05_seeds_0_2": ("cifar100_s%d", (0, 1, 2)),
    "cifar100_d01": ("cifar100_d01_s%d", (0, 1, 2)),
    "cifar10": ("cifar10_s%d", (0, 1, 2)),
    "late70": ("late70_s%d", (0, 1, 2)),
    "cohort05": ("cohort05_s%d", (0, 1, 2)),
    "cohort20": ("cohort20_s%d", (0, 1, 2)),
    "cohort40": ("cohort40_s%d", (0, 1, 2)),
    "nopp": ("nopp_s%d", (0, 1, 2)),
}
FROZEN_THRESHOLD_1PCT = 3.7859094579543044
NUMPY_AUC_TOL = 1e-3
NUMPY_MEAN_RTOL = 1e-12


def same(x, y) -> bool:
    if x is None or y is None:
        return x is None and y is None
    if isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y):
        return True
    return x == y


def portable(v):
    if isinstance(v, str) and os.path.isabs(v):
        try:
            return os.path.relpath(v, HERE).replace(os.sep, "/")
        except ValueError:
            return v
    return v


def archive_for(cfg: dict) -> str:
    tag = cfg.get("tag") or cfg.get("dataset")
    a = os.path.join(ARCHIVES, "arch_%s_c%d_r%d_d%s_s%d.npz"
                     % (tag, cfg["clients"], cfg["rounds"], cfg["dirichlet"], cfg["seed"]))
    b = os.path.join(ARCHIVES, "arch_%s_c%d_r%d_s%d.npz"
                     % (tag, cfg["clients"], cfg["rounds"], cfg["seed"]))
    if not os.path.exists(a) and cfg["dirichlet"] == 0.5 and os.path.exists(b):
        return b
    return a


def load_archive(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    return {"C": z["C"].item(), "present": z["present"].item(), "cbar": z["cbar"].item()}


def with_floor(module, eta: float) -> None:
    module.InnovationAuditor = functools.partial(InnovationAuditor, mad_floor=eta)


def task_innovation(task: dict) -> dict:
    import multiprocessing
    if multiprocessing.current_process().name != "MainProcess":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import innovation_eval as IE
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    with_floor(IE, task["eta"])
    cfg = task["cfg"]
    arch = load_archive(task["path"])
    pos, neg = IE.replay(arch, task["attack"], cfg["attack_start"], cfg["eval_from"],
                         cfg["clients"], cfg["rounds"], cfg["seed"])
    res = {k: {"auc": IE.roc_auc(pos[k], neg[k]),
               "pos_mean": float(np.mean(pos[k])) if pos[k] else None,
               "neg_mean": float(np.mean(neg[k])) if neg[k] else None,
               "n_pos": len(pos[k]), "n_neg": len(neg[k])} for k in pos}
    return {"tag": cfg["tag"], "attack": task["attack"], "eta": task["eta"], "results": res}


def task_sweep(task: dict) -> dict:
    import parameter_sweep as PS
    with_floor(PS, task["eta"])
    arch = PS.load_archive(task["path"])
    r = PS.evaluate(arch, task["kind"], task["strength"], task["n_att"], task["clients"],
                    task["rounds"], 12, 12, task["index"])
    return dict(task, path=os.path.basename(task["path"]), aucs=r)


def collusion_detail(path: str, n_att: int, eta: float, clients: int = 10,
                     rounds: int = 35, attack_start: int = 12, eval_from: int = 12) -> dict:
    arch = load_archive(path)
    attackers = list(range(clients - n_att, clients))
    source = {}
    for cid in attackers:
        others = [j for j in arch["C"] if j != cid]
        source[cid] = others[cid % len(others)]
    aud = InnovationAuditor(mad_floor=eta)
    S = {c: [] for c in range(clients)}
    for r in range(1, rounds + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        payloads = {}
        for cid in range(clients):
            C = arch["C"][cid][r]
            if cid in source and r >= attack_start:
                C = arch["C"][source[cid]][r]
            payloads[cid] = (C, arch["present"][cid][r])
        rs = aud.observe_round(r, payloads)
        if r < max(eval_from, attack_start):
            continue
        for cid, v in rs.per_client.items():
            S[cid].append(float(v.get("suspicion", float("nan"))))
    honest = [c for c in range(clients) if c not in source]
    dup = sorted({s for s in source.values() if s not in source})
    from protocol_eval import roc_auc
    pos = np.concatenate([np.asarray(S[c]) for c in attackers])
    neg = np.concatenate([np.asarray(S[c]) for c in honest])
    negx = np.concatenate([np.asarray(S[c]) for c in honest if c not in dup])
    out = {"attackers": attackers, "copies": {str(k): v for k, v in source.items()},
           "honest_clients_duplicated": dup,
           "attackers_republishing_another_attackers_source": sorted(
               c for c in attackers if source[c] in source),
           "auc": roc_auc(pos, neg), "auc_excl_duplicated_honest": roc_auc(pos, negx)}
    if abs(eta - NEW_FLOOR) < 1e-15:
        thr = FROZEN_THRESHOLD_1PCT
        fl = {c: np.asarray(S[c]) > thr for c in range(clients)}
        out["frozen_threshold_1pct"] = {
            "threshold": thr,
            "attacker_rounds_flagged": int(sum(fl[c].sum() for c in attackers)),
            "attacker_rounds": int(sum(fl[c].size for c in attackers)),
            "per_attacker_rounds_flagged": {str(c): int(fl[c].sum()) for c in attackers},
            "honest_rounds_flagged": int(sum(fl[c].sum() for c in honest)),
            "honest_rounds": int(sum(fl[c].size for c in honest)),
            "duplicated_honest_rounds_flagged": int(sum(fl[c].sum() for c in dup)),
        }
    return out


def run_pool(fn, tasks: List[dict], workers: int) -> List[dict]:
    if workers <= 1 or len(tasks) <= 2:
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, tasks, chunksize=1))


def summarise_group(rows: Dict[str, dict], tags: List[str]) -> dict:
    per = {}
    for a in ATTACKS:
        v = [rows[t][a]["suspicion"]["auc"] for t in tags if t in rows and a in rows[t]]
        v = [x for x in v if x is not None and np.isfinite(x)]
        per[a] = {"mean": float(np.mean(v)) if v else None,
                  "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v),
                  "per_seed": [rows[t][a]["suspicion"]["auc"] for t in tags if t in rows]}
    fin = {a: p["mean"] for a, p in per.items() if p["mean"] is not None}
    w = min(fin, key=lambda a: fin[a])
    return {"per_attack": per, "worst": fin[w], "worst_attack": w}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--out", default=os.path.join(RESULTS, "eta_convention.json"))
    a = ap.parse_args()
    t_all = time.time()

    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    tags = sorted({GROUPS[g][0] % s for g in groups for s in GROUPS[g][1]})
    released = {}
    for t in tags:
        p = os.path.join(RESULTS, "innovation_%s.json" % t)
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            assert d["config"]["tag"] == t, (p, d["config"]["tag"])
            released[t] = d
    missing = [t for t in tags if t not in released or not os.path.exists(
        archive_for(released[t]["config"]))]
    if missing:
        print("missing configurations or archives: %s" % missing)
        return 1

    tasks = []
    for eta, tt in ((NEW_FLOOR, tags), (OLD_FLOOR, [t for t in VERIFY_TAGS if t in tags])):
        for t in tt:
            cfg = released[t]["config"]
            for atk in ATTACKS:
                tasks.append({"eta": eta, "cfg": cfg, "attack": atk, "path": archive_for(cfg),
                              "cost": cfg["clients"] ** 2 * cfg["rounds"]})
    tasks.sort(key=lambda t: -t["cost"])
    t0 = time.time()
    res = run_pool(task_innovation, tasks, a.workers)
    print("innovation replays: %d tasks in %.1fs" % (len(tasks), time.time() - t0))
    rows = {OLD_FLOOR: {}, NEW_FLOOR: {}}
    for r in res:
        rows[r["eta"]].setdefault(r["tag"], {})[r["attack"]] = r["results"]

    checks = []
    exact_bad, tol_bad = [], []
    deviation = {"auc_max_abs": 0.0, "mean_max_rel": 0.0, "auc_cells_differing": []}
    for t, by_atk in rows[OLD_FLOOR].items():
        for atk, stats in by_atk.items():
            for k, v in stats.items():
                ref = released[t]["results"][atk][k]
                for f in ("n_pos", "n_neg"):
                    if v[f] != ref[f]:
                        exact_bad.append((t, atk, k, f))
                if not same(v["auc"], ref["auc"]):
                    if k == "suspicion":
                        exact_bad.append((t, atk, k, "auc"))
                    d = abs(v["auc"] - ref["auc"])
                    deviation["auc_max_abs"] = max(deviation["auc_max_abs"], d)
                    deviation["auc_cells_differing"].append([t, atk, k, v["auc"], ref["auc"]])
                    if not d <= NUMPY_AUC_TOL:
                        tol_bad.append((t, atk, k, "auc", d))
                for f in ("pos_mean", "neg_mean"):
                    if not same(v[f], ref[f]):
                        r = (abs(v[f] - ref[f]) / max(abs(ref[f]), 1e-300)
                             if v[f] is not None and ref[f] is not None else float("inf"))
                        deviation["mean_max_rel"] = max(deviation["mean_max_rel"], r)
                        if not r <= NUMPY_MEAN_RTOL:
                            tol_bad.append((t, atk, k, f, r))
    checks.append({"check": "eta=1e-3 rerun of %s reproduces the released innovation files: "
                            "suspicion AUC and every count exactly" % ", ".join(rows[OLD_FLOOR]),
                   "ok": not exact_bad, "detail": str(exact_bad[:5])})
    checks.append({"check": "eta=1e-3 rerun: the other statistics agree with the released "
                            "files within numpy rounding (|dAUC| <= %g, relative dmean <= %g)"
                            % (NUMPY_AUC_TOL, NUMPY_MEAN_RTOL),
                   "ok": not tol_bad,
                   "detail": "%s; %d AUC cells differ, max |dAUC| %.1e (%s); max relative "
                             "dmean %.1e" % (tol_bad[:5], len(deviation["auc_cells_differing"]),
                                             deviation["auc_max_abs"],
                                             sorted({(c[1], c[2]) for c in
                                                     deviation["auc_cells_differing"]}),
                                             deviation["mean_max_rel"])})
    bad, n_cells = [], 0
    for t, by_atk in rows[OLD_FLOOR].items():
        for atk, stats in by_atk.items():
            for k, v in stats.items():
                if k == "suspicion":
                    continue
                w = rows[NEW_FLOOR][t][atk][k]
                for f in ("auc", "n_pos", "n_neg", "pos_mean", "neg_mean"):
                    n_cells += 1
                    if not same(v[f], w[f]):
                        bad.append((t, atk, k, f))
    checks.append({"check": "floor-independent statistics (raw u1,u2,g,x,rep and raw "
                            "Byzantine scores) at eta=0.01 equal the eta=1e-3 rerun bit for bit",
                   "ok": not bad, "detail": "%d cells compared %s" % (n_cells, bad[:5])})
    bad, n_diff, by_attack = [], 0, {}
    for t, by_atk in rows[NEW_FLOOR].items():
        for atk, stats in by_atk.items():
            for k, v in stats.items():
                if k == "suspicion":
                    continue
                ref = released[t]["results"][atk][k]
                if not same(v["auc"], ref["auc"]):
                    n_diff += 1
                    by_attack[atk] = by_attack.get(atk, 0) + 1
                    if not abs(v["auc"] - ref["auc"]) <= NUMPY_AUC_TOL:
                        bad.append((t, atk, k, abs(v["auc"] - ref["auc"])))
    checks.append({"check": "floor-independent statistics at eta=0.01 agree with the released "
                            "files of every configuration within numpy rounding (|dAUC| <= %g)"
                            % NUMPY_AUC_TOL, "ok": not bad,
                   "detail": "%s; %d AUC cells differ, by attack %s" % (bad[:5], n_diff,
                                                                       by_attack)})

    configs = {}
    for g in groups:
        pat, seeds = GROUPS[g]
        tt = [pat % s for s in seeds]
        rel_rows = {t: released[t]["results"] for t in tt}
        new = summarise_group(rows[NEW_FLOOR], tt)
        old = summarise_group(rel_rows, tt)
        n = released[tt[0]]["config"]["clients"]
        configs[g] = {"clients": n, "rounds": released[tt[0]]["config"]["rounds"],
                      "seeds": list(seeds), "eta_0.01": new, "released_eta_1e-3": old,
                      "tie_ceiling_(N-1.5)/(N-1)": (n - 1.5) / (n - 1)}
        print("%-24s worst AUC  eta=1e-3 %.4f (%s)   eta=0.01 %.4f (%s)   peer copy %.4f -> %.4f"
              % (g, old["worst"], old["worst_attack"], new["worst"], new["worst_attack"],
                 old["per_attack"]["peer_copy"]["mean"], new["per_attack"]["peer_copy"]["mean"]))

    import experiments.protocol_eval as P
    paths = P.discover_archives(ARCHIVES, "arch_cifar100_s")
    sweep_paths = [paths[s] for s in (0, 1, 2)]
    stasks = []
    for eta in (OLD_FLOOR, NEW_FLOOR):
        for n_att in (1, 2, 3, 4, 5):
            for i, p in enumerate(sweep_paths):
                stasks.append({"eta": eta, "sweep": "fraction", "kind": "peer_copy",
                               "strength": 1.0, "n_att": n_att, "index": i, "path": p,
                               "clients": 10, "rounds": 35})
        for kind in ("shrink", "peer_copy", "ar2_replay"):
            for s in (0.1, 0.25, 0.5, 0.75, 1.0):
                for i, p in enumerate(sweep_paths):
                    stasks.append({"eta": eta, "sweep": "strength", "kind": kind,
                                   "strength": s, "n_att": 1, "index": i, "path": p,
                                   "clients": 10, "rounds": 35})
    t0 = time.time()
    sres = run_pool(task_sweep, stasks, a.workers)
    print("sweep replays: %d tasks in %.1fs" % (len(stasks), time.time() - t0))

    def sweep_table(eta):
        out = {"fraction_sweep": {}, "strength_sweep": {}}
        ordered = sorted(sres, key=lambda r: r["index"])
        for n_att in (1, 2, 3, 4, 5):
            runs = [r["aucs"] for r in ordered if r["eta"] == eta and r["sweep"] == "fraction"
                    and r["n_att"] == n_att]
            out["fraction_sweep"][str(n_att)] = {k: float(np.nanmean([x[k] for x in runs]))
                                                 for k in runs[0]}
        for kind in ("shrink", "peer_copy", "ar2_replay"):
            out["strength_sweep"][kind] = {}
            for s in (0.1, 0.25, 0.5, 0.75, 1.0):
                runs = [r["aucs"] for r in ordered if r["eta"] == eta
                        and r["sweep"] == "strength" and r["kind"] == kind
                        and r["strength"] == s]
                out["strength_sweep"][kind][str(s)] = {k: float(np.nanmean([x[k] for x in runs]))
                                                       for k in runs[0]}
        return out

    old_sw, new_sw = sweep_table(OLD_FLOOR), sweep_table(NEW_FLOOR)
    ref_sw = json.load(open(os.path.join(RESULTS, "sweeps.json"), encoding="utf-8"))
    bad = []
    for part in ("fraction_sweep", "strength_sweep"):
        def walk(x, y, path):
            if isinstance(y, dict):
                for k in y:
                    walk(x[k], y[k], path + (k,))
            elif not (x == y or (np.isnan(x) and np.isnan(y))):
                bad.append(path)
        walk(old_sw[part], ref_sw[part], (part,))
    checks.append({"check": "eta=1e-3 rerun reproduces results/sweeps.json exactly",
                   "ok": not bad, "detail": str(bad[:5])})

    detail = {}
    for eta in (OLD_FLOOR, NEW_FLOOR):
        detail["%g" % eta] = {}
        for n_att in (1, 2, 3, 4, 5):
            per = [collusion_detail(p, n_att, eta) for p in sweep_paths]
            ref = new_sw if eta == NEW_FLOOR else old_sw
            ok = abs(float(np.mean([d["auc"] for d in per]))
                     - ref["fraction_sweep"][str(n_att)]["suspicion"]) < 1e-12
            checks.append({"check": "collusion detail AUC equals the sweep at eta=%g, n=%d"
                                    % (eta, n_att), "ok": bool(ok), "detail": ""})
            rec = {"per_archive": per,
                   "auc_mean": float(np.mean([d["auc"] for d in per])),
                   "auc_excl_duplicated_honest_mean": float(np.mean(
                       [d["auc_excl_duplicated_honest"] for d in per]))}
            if "frozen_threshold_1pct" in per[0]:
                k = sum(d["frozen_threshold_1pct"]["attacker_rounds_flagged"] for d in per)
                n = sum(d["frozen_threshold_1pct"]["attacker_rounds"] for d in per)
                hk = sum(d["frozen_threshold_1pct"]["honest_rounds_flagged"] for d in per)
                hn = sum(d["frozen_threshold_1pct"]["honest_rounds"] for d in per)
                rec["frozen_threshold_1pct_pooled"] = {
                    "attacker_detection": k / n, "attacker_rounds": n,
                    "honest_flag_rate": hk / hn, "honest_rounds": hn}
            detail["%g" % eta][str(n_att)] = rec
    print("collusion, five of ten: AUC %.4f (eta=1e-3) -> %.4f (eta=0.01); excluding the "
          "duplicated honest client %.4f -> %.4f"
          % (detail["0.001"]["5"]["auc_mean"], detail["0.01"]["5"]["auc_mean"],
             detail["0.001"]["5"]["auc_excl_duplicated_honest_mean"],
             detail["0.01"]["5"]["auc_excl_duplicated_honest_mean"]))

    for c in checks:
        print("  [%s] %s %s" % ("ok" if c["ok"] else "FAIL", c["check"], c["detail"]))
    from passportfl.reproducibility import environment
    out = {
        "config": {k: portable(v) for k, v in vars(a).items()},
        "command": " ".join(sys.argv),
        "environment": environment(),
        "floors": {"released": OLD_FLOOR, "recomputed": NEW_FLOOR},
        "configuration_controls": configs,
        "innovation_results_eta_0.01": {t: rows[NEW_FLOOR][t] for t in sorted(rows[NEW_FLOOR])},
        "innovation_results_eta_1e-3_rerun": {t: rows[OLD_FLOOR][t]
                                              for t in sorted(rows[OLD_FLOOR])},
        "rerun_deviation_from_released": deviation,
        "sweeps_eta_0.01": new_sw,
        "sweeps_eta_1e-3_rerun": old_sw,
        "collusion_detail": detail,
        "collusion_setup": {
            "source": "experiments/parameter_sweep.py fraction sweep, results/sweeps.json "
                      "fraction_sweep[n]['suspicion']",
            "floor": "released with InnovationAuditor's default eta = 1e-3",
        },
        "consistency_checks": checks,
        "all_checks_ok": all(c["ok"] for c in checks),
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\nwrote %s   (total %.1fs)" % (a.out, time.time() - t_all))
    return 0 if out["all_checks_ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
