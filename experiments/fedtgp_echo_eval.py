from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

import fedtgp_attacks as FA
import protocol_eval as P
from passportfl.baselines import BASELINES, bulyan_deviation, score_round
from passportfl.innovation import InnovationAuditor

F_BYZANTINE = 2
RANKED = ("krum", "median_dev", "trimmed_mean_dev", "bulyan_dev")
WINDOWS = {"rounds_12_20": (12, 20), "rounds_28_35": (28, 35)}
_CACHE: Dict[str, dict] = {}


def load(path: str) -> dict:
    if path not in _CACHE:
        _CACHE.clear()
        _CACHE[path] = FA.load_archive_ext(path)
    return _CACHE[path]


def names() -> List[str]:
    return P.stat_names() + ["bulyan_dev"]


def score_pass(task: dict) -> dict:
    variant, victim = task.get("variant"), task.get("victim")
    base = load(task["path"])
    clients, rounds = P.archive_shape(base)
    arch = dict(base)
    arch["published"] = {i: {} for i in range(clients)}
    fn = FA.VARIANTS[variant] if variant else None
    rng = np.random.default_rng([int(task["seed"]), int(victim) + 1 if victim is not None else 0,
                                 100 + FA.VARIANT_ORDER.index(variant) if variant else 0])
    aud = InnovationAuditor(mad_floor=task["mad_floor"], mask_policy="declared")
    nm = names()
    rec_rounds, mats = [], {k: [] for k in nm}
    for r in range(1, rounds + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        payloads = {}
        for cid in range(clients):
            C = arch["C"][cid][r]
            if fn is not None and cid == victim and r >= task["attack_start"]:
                C = fn(arch, cid, r, rng)
            arch["published"][cid][r] = C
            payloads[cid] = (C, arch["present"][cid][r])
        rs = aud.observe_round(r, payloads)
        base_s: Dict[str, Dict[int, float]] = {}
        Cbar_prev = arch["cbar"].get(r - 1)
        if Cbar_prev is not None:
            innov = {cid: (payloads[cid][0] - Cbar_prev) for cid in payloads}
            if len(innov) >= 3:
                base_s = score_round(innov)
                try:
                    base_s["bulyan_dev"] = bulyan_deviation(innov, n_byzantine=F_BYZANTINE)
                except Exception:
                    base_s["bulyan_dev"] = {c: float("nan") for c in innov}
        if r < task["scope_start"]:
            continue
        rec_rounds.append(r)
        row = {k: [float("nan")] * clients for k in nm}
        for cid, v in rs.per_client.items():
            row["suspicion"][cid] = float(v.get("suspicion", float("nan")))
        for bname, vals in base_s.items():
            for cid, v in vals.items():
                row[bname][cid] = float(v) if np.isfinite(v) else float("nan")
            if bname in BASELINES:
                zz = P.robust_z_score({c: vals.get(c, float("nan")) for c in range(clients)},
                                      task["mad_floor"])
                for cid, v in zz.items():
                    row[f"{bname}_z"][cid] = float(v)
        for k in nm:
            mats[k].append(row[k])
    return {"key": {"seed": task["seed"], "variant": variant, "victim": victim,
                    "attack": variant, "floor": task["mad_floor"], "pool": task.get("pool")},
            "rounds": rec_rounds, "mats": {k: np.asarray(v, dtype=np.float64) for k, v in mats.items()}}


def run(tasks: List[dict], workers: int) -> List[dict]:
    tasks = sorted(tasks, key=lambda t: t["path"])
    if workers <= 1 or len(tasks) <= 2:
        return [score_pass(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(score_pass, tasks, chunksize=max(1, len(tasks) // (workers * 4))))


def max_abs_diff(a: dict, b: dict, stats) -> float:
    worst = 0.0
    for s in stats:
        x, y = a["mats"][s], b["mats"][s]
        if x.shape != y.shape or not np.array_equal(np.isnan(x), np.isnan(y)):
            return float("inf")
        f = np.isfinite(x)
        if f.any():
            worst = max(worst, float(np.abs(x[f] - y[f]).max()))
    return worst


def honest_by_architecture(honest: List[dict], thresholds: dict, labels: List[str]) -> dict:
    out = {}
    for stat, thr in thresholds.items():
        out[stat] = {}
        for t in P.TARGETS:
            f = np.concatenate([P.flag_scores(p["mats"][stat], thr[f"{t:g}"]) for p in honest], axis=0)
            per_seat = [P.rate_record(int(f[:, c].sum()), int(f.shape[0])) for c in range(f.shape[1])]
            per_enc = {}
            for name in dict.fromkeys(labels):
                cols = [c for c in range(f.shape[1]) if labels[c] == name]
                per_enc[name] = P.rate_record(int(f[:, cols].sum()), int(f[:, cols].size))
            out[stat][f"{t:g}"] = {"per_encoder": per_enc,
                                   "per_seat": [r["point"] for r in per_seat]}
    return out


def in_window(p: dict, lo: int, hi: int) -> np.ndarray:
    r = np.asarray(p["rounds"])
    return (r >= lo) & (r <= hi)


def threshold_free(passes: List[dict], attack_start: int, clients: int,
                   last_round: int = 10 ** 9) -> dict:
    by_seed: Dict[int, List[dict]] = {}
    for p in passes:
        by_seed.setdefault(p["key"]["seed"], []).append(p)
    out = {"rounds": [attack_start, last_round], "per_seed": {}, "pooled": {}}
    pool_pos = {s: [] for s in names()}
    pool_neg = {s: [] for s in names()}
    pool_rank = {s: [] for s in RANKED}
    for seed, ps in sorted(by_seed.items()):
        ent = {}
        for s in names():
            pos, neg = [], []
            for p in ps:
                v = p["key"]["victim"]
                m = p["mats"][s][in_window(p, attack_start, last_round)]
                pos.append(m[:, v])
                neg.append(np.delete(m, v, axis=1).ravel())
            pos, neg = np.concatenate(pos), np.concatenate(neg)
            ent[s] = {"auc": P.roc_auc(pos, neg)}
            pool_pos[s].append(pos)
            pool_neg[s].append(neg)
        for s in RANKED:
            rk = []
            for p in ps:
                v = p["key"]["victim"]
                rk.extend(FA.attacker_rank(row, v)
                          for row in p["mats"][s][in_window(p, attack_start, last_round)])
            ent[s].update(FA.rank_summary(rk))
            pool_rank[s].extend(rk)
        out["per_seed"][str(seed)] = ent
    for s in names():
        aucs = np.asarray([out["per_seed"][k][s]["auc"] for k in out["per_seed"]], float)
        fin = aucs[np.isfinite(aucs)]
        e = {"auc_pooled": P.roc_auc(np.concatenate(pool_pos[s]), np.concatenate(pool_neg[s])),
             "auc_mean_over_seeds": float(fin.mean()) if fin.size else float("nan"),
             "auc_min": float(fin.min()) if fin.size else float("nan"),
             "auc_max": float(fin.max()) if fin.size else float("nan")}
        if s in RANKED:
            e.update(FA.rank_summary(pool_rank[s]))
        out["pooled"][s] = e
    return out


def per_round_ranks(passes: List[dict], attack_start: int) -> dict:
    by: Dict[str, Dict[int, Dict[int, list]]] = {s: {} for s in RANKED}
    for p in passes:
        v, seed = p["key"]["victim"], p["key"]["seed"]
        for i, r in enumerate(p["rounds"]):
            if r < attack_start:
                continue
            for s in RANKED:
                by[s].setdefault(int(r), {}).setdefault(seed, []).append(
                    FA.attacker_rank(p["mats"][s][i], v))
    out = {}
    for s, byr in by.items():
        rounds = sorted(byr)
        seeds = sorted({sd for r in rounds for sd in byr[r]})
        allr = [FA.rank_summary([x for sd in byr[r] for x in byr[r][sd]]) for r in rounds]
        out[s] = {"rounds": rounds,
                  "rank_mean": [e["rank_mean"] for e in allr],
                  "rank1_frac": [e["rank1_frac"] for e in allr],
                  "rank_mean_strict": [e["rank_mean_strict"] for e in allr],
                  "per_seed_rank_mean": {str(sd): [FA.rank_summary(byr[r].get(sd, []))["rank_mean"]
                                                   for r in rounds] for sd in seeds}}
    return out


def exact_vs_echo_rank(geometry: Dict[str, dict], echo_passes: List[dict]) -> dict:
    worst, cells = 0.0, 0
    for seed, g in geometry.items():
        ex = {x["round"]: x for x in g.get("exact_per_round", [])}
        ps = [p for p in echo_passes if str(p["key"]["seed"]) == seed]
        if not ex or not ps:
            continue
        for rule in FA.EXACT_RULES:
            for r, x in ex.items():
                hits = []
                for p in ps:
                    i = p["rounds"].index(r) if r in p["rounds"] else None
                    if i is None:
                        continue
                    rk = FA.attacker_rank(p["mats"][rule][i], p["key"]["victim"])
                    hits.append(rk is not None and rk["rank"] == 1)
                if hits:
                    worst = max(worst, abs(float(np.mean(hits)) - x[f"exact_{rule}_inverts_frac_seats"]))
                    cells += 1
    return {"max_abs": worst, "cells": cells}


def archive_geometry(path: str, scope_start: int) -> dict:
    arch = load(path)
    _, rounds = P.archive_shape(arch)
    g = FA.innovation_geometry(arch, arch["cbar"], arch["qbar"], rounds, scope_start)
    tl = arch.get("tgp_log")
    if tl:
        for row in g["per_round"]:
            prev = tl.get(row["round"] - 1, {})
            row["tgp_p_minus_q_fro_received"] = prev.get("p_minus_q_fro")
            row["tgp_margin_received"] = prev.get("margin")
        g["tgp_per_round"] = {"rounds": sorted(tl),
                              "p_minus_q_fro": [tl[r].get("p_minus_q_fro") for r in sorted(tl)],
                              "margin": [tl[r].get("margin") for r in sorted(tl)]}
    g["qbar_source"] = arch["qbar_source"]
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", default=os.path.join(HERE, "results", "archives_rev2"))
    ap.add_argument("--archive-prefix", default="arch_fedtgp_lam1_s",
                    help="one configuration: arch_hetero_s or arch_fedtgp_<lambda arm>_s")
    ap.add_argument("--cal", default="0,1")
    ap.add_argument("--dev", default="2")
    ap.add_argument("--test", default="3,4")
    ap.add_argument("--variants", default=",".join(FA.VARIANT_ORDER))
    ap.add_argument("--attack-start", type=int, default=12)
    ap.add_argument("--eval-from", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arch-group", default="hetero4",
                    help="encoder group of the archives (passportfl/encoders.py), used only to "
                         "label seats in the per-architecture false-alarm table")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "fedtgp_echo_eval.json"))
    a = ap.parse_args()
    t0 = time.time()
    ints = lambda s: [int(x) for x in s.split(",") if x.strip()]
    cal, dev, test = ints(a.cal), ints(a.dev), ints(a.test)
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in FA.VARIANTS]
    if bad:
        raise SystemExit("unknown variant(s) %s" % bad)
    workers = max(1, min(4, a.workers))
    paths = P.discover_archives(a.archives, a.archive_prefix)
    seeds = sorted(set(cal + dev + test) & set(paths))
    if not [s for s in cal if s in paths]:
        raise SystemExit("no CAL archive with prefix %s in %s" % (a.archive_prefix, a.archives))
    cal = [s for s in cal if s in paths]
    clients, rounds = P.archive_shape(load(paths[cal[0]]))
    scope = max(a.eval_from, a.attack_start)
    common = {"scope_start": scope, "attack_start": a.attack_start}
    arch_meta = {str(s): load(paths[s]).get("meta") for s in seeds}
    confs = {FA.config_key(m) for m in arch_meta.values() if m}
    if len(confs) > 1:
        raise SystemExit("archives of different configurations under prefix %s: %s"
                         % (a.archive_prefix, sorted(confs)))
    groups = {m.get("arch_group") for m in arch_meta.values() if m}
    if groups and groups != {a.arch_group}:
        raise SystemExit("--arch-group %s but the archives were trained with %s"
                         % (a.arch_group, sorted(groups)))

    def task(seed, floor, variant=None, victim=None, pool=None):
        return dict(common, path=paths[seed], seed=seed, mad_floor=floor, variant=variant,
                    victim=victim, pool=pool)

    s0 = cal[0]
    mine_h, mine_e = score_pass(task(s0, 1e-3)), score_pass(task(s0, 1e-3, "fedtgp_echo", 0))
    ref = {"path": paths[s0], "seed": s0, "mad_floor": 1e-3, "scope_start": scope,
           "attack_start": a.attack_start, "key": {}}
    ref_h = P.score_one_archive(dict(ref, attack=None, victim=None))
    ref_e = P.score_one_archive(dict(ref, attack="global_echo", victim=0))
    checks = {"honest_vs_protocol_eval_max_abs": max_abs_diff(mine_h, ref_h, P.stat_names()),
              "echo_vs_protocol_eval_global_echo_max_abs": max_abs_diff(mine_e, ref_e, P.stat_names())}
    print("checks: %s" % checks, flush=True)

    gres = run([task(s, f, pool="CAL") for f in P.MAD_FLOOR_GRID for s in cal], workers)
    cal_honest = {f: sorted([r for r in gres if r["key"]["floor"] == f], key=lambda r: r["key"]["seed"])
                  for f in P.MAD_FLOOR_GRID}
    sel = P.select_mad_floor(cal_honest, P.TARGETS)
    floor = sel["selected"]
    stats = ["suspicion"] + list(BASELINES) + [f"{b}_z" for b in BASELINES] + ["bulyan_dev"]
    thresholds = P.fit_thresholds({s: P.stack_scores(cal_honest[floor], s) for s in stats}, P.TARGETS)
    print("MAD floor %.0e (%s); tau suspicion @1%% = %.4g"
          % (floor, sel["criterion"], thresholds["suspicion"]["0.01"]), flush=True)

    tasks = []
    for s in seeds:
        pool = "CAL" if s in cal else ("DEV" if s in dev else "TEST")
        if pool != "CAL":
            tasks.append(task(s, floor, pool=pool))
        for v in variants:
            for victim in range(clients):
                tasks.append(task(s, floor, v, victim, pool))
    res = run(tasks, workers)

    out = {"meta": {"command": "python experiments/fedtgp_echo_eval.py " + " ".join(sys.argv[1:]),
                    "archives": {str(s): os.path.basename(paths[s]) for s in seeds},
                    "pools": {"CAL": cal, "DEV": [s for s in dev if s in paths],
                              "TEST": [s for s in test if s in paths]},
                    "clients": clients, "rounds": rounds, "attack_start": a.attack_start,
                    "scope_start": scope, "offset_rounds": list(FA.OFFSET_ROUNDS),
                    "archive_meta": arch_meta,
                    "checks": checks,
                    "windows": {k: list(w) for k, w in WINDOWS.items()}},
           "mad_floor": {"selected": floor, "criterion": sel["criterion"], "grid": sel["grid"]},
           "thresholds": {s: thresholds[s] for s in stats},
           "geometry": {str(s): archive_geometry(paths[s], scope) for s in seeds},
           "threshold_free": {}, "thresholded": {}}
    for v in variants:
        ps = [r for r in res if r["key"]["variant"] == v]
        out["threshold_free"][v] = {"all_seeds": threshold_free(ps, a.attack_start, clients)}
        for pool in ("CAL", "DEV", "TEST"):
            pp = [r for r in ps if r["key"]["pool"] == pool]
            if pp:
                out["threshold_free"][v][pool] = threshold_free(pp, a.attack_start, clients)["pooled"]
        out["threshold_free"][v]["windows"] = {
            k: threshold_free(ps, max(lo, a.attack_start), clients, hi)
            for k, (lo, hi) in WINDOWS.items() if lo <= rounds}
        out["threshold_free"][v]["per_round"] = per_round_ranks(ps, a.attack_start)
        pr = out["threshold_free"][v]["all_seeds"]["pooled"]
        print("%-20s " % v + "  ".join("%s AUC %.3f rank %.2f r1 %.2f" % (k, pr[k]["auc_pooled"], pr[k]["rank_mean"],
                                                                           pr[k]["rank1_frac"]) for k in RANKED)
              + "  suspicion AUC %.3f" % pr["suspicion"]["auc_pooled"], flush=True)
        for k in out["threshold_free"][v]["windows"]:
            w = out["threshold_free"][v]["windows"][k]["pooled"]
            print("  %-18s " % k + "  ".join("%s AUC %.3f rank %.2f" % (s, w[s]["auc_pooled"], w[s]["rank_mean"])
                                             for s in ("krum", "median_dev"))
                  + "  suspicion AUC %.3f" % w["suspicion"]["auc_pooled"], flush=True)
    g = out["geometry"]
    print("honest-cohort approximation ||c|| < min_j ||I_j - c|| (median centre), fraction of rounds "
          "per seed: " + ", ".join("%s %.2f" % (s, g[s]["median_over_rounds"]["median_inverts_all_frac"])
                                   for s in g), flush=True)
    for rule in FA.EXACT_RULES:
        print("exact condition (%s, centre on the attacked cohort), fraction of (round, seat) pairs "
              "per seed: " % rule + ", ".join("%s %.2f" % (s, g[s]["exact_summary"][rule]["inverts_frac_round_seats"])
                                               for s in g), flush=True)
    if "fedtgp_echo" in variants:
        chk = exact_vs_echo_rank(g, [r for r in res if r["key"]["variant"] == "fedtgp_echo"])
        out["meta"]["checks"]["exact_condition_vs_echo_rank1_max_abs"] = chk["max_abs"]
        out["meta"]["checks"]["exact_condition_vs_echo_rank1_cells"] = chk["cells"]
        print("check: exact condition vs echo rank 1, max |diff| %.3g over %d cells"
              % (chk["max_abs"], chk["cells"]), flush=True)
    from passportfl.encoders import group_assignment
    labels = group_assignment(a.arch_group, clients)
    frozen = {s: thresholds[s] for s in ("suspicion", "krum_z", "median_dev_z", "trimmed_mean_dev_z")}
    out["honest_by_architecture"] = {"encoders": labels}
    for pool in ("CAL", "DEV", "TEST"):
        honest = (cal_honest[floor] if pool == "CAL" else
                  [r for r in res if r["key"]["pool"] == pool and r["key"]["variant"] is None])
        if honest:
            out["honest_by_architecture"][pool] = honest_by_architecture(honest, frozen, labels)
    for pool in ("DEV", "TEST"):
        honest = [r for r in res if r["key"]["pool"] == pool and r["key"]["variant"] is None]
        atk = {v: [r for r in res if r["key"]["pool"] == pool and r["key"]["variant"] == v]
               for v in variants}
        atk = {k: x for k, x in atk.items() if x}
        if honest and atk:
            out["thresholded"][pool] = P.evaluate_pool(
                honest, atk, {s: thresholds[s] for s in ("suspicion", "krum_z", "median_dev_z",
                                                         "trimmed_mean_dev_z")},
                P.TARGETS, clients, a.attack_start)
            e = out["thresholded"][pool]["suspicion"]["0.01"]
            print("%s auditor @1%%: honest FPR %s; " % (pool, e["honest"]["client_round"]["reported"])
                  + "; ".join("%s TPR %s" % (v, e["attacks"][v]["tpr_client_round"]["reported"])
                              for v in atk), flush=True)
    from passportfl.reproducibility import environment
    out["meta"]["environment"] = environment()
    out["meta"]["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("wrote %s (%.0fs)" % (a.out, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
