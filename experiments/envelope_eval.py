from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import pickle
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from passportfl.baselines import BASELINES, bulyan_deviation, score_round
from passportfl.innovation import (InnovationAuditor, SCORE_STRUCTURAL,
                                   SCORE_STAT_MAX)
from passportfl.reference_baselines import FoolsGold
from passportfl.temporal_baselines import TemporalBaselineAuditor
import experiments.protocol_eval as P
import experiments.temporal_eval as T
from published_predicates_eval import make_drift_attack

RESULTS = os.path.join(HERE, "results")
TEMPORAL_REF = os.path.join(RESULTS, "temporal_eval.json")
TEMPORAL_REF_V2 = os.path.join(RESULTS, "temporal_eval_v2.json")

ECHO_REL = (1e-5, 1e-3, 0.1, 0.5, 1.0)
LAGS_BEYOND_W = (6, 8, 10)
NOISE_LAGS = (3, 6)
NOISE_KAPPA = (0.25, 0.5, 1.0, 2.0)
ROT_LAG = 3
ROT_THETA = (0.05, 0.2, 0.5, 1.0)
MOSAIC_M = (3, 9)
SENSITIVITY_LAGS = (12, 16)
F_BULYAN = 2
BEST_SINGLE_STATEFUL = "tb_maha_pca_burnin_zabs"
HEADLINE_TARGET = 0.01


def pre_rounds(attack_start: int) -> List[int]:
    return list(range(2, attack_start))


def own_innovation_norm(arch: dict, cid: int, pre: Sequence[int]) -> float:
    t = arch.setdefault("_own_innov_norm", {})
    if cid not in t:
        t[cid] = float(np.median([np.linalg.norm(arch["C"][cid][k] - arch["cbar"][k - 1])
                                  for k in pre]))
    return t[cid]


def own_payload_step(arch: dict, cid: int, pre: Sequence[int]) -> float:
    t = arch.setdefault("_own_step", {})
    if cid not in t:
        t[cid] = float(np.median([np.linalg.norm(arch["C"][cid][k] - arch["C"][cid][k - 1])
                                  for k in pre]))
    return t[cid]


def unit_direction(rng: np.random.Generator, shape, rows: np.ndarray) -> np.ndarray:
    U = rng.normal(size=shape) * np.asarray(rows, bool)[:, None]
    return U / np.linalg.norm(U)


def cayley_rotation(rng: np.random.Generator, d: int, theta: float) -> np.ndarray:
    A = rng.normal(size=(d, d))
    A = 0.5 * (A - A.T)
    A /= np.linalg.norm(A, 2)
    eye = np.eye(d)
    return np.linalg.solve(eye - 0.5 * theta * A, eye + 0.5 * theta * A)


def clean_replay_source(arch: dict, cid: int, r: int, lag: int, attack_start: int) -> np.ndarray:
    clean = arch.setdefault("_clean", {}).setdefault((cid, lag), {})
    if not clean:
        for k in range(1, attack_start):
            clean[k] = arch["C"][cid][k]
    src = clean[max(1, r - lag)]
    clean[r] = src
    return src


def record_sources(arch: dict, rows_by_peer: Dict[int, int]) -> None:
    s = arch.setdefault("_sources", {})
    for j, n in rows_by_peer.items():
        if n > 0:
            s[int(j)] = s.get(int(j), 0) + int(n)


def peers_of(arch: dict, cid: int, m: int) -> List[int]:
    others = [j for j in arch["C"] if j != cid]
    return [others[(cid + i) % len(others)] for i in range(m)]


def declared_rows(arch: dict, cid: int, r: int) -> np.ndarray:
    return np.asarray(arch["present"][cid][r], bool)


def make_echo_eps(rel: float, pre: Sequence[int]):
    def attack_echo_eps(arch, cid, r, rng):
        base = arch["cbar"][r - 1]
        U = unit_direction(rng, base.shape, declared_rows(arch, cid, r))
        return base + rel * own_innovation_norm(arch, cid, pre) * U
    return attack_echo_eps


def make_frozen_innovation(attack_start: int):
    def attack_frozen_innovation(arch, cid, r, rng):
        I = arch["C"][cid][attack_start - 1] - arch["cbar"][attack_start - 2]
        return arch["cbar"][r - 1] + I
    return attack_frozen_innovation


def make_lagged(lag: int):
    def attack_lagged(arch, cid, r, rng):
        pub = arch["published"][cid]
        return pub[max(min(pub), r - lag)].copy()
    return attack_lagged


def make_lag_noise(lag: int, kappa: float, pre: Sequence[int], attack_start: int):
    def attack_lag_noise(arch, cid, r, rng):
        src = clean_replay_source(arch, cid, r, lag, attack_start)
        U = unit_direction(rng, src.shape, np.ones(src.shape[0], bool))
        return src + kappa * own_payload_step(arch, cid, pre) * U
    return attack_lag_noise


def make_lag_rotation(lag: int, theta: float, attack_start: int):
    def attack_lag_rotation(arch, cid, r, rng):
        src = clean_replay_source(arch, cid, r, lag, attack_start)
        return src @ cayley_rotation(rng, src.shape[1], theta)
    return attack_lag_rotation


def make_peer_average(m: int):
    def attack_peer_average(arch, cid, r, rng):
        rows = declared_rows(arch, cid, r)
        peers = peers_of(arch, cid, m)
        C = arch["cbar"][r - 1].copy()
        C[rows] = np.mean([arch["C"][j][r][rows] for j in peers], axis=0)
        record_sources(arch, {j: int((rows & np.asarray(arch["present"][j][r], bool)).sum())
                              for j in peers})
        return C
    return attack_peer_average


def half_shared_rows(arch: dict, cid: int, j: int, r: int) -> np.ndarray:
    own = np.where(declared_rows(arch, cid, r))[0]
    shared = own[np.asarray(arch["present"][j][r], bool)[own]]
    return shared[:int(round(0.5 * len(own)))]


def make_rows_half(fill: str):
    def attack_rows_half(arch, cid, r, rng):
        rows = declared_rows(arch, cid, r)
        j = peers_of(arch, cid, 1)[0]
        C = arch["cbar"][r - 1].copy()
        if fill == "lag3":
            pub = arch["published"][cid]
            C[rows] = pub[max(min(pub), r - 3)][rows]
        S = half_shared_rows(arch, cid, j, r)
        C[S] = arch["C"][j][r][S]
        record_sources(arch, {j: len(S)})
        return C
    return attack_rows_half


def make_mosaic(m: int):
    def attack_mosaic(arch, cid, r, rng):
        rows = np.where(declared_rows(arch, cid, r))[0]
        peers = peers_of(arch, cid, m)
        C = arch["cbar"][r - 1].copy()
        used: Dict[int, int] = {}
        for k in rows:
            cands = [j for j in peers if arch["present"][j][r][k]]
            if not cands:
                continue
            j = cands[k % len(cands)]
            C[k] = arch["C"][j][r][k]
            used[j] = used.get(j, 0) + 1
        record_sources(arch, used)
        return C
    return attack_mosaic


def make_reference(name: str):
    fn = P.ATTACKS[name]
    if name != "peer_copy":
        return fn

    def attack_peer_copy_tracked(arch, cid, r, rng):
        acc = P.accomplice("peer_copy", cid, len(arch["C"]))
        record_sources(arch, {acc: int(np.asarray(arch["present"][acc][r], bool).sum())})
        return fn(arch, cid, r, rng)
    return attack_peer_copy_tracked


def _g(x: float) -> str:
    return f"{x:g}"


def variant_table() -> Dict[str, dict]:
    v: Dict[str, dict] = {}
    for a in P.ATTACK_ORDER:
        v[a] = {"family": "reference", "params": {}}
    for rel in ECHO_REL:
        v[f"echo_eps_{_g(rel)}"] = {"family": "a", "params": {"rel": rel}}
    v["agg_frozen_innov"] = {"family": "a", "params": {}}
    v["drift_pos"] = {"family": "a", "params": {"sign": 1.0}}
    v["drift_neg"] = {"family": "a", "params": {"sign": -1.0}}
    for L in LAGS_BEYOND_W:
        v[f"lagged_{L}"] = {"family": "b", "params": {"lag": L}}
    for L in NOISE_LAGS:
        for k in NOISE_KAPPA:
            v[f"lag{L}_noise_{_g(k)}"] = {"family": "c", "params": {"lag": L, "kappa": k}}
    for t in ROT_THETA:
        v[f"lag{ROT_LAG}_rot_{_g(t)}"] = {"family": "c", "params": {"lag": ROT_LAG, "theta": t}}
    v["peer_avg_2"] = {"family": "d", "params": {"m": 2}}
    v["rows_half_echo"] = {"family": "d", "params": {"fill": "echo"}}
    v["rows_half_lag3"] = {"family": "d", "params": {"fill": "lag3"}}
    for m in MOSAIC_M:
        v[f"mosaic_{m}"] = {"family": "d", "params": {"m": m}}
    for L in SENSITIVITY_LAGS:
        v[f"lagged_{L}"] = {"family": "b", "params": {"lag": L}, "default": False}
    return v


VARIANTS = variant_table()
VARIANT_ORDER = list(VARIANTS)
DEFAULT_VARIANTS = [k for k in VARIANT_ORDER if VARIANTS[k].get("default", True)]


def build_attack(name: str, attack_start: int):
    spec = VARIANTS[name]
    fam, p = spec["family"], spec["params"]
    pre = pre_rounds(attack_start)
    if fam == "reference":
        return make_reference(name)
    if name.startswith("echo_eps_"):
        return make_echo_eps(p["rel"], pre)
    if name == "agg_frozen_innov":
        return make_frozen_innovation(attack_start)
    if name.startswith("drift_"):
        return make_drift_attack(pre, p["sign"])
    if name.startswith("lagged_"):
        return make_lagged(p["lag"])
    if "_noise_" in name:
        return make_lag_noise(p["lag"], p["kappa"], pre, attack_start)
    if "_rot_" in name:
        return make_lag_rotation(p["lag"], p["theta"], attack_start)
    if name.startswith("peer_avg_"):
        return make_peer_average(p["m"])
    if name.startswith("rows_half_"):
        return make_rows_half(p["fill"])
    if name.startswith("mosaic_"):
        return make_mosaic(p["m"])
    raise KeyError(name)


def rng_tag(name: Optional[str]) -> int:
    if name is None:
        return 0
    if name in P.ATTACK_ORDER:
        return P.ATTACK_ORDER.index(name) + 1
    return 1000 + zlib.crc32(name.encode()) % 1_000_000


RAW_FEATURES = ("u1", "u2", "g", "x", "rep")
EXTRA = ["bulyan", "bulyan_z"] + [f"raw_{k}" for k in RAW_FEATURES]


def peer_columns() -> List[str]:
    return T.peer_raw_names() + list(T.PEER_ENSEMBLES)


def kept_columns(auditor_only: bool = False) -> List[str]:
    cols = (["suspicion"] + T.ABL_NAMES + list(BASELINES) + [f"{b}_z" for b in BASELINES]
            + EXTRA)
    if auditor_only:
        return cols
    return cols + [BEST_SINGLE_STATEFUL] + list(T.ENSEMBLES) + peer_columns()


ROBUST = ["krum_z", "median_dev_z", "trimmed_mean_dev_z", "bulyan_z", "norm_bound_z"]
AUDITOR_ROWS = ["suspicion", "abl_full_nostruct", "abl_no_u1", "abl_no_u2", "abl_no_g",
                "abl_no_x", "abl_no_rep", "abl_x_rep", "abl_x_rep_nostruct", "abl_u1g",
                "abl_ownhist", "abl_crosshist", "abl_memoryless", "abl_struct_only",
                "abl_u1", "abl_u2", "abl_g", "abl_x", "abl_rep"]
EVAL_COLS = (AUDITOR_ROWS + ROBUST + ["cosine_cohort_z", "foolsgold_maxcos_z",
                                      BEST_SINGLE_STATEFUL] + list(T.ENSEMBLES)
             + [f"{k}_z" for k in T.FG_VARIANTS] + list(T.PEER_ENSEMBLES))
ALL_TARGET_COLS = ("suspicion", "abl_full_nostruct", "ens_stateful", "ens_stateless", "ens_all")


def eval_columns(auditor_only: bool = False) -> List[str]:
    keep = set(kept_columns(auditor_only))
    return [c for c in EVAL_COLS if c in keep]


def replay_pass(task: dict, keep_all: bool = False) -> dict:
    path = task["path"]
    variant = task.get("attack")
    victim = task.get("victim")
    mad_floor = task["mad_floor"]
    scope_start = task["scope_start"]
    attack_start = task["attack_start"]
    seed = task["seed"]

    base_arch = P.load_archive(path)
    clients, rounds = P.archive_shape(base_arch)
    arch = dict(base_arch)
    arch["published"] = {i: {} for i in range(clients)}

    fn = build_attack(variant, attack_start) if variant else None
    rng = np.random.default_rng([int(seed), int(victim) + 1 if victim is not None else 0,
                                 rng_tag(variant)])

    auditor_only = bool(task.get("auditor_only", False))
    aud = InnovationAuditor(lag_window=int(task.get("lag_window", 5)), mad_floor=mad_floor,
                            mask_policy=task.get("mask_policy", "declared"))
    tb = None if auditor_only else TemporalBaselineAuditor(burnin_end=attack_start, seed=int(seed))
    fg = ({} if auditor_only else
          {k: FoolsGold(importance=imp) for k, imp in T.FG_VARIANTS.items()})

    names = (kept_columns(True) if auditor_only
             else T.all_stat_names() + EXTRA + T.peer_raw_names())
    tb_names = [] if auditor_only else T.temporal_raw_names()
    rec_rounds: List[int] = []
    mats = {k: [] for k in names}

    for r in range(1, rounds + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
            if tb is not None:
                tb.set_aggregate(r - 1, arch["cbar"][r - 1])
        payloads = {}
        for cid in range(clients):
            C = arch["C"][cid][r]
            if fn is not None and cid == victim and r >= attack_start:
                C = fn(arch, cid, r, rng)
            arch["published"][cid][r] = C
            payloads[cid] = (C, arch["present"][cid][r])

        rs = aud.observe_round(r, payloads)
        tbs = tb.observe_round(r, payloads) if tb is not None else {}

        base: Dict[str, Dict[int, float]] = {}
        bul: Dict[int, float] = {}
        fgs: Dict[str, Dict[int, float]] = {}
        Cbar_prev = arch["cbar"].get(r - 1)
        if Cbar_prev is not None:
            innov = {cid: (payloads[cid][0] - Cbar_prev) for cid in payloads}
            if len(innov) >= 3:
                base = score_round(innov)
                if r >= scope_start:
                    bul = bulyan_deviation(innov, n_byzantine=F_BULYAN)
            for k, f in fg.items():
                fgs[k] = f.update(innov)

        if r < scope_start:
            continue
        rec_rounds.append(r)
        row = {k: [float("nan")] * clients for k in names}
        for cid, v in rs.per_client.items():
            row["suspicion"][cid] = float(v.get("suspicion", float("nan")))
            for nm in T.ABL_NAMES:
                keys, st = T.ABLATIONS[nm]
                row[nm][cid] = T.combine_scores(v, keys, st)
            for k in RAW_FEATURES:
                row[f"raw_{k}"][cid] = float(v.get(k, float("nan")))
        for bname, vals in base.items():
            for cid, v in vals.items():
                row[bname][cid] = float(v) if np.isfinite(v) else float("nan")
            zz = P.robust_z_score({c: vals.get(c, float("nan")) for c in range(clients)}, mad_floor)
            for cid, v in zz.items():
                row[f"{bname}_z"][cid] = float(v)
        if bul:
            for cid, v in bul.items():
                row["bulyan"][cid] = float(v) if np.isfinite(v) else float("nan")
            zz = P.robust_z_score({c: bul.get(c, float("nan")) for c in range(clients)}, mad_floor)
            for cid, v in zz.items():
                row["bulyan_z"][cid] = float(v)
        for nm in tb_names:
            vals = {cid: tbs[cid].get(nm, float("nan")) for cid in range(clients)}
            for cid, v in vals.items():
                row[nm][cid] = float(v)
            zz = P.robust_z_score(vals, mad_floor)
            for cid, v in zz.items():
                row[f"{nm}_z"][cid] = float(v)
        for k, vals in fgs.items():
            for cid, v in vals.items():
                row[k][cid] = float(v) if np.isfinite(v) else float("nan")
            zz = P.robust_z_score({c: vals.get(c, float("nan")) for c in range(clients)}, mad_floor)
            for cid, v in zz.items():
                row[f"{k}_z"][cid] = float(v)
        for k in names:
            mats[k].append(row[k])

    out = {"key": task["key"], "rounds": rec_rounds,
           "mats": {k: np.asarray(v, dtype=np.float64) for k, v in mats.items()},
           "sources": {int(j): int(n) for j, n in sorted(arch.get("_sources", {}).items())
                       if victim is None or int(j) != int(victim)}}
    if not auditor_only:
        T.add_absolute_z([out])
        T.add_ensembles([out])
        T.add_peer_ensembles([out])
    if not keep_all:
        out["mats"] = {k: out["mats"][k] for k in kept_columns(auditor_only)}
    return out


def run_tasks(tasks: List[dict], workers: int, log_every: int = 100) -> List[dict]:
    if not tasks:
        return []
    tasks = sorted(tasks, key=lambda t: (t["path"], str(t["attack"]), str(t["victim"])))
    t0 = time.time()
    out = []
    if workers <= 1 or len(tasks) <= 2:
        it = map(replay_pass, tasks)
        ex = None
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        it = ex.map(replay_pass, tasks, chunksize=max(1, len(tasks) // (workers * 8)))
    try:
        for i, r in enumerate(it, 1):
            out.append(r)
            if i % log_every == 0 or i == len(tasks):
                el = time.time() - t0
                print("  replay %5d/%d  %.0fs elapsed, ~%.0fs left"
                      % (i, len(tasks), el, el / i * (len(tasks) - i)), flush=True)
    finally:
        if ex is not None:
            ex.shutdown()
    return out


def check_loop_identity(path: str, seed: int, mad_floor: float, attack_start: int,
                        auditor_only: bool = False) -> dict:
    cases = [(None, None), ("noise", 3), ("peer_copy", 0), ("lagged_self", 7)]
    rows = []
    ok = True
    for a, v in cases:
        task = {"key": {"pool": "CHECK", "seed": seed, "attack": a, "victim": v},
                "path": path, "attack": a, "victim": v, "seed": seed, "mad_floor": mad_floor,
                "scope_start": attack_start, "attack_start": attack_start}
        if auditor_only:
            ref = P.score_one_archive(dict(task))
        else:
            ref = T.score_one_archive_temporal(dict(task, peer_aware=True))
            T.add_absolute_z([ref])
            T.add_ensembles([ref])
            T.add_peer_ensembles([ref])
        mine = replay_pass(dict(task, auditor_only=auditor_only), keep_all=True)
        bad = [k for k in ref["mats"] if not np.array_equal(ref["mats"][k], mine["mats"][k],
                                                            equal_nan=True)]
        ok = ok and not bad and ref["rounds"] == mine["rounds"]
        rows.append({"attack": a, "victim": v, "columns": len(ref["mats"]), "differing": bad})
    return {"identical": ok, "cases": rows}


def check_attack_constructions(path: str, seed: int, attack_start: int) -> dict:
    arch0 = P.load_archive(path)
    clients, rounds = P.archive_shape(arch0)
    pre = pre_rounds(attack_start)
    out: Dict[str, object] = {}

    def run(fn, victim, seed_tag=7):
        arch = dict(arch0)
        arch["published"] = {i: {} for i in range(clients)}
        rng = np.random.default_rng([seed, victim + 1, seed_tag])
        pays = {}
        for r in range(1, rounds + 1):
            for cid in range(clients):
                C = arch0["C"][cid][r]
                if cid == victim and r >= attack_start:
                    C = fn(arch, cid, r, rng)
                arch["published"][cid][r] = C
            pays[r] = arch["published"][victim][r]
        return arch, pays

    _, ref = run(P.ATTACKS["lagged_self"], 2)
    _, l3 = run(make_lagged(3), 2)
    _, n0 = run(make_lag_noise(3, 0.0, pre, attack_start), 2)
    out["lagged3_equals_lagged_self"] = all(np.array_equal(ref[r], l3[r]) for r in ref)
    out["lag3_noise0_equals_lagged_self"] = all(np.array_equal(ref[r], n0[r]) for r in ref)
    errs = []
    for rel in ECHO_REL:
        arch, pays = run(make_echo_eps(rel, pre), 4)
        tgt = rel * own_innovation_norm(arch, 4, pre)
        errs += [abs(np.linalg.norm(pays[r] - arch0["cbar"][r - 1]) - tgt) / tgt
                 for r in range(attack_start, rounds + 1)]
    out["echo_eps_norm_max_rel_error"] = float(max(errs))
    arch, pays = run(make_lag_noise(6, 1.0, pre, attack_start), 5)
    clean = arch["_clean"][(5, 6)]
    stp = own_payload_step(arch, 5, pre)
    out["lag_noise_norm_max_rel_error"] = float(max(
        abs(np.linalg.norm(pays[r] - clean[r]) - stp) / stp for r in range(attack_start, rounds + 1)))
    g = np.random.default_rng(0)
    R1, R2 = cayley_rotation(g, 128, 1.0), cayley_rotation(g, 128, 1.0)
    out["cayley_orthogonality_max_error"] = float(np.abs(R1.T @ R1 - np.eye(128)).max())
    out["cayley_fresh_differs"] = bool(np.abs(R1 - R2).max() > 1e-3)
    arch, pays = run(make_mosaic(9), 1)
    r = attack_start + 3
    rows = declared_rows(arch0, 1, r)
    match = 0
    for k in np.where(rows)[0]:
        match += int(any(np.array_equal(pays[r][k], arch0["C"][j][r][k])
                         for j in range(clients) if j != 1 and arch0["present"][j][r][k]))
    out["mosaic9_rows_copied"] = [match, int(rows.sum())]
    out["mosaic9_sources"] = sorted(arch["_sources"])
    out["mosaic9_undeclared_carry"] = bool(np.array_equal(pays[r][~rows], arch0["cbar"][r - 1][~rows]))
    arch, pays = run(make_rows_half("echo"), 0)
    out["rows_half_echo_rows_per_round"] = {str(j): n // (rounds - attack_start + 1)
                                            for j, n in arch["_sources"].items()}
    return out


def evaluate_variant(passes: List[dict], stat: str, thr: float, attack_start: int) -> dict:
    c = {k: [0, 0] for k in ("tpr", "run", "onset", "run_ge2", "byst", "uninv", "src", "top1")}
    per_seed: Dict[str, List[int]] = {}
    lat: List[int] = []
    for p in passes:
        v = p["key"]["victim"]
        rounds = np.asarray(p["rounds"])
        sel = rounds >= attack_start
        mat = p["mats"][stat][sel]
        rr = rounds[sel]
        f = P.flag_scores(mat, thr)
        fv = f[:, v]
        c["tpr"][0] += int(fv.sum())
        c["tpr"][1] += int(fv.size)
        ps = per_seed.setdefault(str(p["key"]["seed"]), [0, 0])
        ps[0] += int(fv.sum())
        ps[1] += int(fv.size)
        c["run"][0] += int(fv.any())
        c["run"][1] += 1
        c["run_ge2"][0] += int(fv.sum() >= 2)
        c["run_ge2"][1] += 1
        on = rr == attack_start
        if on.any():
            c["onset"][0] += int(fv[on].any())
            c["onset"][1] += 1
        if fv.any():
            lat.append(int(rr[np.argmax(fv)] - attack_start))
        srcs = [int(j) for j in p.get("sources", {}) if int(j) != v]
        others = np.ones(f.shape[1], bool)
        others[v] = False
        uninv = others.copy()
        uninv[srcs] = False
        c["byst"][0] += int(f[:, others].sum())
        c["byst"][1] += int(f[:, others].size)
        c["uninv"][0] += int(f[:, uninv].sum())
        c["uninv"][1] += int(f[:, uninv].size)
        if srcs:
            c["src"][0] += int(f[:, srcs].sum())
            c["src"][1] += int(f[:, srcs].size)
        for i in range(f.shape[0]):
            if f[i].any():
                c["top1"][1] += 1
                vals = np.where(np.isfinite(mat[i]) & f[i], mat[i], -np.inf)
                c["top1"][0] += int(int(np.argmax(vals)) == v)
    out: Dict[str, object] = dict(c)
    out["per_seed_tpr"] = per_seed
    out["latency_median"] = float(np.median(lat)) if lat else None
    out["latency_max"] = int(max(lat)) if lat else None
    return out


def rate(kn) -> Optional[float]:
    return (kn[0] / kn[1]) if kn and kn[1] else None


def auc_by_pool(res: List[dict], pools: Dict[str, List[int]], stats: Sequence[str],
                attack_start: int) -> dict:
    by = {}
    for p in res:
        if p["key"]["attack"] is None:
            continue
        by.setdefault((p["key"]["attack"], p["key"]["seed"]), []).append(p)
    out: Dict[str, dict] = {}
    for a in VARIANT_ORDER:
        if not any(k[0] == a for k in by):
            continue
        rec = {}
        for stat in stats:
            per = {"all": {}, "excl_sources": {}}
            for (aa, s), ps in by.items():
                if aa != a:
                    continue
                pos, neg, neg_ex = [], [], []
                for p in ps:
                    m = p["mats"][stat][np.asarray(p["rounds"]) >= attack_start]
                    v = p["key"]["victim"]
                    srcs = [int(j) for j in p.get("sources", {}) if int(j) != v]
                    pos.append(m[:, v])
                    neg.append(np.delete(m, [v], axis=1).ravel())
                    neg_ex.append(np.delete(m, [v] + srcs, axis=1).ravel())
                pos = np.concatenate(pos)
                per["all"][str(s)] = P.roc_auc(pos, np.concatenate(neg))
                per["excl_sources"][str(s)] = P.roc_auc(pos, np.concatenate(neg_ex))
            r2 = {}
            for kind, d in per.items():
                r2[kind] = {}
                for pool, seeds in pools.items():
                    vals = [d[str(s)] for s in seeds if str(s) in d and np.isfinite(d[str(s)])]
                    r2[kind][pool] = {"mean": float(np.mean(vals)) if vals else None,
                                      "min": float(np.min(vals)) if vals else None,
                                      "n_seeds": len(vals)}
                r2[kind]["per_seed"] = {k: (float(x) if np.isfinite(x) else None)
                                        for k, x in sorted(d.items())}
            rec[stat] = r2
        out[a] = rec
    return out


def trigger_breakdown(passes: List[dict], thr: float, attack_start: int) -> Dict[str, int]:
    comp = {"u1": "abl_u1", "u2": "abl_u2", "g": "abl_g", "x": "abl_x", "rep": "abl_rep"}
    out: Dict[str, int] = {}
    for p in passes:
        v = p["key"]["victim"]
        sel = np.asarray(p["rounds"]) >= attack_start
        s = p["mats"]["suspicion"][sel][:, v]
        parts = {k: p["mats"][c][sel][:, v] for k, c in comp.items()}
        for i in np.where(P.flag_scores(s, thr))[0]:
            if s[i] >= SCORE_STRUCTURAL:
                key = "structural"
            else:
                vals = {k: x[i] for k, x in parts.items() if np.isfinite(x[i])}
                key = max(vals, key=vals.get) if vals else "?"
            out[key] = out.get(key, 0) + 1
    return out


def feature_medians(passes: List[dict], honest: List[dict], attack_start: int) -> dict:
    out = {}
    for k in RAW_FEATURES:
        col = f"raw_{k}"
        vic = np.concatenate([p["mats"][col][np.asarray(p["rounds"]) >= attack_start][:, p["key"]["victim"]]
                              for p in passes])
        bys = np.concatenate([np.delete(p["mats"][col][np.asarray(p["rounds"]) >= attack_start],
                                        [p["key"]["victim"]], axis=1).ravel() for p in passes])
        hon = np.concatenate([p["mats"][col][np.asarray(p["rounds"]) >= attack_start].ravel()
                              for p in honest])
        vic, bys, hon = vic[np.isfinite(vic)], bys[np.isfinite(bys)], hon[np.isfinite(hon)]
        out[k] = {"victim_median": float(np.median(vic)) if vic.size else None,
                  "bystander_median": float(np.median(bys)) if bys.size else None,
                  "honest_median": float(np.median(hon)) if hon.size else None,
                  "honest_q01": float(np.quantile(hon, 0.01)) if hon.size else None,
                  "honest_q99": float(np.quantile(hon, 0.99)) if hon.size else None}
    return out


def integrity(res: List[dict]) -> dict:
    bad = clipped = tot = 0
    for p in res:
        x, y = p["mats"]["suspicion"], p["mats"]["abl_full"]
        tot += int((np.isfinite(x) | np.isfinite(y)).sum())
        diff = np.nan_to_num(x, nan=-1.0) != np.nan_to_num(y, nan=-1.0)
        clip = diff & (x == SCORE_STAT_MAX) & (y > SCORE_STAT_MAX)
        clipped += int(clip.sum())
        bad += int((diff & ~clip).sum())
    return {"cells": tot, "differing_unexplained": bad, "differing_by_clip": clipped}


COMPARATORS = ("ens_stateful", "ens_stateful_causal", "ens_stateful_burnin", "ens_all",
               "ens_stateless")
RANKING_RULES = ("krum", "median_dev", "trimmed_mean_dev", "bulyan")
RANKING_RULES_Z = ("krum_z", "median_dev_z", "trimmed_mean_dev_z", "bulyan_z")
MEMORYLESS_MEMBERS = tuple(f"{b}_z" for b in BASELINES)


def comparator_summary(pr: dict, auc: dict, pool: str, attacks: Sequence[str],
                       ref_col: str = "suspicion", cols: Sequence[str] = COMPARATORS,
                       target: float = HEADLINE_TARGET,
                       ref_low_target: Optional[float] = None) -> dict:
    t = f"{target:g}"
    allc = [ref_col] + [c for c in cols if c != ref_col]
    per = {}
    for at in attacks:
        per[at] = {}
        for c in allc:
            per[at][c] = {"tpr": rate(pr["attacks"][at][c][t]["tpr"]),
                          "auc": auc[at][c]["all"][pool]["mean"],
                          "auc_excl_sources": auc[at][c]["excl_sources"][pool]["mean"]}

    def cmp(at, c, m):
        x, y = per[at][c][m], per[at][ref_col][m]
        if x is None or y is None:
            return None
        return "gt" if x > y else ("lt" if x < y else "tie")
    out = {"pool": pool, "target": target, "reference": ref_col, "attacks": list(attacks),
           "honest_client_round": {c: pr["honest"][c][t]["client_round"] for c in allc},
           "per_attack": per}
    for m in ("tpr", "auc", "auc_excl_sources"):
        for rel, key in (("gt", f"{m}_gt_reference"), ("lt", f"{m}_lt_reference"),
                         ("tie", f"{m}_tie")):
            out[key] = {c: [at for at in attacks if cmp(at, c, m) == rel]
                        for c in allc if c != ref_col}
            out["n_" + key] = {c: len(v) for c, v in out[key].items()}
    tl = None if ref_low_target is None else f"{ref_low_target:g}"
    if tl is not None and tl in pr["honest"].get(ref_col, {}):
        low = {at: rate(pr["attacks"][at][ref_col][tl]["tpr"]) for at in attacks}
        still = {c: [at for at in attacks if low[at] is not None and per[at][c]["tpr"] is not None
                     and low[at] > per[at][c]["tpr"]] for c in allc if c != ref_col}
        out["reference_at_lower_target"] = {
            "target": ref_low_target,
            "honest_client_round": pr["honest"][ref_col][tl]["client_round"],
            "tpr": low, "still_gt": still, "n_still_gt": {c: len(v) for c, v in still.items()}}
        cl = {}
        for c in allc:
            if c == ref_col or tl not in pr["honest"].get(c, {}):
                continue
            ctpr = {at: rate(pr["attacks"][at][c][tl]["tpr"]) for at in attacks}
            gt = [at for at in attacks if ctpr[at] is not None and per[at][ref_col]["tpr"] is not None
                  and ctpr[at] > per[at][ref_col]["tpr"]]
            cl[c] = {"honest_client_round": pr["honest"][c][tl]["client_round"], "tpr": ctpr,
                     "still_gt_reference": gt, "n_still_gt_reference": len(gt)}
        out["comparators_at_lower_target"] = {"target": ref_low_target, "columns": cl}
    return out


def ranking_inversion(auc: dict, pool: str, attacks: Sequence[str],
                      rules: Sequence[str] = RANKING_RULES,
                      seeds: Optional[Sequence[int]] = None) -> Dict[str, dict]:
    out = {}
    for at in attacks:
        vals = {c: auc[at][c]["all"][pool]["mean"] for c in rules}
        vals = {c: x for c, x in vals.items() if x is not None}
        if vals:
            c = max(vals, key=vals.get)
            out[at] = {"max_auc": vals[c], "rule": c, "auc": vals}
            if seeds is not None:
                per = [auc[at][r]["all"]["per_seed"].get(str(s)) for r in rules for s in seeds]
                per = [x for x in per if x is not None]
                out[at]["max_seed_auc"] = max(per) if per else None
    return out


def inversion_block(inv: Dict[str, dict], attacks: Sequence[str], rules: Sequence[str]) -> dict:
    att = [at for at in attacks if at in inv]
    if not att:
        return {"rules": list(rules), "per_attack": inv, "families_a_d": None}
    top = max(att, key=lambda at: inv[at]["max_auc"])
    seed_max = [inv[at].get("max_seed_auc") for at in att]
    seed_max = [x for x in seed_max if x is not None]
    return {"rules": list(rules), "per_attack": inv, "families_a_d": {
        "attacks": att, "max_auc": inv[top]["max_auc"], "attack": top, "rule": inv[top]["rule"],
        "n_all_below_half": sum(inv[at]["max_auc"] < 0.5 for at in att),
        "max_seed_auc": max(seed_max) if seed_max else None}}


def member_breakdown(passes: List[dict], ens: str, members: Sequence[str], thr: float,
                     attack_start: int) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in passes:
        sel = np.asarray(p["rounds"]) >= attack_start
        e = p["mats"][ens][sel]
        z = np.stack([p["mats"][c][sel] for c in members])
        v = p["key"]["victim"]
        for j in (range(e.shape[1]) if v is None else [v]):
            for i in np.where(P.flag_scores(e[:, j], thr))[0]:
                zz = z[:, i, j]
                az = np.where(np.isfinite(zz), np.abs(zz), -np.inf)
                k = int(np.argmax(az))
                if az[k] != e[i, j]:
                    out["mismatch"] = out.get("mismatch", 0) + 1
                key = members[k] + ("+" if zz[k] > 0 else "-")
                out[key] = out.get(key, 0) + 1
    return out


def command_line() -> str:
    return " ".join(["python experiments/envelope_eval.py"] + sys.argv[1:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", default=P.ARCHIVE_DIR)
    ap.add_argument("--archive-prefix", default="arch_cifar100_s")
    ap.add_argument("--cal", default="0,1")
    ap.add_argument("--dev", default="2")
    ap.add_argument("--test", default="3,4,5,6,7")
    ap.add_argument("--attack-start", type=int, default=12)
    ap.add_argument("--mad-floor", type=float, default=0.01,
                    help="the floor selected on CAL honest data in the release (checked)")
    ap.add_argument("--variants", default="all",
                    help="comma-separated attacks; 'all' expands to the default set")
    ap.add_argument("--victims", default="", help="comma-separated victim seats (default all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--save-scores", default="",
                    help="also pickle the per-pass score matrices here (hundreds of MB)")
    ap.add_argument("--load-scores", default="",
                    help="re-summarise a pickle written by --save-scores instead of replaying")
    ap.add_argument("--smoke", action="store_true",
                    help="CAL seed 0 only, victims 0 and 5, a few attacks; writes nothing")
    ap.add_argument("--lag-window", type=int, default=5,
                    help="POST-HOC SENSITIVITY ONLY: the auditor's lag window W (release: 5); "
                         "thresholds are refitted on CAL honest data at the new W")
    ap.add_argument("--auditor-only", action="store_true",
                    help="skip the 48 stateful detectors and their ensembles (about 4x faster)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "envelope_eval.json"))
    a = ap.parse_args()
    t_all = time.time()

    def ints(s):
        return [int(x) for x in s.split(",") if x.strip() != ""]

    workers = max(1, a.workers)
    cal, dev, test = ints(a.cal), ints(a.dev), ints(a.test)
    release_w = a.lag_window == 5
    variants = []
    for x in a.variants.split(","):
        for y in (DEFAULT_VARIANTS if x.strip() == "all" else [x.strip()] if x.strip() else []):
            if y not in variants:
                variants.append(y)
    if a.smoke:
        cal, dev, test = [0], [], []
        variants = ["lagged_self", "peer_copy", "echo_eps_1e-05", "lagged_6", "lag3_noise_0.5",
                    "lag3_rot_0.5", "peer_avg_2", "rows_half_lag3", "mosaic_9"]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit("unknown attacks: %s" % unknown)
    paths = P.discover_archives(a.archives, a.archive_prefix)
    pools = {"CAL": cal, "DEV": dev, "TEST": test}
    seeds = sorted(set(cal + dev + test))
    missing = [s for s in seeds if s not in paths]
    if missing:
        raise SystemExit("no archive for seeds %s" % missing)
    clients, rounds = P.archive_shape(P.load_archive(paths[seeds[0]]))
    victims = ints(a.victims) if a.victims else list(range(clients))
    if a.smoke:
        victims = [0, 5]
    scope_start = a.attack_start

    print("=" * 110)
    print("ATTACKS OUTSIDE THE AUDITOR'S DESIGN ENVELOPE")
    print("=" * 110)
    print("  pools         : CAL %s | DEV %s | TEST %s   victims %s" % (cal, dev, test, victims))
    print("  attacks       : %d (%s)" % (len(variants), ", ".join(variants)))
    print("  auditor       : lag window W = %d%s%s" % (
        a.lag_window, "" if release_w else "  (POST-HOC SENSITIVITY; release W = 5)",
        "; auditor and memoryless rules only" if a.auditor_only else ""))

    ref = None
    if os.path.exists(TEMPORAL_REF):
        with open(TEMPORAL_REF, encoding="utf-8") as fh:
            ref = json.load(fh)
    checks: Dict[str, object] = {"attack_definitions": P.check_attack_definitions()}
    if ref is not None:
        checks["mad_floor_matches_release"] = bool(ref["protocol"]["mad_floor"] == a.mad_floor)
    if not a.skip_check:
        t0 = time.time()
        if release_w:
            checks["loop_identity_vs_temporal_eval"] = check_loop_identity(
                paths[seeds[0]], seeds[0], a.mad_floor, a.attack_start, a.auditor_only)
        else:
            checks["loop_identity_vs_temporal_eval"] = {
                "identical": None}
        checks["attack_constructions"] = check_attack_constructions(
            paths[seeds[0]], seeds[0], a.attack_start)
        print("  loop identity : %s  (%.0fs)" % (checks["loop_identity_vs_temporal_eval"]["identical"],
                                                time.time() - t0))
        print("  constructions : %s" % json.dumps(checks["attack_constructions"]))
        if checks["loop_identity_vs_temporal_eval"]["identical"] is False:
            print("  ABORT: the local loop does not reproduce temporal_eval.py.")
            return 2

    tasks = []
    for pool, ss in pools.items():
        for s in ss:
            base = {"path": paths[s], "seed": s, "mad_floor": a.mad_floor,
                    "scope_start": scope_start, "attack_start": a.attack_start,
                    "lag_window": a.lag_window, "auditor_only": a.auditor_only}
            tasks.append(dict(base, key={"pool": pool, "seed": s, "attack": None, "victim": None},
                              attack=None, victim=None))
            for v_name in variants:
                for v in victims:
                    tasks.append(dict(base, key={"pool": pool, "seed": s, "attack": v_name,
                                                 "victim": v}, attack=v_name, victim=v))
    print("  replay passes : %d on %d workers" % (len(tasks), workers))
    t0 = time.time()
    if a.load_scores:
        with open(a.load_scores, "rb") as fh:
            cached = pickle.load(fh)
        want = sorted(json.dumps(t["key"], sort_keys=True) for t in tasks)
        have = sorted(json.dumps(p["key"], sort_keys=True) for p in cached["res"])
        if want != have:
            raise SystemExit("cached scores do not match the requested task list")
        res = cached["res"]
        wall_replay = cached["wall_replay_s"]
        sha = hashlib.sha256()
        with open(a.load_scores, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                sha.update(block)
        checks["scores_loaded_from_cache"] = {
            "file": os.path.basename(a.load_scores), "sha256": sha.hexdigest(),
            "replay_command": cached.get("command")}
    else:
        res = run_tasks(tasks, workers)
        wall_replay = time.time() - t0
        if a.save_scores:
            with open(a.save_scores, "wb") as fh:
                pickle.dump({"res": res, "wall_replay_s": wall_replay,
                             "command": command_line()}, fh, protocol=4)
    print("  replay done   : %.0fs" % wall_replay)

    checks["integrity_abl_full_vs_suspicion"] = integrity(res)
    print("  integrity     : %s" % checks["integrity_abl_full_vs_suspicion"])
    if checks["integrity_abl_full_vs_suspicion"]["differing_unexplained"]:
        print("  ABORT: the ablation does not reproduce the detector it ablates.")
        return 2

    cols = kept_columns(a.auditor_only)
    evcols = eval_columns(a.auditor_only)
    cal_honest = [p for p in res if p["key"]["pool"] == "CAL" and p["key"]["attack"] is None]
    thresholds = P.fit_thresholds({s: P.stack_scores(cal_honest, s) for s in cols}, P.TARGETS)
    if ref is not None and cal == ref["protocol"]["pools"]["CAL"]:
        common = [s for s in cols if s in ref["thresholds"]
                  and (release_w or s in BASELINES or s[:-2] in BASELINES)]
        dev_max = max(abs(thresholds[s][t] - ref["thresholds"][s][t]) for s in common
                      for t in thresholds[s] if np.isfinite(thresholds[s][t]))
        checks["thresholds_vs_release"] = {"columns": len(common), "max_abs_diff": dev_max}
        print("  thresholds    : %d columns vs release, max |diff| %.3g"
              % (len(common), dev_max))

    auc = auc_by_pool(res, pools, cols, a.attack_start)

    def auc_reproduction(refj: dict, stats: Sequence[str]) -> dict:
        worst, n, seeds_cmp = 0.0, 0, set()
        for at in P.ATTACK_ORDER:
            if at not in auc:
                continue
            for stat in stats:
                if stat not in cols:
                    continue
                for s, x in auc[at][stat]["all"]["per_seed"].items():
                    r_ = refj["auc"][at].get(stat, {}).get("per_seed", {}).get(s)
                    if r_ is not None and x is not None:
                        worst = max(worst, abs(x - r_))
                        n += 1
                        seeds_cmp.add(int(s))
        return {"compared": n, "seeds": sorted(seeds_cmp), "max_abs_diff": worst}

    if release_w:
        base_stats = ("suspicion", "abl_full_nostruct", "abl_no_rep", "abl_x", "krum_z",
                      "median_dev_z", "norm_bound_z", "ens_stateful", "ens_stateless",
                      "ens_all", BEST_SINGLE_STATEFUL)
        if ref is not None:
            checks["release_auc_reproduction"] = auc_reproduction(ref, base_stats)
            print("  release AUCs  : %s" % checks["release_auc_reproduction"])
        if os.path.exists(TEMPORAL_REF_V2) and not a.auditor_only:
            with open(TEMPORAL_REF_V2, encoding="utf-8") as fh:
                ref2 = json.load(fh)
            checks["temporal_eval_v2_auc_reproduction"] = auc_reproduction(
                ref2, base_stats + tuple(c for c in peer_columns() if c.startswith("ens_")
                                         or c.endswith("_z")))
            print("  v2 AUCs       : %s" % checks["temporal_eval_v2_auc_reproduction"])

    results: Dict[str, dict] = {}
    for pool, ss in pools.items():
        ps = [p for p in res if p["key"]["pool"] == pool]
        if not ps:
            continue
        honest = [p for p in ps if p["key"]["attack"] is None]
        by_att = {}
        for p in ps:
            if p["key"]["attack"] is not None:
                by_att.setdefault(p["key"]["attack"], []).append(p)
        pr = {"seeds": ss, "honest": {}, "attacks": {}, "triggers": {}, "features": {},
              "sources_per_pass": {}}
        for stat in evcols:
            tg = P.TARGETS if stat in ALL_TARGET_COLS else (HEADLINE_TARGET,)
            for t in tg:
                thr = thresholds[stat][f"{t:g}"]
                h = P.evaluate_honest(honest, stat, thr, clients)
                pr["honest"].setdefault(stat, {})[f"{t:g}"] = {
                    "threshold": thr, "client_round": [h["client_round"]["k"], h["client_round"]["n"]],
                    "client_run": [h["client_run"]["k"], h["client_run"]["n"]]}
                for at, aps in by_att.items():
                    pr["attacks"].setdefault(at, {}).setdefault(stat, {})[f"{t:g}"] = \
                        evaluate_variant(aps, stat, thr, a.attack_start)
        thr1 = thresholds["suspicion"][f"{HEADLINE_TARGET:g}"]
        for at, aps in by_att.items():
            pr["triggers"][at] = trigger_breakdown(aps, thr1, a.attack_start)
            pr["features"][at] = feature_medians(aps, honest, a.attack_start)
            srcs = [len(p.get("sources", {})) for p in aps]
            pr["sources_per_pass"][at] = {"min": int(min(srcs)), "max": int(max(srcs))}
        if "ens_stateless" in cols:
            thr_ml = thresholds["ens_stateless"][f"{HEADLINE_TARGET:g}"]
            pr["members_ens_stateless"] = {"honest": member_breakdown(
                honest, "ens_stateless", MEMORYLESS_MEMBERS, thr_ml, a.attack_start)}
            for at, aps in by_att.items():
                pr["members_ens_stateless"][at] = member_breakdown(
                    aps, "ens_stateless", MEMORYLESS_MEMBERS, thr_ml, a.attack_start)
        results[pool] = pr

    head = "TEST" if "TEST" in results else ("DEV" if "DEV" in results else "CAL")
    t1 = f"{HEADLINE_TARGET:g}"
    summary = {"pool": head, "target": HEADLINE_TARGET, "rows": {}}
    for at in variants:
        if at not in results[head]["attacks"]:
            continue
        e = results[head]["attacks"][at]
        row = {"family": VARIANTS[at]["family"],
               "auc_auditor": auc[at]["suspicion"]["all"][head]["mean"],
               "auc_auditor_min_seed": auc[at]["suspicion"]["all"][head]["min"],
               "auc_auditor_excl_sources": auc[at]["suspicion"]["excl_sources"][head]["mean"],
               "triggers": results[head]["triggers"][at]}
        for stat in evcols:
            row[f"tpr_{stat}"] = rate(e[stat][t1]["tpr"])
        for stat in (["abl_full_nostruct", BEST_SINGLE_STATEFUL, "ens_stateful", "ens_stateless",
                      "ens_all"] + [f"{k}_z" for k in T.FG_VARIANTS] + list(T.PEER_ENSEMBLES)
                     + ROBUST):
            row[f"auc_{stat}"] = auc[at][stat]["all"][head]["mean"] if stat in cols else None
        row["auc_best_robust"] = max((auc[at][c]["all"][head]["mean"] or 0.0) for c in ROBUST)
        row["tpr_best_robust"] = max((row[f"tpr_{c}"] or 0.0) for c in ROBUST)
        s = e["suspicion"][t1]
        row["runs_flagged"] = s["run"]
        row["onset_flagged"] = s["onset"]
        row["runs_flagged_ge2"] = s["run_ge2"]
        row["latency_median"] = s["latency_median"]
        row["fpr_bystander"] = rate(s["byst"])
        row["fpr_uninvolved"] = rate(s["uninv"])
        row["flag_rate_sources"] = rate(s["src"])
        row["attribution_top1"] = rate(s["top1"])
        summary["rows"][at] = row
    new = [at for at in summary["rows"] if VARIANTS[at]["family"] != "reference"]
    if new:
        for stat in [c for c in ALL_TARGET_COLS if c in evcols]:
            w = min(new, key=lambda at: summary["rows"][at][f"tpr_{stat}"] or 0.0)
            summary.setdefault("worst_new_attack", {})[stat] = {
                "attack": w, "tpr": summary["rows"][w][f"tpr_{stat}"]}
    if new and not a.auditor_only:
        lower = [x for x in P.TARGETS if x < HEADLINE_TARGET]
        summary["comparators"] = comparator_summary(
            results[head], auc, head, new, ref_low_target=max(lower) if lower else None)
        fam_ad = [at for at in new if VARIANTS[at]["family"] in ("a", "d")]
        summary["ranking_inversion"] = {
            "raw": inversion_block(ranking_inversion(auc, head, list(summary["rows"]),
                                                     RANKING_RULES, pools[head]),
                                   fam_ad, RANKING_RULES),
            "cohort_z": inversion_block(ranking_inversion(auc, head, list(summary["rows"]),
                                                          RANKING_RULES_Z, pools[head]),
                                        fam_ad, RANKING_RULES_Z)}
        summary["robust_rules"] = {
            "honest_client_round": {c: results[head]["honest"][c][t1]["client_round"]
                                    for c in ROBUST},
            "best_by_tpr": {at: max(ROBUST, key=lambda c: summary["rows"][at][f"tpr_{c}"] or 0.0)
                            for at in summary["rows"]}}

    print("\n" + "=" * 128)
    print("%s pool, frozen CAL threshold at nominal %g%% (auditor %.3f). TPR per client-round; runs and"
          % (head, 100 * HEADLINE_TARGET, thresholds["suspicion"][t1]))
    print("onset are victim runs flagged at least once / in the onset round; uninv = uninvolved")
    print("bystander false-alarm rate; src = flag rate of the copy sources.")
    print("=" * 128)
    hdr = ("%-18s %6s %6s %6s %6s %6s %7s %7s %7s %7s %7s %7s %7s %6s %6s"
           % ("attack", "AUC", "TPR", "nostr", "minLOO", "robust", "ensSF", "ensML", "ensAll",
              "SF+x+r", "runs", "onset", "uninv", "src", "trig"))
    print(hdr)
    print("-" * len(hdr))

    def pc(x, w=6, d=1):
        return ("%*.*f%%" % (w - 1, d, 100 * x)) if x is not None else "%*s" % (w, "-")

    for at, row in summary["rows"].items():
        loo = min((row[f"tpr_abl_no_{k}"] or 0.0) for k in RAW_FEATURES)
        trig = max(row["triggers"], key=row["triggers"].get) if row["triggers"] else "-"
        print("%-18s %6.3f %s %s %s %s %s %s %s %s %3d/%-3d %3d/%-3d %s %s %6s"
              % (at[:18], row["auc_auditor"], pc(row["tpr_suspicion"]),
                 pc(row["tpr_abl_full_nostruct"]), pc(loo), pc(row["tpr_best_robust"]),
                 pc(row.get("tpr_ens_stateful"), 7), pc(row.get("tpr_ens_stateless"), 7),
                 pc(row.get("tpr_ens_all"), 7), pc(row.get("tpr_ens_stateful_x_rep"), 7),
                 row["runs_flagged"][0], row["runs_flagged"][1],
                 row["onset_flagged"][0], row["onset_flagged"][1],
                 pc(row["fpr_uninvolved"], 7, 2), pc(row["flag_rate_sources"]), trig))
    hh = results[head]["honest"]
    print("  honest %s false-alarm rate: %s"
          % (head, ", ".join("%s %s" % (c, hh[c][t1]["client_round"])
                             for c in ("suspicion", "ens_stateful", "ens_stateless") if c in hh)))
    if "comparators" in summary:
        cm = summary["comparators"]
        print("  new attacks on which a comparator beats the auditor (TPR / AUC, of %d): %s"
              % (len(cm["attacks"]), ", ".join(
                  "%s %d/%d" % (c, cm["n_tpr_gt_reference"][c], cm["n_auc_gt_reference"][c])
                  for c in COMPARATORS)))
        print("  ... and on which the auditor is higher (TPR / AUC): %s"
              % ", ".join("%s %d/%d" % (c, cm["n_tpr_lt_reference"][c], cm["n_auc_lt_reference"][c])
                          for c in COMPARATORS))
        if "comparators_at_lower_target" in cm:
            cl = cm["comparators_at_lower_target"]
            print("  comparators at their own %g%% thresholds, still above the auditor (TPR): %s"
                  % (100 * cl["target"], ", ".join(
                      "%s %d (honest %d/%d)" % (c, v["n_still_gt_reference"], *v["honest_client_round"])
                      for c, v in cl["columns"].items())))
        for scale in ("raw", "cohort_z"):
            fa = summary["ranking_inversion"][scale]["families_a_d"]
            if fa:
                print("  largest ranking-rule AUC over families (a) and (d), %s scores: %.3f "
                      "(%s, %s); %d/%d below 0.5" % (scale, fa["max_auc"], fa["rule"], fa["attack"],
                                                     fa["n_all_below_half"], len(fa["attacks"])))

    if a.smoke:
        print("\nsmoke run: nothing written (%.0fs)" % (time.time() - t_all))
        return 0

    from passportfl.reproducibility import environment
    payload = {
        "config": {"dataset": "cifar100", "dirichlet": 0.5, "clients": clients, "rounds": rounds,
                   "pools": pools, "victims": victims, "attack_start": a.attack_start,
                   "scope_start": scope_start, "mad_floor": a.mad_floor,
                   "lag_window": a.lag_window, "auditor_only": a.auditor_only,
                   "post_hoc_sensitivity": (None if release_w else
                                            "lag window W changed from the release value 5 "
                                            "after the W = 5 results were seen; thresholds "
                                            "refitted on CAL honest data; not a headline"),
                   "mask_policy": "declared", "targets": list(P.TARGETS),
                   "headline_target": HEADLINE_TARGET,
                   "archives": {str(s): os.path.basename(paths[s]) for s in seeds},
                   "variants": {k: VARIANTS[k] for k in variants},
                   "grids": {"echo_rel": ECHO_REL, "lags_beyond_W": LAGS_BEYOND_W,
                             "noise_lags": NOISE_LAGS, "noise_kappa": NOISE_KAPPA,
                             "rotation_lag": ROT_LAG, "rotation_theta": ROT_THETA,
                             "mosaic_m": MOSAIC_M, "bulyan_f_requested": F_BULYAN},
                   "rng": "default_rng([seed, victim + 1, tag]); tag = release index + 1 for "
                          "release attacks, 1000 + crc32(name) % 1e6 otherwise",
                   "columns": cols, "eval_columns": evcols, "robust_rules": ROBUST,
                   "best_single_stateful": BEST_SINGLE_STATEFUL,
                   "peer_aware": (None if a.auditor_only else
                                  {"foolsgold": dict(T.FG_VARIANTS)}),
                   "command": command_line()},
        "thresholds": {s: thresholds[s] for s in cols},
        "auc": auc,
        "results": results,
        "summary": summary,
        "meta": {"environment": environment(), "checks": checks, "workers": workers,
                 "passes": len(tasks), "wall_replay_s": round(wall_replay, 1),
                 "wall_s": round(time.time() - t_all, 1)},
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=float)
    print("\nwrote %s   (total %.0fs)" % (a.out, time.time() - t_all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
