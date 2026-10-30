from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import experiments.protocol_eval as P
import experiments.temporal_eval as T

RESULTS = os.path.join(HERE, "results")
CAL = (0, 1)
POOLS = {"DEV": (2,), "TEST": (3, 4), "FRESH": (5, 6, 7), "HELDOUT": (3, 4, 5, 6, 7)}
TARGETS = P.TARGETS
ATTACKS = P.ATTACK_ORDER
BEST = "BEST_STATEFUL_CAL"

DETECTORS: List[Tuple[str, str]] = [
    ("suspicion", "Payload auditor"),
    ("abl_full_nostruct", "Payload auditor without the structural tier"),
    ("ens_stateful", "Stateful ensemble (24 families x 2 fit regimes)"),
    ("ens_stateful_causal", "Stateful ensemble, causal fits only"),
    ("ens_stateful_burnin", "Stateful ensemble, burn-in fits only (clean window)"),
    (BEST, "Best single stateful, selected on CAL seeds only"),
    ("tb_ar1_burnin_zabs", "AR(1) residual"),
    ("ens_stateless", "Ensemble of six memoryless rules"),
    ("ens_all", "Stateful + memoryless ensemble"),
    ("ens_stateful_x", "Stateful ensemble + auditor's x"),
    ("ens_stateful_fg", "Stateful ensemble + FoolsGold (cumulative)"),
    ("ens_stateful_x_rep", "Stateful ensemble + auditor's x and rep"),
    ("ens_all_fg", "Stateful + memoryless + FoolsGold ensemble"),
    ("abl_x", "auditor's x alone"),
    ("abl_rep", "auditor's rep alone"),
    ("fg_cum_z", "FoolsGold cumulative, uniform weights (z)"),
    ("fg_cum_imp10_z", "FoolsGold cumulative, indicative features (z)"),
]
PAIRED = [d for d, _ in DETECTORS if d != "suspicion"]

K_RULES = (1, 2, 3)

FPR_GRID = np.unique(np.round(np.concatenate([[0.0], np.logspace(-4, 0, 41),
                                              np.linspace(0.02, 1.0, 50)]), 10))


class Scores:
    def __init__(self, path: str):
        self.path = path
        self.z = np.load(path, allow_pickle=False)
        self.columns = [str(c) for c in self.z["columns"]]
        self.seed = self.z["seed"].astype(int)
        self.victim = self.z["victim"].astype(int)
        self.attack = self.z["attack"].astype(int)
        self.attack_names = [str(a) for a in self.z["attack_names"]]
        assert self.attack_names == list(ATTACKS), "attack order differs from protocol_eval"
        self.pool = [str(p) for p in self.z["pool"]]
        self.rounds = [int(r) for r in self.z["rounds"]]
        self.clients = int(self.z["clients"].size)
        self.meta = json.loads(str(self.z["meta_json"]))
        self.index = {}
        for i in range(self.seed.size):
            a = None if self.attack[i] < 0 else self.attack_names[self.attack[i]]
            v = None if self.victim[i] < 0 else int(self.victim[i])
            self.index[(int(self.seed[i]), a, v)] = i
        self.seeds = sorted(set(self.seed.tolist()))
        self._cache: Dict[str, np.ndarray] = {}

    def col(self, name: str) -> np.ndarray:
        if name not in self._cache:
            self._cache[name] = self.z["s__" + name]
        return self._cache[name]

    def mat(self, name: str, seed: int, attack: Optional[str], victim: Optional[int]) -> np.ndarray:
        return self.col(name)[self.index[(seed, attack, victim)]]

    def passes(self, seeds: Sequence[int], columns: Sequence[str], honest: bool = True,
               attacks: bool = True) -> List[dict]:
        out = []
        for i in range(self.seed.size):
            s = int(self.seed[i])
            if s not in seeds:
                continue
            a = int(self.attack[i])
            if (a < 0 and not honest) or (a >= 0 and not attacks):
                continue
            out.append({"key": {"pool": self.pool[i], "seed": s,
                                "attack": None if a < 0 else self.attack_names[a],
                                "victim": None if a < 0 else int(self.victim[i])},
                        "rounds": self.rounds,
                        "mats": {c: self.col(c)[i] for c in columns}})
        return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def same(x, y) -> bool:
    if x is None or y is None:
        return x is None and y is None
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    return x == y


def load_json(name: str) -> Optional[dict]:
    p = name if os.path.isabs(name) else os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def select_stateful(sc: Scores, seeds: Sequence[int], attack_start: int) -> dict:
    cands = [c for c in sc.columns if c.startswith("tb_")]
    passes = sc.passes(seeds, cands, honest=False)
    auc = T.auc_table(passes, list(seeds), cands, attack_start)
    ranked = sorted(((T.worst_case(auc, c)[0], c) for c in cands
                     if np.isfinite(T.worst_case(auc, c)[0])), reverse=True)
    fam: Dict[str, dict] = {}
    for f in T.BASE_STATS:
        best, bv = None, -1.0
        for c in cands:
            base = c[:-5] if c.endswith("_zabs") else (c[:-2] if c.endswith("_z") else c)
            if T.family_of(base) != f:
                continue
            wc = T.worst_case(auc, c)[0]
            if np.isfinite(wc) and wc > bv:
                best, bv = c, wc
        if best:
            fam[f] = {"column": best, "worst_auc": bv}
    return {"seeds": list(seeds), "selected": ranked[0][1], "worst_auc": ranked[0][0],
            "top10": [{"column": c, "worst_auc": v,
                       "worst_attack": T.worst_case(auc, c)[1]} for v, c in ranked[:10]],
            "family_best_variant": fam, "n_candidates": len(cands)}


def pool_auc(auc: dict, det: str, seeds: Sequence[int]) -> Dict[str, Optional[float]]:
    out = {}
    for a in ATTACKS:
        per = auc[a][det]["per_seed"]
        v = [per[str(s)] for s in seeds if per.get(str(s)) is not None
             and np.isfinite(per[str(s)])]
        out[a] = float(np.mean(v)) if v else None
    return out


def evaluate_frozen(sc: Scores, seeds: Sequence[int], det: str, thr: Dict[str, float],
                    attack_start: int) -> dict:
    hon = sc.passes(seeds, [det], attacks=False)
    allp = sc.passes(seeds, [det], honest=False)
    atk = {a: [p for p in allp if p["key"]["attack"] == a] for a in ATTACKS}
    out = {}
    for t in TARGETS:
        th = thr[f"{t:g}"]
        h = P.evaluate_honest(hon, det, th, sc.clients)
        per, uk, un = {}, 0, 0
        for a in ATTACKS:
            e = P.evaluate_attack(atk[a], det, th, sc.clients, attack_start, a)
            fp = e["fpr_uninvolved_client_round"]
            uk += fp["k"]
            un += fp["n"]
            tr = e["tpr_client_round"]
            per[a] = {"tpr": tr["point"], "k": tr["k"], "n": tr["n"],
                      "ci": [tr["lo"], tr["hi"]],
                      "trajectories_detected": e["tpr_client_run"]["k"],
                      "trajectories": e["tpr_client_run"]["n"],
                      "fpr_uninvolved": fp["point"], "uninvolved_k": fp["k"],
                      "uninvolved_n": fp["n"],
                      "fpr_bystander": e["fpr_bystander_client_round"]["point"],
                      "flag_rate_copy_source": (e["flag_rate_copy_source"]["point"]
                                                if e["flag_rate_copy_source"] else None),
                      "latency_median": e["latency_rounds"]["median"],
                      "attribution_precision": e["attribution_precision"]["point"]}
        worst = min(ATTACKS, key=lambda a: per[a]["tpr"])
        cr = h["client_round"]
        out[f"{t:g}"] = {
            "threshold": th,
            "clean_fpr": cr["point"], "clean_k": cr["k"], "clean_n": cr["n"],
            "clean_ci": [cr["lo"], cr["hi"]],
            "clean_trajectories_flagged": h["client_run"]["k"],
            "clean_trajectories": h["client_run"]["n"],
            "uninvolved_fpr": (uk / un) if un else None, "uninvolved_k": uk, "uninvolved_n": un,
            "worst_tpr": per[worst]["tpr"], "worst_attack": worst,
            "mean_tpr": float(np.mean([per[a]["tpr"] for a in ATTACKS])),
            "per_attack": per,
        }
    return out


def operational(v: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(v), v, -np.inf)


def roc_on_grid(pos: np.ndarray, neg: np.ndarray, grid: np.ndarray) -> Tuple[List[float], List[float]]:
    ns = np.sort(operational(neg))
    ps = np.sort(operational(pos))
    n = ns.size
    fpr, tpr = [], []
    for f in grid:
        m = int(math.floor(f * n))
        idx = n - 1 - m
        thr = ns[idx] if idx >= 0 else -np.inf
        fpr.append(float((n - np.searchsorted(ns, thr, side="right")) / n))
        tpr.append(float((ps.size - np.searchsorted(ps, thr, side="right")) / ps.size))
    return fpr, tpr


def operational_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    p, n = operational(pos), operational(neg)
    fin = np.concatenate([p[np.isfinite(p)], n[np.isfinite(n)]])
    low = (fin.min() - 1.0) if fin.size else 0.0
    return P.roc_auc(np.where(np.isfinite(p), p, low), np.where(np.isfinite(n), n, low))


def roc_curves(sc: Scores, seeds: Sequence[int], det: str) -> dict:
    N = sc.clients
    clean = np.concatenate([sc.mat(det, s, None, None).ravel() for s in seeds])
    out = {"bystanders": {}, "clean_runs": {}}
    env_b, env_c, env_bx = None, None, None
    for a in ATTACKS:
        pos, neg, negx = [], [], []
        for s in seeds:
            for v in range(N):
                m = sc.mat(det, s, a, v)
                pos.append(m[:, v])
                keep = [c for c in range(N) if c != v]
                neg.append(m[:, keep].ravel())
                acc = P.accomplice(a, v, N)
                negx.append(m[:, [c for c in keep if c != acc]].ravel())
        pos, neg, negx = np.concatenate(pos), np.concatenate(neg), np.concatenate(negx)
        fb, tb = roc_on_grid(pos, neg, FPR_GRID)
        fc, tc = roc_on_grid(pos, clean, FPR_GRID)
        out["bystanders"][a] = {"fpr": fb, "tpr": tb, "auc_pooled": operational_auc(pos, neg),
                                "n_pos": int(pos.size), "n_neg": int(neg.size)}
        tbx = tb
        if a == "peer_copy":
            fx, tx = roc_on_grid(pos, negx, FPR_GRID)
            tbx = tx
            out["bystanders"]["peer_copy_excl_copy_source"] = {
                "fpr": fx, "tpr": tx, "auc_pooled": operational_auc(pos, negx),
                "n_pos": int(pos.size), "n_neg": int(negx.size)}
        out["clean_runs"][a] = {"fpr": fc, "tpr": tc, "auc_pooled": operational_auc(pos, clean),
                                "n_pos": int(pos.size), "n_neg": int(clean.size)}
        env_b = np.asarray(tb) if env_b is None else np.minimum(env_b, tb)
        env_bx = np.asarray(tbx) if env_bx is None else np.minimum(env_bx, tbx)
        env_c = np.asarray(tc) if env_c is None else np.minimum(env_c, tc)
    out["bystanders"]["worst_case_envelope"] = {"nominal_fpr": FPR_GRID.tolist(),
                                                "tpr": env_b.tolist()}
    out["bystanders"]["worst_case_envelope_excl_copy_source"] = {
        "nominal_fpr": FPR_GRID.tolist(), "tpr": env_bx.tolist()}
    out["clean_runs"]["worst_case_envelope"] = {"fpr": out["clean_runs"][ATTACKS[0]]["fpr"],
                                                "tpr": env_c.tolist()}
    return out


def flags(m: np.ndarray, thr: float) -> np.ndarray:
    return P.flag_scores(m, thr)


def kth_flag_rounds(f: np.ndarray, kmax: int = max(K_RULES)) -> np.ndarray:
    idx = np.flatnonzero(np.asarray(f))
    out = np.full(kmax, -1, dtype=int)
    out[:min(kmax, idx.size)] = idx[:kmax]
    return out


def seat_statistics(sc: Scores, seed: int, det: str, thr: Dict[str, float]) -> dict:
    N = sc.clients
    R = len(sc.rounds)
    out = {"clean": {}, "victim": {}, "victim_kth": {}, "bystander": {}, "copy_source": {},
           "U": {}, "Ux": {}}
    mh = sc.mat(det, seed, None, None)
    for t in TARGETS:
        out["clean"][f"{t:g}"] = flags(mh, thr[f"{t:g}"]).sum(axis=0).astype(int)
    for a in ATTACKS:
        ms = [sc.mat(det, seed, a, v) for v in range(N)]
        src = [P.accomplice(a, v, N) for v in range(N)]
        for t in TARGETS:
            th = thr[f"{t:g}"]
            fl = [flags(m, th) for m in ms]
            out["victim"].setdefault(f"{t:g}", {})[a] = np.asarray(
                [int(fl[v][:, v].sum()) for v in range(N)])
            out["victim_kth"].setdefault(f"{t:g}", {})[a] = np.stack(
                [kth_flag_rounds(fl[v][:, v]) for v in range(N)])
            if all(c is not None for c in src):
                out["copy_source"].setdefault(f"{t:g}", {})[a] = np.asarray(
                    [int(fl[v][:, src[v]].sum()) for v in range(N)])
            by = []
            for v in range(N):
                acc = P.accomplice(a, v, N)
                for c in range(N):
                    if c != v and c != acc:
                        by.append(int(fl[v][:, c].sum()))
            out["bystander"].setdefault(f"{t:g}", {})[a] = np.asarray(by)
        pos = [ms[v][:, v][np.isfinite(ms[v][:, v])] for v in range(N)]
        for excl, key in ((False, "U"), (True, "Ux")):
            negs = []
            for c in range(N):
                parts = []
                for v in range(N):
                    if v == c or (excl and P.accomplice(a, v, N) == c):
                        continue
                    col = ms[v][:, c]
                    parts.append(col[np.isfinite(col)])
                negs.append(np.sort(np.concatenate(parts)))
            U = np.zeros((N, N))
            for c in range(N):
                for c2 in range(N):
                    lt = np.searchsorted(negs[c2], pos[c], side="left")
                    le = np.searchsorted(negs[c2], pos[c], side="right")
                    U[c, c2] = float((lt + 0.5 * (le - lt)).sum())
            out[key][a] = (U, np.asarray([p.size for p in pos], float),
                           np.asarray([n.size for n in negs], float))
    out["R"] = R
    return out


def weighted_auc(U: np.ndarray, npos: np.ndarray, nneg: np.ndarray, W: np.ndarray) -> np.ndarray:
    num = np.einsum("bi,ij,bj->b", W, U, W)
    den = (W @ npos) * (W @ nneg)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def metrics(st: Dict[int, dict], seeds: Sequence[int], Wseat: Optional[np.ndarray],
            slots: Optional[np.ndarray], B: int) -> Dict[str, np.ndarray]:
    S = len(seeds)
    N = next(iter(st.values()))["clean"][f"{TARGETS[0]:g}"].size
    R = next(iter(st.values()))["R"]
    if slots is None:
        slots = np.tile(np.arange(S), (B, 1))
    if Wseat is None:
        Wseat = np.ones((B, slots.shape[1], N))
    out: Dict[str, np.ndarray] = {}
    wsum = Wseat.sum(axis=2)

    def gather(arr_by_seed: List[np.ndarray]) -> np.ndarray:
        A = np.stack(arr_by_seed)
        return A[slots]

    for t in TARGETS:
        key = f"{t:g}"
        ck = gather([st[s]["clean"][key] for s in seeds])
        out[f"clean_fpr@{key}"] = (Wseat * ck).sum(axis=(1, 2)) / (R * wsum.sum(axis=1))
        tprs = []
        for a in ATTACKS:
            vk = gather([st[s]["victim"][key][a] for s in seeds])
            tp = (Wseat * vk).sum(axis=(1, 2)) / (R * wsum.sum(axis=1))
            out[f"tpr@{key}/{a}"] = tp
            tprs.append(tp)
        out[f"worst_tpr@{key}"] = np.min(np.stack(tprs), axis=0)
    for key, name in (("U", "auc"), ("Ux", "auc_excl_copy_source")):
        aucs = []
        for a in ATTACKS:
            vals = np.full(slots.shape, np.nan)
            for si, s in enumerate(seeds):
                U, npos, nneg = st[s][key][a]
                msk = slots == si
                if msk.any():
                    vals[msk] = weighted_auc(U, npos, nneg, Wseat[msk])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmean(vals, axis=1)
            out[f"{name}/{a}"] = m
            aucs.append(m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[f"worst_{name}"] = np.nanmin(np.stack(aucs), axis=0)
    return out


def summarise(draws: np.ndarray, point: float) -> dict:
    d = draws[np.isfinite(draws)]
    if not d.size:
        return {"point": point, "lo": None, "hi": None, "se": None}
    return {"point": float(point), "lo": float(np.quantile(d, 0.025)),
            "hi": float(np.quantile(d, 0.975)), "se": float(d.std(ddof=1)),
            "n_distinct": int(np.unique(np.round(d, 12)).size)}


def bootstrap(stats: Dict[str, Dict[int, dict]], seeds: Sequence[int], scheme: str,
              B: int, rng: np.random.Generator) -> dict:
    S = len(seeds)
    N = next(iter(next(iter(stats.values())).values()))["clean"][f"{TARGETS[0]:g}"].size
    if scheme == "seed":
        slots = rng.integers(0, S, size=(B, S))
        Wseat = None
    elif scheme == "seat":
        slots = None
        Wseat = rng.multinomial(N, np.full(N, 1.0 / N), size=(B, S)).astype(float)
    elif scheme == "two_stage":
        slots = rng.integers(0, S, size=(B, S))
        Wseat = rng.multinomial(N, np.full(N, 1.0 / N), size=(B, S)).astype(float)
    else:
        raise ValueError(scheme)
    res: Dict[str, dict] = {}
    draws: Dict[str, Dict[str, np.ndarray]] = {}
    points: Dict[str, Dict[str, float]] = {}
    for det, st in stats.items():
        draws[det] = metrics(st, seeds, Wseat, slots, B)
        points[det] = {k: float(v[0]) for k, v in metrics(st, seeds, None, None, 1).items()}
        res[det] = {k: summarise(v, points[det][k]) for k, v in draws[det].items()}
    if "suspicion" in draws:
        for det in stats:
            if det == "suspicion":
                continue
            for k in ("worst_tpr@0.01", "worst_tpr@0.05", "worst_auc",
                      "worst_auc_excl_copy_source"):
                diff = draws["suspicion"][k] - draws[det][k]
                res[det][f"auditor_minus_this/{k}"] = summarise(
                    diff, points["suspicion"][k] - points[det][k])
                res[det][f"auditor_minus_this/{k}"]["share_draws_le_0"] = float(np.mean(diff <= 0))
    return res


def trajectory_tables(st_by_seed: Dict[int, dict], seeds: Sequence[int]) -> dict:
    R = st_by_seed[seeds[0]]["R"]
    copy_attacks = [a for a in ATTACKS if a in st_by_seed[seeds[0]]["copy_source"].get(
        f"{TARGETS[0]:g}", {})]
    out = {}
    for t in TARGETS:
        key = f"{t:g}"
        clean = np.concatenate([st_by_seed[s]["clean"][key] for s in seeds])
        per_seed_clean = {str(s): st_by_seed[s]["clean"][key].tolist() for s in seeds}
        hist = np.bincount(clean, minlength=R + 1)
        rec = {"clean": {"trajectories": int(clean.size),
                         "flagged_rounds_histogram": {str(i): int(hist[i]) for i in range(R + 1)
                                                      if hist[i]},
                         "flagged_rounds_per_seed_and_seat": per_seed_clean,
                         "max_flags_in_one_trajectory": int(clean.max()),
                         "flags_total": int(clean.sum()), "rounds_total": int(clean.size * R)},
               "attacked": {}, "uninvolved": {}, "copy_source": {}, "k_of_24_rules_post_hoc": {}}
        ck = (np.concatenate([st_by_seed[s]["copy_source"][key][a] for s in seeds
                              for a in copy_attacks]) if copy_attacks else np.zeros(0, int))
        for k in K_RULES:
            hk = int((clean >= k).sum())
            p, lo, hi = P.clopper_pearson(hk, int(clean.size))
            rule = {"clean_trajectories_flagged": hk, "clean_trajectories": int(clean.size),
                    "clean_rate": p, "clean_ci": [lo, hi],
                    "clean_one_sided_95_upper": P.clopper_pearson(hk, int(clean.size), 0.90)[2],
                    "attacked": {}}
            for a in ATTACKS:
                vk = np.concatenate([st_by_seed[s]["victim"][key][a] for s in seeds])
                d = int((vk >= k).sum())
                p2, lo2, hi2 = P.clopper_pearson(d, int(vk.size))
                rk = np.concatenate([st_by_seed[s]["victim_kth"][key][a][:, k - 1]
                                     for s in seeds])
                reached = rk[rk >= 0]
                rule["attacked"][a] = {"detected": d, "trajectories": int(vk.size),
                                       "rate": p2, "ci": [lo2, hi2],
                                       "one_sided_95_lower": P.clopper_pearson(
                                           d, int(vk.size), 0.90)[1],
                                       "round_of_kth_flag": {
                                           "median": (float(np.median(reached))
                                                      if reached.size else None),
                                           "max": int(reached.max()) if reached.size else None,
                                           "not_reached": int((rk < 0).sum()),
                                           "at_earliest_possible_round": int(
                                               (rk == k - 1).sum())}}
            rule["attacked_all_attacks"] = {
                "detected": sum(r["detected"] for r in rule["attacked"].values()),
                "trajectories": sum(r["trajectories"] for r in rule["attacked"].values()),
                "kth_flag_at_earliest_possible_round": sum(
                    r["round_of_kth_flag"]["at_earliest_possible_round"]
                    for r in rule["attacked"].values())}
            uk = np.concatenate([st_by_seed[s]["bystander"][key][a] for s in seeds
                                 for a in ATTACKS])
            rule["uninvolved_trajectories_flagged"] = int((uk >= k).sum())
            rule["uninvolved_trajectories"] = int(uk.size)
            rule["copy_source_trajectories_flagged"] = int((ck >= k).sum())
            rule["copy_source_trajectories"] = int(ck.size)
            rule["honest_trajectories_in_attacked_runs_flagged"] = int((uk >= k).sum()
                                                                       + (ck >= k).sum())
            rule["honest_trajectories_in_attacked_runs"] = int(uk.size + ck.size)
            rec["k_of_24_rules_post_hoc"][str(k)] = rule
        for a in copy_attacks:
            cs = np.concatenate([st_by_seed[s]["copy_source"][key][a] for s in seeds])
            rec["copy_source"][a] = {"trajectories": int(cs.size),
                                     "flagged_at_least_once": int((cs >= 1).sum()),
                                     "all_rounds_flagged": int((cs == R).sum()),
                                     "flags_total": int(cs.sum()),
                                     "rounds_total": int(cs.size * R)}
        for a in ATTACKS:
            vk = np.concatenate([st_by_seed[s]["victim"][key][a] for s in seeds])
            rec["attacked"][a] = {"trajectories": int(vk.size),
                                  "flagged_at_least_once": int((vk >= 1).sum()),
                                  "all_rounds_flagged": int((vk == R).sum()),
                                  "median_fraction_of_rounds_flagged": float(np.median(vk) / R),
                                  "min_fraction_of_rounds_flagged": float(vk.min() / R)}
            bk = np.concatenate([st_by_seed[s]["bystander"][key][a] for s in seeds])
            rec["uninvolved"][a] = {"trajectories": int(bk.size),
                                    "flagged_at_least_once": int((bk >= 1).sum()),
                                    "flagged_at_least_twice": int((bk >= 2).sum()),
                                    "flags_total": int(bk.sum())}
        out[key] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=os.path.join(HERE, ".reproduction", "scores",
                                                     "temporal_eval_v2_scores.npz"))
    ap.add_argument("--temporal", default="temporal_eval_v2.json",
                    help="the temporal_eval output that wrote --scores (cross-checked)")
    ap.add_argument("--released", default="temporal_eval.json",
                    help="the released two-seed TEST evaluation (cross-checked)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=20261019)
    ap.add_argument("--out", default=os.path.join(RESULTS, "auditor_statistics.json"))
    a = ap.parse_args()
    t_all = time.time()

    sc = Scores(a.scores)
    attack_start = int(sc.meta["attack_start"])
    assert sc.rounds[0] >= attack_start, "scores must start at the attack onset"
    print("scores        : %s (%d passes, seeds %s, %d columns, %d rounds x %d clients)"
          % (a.scores, sc.seed.size, sc.seeds, len(sc.columns), len(sc.rounds), sc.clients))
    checks: List[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print("  [%s] %s %s" % ("ok" if ok else "FAIL", name, detail))

    te = load_json(a.temporal)
    rel = load_json(a.released)

    t0 = time.time()
    sel_cal = select_stateful(sc, CAL, attack_start)
    sel_oracle = select_stateful(sc, (0, 1, 2, 3, 4), attack_start)
    best = sel_cal["selected"]
    print("\nselection     : CAL-only %s (worst AUC %.4f); seeds 0-4 %s (%.4f)  [%.1fs]"
          % (best, sel_cal["worst_auc"], sel_oracle["selected"], sel_oracle["worst_auc"],
             time.time() - t0))
    check("CAL-only selection picks tb_maha_pca_burnin_zabs", best == "tb_maha_pca_burnin_zabs",
          "(selected %s)" % best)
    if rel:
        rb = rel["verdict"]["best_stateful_baseline"]
        check("seeds 0-4 selection reproduces the released oracle comparator (AUC within "
              "1e-5: numpy-sensitive column)",
              sel_oracle["selected"] == rb["column"]
              and abs(sel_oracle["worst_auc"] - rb["worst_auc"]) < 1e-5,
              "(%s %.6f vs %s %.6f)" % (sel_oracle["selected"], sel_oracle["worst_auc"],
                                        rb["column"], rb["worst_auc"]))

    dets = [(best if d == BEST else d, lab) for d, lab in DETECTORS]
    missing = [d for d, _ in dets if d not in sc.columns]
    if missing:
        print("missing columns (run temporal_eval.py with --peer-aware): %s" % missing)
        return 1
    names = [d for d, _ in dets]
    labels = dict(dets)

    cal_h = sc.passes(CAL, names, attacks=False)
    thr = P.fit_thresholds({d: P.stack_scores(cal_h, d) for d in names}, TARGETS)
    if te:
        bad = [d for d in names if thr[d] != te["thresholds"][d]]
        check("CAL thresholds equal %s" % a.temporal, not bad, str(bad) if bad else "")
    if rel:
        shared = [d for d in names if d in rel["thresholds"]]
        drift = {d for d in shared if "maha" in d or "iforest" in d or d.startswith("ens_")}
        bad = [d for d in shared if d not in drift and thr[d] != rel["thresholds"][d]]
        dev = max([abs(thr[d][k] - rel["thresholds"][d][k]) / max(1.0, abs(rel["thresholds"][d][k]))
                   for d in drift for k in thr[d]] + [0.0])
        check("CAL thresholds equal the released temporal_eval.json (%d shared columns; "
              "numpy-sensitive columns within 1e-9 relative)" % len(shared),
              not bad and dev < 1e-9, "%s max relative deviation %.1e" % (bad or "", dev))

    allp = sc.passes(sc.seeds, names, honest=False)
    auc = T.auc_table(allp, sc.seeds, names, attack_start)
    auc_ex = T.auc_table(allp, sc.seeds, names, attack_start, excl_accomplice=True)
    if te:
        bad = [(x, d, s) for x in ATTACKS for d in names
               for s, v in te["auc"][x][d]["per_seed"].items()
               if not same(auc[x][d]["per_seed"].get(s), v)]
        check("per-seed AUCs equal %s" % a.temporal, not bad, str(bad[:5]) if bad else "")
    if rel:
        bad, dev = [], {}
        for x in ATTACKS:
            for d in names:
                if d not in rel["auc"][x]:
                    continue
                for s, v in rel["auc"][x][d]["per_seed"].items():
                    if v is None or same(auc[x][d]["per_seed"].get(s), v):
                        continue
                    if "maha" in d or "iforest" in d or d.startswith("ens_"):
                        dev[d] = max(dev.get(d, 0.0), abs(auc[x][d]["per_seed"][s] - v))
                    else:
                        bad.append((x, d, s))
        check("per-seed AUCs on seeds 0-4 equal the released temporal_eval.json "
              "(numpy-sensitive columns within 1e-2)",
              not bad and max(dev.values(), default=0.0) < 1e-2,
              "%s numpy-sensitive max |diff| %s" % (bad[:5] or "",
                                                    {k: "%.1e" % v for k, v in dev.items()}))

    t0 = time.time()
    frozen: Dict[str, dict] = {}
    for pool, seeds in POOLS.items():
        if not all(s in sc.seeds for s in seeds):
            continue
        frozen[pool] = {"seeds": list(seeds), "detectors": {}}
        for d in names:
            ev = evaluate_frozen(sc, seeds, d, thr[d], attack_start)
            pa, px = pool_auc(auc, d, seeds), pool_auc(auc_ex, d, seeds)
            frozen[pool]["detectors"][d] = {
                "label": labels[d], "operating_points": ev,
                "auc": pa, "auc_excl_copy_source": px,
                "worst_auc": min(v for v in pa.values() if v is not None),
                "worst_auc_attack": min((k for k in pa if pa[k] is not None), key=lambda k: pa[k]),
                "worst_auc_excl_copy_source": min(v for v in px.values() if v is not None)}
    print("frozen tables : %.1fs" % (time.time() - t0))
    if rel and "TEST" in frozen:
        bad = []
        rt = rel["results"]["TEST"]
        for d in names:
            if d not in rt:
                continue
            for t in TARGETS:
                k = f"{t:g}"
                mine = frozen["TEST"]["detectors"][d]["operating_points"][k]
                theirs = rt[d][k]
                if mine["clean_k"] != theirs["honest"]["client_round"]["k"]:
                    bad.append((d, k, "clean"))
                for x in ATTACKS:
                    if mine["per_attack"][x]["k"] != theirs["attacks"][x]["tpr_client_round"]["k"]:
                        bad.append((d, k, x))
                    if (mine["per_attack"][x]["uninvolved_k"]
                            != theirs["attacks"][x]["fpr_uninvolved_client_round"]["k"]):
                        bad.append((d, k, x, "uninvolved"))
        check("TEST (3,4) counts equal the released frozen-threshold results", not bad,
              str(bad[:5]) if bad else "")
    fs = load_json("fresh_seed_transfer.json")
    if fs and "FRESH" in frozen:
        st = frozen["FRESH"]["detectors"]["suspicion"]["operating_points"]["0.01"]
        per_seed_flags = {}
        for s in POOLS["FRESH"]:
            per_seed_flags[str(s)] = int(flags(sc.mat("suspicion", s, None, None),
                                               thr["suspicion"]["0.01"]).sum())
        ok = all(per_seed_flags[s] == fs["honest"]["per_seed"][s]["flags"]
                 for s in per_seed_flags)
        check("seeds 5-7 clean flags equal fresh_seed_transfer.json", ok,
              "(%s vs %s)" % (per_seed_flags, {s: v["flags"] for s, v in
                                               fs["honest"]["per_seed"].items()}))
        ok2 = all(st["per_attack"][x]["k"] == fs["tpr"][x]["k"] for x in ATTACKS
                  if x != "noise")
        check("seeds 5-7 detection counts equal fresh_seed_transfer.json (noise uses a "
              "different generator seed there)", ok2,
              "(noise %d vs %d)" % (st["per_attack"]["noise"]["k"], fs["tpr"]["noise"]["k"]))
    ho = load_json("held_out_summary.json")
    if ho and "HELDOUT" in frozen:
        st = frozen["HELDOUT"]["detectors"]["suspicion"]["operating_points"]["0.01"]
        check("seeds 3-7 clean flags equal held_out_summary.json",
              st["clean_k"] == ho["honest"]["flags"] and st["clean_n"] == ho["honest"]["rounds"],
              "(%d/%d)" % (st["clean_k"], st["clean_n"]))

    t0 = time.time()
    tb_cols = [c for c in sc.columns if c.startswith("tb_")]
    cal_tb = sc.passes(CAL, tb_cols, attacks=False)
    thr_tb = P.fit_thresholds({d: P.stack_scores(cal_tb, d) for d in tb_cols}, TARGETS)
    singles: Dict[str, dict] = {}
    bad = []
    for pool in ("TEST", "HELDOUT"):
        if pool not in frozen:
            continue
        per_col = {d: evaluate_frozen(sc, POOLS[pool], d, thr_tb[d], attack_start)
                   for d in tb_cols}
        rec = {"outputs": len(tb_cols), "nominal": {}}
        for t in TARGETS:
            k = f"{t:g}"
            top = min(tb_cols, key=lambda d: (-per_col[d][k]["worst_tpr"], d))
            rec["nominal"][k] = {
                "max_worst_tpr": per_col[top][k]["worst_tpr"], "argmax": top,
                "argmax_clean_fpr": per_col[top][k]["clean_fpr"],
                "worst_tpr": {d: per_col[d][k]["worst_tpr"] for d in tb_cols},
                "clean_fpr": {d: per_col[d][k]["clean_fpr"] for d in tb_cols}}
            for d in names:
                if d in per_col:
                    mine = frozen[pool]["detectors"][d]["operating_points"][k]
                    if (per_col[d][k]["worst_tpr"] != mine["worst_tpr"]
                            or per_col[d][k]["clean_fpr"] != mine["clean_fpr"]):
                        bad.append((pool, d, k))
        singles[pool] = rec
    check("single-output tables equal the frozen tables on the shared columns", not bad,
          str(bad[:5]) if bad else "")
    print("single outputs: %.1fs (%d outputs)" % (time.time() - t0, len(tb_cols)))

    t0 = time.time()
    rocs = {pool: {d: roc_curves(sc, POOLS[pool], d) for d in names}
            for pool in ("TEST", "HELDOUT") if pool in frozen}
    print("ROC curves    : %.1fs" % (time.time() - t0))

    t0 = time.time()
    seat = {d: {s: seat_statistics(sc, s, d, thr[d]) for s in sc.seeds if s not in CAL}
            for d in names}
    worst_dev, undefined, mismatched = 0.0, 0, []
    for d in names:
        for s in seat[d]:
            for x in ATTACKS:
                for key, table in (("U", auc), ("Ux", auc_ex)):
                    U, npos, nneg = seat[d][s][key][x]
                    den = npos.sum() * nneg.sum()
                    v = U.sum() / den if den > 0 else float("nan")
                    ref = table[x][d]["per_seed"][str(s)]
                    ref = float("nan") if ref is None else float(ref)
                    if math.isnan(v) or math.isnan(ref):
                        if math.isnan(v) and math.isnan(ref):
                            undefined += 1
                        else:
                            mismatched.append((d, s, x, key))
                        continue
                    worst_dev = max(worst_dev, abs(v - ref))
    check("seat decomposition reproduces every per-seed AUC",
          worst_dev < 1e-12 and not mismatched,
          "(max |diff| %.1e; %d cells undefined in both; mismatched %s)"
          % (worst_dev, undefined, mismatched[:5]))
    print("seat stats    : %.1fs" % (time.time() - t0))

    t0 = time.time()
    rng = np.random.default_rng(a.boot_seed)
    boot = {}
    for pool, schemes in (("HELDOUT", ("seed", "seat", "two_stage")), ("TEST", ("seat",))):
        if pool not in frozen:
            continue
        seeds = POOLS[pool]
        boot[pool] = {}
        for scheme in schemes:
            boot[pool][scheme] = bootstrap({d: {s: seat[d][s] for s in seeds} for d in names},
                                           seeds, scheme, a.boot, rng)
    print("bootstrap     : %.1fs (B=%d)" % (time.time() - t0, a.boot))

    traj = {pool: {d: trajectory_tables(seat[d], list(POOLS[pool])) for d in names}
            for pool in frozen}
    bad = []
    for pool in traj:
        for d in names:
            for k, rec in traj[pool][d].items():
                op = frozen[pool]["detectors"][d]["operating_points"][k]
                if rec["clean"]["flags_total"] != op["clean_k"]:
                    bad.append((pool, d, k, "clean"))
                for x in ATTACKS:
                    if rec["uninvolved"][x]["flags_total"] != op["per_attack"][x]["uninvolved_k"]:
                        bad.append((pool, d, k, x, "uninvolved"))
                for x, cs in rec["copy_source"].items():
                    if abs(cs["flags_total"] / cs["rounds_total"]
                           - op["per_attack"][x]["flag_rate_copy_source"]) > 1e-12:
                        bad.append((pool, d, k, x, "copy source"))
    check("trajectory flag totals equal the frozen client-round counts (clean, uninvolved, "
          "copy source)", not bad, str(bad[:5]) if bad else "")
    bad = [(pool, d, t, k, x) for pool in traj for d in names
           for t, rec in traj[pool][d].items()
           for k, rule in rec["k_of_24_rules_post_hoc"].items()
           for x, r in rule["attacked"].items()
           if r["trajectories"] - r["detected"] != r["round_of_kth_flag"]["not_reached"]
           or r["round_of_kth_flag"]["at_earliest_possible_round"] > r["detected"]]
    check("k-th flag rounds agree with the k-of-24 trajectory counts", not bad,
          str(bad[:5]) if bad else "")

    nostruct = {}
    for s in sorted(seat["abl_full_nostruct"]):
        stn = seat["abl_full_nostruct"][s]
        nostruct[str(s)] = {
            "auc_global_echo": auc["global_echo"]["abl_full_nostruct"]["per_seed"][str(s)],
            "auc_by_attack": {x: auc[x]["abl_full_nostruct"]["per_seed"][str(s)] for x in ATTACKS},
            "global_echo_detected_rounds": {k: int(stn["victim"][k]["global_echo"].sum())
                                            for k in stn["victim"]},
            "global_echo_rounds": int(sc.clients * stn["R"]),
            "global_echo_trajectories_flagged": {
                k: int((stn["victim"][k]["global_echo"] >= 1).sum()) for k in stn["victim"]},
            "clean_flags": {k: int(stn["clean"][k].sum()) for k in stn["clean"]},
            "auditor_clean_flags": {k: int(seat["suspicion"][s]["clean"][k].sum())
                                    for k in stn["clean"]},
        }

    def pct(x):
        return "  n/a " if x is None else "%5.1f%%" % (100 * x)

    for pool in ("TEST", "HELDOUT", "FRESH", "DEV"):
        if pool not in frozen:
            continue
        print("\n" + "=" * 132)
        print("%s seeds %s: worst-case detection [clean false alarms] at nominal 10 / 5 / 1 / "
              "0.5%%, worst AUC (copy source excluded)" % (pool, POOLS[pool]))
        print("=" * 132)
        for d in names:
            r = frozen[pool]["detectors"][d]
            cells = []
            for t in TARGETS:
                o = r["operating_points"][f"{t:g}"]
                cells.append("%s [%s]" % (pct(o["worst_tpr"]), pct(o["clean_fpr"]).strip()))
            print("  %-46s %s   %.3f (%.3f)" % (labels[d][:46], "  ".join(cells),
                                               r["worst_auc"], r["worst_auc_excl_copy_source"]))
    for pool, rec in singles.items():
        print("\n%s: best of the %d single stateful outputs at frozen thresholds, worst-case "
              "detection [clean false alarms]: %s" % (pool, rec["outputs"], "  ".join(
                  "%s %s [%s] %s" % (k, pct(v["max_worst_tpr"]), pct(v["argmax_clean_fpr"]).strip(),
                                     v["argmax"]) for k, v in rec["nominal"].items())))
    print("\nper-attack detection at nominal 1%, five held-out seeds:")
    if "HELDOUT" in frozen:
        print("  %-46s" % "" + "".join("%13s" % x[:12] for x in ATTACKS) + "   uninvolved")
        for d in names:
            o = frozen["HELDOUT"]["detectors"][d]["operating_points"]["0.01"]
            print("  %-46s" % labels[d][:46]
                  + "".join("%12.1f%%" % (100 * o["per_attack"][x]["tpr"]) for x in ATTACKS)
                  + "   %.2f%%" % (100 * o["uninvolved_fpr"]))
    if "HELDOUT" in boot:
        print("\nbootstrap 95%% intervals, five held-out seeds, worst-case detection at 1%% and "
              "worst AUC:")
        for d in names:
            cells = []
            for sch in ("seed", "seat", "two_stage"):
                b1 = boot["HELDOUT"][sch][d]["worst_tpr@0.01"]
                b2 = boot["HELDOUT"][sch][d]["worst_auc"]
                cells.append("%s %.3f[%.3f,%.3f] auc[%.3f,%.3f]"
                             % (sch[:4], b1["point"], b1["lo"], b1["hi"], b2["lo"], b2["hi"]))
            print("  %-34s %s" % (labels[d][:34], " | ".join(cells)))
    if "HELDOUT" in traj:
        print("\nk-of-24 trajectory rules at nominal 1%, five held-out seeds (post hoc): "
              "attacked / clean / honest trajectories of attacked runs flagged")
        for d in names:
            cells = []
            for k, rule in traj["HELDOUT"][d]["0.01"]["k_of_24_rules_post_hoc"].items():
                aa = rule["attacked_all_attacks"]
                cells.append("k>=%s %d/%d (earliest %d) %d/%d %d/%d" % (
                    k, aa["detected"], aa["trajectories"],
                    aa["kth_flag_at_earliest_possible_round"], rule["clean_trajectories_flagged"],
                    rule["clean_trajectories"], rule["honest_trajectories_in_attacked_runs_flagged"],
                    rule["honest_trajectories_in_attacked_runs"]))
            print("  %-34s %s" % (labels[d][:34], " | ".join(cells)))

    from passportfl.reproducibility import environment
    out = {
        "config": {k: T.portable_path(v) for k, v in vars(a).items()},
        "command": " ".join(sys.argv),
        "environment": environment(),
        "inputs": {"scores": {"file": os.path.basename(a.scores),
                              "sha256": sha256_file(a.scores),
                              "meta": sc.meta},
                   "temporal": a.temporal, "released": a.released},
        "detectors": [{"column": d, "label": labels[d]} for d in names],
        "thresholds": {d: thr[d] for d in names},
        "selection_best_single_stateful": {"cal_only": sel_cal, "seeds_0_4_oracle": sel_oracle},
        "frozen_threshold": frozen,
        "roc": {"nominal_fpr_grid": FPR_GRID.tolist(), "pools": rocs},
        "bootstrap": {"B": a.boot, "rng_seed": a.boot_seed, "pools": boot},
        "trajectories": traj,
        "auditor_without_structural_tier": nostruct,
        "single_stateful_outputs": singles,
        "consistency_checks": checks,
        "all_checks_ok": all(c["ok"] for c in checks),
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\nchecks: %d/%d ok" % (sum(c["ok"] for c in checks), len(checks)))
    print("wrote %s   (total %.1fs)" % (a.out, time.time() - t_all))
    return 0 if out["all_checks_ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
