from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

import numpy as np

from passportfl import identity as idm
from passportfl.identity import LearningRecord
from passportfl.probe import (MerkleTree, ProbeReveal, check_reveals, leaf_hash, mean_separation,
                              nearest_prototype, prototype_drift_ratio, select_subset)
from passportfl.protocol import Contribution, GenesisCert, Receiver, VerifyResult
from passportfl.reproducibility import environment

from negative_results import DEFAULT_ENTROPY, ghost_client, seed_bytes
from protocol_eval import clopper_pearson

REPO = os.path.normpath(os.path.join(_HERE, ".."))

K, D, M, T = 100, 128, 256, 32
DELTA, MAX_RATIO, MIN_STAB = 1.0, 1.0, 0.5
STATS = ("proto_ratio", "label_stability", "max_drift")
LOG_KEY = {"proto_ratio": "proto_ratio", "label_stability": "stability", "max_drift": "max_drift"}
BAND_PCT = (1.0, 99.0)
BEACON_DOMAIN = b"passportfl/ghost-beacon/v1|"

ORIG_SEED, ORIG_STEP = 0, 0.10
ARCH = idm.canonical_json({"family": "PassportResNet18", "embed_dim": D, "num_classes": K,
                           "passport_in_channels": [64, 128, 128, 128, 128, 256, 256, 256,
                                                    256, 512, 512, 512]})

SEARCH_STAGES = (
    ("coarse", {"s_c": (0.155,), "tau": (0.5, 1.0, 1.5, 2.0, 3.0), "beta": (0.0, 0.5, 0.85),
                "s_z": (0.0, 0.10)}),
    ("fine", {"s_c": (0.15, 0.16, 0.17, 0.18), "tau": (2.4, 2.8, 3.2, 3.6),
              "beta": (0.3, 0.4, 0.5, 0.6), "s_z": (0.0, 0.05)}),
)
CAL_SEED = 0


def jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bytes):
        return o.hex()
    raise TypeError(type(o))


def pct(v: Sequence[float]) -> dict:
    v = np.asarray(v, float)
    q = np.percentile(v, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {"n": int(len(v)), "min": q[0], "p1": q[1], "p5": q[2], "p25": q[3], "median": q[4],
            "p75": q[5], "p95": q[6], "p99": q[7], "max": q[8], "mean": float(v.mean())}


@dataclass
class Trajectory:
    Z0: np.ndarray
    C: List[np.ndarray] = field(default_factory=list)
    Z: List[np.ndarray] = field(default_factory=list)
    records: List[LearningRecord] = field(default_factory=list)
    params: dict = field(default_factory=dict)


def _record(r: int, C: np.ndarray, C_prev: Optional[np.ndarray]) -> LearningRecord:
    drift = 0.0 if C_prev is None else float(np.linalg.norm(C - C_prev, axis=1).mean())
    return LearningRecord(round=r, n_samples=5000, train_loss=2.0 / (1 + 0.1 * r), local_acc=0.4,
                          proto_drift=drift, intra_compactness=2.0, inter_separation=4.0)


def original_trajectory(rounds: int, seed: int = ORIG_SEED, step: float = ORIG_STEP) -> Trajectory:
    rng = np.random.default_rng(seed)
    Cs = rng.normal(0, 1.0, (K, D)) * 3.0
    Zs = rng.normal(0, 1.0, (M, D)) * 3.0
    vC = rng.normal(0, 1, (K, D))
    vZ = rng.normal(0, 1, (M, D))
    tr = Trajectory(Z0=idm.dequantize(Zs), params={"seed": seed, "step": step})
    C_prev = None
    for r in range(1, rounds + 1):
        d = step / (1 + 0.06 * r)
        vC = 0.85 * vC + 0.15 * rng.normal(0, 1, (K, D))
        vZ = 0.85 * vZ + 0.15 * rng.normal(0, 1, (M, D))
        Cs = Cs + d * vC / np.linalg.norm(vC, axis=1, keepdims=True)
        Zs = Zs + d * vZ / np.linalg.norm(vZ, axis=1, keepdims=True)
        C, Z = idm.dequantize(Cs), idm.dequantize(Zs)
        tr.C.append(C); tr.Z.append(Z); tr.records.append(_record(r, C, C_prev))
        C_prev = C
    return tr


def calibrated_trajectory(rounds: int, seed: int, s_c: float, tau: float, beta: float,
                          s_z: float) -> Trajectory:
    rng = np.random.default_rng(seed)
    C = rng.normal(0, 1.0, (K, D)) * 3.0
    sep0 = mean_separation(C)
    a = rng.integers(0, K, M)
    U = rng.normal(0, 1.0, (M, D))
    U *= tau * sep0 / np.linalg.norm(U, axis=1, keepdims=True)
    vC = rng.normal(0, 1, (K, D))
    vU = rng.normal(0, 1, (M, D))
    tr = Trajectory(Z0=idm.dequantize(C[a] + U),
                    params={"seed": seed, "s_c": s_c, "tau": tau, "beta": beta, "s_z": s_z})
    C_prev = None
    for r in range(1, rounds + 1):
        vC = beta * vC + (1 - beta) * rng.normal(0, 1, (K, D))
        vU = beta * vU + (1 - beta) * rng.normal(0, 1, (M, D))
        C = C + s_c * sep0 * vC / np.linalg.norm(vC, axis=1, keepdims=True)
        mu = C.mean(0)
        C = mu + (C - mu) * (sep0 / mean_separation(C))
        U = U + s_z * sep0 * vU / np.linalg.norm(vU, axis=1, keepdims=True)
        U *= tau * sep0 / np.linalg.norm(U, axis=1, keepdims=True)
        Cq, Zq = idm.dequantize(C), idm.dequantize(C[a] + U)
        tr.C.append(Cq); tr.Z.append(Zq); tr.records.append(_record(r, Cq, C_prev))
        C_prev = Cq
    return tr


class GhostProver:
    def __init__(self, tr: Trajectory, scheme: str, key_seed: bytes):
        self.tr = tr
        self.signer = idm.SCHEMES[scheme](key_seed)
        pk = self.signer.public_key
        F0 = MerkleTree([leaf_hash(e) for e in tr.Z0]).root
        self.cert = GenesisCert(idm.genesis_id(ARCH, pk, F0), pk, ARCH, F0, scheme)
        self.E = idm.GENESIS_E
        self.round = 0
        self._tree: Optional[MerkleTree] = None
        self._Z: Optional[np.ndarray] = None

    def commit(self) -> Contribution:
        self.round += 1
        r = self.round
        C, Z, rec = self.tr.C[r - 1], self.tr.Z[r - 1], self.tr.records[r - 1]
        self._tree, self._Z = MerkleTree([leaf_hash(e) for e in Z]), Z
        F = self._tree.root
        KC, Rd = idm.hash_prototypes(C), rec.digest()
        sig = self.signer.sign(idm.signed_message(r, self.E, KC, Rd, F))
        En = idm.evolving_id(r, self.E, KC, Rd, F, sig)
        c = Contribution(self.cert.G, r, self.E, En, C, rec, F, [], sig,
                         genesis=self.cert if r == 1 else None)
        self.E = En
        return c

    def open(self, idxs: Sequence[int]) -> List[ProbeReveal]:
        return [ProbeReveal(i, self._Z[i], self._tree.proof(i)) for i in idxs]


class ChainHeadChallenge:
    name = "chain_head"

    def seed(self, c: Contribution) -> bytes:
        return c.E_prev


class BeaconChallenge:
    name = "beacon"

    def __init__(self, key: bytes):
        self.key = key

    def seed(self, c: Contribution) -> bytes:
        return idm.H(BEACON_DOMAIN + self.key + c.round.to_bytes(4, "little"))


class ChallengeReceiver(Receiver):
    def __init__(self, warmup_rounds: int, bands: Optional[Dict[str, Tuple[float, float]]] = None):
        super().__init__(M, T, DELTA, MAX_RATIO, MIN_STAB, warmup_rounds)
        self.bands = bands
        self.last: dict = {}

    def verify(self, c: Contribution, challenge_seed: Optional[bytes] = None) -> VerifyResult:
        self.last = {}
        if c.client_G not in self.peers:
            if c.genesis is None or c.genesis.G != c.client_G:
                return VerifyResult(False, "unknown peer and no genesis certificate")
            err = self._register(c.genesis)
            if err:
                return VerifyResult(False, err)
        st = self.peers[c.client_G]
        if c.round != st.round + 1:
            return VerifyResult(False, f"round mismatch: expected {st.round + 1}, got {c.round}")
        if c.E_prev != st.E:
            return VerifyResult(False, "E_prev does not match stored chain head")
        KC = idm.hash_prototypes(c.C)
        Rd = c.record.digest()
        if c.record.round != c.round:
            return VerifyResult(False, "record round mismatch")
        msg = idm.signed_message(c.round, c.E_prev, KC, Rd, c.F)
        if not idm.SCHEMES[st.cert.scheme].verify(st.cert.pk, msg, c.sig):
            return VerifyResult(False, "signature invalid")
        if idm.evolving_id(c.round, c.E_prev, KC, Rd, c.F, c.sig) != c.E_new:
            return VerifyResult(False, "E_new does not recompute")
        st.E = c.E_new
        st.round = c.round
        scale = mean_separation(c.C) + 1e-12
        rates = [float(np.linalg.norm(r.embedding - st.probe_history[r.index][1])) / scale
                 / max(1, c.round - st.probe_history[r.index][0])
                 for r in c.reveals if r.index in st.probe_history]
        defined = {"proto_ratio": st.C_prev is not None,
                   "label_stability": st.C_prev is not None and bool(st.probe_history),
                   "max_drift": bool(rates)}
        seed = c.E_prev if challenge_seed is None else challenge_seed
        pr = check_reveals(c.F, c.reveals, c.C, st.C_prev, st.C_prev_round, seed, self.m, self.t,
                           st.probe_history, c.round, self.delta, self.max_ratio, self.min_stab,
                           self.warmup)
        if pr.merkle_ok and pr.subset_ok:
            st.C_prev = c.C
            st.C_prev_round = c.round
        values = {"proto_ratio": pr.proto_drift_ratio, "label_stability": pr.label_stability,
                  "max_drift": pr.max_drift}
        band_fail = []
        if self.bands is not None and c.round > self.warmup:
            band_fail = [s for s in STATS if defined[s]
                         and not (self.bands[s][0] <= values[s] <= self.bands[s][1])]
        mean_rate = (pr.probe_velocity * pr.proto_drift_ratio / pr.proto_velocity
                     if pr.proto_velocity > 0 and pr.n_velocity_probes else None)
        self.last = {"max_drift_rate": max(rates) if rates else None, "mean_drift_rate": mean_rate,
                     "n_opened_with_history": len(rates), "band_fail": band_fail}
        why = [k for k, v in (("merkle", pr.merkle_ok), ("subset", pr.subset_ok), ("proto", pr.proto_ok),
                              ("stability", pr.stability_ok), ("drift", pr.drift_ok)) if not v]
        if why:
            return VerifyResult(False, "probe check failed: " + ",".join(why), True, pr)
        if band_fail:
            return VerifyResult(False, "two-sided band failed: " + ",".join(band_fail), True, pr)
        return VerifyResult(True, "ok", True, pr)


def verifier_set(bands: Dict[str, Tuple[float, float]]) -> Dict[str, ChallengeReceiver]:
    return {"deployed_w5": ChallengeReceiver(5), "deployed_w0": ChallengeReceiver(0),
            "two_sided_w5": ChallengeReceiver(5, bands), "two_sided_w0": ChallengeReceiver(0, bands)}


def run_interaction(prover: GhostProver, challenge, verifiers: Dict[str, ChallengeReceiver],
                    rounds: int, opened: Optional[list] = None) -> List[dict]:
    log = []
    for _ in range(rounds):
        c = prover.commit()
        seed = challenge.seed(c)
        idxs = select_subset(seed, M, T)
        if opened is not None:
            opened.append(idxs)
        c.reveals = prover.open(idxs)
        row, seen = {"round": c.round, "stat_mismatch_across_verifiers": 0}, None
        for name, v in verifiers.items():
            res = v.verify(c, seed)
            p = res.probe
            row[name] = {"accepted": bool(res.accepted), "reason": res.reason}
            if p is None:
                continue
            stats = (p.proto_drift_ratio, p.label_stability, p.max_drift, v.last["max_drift_rate"])
            if seen is None:
                seen = stats
                row.update({"proto_ratio": p.proto_drift_ratio, "label_stability": p.label_stability,
                            "max_drift": p.max_drift, **{k: v.last[k] for k in
                                                          ("max_drift_rate", "mean_drift_rate",
                                                           "n_opened_with_history")}})
            elif stats != seen:
                row["stat_mismatch_across_verifiers"] += 1
        log.append(row)
    return log


def summarise_run(log: List[dict], verifiers: Sequence[str], warmup_for_medians: int = 5) -> dict:
    post = [r for r in log if r["round"] > warmup_for_medians]
    out = {"rounds": len(log),
           "stat_mismatch_across_verifiers": int(sum(r["stat_mismatch_across_verifiers"] for r in log)),
           "accepted": {v: int(sum(r[v]["accepted"] for r in log)) for v in verifiers},
           "rejected_rounds": {v: [r["round"] for r in log if not r[v]["accepted"]] for v in verifiers},
           "medians_rounds_gt_%d" % warmup_for_medians: {
               s: float(np.median([r[s] for r in post if r.get(s) is not None]))
               for s in STATS + ("max_drift_rate", "mean_drift_rate")}}
    return out


def max_flag_fraction(last: np.ndarray, old: np.ndarray, t: int) -> float:
    n10 = int(np.sum(last & ~old)); n11 = int(np.sum(last & old))
    n00 = int(np.sum(~last & ~old)); n01 = int(np.sum(~last & old))
    a10 = min(t, n10); a11 = min(t - a10, n11); a00 = min(t - a10 - a11, n00)
    a01 = t - a10 - a11 - a00
    older = (n11 - a11) + (n01 - a01)
    return (a10 + a11 + older) / float(t + older)


def _brute_max_flag_fraction(last: np.ndarray, old: np.ndarray, t: int) -> float:
    n, best = len(last), 0.0
    for A in itertools.combinations(range(n), t):
        rest = [i for i in range(n) if i not in A]
        for k in range(len(rest) + 1):
            for B in itertools.combinations(rest, k):
                num = sum(bool(last[i]) for i in A) + sum(bool(old[i]) for i in B)
                best = max(best, num / float(t + k))
    return best


def universal_check(tr: Trajectory) -> List[dict]:
    R = len(tr.C)
    seps = [mean_separation(C) + 1e-12 for C in tr.C]
    nn = {q: {} for q in range(1, R + 1)}
    for q in range(1, R + 1):
        for s in range(1, q):
            nn[q][s] = nearest_prototype(tr.Z[s - 1], tr.C[q - 1])
    rows = []
    for r in range(2, R + 1):
        Zr = tr.Z[r - 1]
        dist = np.stack([np.linalg.norm(Zr - tr.Z[r0 - 1], axis=1) / seps[r - 1]
                         for r0 in range(1, r)])
        el = (r - np.arange(1, r))[:, None].astype(float)
        unstable = {s: nn[r][s] != (nn[r - 1][s] if s < r - 1 else
                                    nearest_prototype(tr.Z[s - 1], tr.C[r - 2]))
                    for s in range(1, r)}
        last_u = unstable[r - 1]
        old_u = (np.any(np.stack([unstable[s] for s in range(1, r - 1)]), axis=0)
                 if r > 2 else np.zeros(M, bool))
        last_s, old_s = ~last_u, (np.any(np.stack([~unstable[s] for s in range(1, r - 1)]), axis=0)
                                  if r > 2 else np.zeros(M, bool))
        rows.append({"round": r,
                     "worst_drift_rate": float((dist / el).max()),
                     "max_drift_ub": float(dist.max()),
                     "proto_ratio": prototype_drift_ratio(tr.C[r - 1], tr.C[r - 2]),
                     "proto_ratio_prev_sep_only": float(
                         np.linalg.norm(tr.C[r - 1] - tr.C[r - 2], axis=1).mean()
                         / (mean_separation(tr.C[r - 2]) + 1e-12)),
                     "stability_lb": 1.0 - max_flag_fraction(last_u, old_u, T),
                     "stability_ub": max_flag_fraction(last_s, old_s, T)})
    return rows


class HistoryReplay:
    def __init__(self, tr: Trajectory):
        R = len(tr.C)
        self.R = R
        self.sep = np.array([0.0] + [mean_separation(C) + 1e-12 for C in tr.C])
        self.ratio = np.array([0.0, 0.0] + [prototype_drift_ratio(tr.C[r - 1], tr.C[r - 2])
                                            for r in range(2, R + 1)])
        self.dist = np.zeros((R + 1, R + 1, M))
        self.unst = np.zeros((R + 1, R + 1, M), bool)
        for r in range(2, R + 1):
            for s in range(1, r):
                self.dist[r, s] = [float(np.linalg.norm(tr.Z[r - 1][i] - tr.Z[s - 1][i])) / self.sep[r]
                                   for i in range(M)]
                self.unst[r, s] = (nearest_prototype(tr.Z[s - 1], tr.C[r - 1])
                                   != nearest_prototype(tr.Z[s - 1], tr.C[r - 2]))

    def run(self, subsets) -> Dict[str, np.ndarray]:
        out = {k: [] for k in ("proto_ratio", "label_stability", "max_drift", "max_drift_rate",
                               "drift_ok", "has_drift")}
        last = None
        for r in range(1, self.R + 1):
            S = subsets(r)
            N = len(S)
            if last is None:
                last = np.zeros((N, M), np.int64)
            ls = np.take_along_axis(last, S, 1)
            has = ls > 0
            el = np.maximum(1, r - ls)
            d = np.where(has, self.dist[r][ls, S], 0.0)
            seen = last > 0
            n_seen = seen.sum(1)
            flips = (self.unst[r][last, np.arange(M)[None, :]] & seen).sum(1)
            stab = np.where(n_seen > 0, (n_seen - flips) / np.maximum(1, n_seen), 1.0) if r > 1 else np.ones(N)
            out["proto_ratio"].append(np.full(N, self.ratio[r]))
            out["label_stability"].append(stab)
            out["max_drift"].append(d.max(1))
            out["max_drift_rate"].append(np.where(has, d / el, 0.0).max(1))
            out["drift_ok"].append(np.all(~has | (d <= DELTA * el), axis=1))
            out["has_drift"].append(has.any(1))
            np.put_along_axis(last, S, r, 1)
        return {k: np.stack(v, 1) for k, v in out.items()}

    @staticmethod
    def verdicts(st: Dict[str, np.ndarray], warmup: int,
                 bands: Optional[Dict[str, Tuple[float, float]]]) -> np.ndarray:
        N, R = st["proto_ratio"].shape
        rounds = np.arange(1, R + 1)[None, :]
        warm = rounds <= warmup
        ok = warm | ((st["proto_ratio"] <= MAX_RATIO) & (st["label_stability"] >= MIN_STAB) & st["drift_ok"])
        if bands is not None:
            defined = {"proto_ratio": rounds > 1, "label_stability": rounds > 1, "max_drift": st["has_drift"]}
            for s in STATS:
                lo, hi = bands[s]
                ok &= warm | ~defined[s] | ((st[s] >= lo) & (st[s] <= hi))
        return ok

    def monte_carlo(self, n_seq: int, seed: int, bands: Dict[str, Tuple[float, float]]) -> dict:
        rng = np.random.default_rng(seed)
        st = self.run(lambda r: np.argsort(rng.random((n_seq, M)), axis=1)[:, :T])
        out = {"sequences": n_seq, "rng_seed": seed}
        for name, b in (("deployed", None), ("two_sided", bands)):
            for w in (5, 0):
                ok = self.verdicts(st, w, b)
                full = int(ok.all(1).sum())
                _, lo, hi = clopper_pearson(n_seq - full, n_seq)
                out["%s_w%d" % (name, w)] = {
                    "sequences_all_rounds_accepted": full,
                    "mean_accepted_rounds": float(ok.sum(1).mean()),
                    "min_accepted_rounds": int(ok.sum(1).min()),
                    "round_rejection_rate": float(1 - ok.mean()),
                    "max_per_round_rejection_rate": float(1 - ok.mean(0).min()),
                    "seq_rejection_rate_cp95": [lo, hi]}
        post = {s: st[s][:, 5:] for s in STATS}
        out["medians_rounds_gt_5_pooled"] = {s: float(np.median(v)) for s, v in post.items()}
        out["stability_rounds_gt_5_pooled"] = pct(post["label_stability"].ravel())
        return out


def summarise_universal(rows: List[dict], bands: Dict[str, Tuple[float, float]]) -> dict:
    out = {}
    for w in (0, 5):
        sel = [x for x in rows if x["round"] > w]
        ok = [x["worst_drift_rate"] <= DELTA and x["proto_ratio"] <= MAX_RATIO
              and x["stability_lb"] >= MIN_STAB for x in sel]
        out["warmup_%d" % w] = {
            "rounds_checked": len(sel),
            "rounds_passing_every_admissible_challenge": int(sum(ok)),
            "max_worst_drift_rate": max(x["worst_drift_rate"] for x in sel),
            "max_proto_ratio": max(x["proto_ratio"] for x in sel),
            "min_stability_lb": min(x["stability_lb"] for x in sel),
            "max_max_drift_ub": max(x["max_drift_ub"] for x in sel),
            "rounds_ratio_in_band": int(sum(bands["proto_ratio"][0] <= x["proto_ratio"]
                                            <= bands["proto_ratio"][1] for x in sel)),
            "rounds_stability_interval_inside_band": int(sum(
                x["stability_lb"] >= bands["label_stability"][0]
                and x["stability_ub"] <= bands["label_stability"][1] for x in sel)),
        }
    return out


def honest_reference(path: str) -> dict:
    fp = json.load(open(path, encoding="utf-8"))
    warm = int(fp["config"]["warmup"])
    log = fp["verify_log"]
    rows = [e for e in log if int(e["round"]) > warm]
    bands = {s: tuple(float(x) for x in np.percentile([e[LOG_KEY[s]] for e in rows], BAND_PCT))
             for s in STATS}

    def in_band(e):
        return [s for s in STATS if not bands[s][0] <= e[LOG_KEY[s]] <= bands[s][1]]

    dep = [bool(e["accepted"]) for e in rows]
    two = [bool(e["accepted"]) and not in_band(e) for e in rows]
    early = [e for e in log if 5 < int(e["round"]) <= warm]
    clients = sorted({e["G"] for e in rows})
    per_client = {g: [(bool(e["accepted"]), bool(e["accepted"]) and not in_band(e))
                      for e in rows if e["G"] == g] for g in clients}
    return {
        "source": os.path.relpath(path, REPO).replace("\\", "/"),
        "config": {k: fp["config"][k] for k in ("clients", "rounds", "dirichlet", "warmup", "scheme",
                                                "m_probes", "t_reveal", "delta", "max_proto_ratio",
                                                "min_label_stability", "seed")},
        "n_rows": len(rows),
        "distributions": {s: pct([e[LOG_KEY[s]] for e in rows]) for s in STATS},
        "bands_p1_p99": {s: list(b) for s, b in bands.items()},
        "acceptance": {
            "deployed_logged": int(sum(dep)),
            "deployed_rejections": [{"round": e["round"], "G": e["G"], "reason": e["reason"]}
                                    for e in rows if not e["accepted"]],
            "two_sided": int(sum(two)),
            "outside_band_by_statistic": {s: int(sum(s in in_band(e) for e in rows)) for s in STATS},
            "clients": len(clients),
            "clients_all_post_warmup_rounds_accepted": {
                "deployed": int(sum(all(x[0] for x in v) for v in per_client.values())),
                "two_sided": int(sum(all(x[1] for x in v) for v in per_client.values()))},
        },
        "rounds_6_to_warmup_inside_all_bands": {
            "n_rows": len(early), "inside": int(sum(not in_band(e) for e in early))},
    }, bands


def supplementary_mean_rate(paths: Sequence[str], after_round: int = 28) -> dict:
    rows = []
    for p in paths:
        fp = json.load(open(p, encoding="utf-8"))
        rows += [e for e in fp["verify_log"] if int(e["round"]) > after_round and e.get("proto_velocity")]
    rate = [e["probe_velocity"] * e["proto_ratio"] / e["proto_velocity"] for e in rows]
    return {"sources": [os.path.relpath(p, REPO).replace("\\", "/") for p in paths],
            "mean_drift_rate": pct(rate)}


def evaluate_generator(args: Tuple[dict, int, str, int]) -> dict:
    params, rounds, entropy_hex, seed = args
    tr = calibrated_trajectory(rounds, seed, **params)
    prover = GhostProver(tr, "Ed25519", seed_bytes(bytes.fromhex(entropy_hex), "n3-signer"))
    v = {"deployed_w5": ChallengeReceiver(5)}
    log = run_interaction(prover, ChainHeadChallenge(), v, rounds)
    s = summarise_run(log, list(v))
    return {"params": params, "accepted": s["accepted"]["deployed_w5"],
            "medians": s["medians_rounds_gt_5"]}


def objective(med: dict, target: dict) -> float:
    return max(abs(med[s] / target[s] - 1.0) for s in STATS)


def calibration_search(target: dict, rounds: int, entropy_hex: str, workers: int) -> dict:
    jobs, stage_of = [], []
    for stage, grid in SEARCH_STAGES:
        keys = list(grid)
        for vals in itertools.product(*(grid[k] for k in keys)):
            jobs.append((dict(zip(keys, vals)), rounds, entropy_hex, CAL_SEED))
            stage_of.append(stage)
    with ProcessPoolExecutor(max(1, workers)) as ex:
        res = list(ex.map(evaluate_generator, jobs))
    for r, stage in zip(res, stage_of):
        r["stage"] = stage
        r["objective"] = objective(r["medians"], target)
    best = min(range(len(res)), key=lambda i: (res[i]["objective"], i))
    return {"stages": [{"name": s, "grid": {k: list(v) for k, v in g.items()},
                        "n_points": int(np.prod([len(v) for v in g.values()]))} for s, g in SEARCH_STAGES],
            "trajectory_seed": CAL_SEED, "rounds": rounds, "signature_scheme": "Ed25519",
            "target_medians": target,
            "points_within_5pct": int(sum(r["objective"] <= 0.05 for r in res)),
            "selected": res[best], "points": res}


def ghost_job(args: Tuple[str, dict, str, str, Optional[int], dict, int]) -> dict:
    kind, params, scheme, entropy_hex, beacon_idx, bands, rounds = args
    entropy = bytes.fromhex(entropy_hex)
    tr = (original_trajectory(rounds, **params) if kind == "original"
          else calibrated_trajectory(rounds, **params))
    prover = GhostProver(tr, scheme, seed_bytes(entropy, "n3-signer"))
    chal = (ChainHeadChallenge() if beacon_idx is None
            else BeaconChallenge(seed_bytes(entropy, "ghost-beacon-seq-%d" % beacon_idx)))
    vs = verifier_set({k: tuple(v) for k, v in bands.items()})
    opened: list = []
    log = run_interaction(prover, chal, vs, rounds, opened)
    return {"kind": kind, "scheme": scheme, "regime": chal.name, "beacon_seq": beacon_idx,
            "summary": summarise_run(log, list(vs)), "log": log, "opened": opened}


def locate(relpath: str, needle: str, after: str = "") -> str:
    with open(os.path.join(REPO, relpath), encoding="utf-8") as fh:
        armed = not after
        for i, line in enumerate(fh, 1):
            armed = armed or after in line
            if armed and needle in line:
                return "%s:%d" % (relpath, i)
    raise SystemExit("code audit: %r not found in %s" % (needle, relpath))


def code_audit(honest_scheme: str) -> dict:
    P, R, N = "passportfl/probe.py", "passportfl/protocol.py", "experiments/negative_results.py"
    return {
        "prototype_ratio_normalisation": {
            "where": [locate(P, "sep = max(mean_separation(C), mean_separation(C_prev))"),
                      locate(P, "elapsed = max(1, round_ - C_prev_round)"),
                      locate(P, "ratio = prototype_drift_ratio(C, C_prev) / elapsed")]},
        "drift_units": {
            "where": [locate(P, "scale = mean_separation(C) + 1e-12"),
                      locate(P, "d = float(np.linalg.norm(r.embedding - e0)) / scale"),
                      locate(P, "allowed = delta_per_round * max(1, round_ - r0)"),
                      locate(P, "max_drift = max(max_drift, d)"),
                      locate("passportfl/fedproto.py", '"max_drift": res.probe.max_drift')]},
        "ghost_signature_scheme": {
            "where": [locate(N, 'signer = idm.SCHEMES["Ed25519"](seed_bytes(entropy, "n3-signer"))'),
                      locate(N, 'cert = GenesisCert(G, pk, arch, F0, "Ed25519")', after="def ghost_client(")]},
        "challenge_fixed_before_commitment": {
            "where": [locate(P, "idxs = select_subset(E_prev, len(embs), t)"),
                      locate(P, "expected = select_subset(E_prev, m, t)"),
                      locate(R, "F, reveals = commit_and_reveal(embs, self.E, self.t)"),
                      locate(N, "F, reveals = commit_and_reveal(Z, E, T)"),
                      locate(N, 'S = select_subset(seed_bytes(entropy, "n3-beacon-%d" % bi), M, T)')]},
    }


def reproduction(entropy: bytes, rounds: int) -> dict:
    stored = json.load(open(os.path.join(REPO, "results", "negative_results.json"),
                            encoding="utf-8"))["N3_ghost_prover"]
    fresh = ghost_client(rounds=rounds, warmup=5, entropy=entropy)
    keys = ("accepted", "reason", "proto_ratio", "label_stability", "max_drift", "coupling",
            "drift_alignment")

    def diff(a, b):
        if isinstance(a, float) and isinstance(b, float):
            return abs(a - b)
        return 0.0 if a == b else float("inf")

    max_diff_fresh_vs_stored = max(diff(x[k], y[k]) for x, y in zip(fresh["per_round"], stored["per_round"])
                                   for k in keys)
    tr = original_trajectory(rounds)
    prover = GhostProver(tr, "Ed25519", seed_bytes(entropy, "n3-signer"))
    mine, lib = ChallengeReceiver(5), Receiver(M, T, DELTA, MAX_RATIO, MIN_STAB, 5)
    wrapper_vs_library, wrapper_vs_fresh = 0, 0.0
    rows = []
    for _ in range(rounds):
        c = prover.commit()
        c.reveals = prover.open(select_subset(c.E_prev, M, T))
        a, b = mine.verify(c), lib.verify(c)
        same = (a.accepted == b.accepted and a.reason == b.reason and a.chain_ok == b.chain_ok
                and repr(asdict(a.probe)) == repr(asdict(b.probe)))
        wrapper_vs_library += int(not same)
        p = a.probe
        rows.append({"accepted": bool(a.accepted), "reason": a.reason, "proto_ratio": p.proto_drift_ratio,
                     "label_stability": p.label_stability, "max_drift": p.max_drift,
                     "coupling": float(p.coupling) if p.probe_velocity > 0 else None,
                     "drift_alignment": p.drift_alignment})
    wrapper_vs_fresh = max(diff(x[k], y[k]) for x, y in zip(rows, fresh["per_round"]) for k in keys)
    post = [x for x in stored["per_round"] if x["round"] > 5]
    return {
        "stored_accepted": stored["accepted_rounds"], "fresh_accepted": fresh["accepted_rounds"],
        "rounds": rounds,
        "max_abs_diff_fresh_vs_stored_per_round": max_diff_fresh_vs_stored,
        "wrapper_rounds_differing_from_library_receiver": wrapper_vs_library,
        "max_abs_diff_wrapper_vs_fresh_per_round": wrapper_vs_fresh,
        "stored_medians_rounds_6_40": {
            "label_stability": float(np.median([x["label_stability"] for x in post])),
            "proto_ratio": float(np.median([x["proto_ratio"] for x in post]))},
        "stored_signature_scheme": "Ed25519",
    }


def self_test_bound() -> dict:
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(200):
        n, t = 7, int(rng.integers(1, 4))
        last, old = rng.random(n) < rng.random(), rng.random(n) < rng.random()
        worst = max(worst, abs(max_flag_fraction(last, old, t) - _brute_max_flag_fraction(last, old, t)))
    return {"instances": 200, "max_abs_diff_greedy_vs_exhaustive": worst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--beacon-seqs", type=int, default=20)
    ap.add_argument("--robust-seeds", type=int, default=10,
                    help="further trajectory seeds of the calibrated ghost (chain-head challenge)")
    ap.add_argument("--mc-seqs", type=int, default=10000,
                    help="uniformly random challenge sequences replayed per ghost")
    ap.add_argument("--entropy", default=DEFAULT_ENTROPY)
    ap.add_argument("--honest", default=os.path.join(REPO, "results", "fedproto_cifar100.json"))
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "ghost_extended.json"))
    a = ap.parse_args()
    t_all = time.time()
    entropy = bytes.fromhex(a.entropy)
    out = {"meta": {"script": "experiments/ghost_extended_eval.py", "entropy": a.entropy,
                    "rounds": a.rounds, "K": K, "D": D, "m_probes": M, "t_reveal": T,
                    "deployed_limits": {"delta_per_round": DELTA, "max_proto_ratio": MAX_RATIO,
                                        "min_label_stability": MIN_STAB},
                    "beacon_sequences": a.beacon_seqs, "mc_sequences": a.mc_seqs,
                    "robust_seeds": a.robust_seeds, "environment": environment()}}

    print("[1/7] reproduction of the original ghost ...")
    out["reproduction"] = reproduction(entropy, a.rounds)
    rp = out["reproduction"]
    print("      stored %d/%d, fresh %d/%d, fresh-vs-stored max diff %.3g, wrapper-vs-library rounds "
          "differing %d, wrapper-vs-fresh max diff %.3g; medians stab %.3f ratio %.4f"
          % (rp["stored_accepted"], a.rounds, rp["fresh_accepted"], a.rounds,
             rp["max_abs_diff_fresh_vs_stored_per_round"], rp["wrapper_rounds_differing_from_library_receiver"],
             rp["max_abs_diff_wrapper_vs_fresh_per_round"], rp["stored_medians_rounds_6_40"]["label_stability"],
             rp["stored_medians_rounds_6_40"]["proto_ratio"]))
    out["self_test_history_bound"] = self_test_bound()

    print("[2/7] honest reference ...")
    honest, bands = honest_reference(a.honest)
    out["honest_reference"] = honest
    out["honest_supplementary_mean_drift_rate"] = supplementary_mean_rate(
        [os.path.join(REPO, "results", "fedproto_abl_pp_s%d.json" % s) for s in range(3)])
    target = {s: honest["distributions"][s]["median"] for s in STATS}
    print("      %d rows; medians ratio %.3f stability %.3f max_drift %.3f; bands %s; "
          "accepted deployed %d, two-sided %d"
          % (honest["n_rows"], target["proto_ratio"], target["label_stability"], target["max_drift"],
             {s: tuple(round(x, 3) for x in b) for s, b in bands.items()},
             honest["acceptance"]["deployed_logged"], honest["acceptance"]["two_sided"]))

    print("[3/7] calibration search (%d points) ..."
          % sum(int(np.prod([len(v) for v in g.values()])) for _, g in SEARCH_STAGES))
    t0 = time.time()
    out["calibration_search"] = cs = calibration_search(target, a.rounds, a.entropy, a.workers)
    cs["seconds"] = time.time() - t0
    cal = {**cs["selected"]["params"], "seed": CAL_SEED}
    print("      selected %s objective %.4f medians %s (%.0fs)"
          % (cs["selected"]["params"], cs["selected"]["objective"],
             {k: round(v, 4) for k, v in cs["selected"]["medians"].items()}, cs["seconds"]))

    print("[4-5/7] both ghosts x {chain head, %d beacon sequences} x four verifiers ..." % a.beacon_seqs)
    variants = {"original": {"seed": ORIG_SEED, "step": ORIG_STEP}, "calibrated": cal}
    bands_l = {k: list(v) for k, v in bands.items()}
    specs = []
    for kind, params in variants.items():
        specs.append(((kind, "chain_head", "Ed25519", None),
                      (kind, params, "Ed25519", a.entropy, None, bands_l, a.rounds)))
        specs += [((kind, "beacon", "Ed25519", b), (kind, params, "Ed25519", a.entropy, b, bands_l, a.rounds))
                  for b in range(a.beacon_seqs)]
        specs.append(((kind, "chain_head", "ML-DSA-44", None),
                      (kind, params, "ML-DSA-44", a.entropy, None, bands_l, a.rounds)))
    specs += [(("calibrated", "seed", "Ed25519", s),
               ("calibrated", {**cal, "seed": s}, "Ed25519", a.entropy, None, bands_l, a.rounds))
              for s in range(1, a.robust_seeds + 1)]
    t0 = time.time()
    with ProcessPoolExecutor(max(1, a.workers)) as ex:
        res = dict(zip([t for t, _ in specs], ex.map(ghost_job, [j for _, j in specs])))
    print("      %d interaction runs in %.0fs" % (len(res), time.time() - t0))
    vnames = ["deployed_w5", "deployed_w0", "two_sided_w5", "two_sided_w0"]
    table, ghosts = [], {}
    for kind, params in variants.items():
        ch = res[(kind, "chain_head", "Ed25519", None)]
        bc = [res[(kind, "beacon", "Ed25519", b)] for b in range(a.beacon_seqs)]
        ml = res[(kind, "chain_head", "ML-DSA-44", None)]
        ghosts[kind] = {"params": params, "chain_head": {**ch["summary"], "per_round": ch["log"]},
                        "beacon": {"per_sequence": [{"seq": r["beacon_seq"], **r["summary"]} for r in bc]},
                        "scheme_check_ML-DSA-44_chain_head": ml["summary"]}
        sup = out["honest_supplementary_mean_drift_rate"]["mean_drift_rate"]
        mr = [x["mean_drift_rate"] for r in [ch] + bc for x in r["log"]
              if x["round"] > 5 and x["mean_drift_rate"] is not None]
        ghosts[kind]["mean_drift_rate_vs_supplementary_honest"] = {
            "ghost": pct(mr), "honest_p1_p99": [sup["p1"], sup["p99"]],
            "fraction_inside": float(np.mean([sup["p1"] <= x <= sup["p99"] for x in mr])),
            "fraction_below_p1": float(np.mean([x < sup["p1"] for x in mr]))}
        for v in vnames:
            w, ver = int(v[-1]), v.rsplit("_", 1)[0]
            table.append({"ghost": kind, "verifier": ver, "regime": "chain_head", "scheme": "Ed25519",
                          "warmup": w, "accepted": ch["summary"]["accepted"][v], "rounds": a.rounds})
            acc = [r["summary"]["accepted"][v] for r in bc]
            table.append({"ghost": kind, "verifier": ver, "regime": "beacon_post_commit", "scheme": "Ed25519",
                          "warmup": w, "sequences": len(acc), "accepted_per_sequence": acc,
                          "accepted_total": int(sum(acc)), "rounds_total": a.rounds * len(acc),
                          "min": int(min(acc)), "max": int(max(acc))})
            table.append({"ghost": kind, "verifier": ver, "regime": "chain_head", "scheme": "ML-DSA-44",
                          "warmup": w, "accepted": ml["summary"]["accepted"][v], "rounds": a.rounds})
    out["acceptance_table"] = table
    out["calibrated_seed_robustness"] = {
        "trajectory_seeds": list(range(1, a.robust_seeds + 1)), "regime": "chain_head",
        "per_seed": [{"seed": s, **res[("calibrated", "seed", "Ed25519", s)]["summary"]}
                     for s in range(1, a.robust_seeds + 1)]}

    print("[6/7] universal check ...")
    for kind, params in variants.items():
        tr = (original_trajectory(a.rounds, **params) if kind == "original"
              else calibrated_trajectory(a.rounds, **params))
        rows = universal_check(tr)
        realised = [r["log"] for t, r in res.items() if t[0] == kind and t[1] != "seed"]
        viol = 0
        for log in realised:
            for x, u in zip(log[1:], rows):
                viol += int(not (u["stability_lb"] - 1e-12 <= x["label_stability"] <= u["stability_ub"] + 1e-12))
                viol += int(x["max_drift_rate"] is not None and x["max_drift_rate"] > u["worst_drift_rate"] + 1e-12)
                viol += int(x["max_drift"] > u["max_drift_ub"] + 1e-12)
                viol += int(abs(x["proto_ratio"] - u["proto_ratio"]) > 1e-12)
        ghosts[kind]["universal"] = {"summary": summarise_universal(rows, bands),
                                     "realised_runs_checked": len(realised),
                                     "realised_values_outside_bounds": viol,
                                     "per_round": rows}
        s5 = ghosts[kind]["universal"]["summary"]["warmup_5"]
        print("      %-10s rounds>5 passing every admissible challenge %d/%d; worst drift rate %.3f, "
              "max ratio %.4f, min stability LB %.3f; realised outside bounds %d"
              % (kind, s5["rounds_passing_every_admissible_challenge"], s5["rounds_checked"],
                 s5["max_worst_drift_rate"], s5["max_proto_ratio"], s5["min_stability_lb"], viol))

        hr = HistoryReplay(tr)
        vmap = {"deployed_w5": (5, None), "deployed_w0": (0, None),
                "two_sided_w5": (5, bands), "two_sided_w0": (0, bands)}
        stat_diff, verdict_diff, n_runs = 0.0, 0, 0
        for t, r in res.items():
            if t[0] != kind or t[1] == "seed":
                continue
            n_runs += 1
            S = np.asarray(r["opened"])
            st = hr.run(lambda q: S[q - 1][None, :])
            for x in r["log"]:
                i = x["round"] - 1
                for s in STATS:
                    stat_diff = max(stat_diff, abs(float(st[s][0, i]) - x[s]))
                stat_diff = max(stat_diff, abs(float(st["max_drift_rate"][0, i]) - (x["max_drift_rate"] or 0.0)))
            for v, (w, b) in vmap.items():
                ok = hr.verdicts(st, w, b)[0]
                verdict_diff += int(sum(bool(ok[x["round"] - 1]) != x[v]["accepted"] for x in r["log"]))
        mc = hr.monte_carlo(a.mc_seqs, int.from_bytes(seed_bytes(entropy, "ghost-mc-" + kind)[:8], "little"),
                            bands)
        ghosts[kind]["random_history_monte_carlo"] = {
            "replay_check": {"runs_replayed": n_runs, "max_abs_stat_diff": stat_diff,
                             "verdict_mismatches": verdict_diff},
            **mc}
        print("      %-10s replay of %d realised runs: max stat diff %.2e, verdict mismatches %d; "
              "MC %d sequences fully accepted: deployed w5 %d, two-sided w5 %d, two-sided w0 %d"
              % (kind, n_runs, stat_diff, verdict_diff, a.mc_seqs,
                 mc["deployed_w5"]["sequences_all_rounds_accepted"],
                 mc["two_sided_w5"]["sequences_all_rounds_accepted"],
                 mc["two_sided_w0"]["sequences_all_rounds_accepted"]))
    for r in res.values():
        r.pop("opened", None)
    out["ghosts"] = ghosts

    print("[7/7] code audit ...")
    out["code_audit"] = code_audit(honest["config"]["scheme"])
    for kind in variants:
        u = ghosts[kind]["universal"]["per_round"]
        rel = [abs(x["proto_ratio_prev_sep_only"] / x["proto_ratio"] - 1) for x in u if x["proto_ratio"] > 0]
        out["code_audit"]["prototype_ratio_normalisation"]["ghost_%s_max_rel_diff_vs_prev_sep_only" % kind] = max(rel)

    out["meta"]["total_seconds"] = time.time() - t_all
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=jsonable)
    print("\nacceptance (accepted / rounds; beacon: total over %d sequences, min-max per sequence)"
          % a.beacon_seqs)
    for row in table:
        if row["regime"] == "beacon_post_commit":
            print("  %-10s %-9s %-18s %-9s w%d  %4d/%-4d (%d-%d)" % (
                row["ghost"], row["verifier"], row["regime"], row["scheme"], row["warmup"],
                row["accepted_total"], row["rounds_total"], row["min"], row["max"]))
        else:
            print("  %-10s %-9s %-18s %-9s w%d  %4d/%-4d" % (
                row["ghost"], row["verifier"], row["regime"], row["scheme"], row["warmup"],
                row["accepted"], row["rounds"]))
    print("wrote %s (%.0fs)" % (a.out, out["meta"]["total_seconds"]))


if __name__ == "__main__":
    main()
