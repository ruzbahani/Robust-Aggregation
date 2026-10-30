from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from . import adaptive_attack as _aa
from . import combined_attack as _ca
from .adaptive_attack import (FIXED_POINT_SCALE, CohortView, Entry, SuspicionSurrogate,
                              WhiteBoxAttacker, _entry_from, _gram_schmidt, _masked_cosine)
from .combined_attack import CombinedAttacker, CombinedSurrogate
from .innovation import (EPS, MAD_SCALE, SCORE_STRUCTURAL, ClientRound, InnovationAuditor,
                         RoundScores)

KINDS = ("current", "delayed", "query", "decision", "feedback", "pseudo", "shadow")


@dataclass(frozen=True)
class AccessLevel:
    kind: str
    delay: int = 1
    queries: int = 0
    rescale: str = "agg_step"
    shadow_seed: int = -1

    @property
    def name(self) -> str:
        if self.kind == "delayed":
            return "delayed:%d" % self.delay
        if self.kind in ("query", "decision"):
            return "%s:%d" % (self.kind, self.queries)
        if self.kind == "pseudo" and self.rescale != "agg_step":
            return "pseudo:%s" % self.rescale
        if self.kind == "shadow":
            return "shadow:%d" % self.shadow_seed
        return self.kind

    @property
    def sees_peers(self) -> bool:
        return self.kind not in ("pseudo", "shadow")


def parse_level(text: str) -> AccessLevel:
    kind, _, arg = text.strip().partition(":")
    if kind not in KINDS:
        raise ValueError("unknown access level %r" % text)
    if kind == "delayed":
        d = int(arg or 1)
        if d < 1:
            raise ValueError("delay must be at least one round")
        return AccessLevel(kind, delay=d)
    if kind in ("query", "decision"):
        q = int(arg)
        if q < 1:
            raise ValueError("query budget must be positive")
        return AccessLevel(kind, delay=1, queries=q)
    if kind == "pseudo":
        return AccessLevel(kind, delay=0, rescale=arg or "agg_step")
    if kind == "shadow":
        if not arg:
            raise ValueError("the shadow level needs the shadow federation's seed, e.g. shadow:5")
        return AccessLevel(kind, delay=0, shadow_seed=int(arg))
    if arg:
        raise ValueError("level %r takes no argument" % kind)
    return AccessLevel(kind, delay=0 if kind == "current" else 1)


@dataclass
class InformationSet:
    round: int
    victim: int
    delay: int
    cbar: Dict[int, np.ndarray]
    own_hist: List[ClientRound]
    present_own: np.ndarray
    peer_hist: Dict[int, List[ClientRound]] = field(default_factory=dict)
    peer_scores: Dict[int, float] = field(default_factory=dict)


def extract_information(aud: InnovationAuditor, score_hist: Dict[int, Dict[int, float]],
                        r: int, victim: int, present_own: np.ndarray,
                        level: AccessLevel) -> InformationSet:
    if level.kind == "current":
        raise ValueError("the current level uses the replay state directly")
    cbar = {t: c for t, c in aud.cbar.items() if t <= r - 1}
    own = [c for c in aud.hist.get(victim, []) if c.round <= r - 1]
    info = InformationSet(round=r, victim=victim, delay=level.delay if level.sees_peers else 0,
                          cbar=cbar, own_hist=own,
                          present_own=np.asarray(present_own, bool).copy())
    if level.sees_peers:
        last = r - level.delay
        info.peer_hist = {j: [c for c in h if c.round <= last]
                          for j, h in aud.hist.items() if j != victim}
        info.peer_scores = {j: float(s) for j, s in score_hist.get(last, {}).items()
                            if j != victim}
    return info


def _own_part(info: InformationSet, lag_window: int):
    r = info.round
    Cbar_prev = info.cbar.get(r - 1)
    if Cbar_prev is None:
        return None
    K = Cbar_prev.shape[0]
    h_own = info.own_hist
    prev = h_own[-1] if h_own else None
    if prev is None:
        return None
    T = np.asarray(info.present_own, bool) & prev.present
    if T.sum() < 2:
        return None
    t_own = np.zeros(K, dtype=np.float64)
    t_own[np.where(T)[0]] = 1.0
    L_own = len(h_own)
    return {"Cbar_prev": Cbar_prev, "K": K, "t_own": t_own,
            "own_prev": _entry_from(prev, K),
            "own_prev2": _entry_from(h_own[-2], K) if len(h_own) >= 2 else None,
            "own_C_prev": prev.C,
            "own_C_lags": [h_own[-lag].C for lag in range(2, min(lag_window, L_own) + 1)],
            "own_rep_lag_ok": L_own >= 2}


def _max_cosine(e: Entry, cands: List[Entry], best: float = float("nan")) -> float:
    for c in cands:
        v = _masked_cosine(e.I, e.mask, c.I, c.mask)
        if np.isfinite(v) and (not np.isfinite(best) or v > best):
            best = v
    return best


def delayed_view(info: InformationSet, lag_window: int = 5) -> Optional[CohortView]:
    own = _own_part(info, lag_window)
    if own is None:
        return None
    r, d, K = info.round, info.delay, own["K"]
    Cbar_prev = own["Cbar_prev"]
    Cbar_lag = info.cbar.get(r - d - 1)
    h_own = info.own_hist
    L_own = len(h_own)
    peer_ids = sorted(info.peer_hist)

    def pool_entries(hj: List[ClientRound]) -> List[Entry]:
        out = []
        for lag in range(d, lag_window + 1):
            idx = len(hj) - 1 - (lag - d)
            if idx < 0:
                break
            if hj[idx].round != r - lag:
                raise ValueError("peer history is not one record per round")
            e = _entry_from(hj[idx], K)
            if e is not None:
                out.append(e)
        return out

    peer_I: List[Optional[Entry]] = []
    peer_full: List[np.ndarray] = []
    pu1, pu2, pnorm, pstep, prep = [], [], [], [], []
    for j in peer_ids:
        hj = info.peer_hist[j]
        cj = hj[-1]
        if cj.round != r - d:
            raise ValueError("peer %d has no record for round %d" % (j, r - d))
        peer_I.append(_entry_from(cj, K))
        peer_full.append(cj.C - Cbar_lag if Cbar_lag is not None else np.zeros_like(Cbar_prev))
        pu1.append(cj.u1); pu2.append(cj.u2); pnorm.append(cj.norm)
        pstep.append(float(np.linalg.norm(cj.C - hj[-2].C)) if len(hj) >= 2 else float("nan"))
        best = float("nan")
        for lag in range(2, min(lag_window, len(hj) - 1) + 1):
            dd = float(np.linalg.norm(cj.C - hj[-1 - lag].C))
            if np.isfinite(dd) and (not np.isfinite(best) or dd < best):
                best = dd
        prep.append(best)

    pools = {j: pool_entries(info.peer_hist[j]) for j in peer_ids}
    pool = [e for j in peer_ids for e in pools[j]]
    own_past = []
    for lag in range(1, lag_window + 1):
        idx = L_own - lag
        if idx < 0:
            break
        e = _entry_from(h_own[idx], K)
        if e is not None:
            own_past.append(e)
    x_const = []
    for k, j in enumerate(peer_ids):
        ej = peer_I[k]
        if ej is None:
            x_const.append(float("nan"))
            continue
        best = float("nan")
        for j2 in peer_ids:
            if j2 != j:
                best = _max_cosine(ej, pools[j2], best)
        x_const.append(_max_cosine(ej, own_past, best))

    live = [v for v in pnorm if np.isfinite(v) and v > EPS]
    return CohortView(
        victim=info.victim, K=K, d=Cbar_prev.shape[1], Cbar_prev=Cbar_prev,
        t_own=own["t_own"], own_prev=own["own_prev"], own_prev2=own["own_prev2"],
        own_C_prev=own["own_C_prev"], own_C_lags=own["own_C_lags"],
        own_rep_lag_ok=own["own_rep_lag_ok"],
        peer_ids=peer_ids, peer_I=peer_I,
        peer_u1=np.asarray(pu1), peer_u2=np.asarray(pu2), peer_norm=np.asarray(pnorm),
        peer_step=np.asarray(pstep), peer_rep_raw=np.asarray(prep),
        peer_x_const=np.asarray(x_const), pool_I=pool, peer_I_full=peer_full,
        cohort_scale=float(np.median(live)) if live else float("nan"))


def pseudo_view(info: InformationSet, n_pseudo: int, onset: int, lag_window: int = 5,
                rescale: str = "agg_step") -> Optional[CohortView]:
    own = _own_part(info, lag_window)
    if own is None:
        return None
    r, K = info.round, own["K"]
    Cbar_prev = own["Cbar_prev"]
    h_own = info.own_hist
    honest = {c.round: c for c in h_own if c.round < onset}
    base = [t for t in sorted(honest, reverse=True) if _entry_from(honest[t], K) is not None]
    base = base[:n_pseudo]
    if not base:
        return None

    rho = 1.0
    if rescale == "agg_step":
        a, b = info.cbar.get(r - 1), info.cbar.get(r - 2)
        a0, b0 = info.cbar.get(onset - 1), info.cbar.get(onset - 2)
        if a is not None and b is not None and a0 is not None and b0 is not None:
            den = float(np.linalg.norm(a0 - b0))
            if den > EPS:
                rho = float(np.linalg.norm(a - b)) / den
    elif rescale != "none":
        raise ValueError(rescale)

    def scaled(e: Optional[Entry]) -> Optional[Entry]:
        return None if e is None else Entry(I=rho * e.I, mask=e.mask, norm=rho * e.norm)

    entry = {t: scaled(_entry_from(c, K)) for t, c in honest.items()}
    peer_I, peer_full, pu1, pu2, pnorm, pstep, prep = [], [], [], [], [], [], []
    for t in base:
        c = honest[t]
        peer_I.append(entry[t])
        Cb = info.cbar.get(t - 1)
        peer_full.append(rho * (c.C - Cb) if Cb is not None else np.zeros_like(Cbar_prev))
        pu1.append(c.u1); pu2.append(c.u2); pnorm.append(rho * c.norm)
        p1 = honest.get(t - 1)
        pstep.append(rho * float(np.linalg.norm(c.C - p1.C)) if p1 is not None else float("nan"))
        best = float("nan")
        for lag in range(2, min(lag_window, t - 1) + 1):
            pl = honest.get(t - lag)
            if pl is None:
                continue
            dd = float(np.linalg.norm(c.C - pl.C))
            if np.isfinite(dd) and (not np.isfinite(best) or dd < best):
                best = dd
        prep.append(rho * best if np.isfinite(best) else best)

    pool_rounds = sorted({t - l for t in base for l in range(0, lag_window + 1)
                          if entry.get(t - l) is not None}, reverse=True)
    pool = [entry[t] for t in pool_rounds]
    own_past = []
    for lag in range(1, lag_window + 1):
        idx = len(h_own) - lag
        if idx < 0:
            break
        e = _entry_from(h_own[idx], K)
        if e is not None:
            own_past.append((h_own[idx].round, e))
    x_const = []
    for k, t in enumerate(base):
        others = [entry[u] for u in pool_rounds if u != t]
        others += [e for u, e in own_past if u != t]
        x_const.append(_max_cosine(peer_I[k], others))

    live = [v for v in pnorm if np.isfinite(v) and v > EPS]
    return CohortView(
        victim=info.victim, K=K, d=Cbar_prev.shape[1], Cbar_prev=Cbar_prev,
        t_own=own["t_own"], own_prev=own["own_prev"], own_prev2=own["own_prev2"],
        own_C_prev=own["own_C_prev"], own_C_lags=own["own_C_lags"],
        own_rep_lag_ok=own["own_rep_lag_ok"],
        peer_ids=[-(k + 1) for k in range(len(base))], peer_I=peer_I,
        peer_u1=np.asarray(pu1), peer_u2=np.asarray(pu2), peer_norm=np.asarray(pnorm),
        peer_step=np.asarray(pstep), peer_rep_raw=np.asarray(prep),
        peer_x_const=np.asarray(x_const), pool_I=pool, peer_I_full=peer_full,
        cohort_scale=float(np.median(live)) if live else float("nan"))


@dataclass
class ShadowFederation:
    seed: int
    source: str
    hist: Dict[int, List[ClientRound]]
    cbar: Dict[int, np.ndarray]
    scores: Dict[int, Dict[int, float]]


def replay_shadow(arch: dict, seed: int, source: str, mad_floor: float,
                  score_fn: Callable) -> ShadowFederation:
    n = len(arch["C"])
    rounds = max(max(v) for v in arch["C"].values())
    aud = InnovationAuditor(mad_floor=mad_floor)
    scores: Dict[int, Dict[int, float]] = {}
    for r in range(1, rounds + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        pay = {c: (arch["C"][c][r], arch["present"][c][r]) for c in range(n)}
        rs = aud.observe_round(r, pay)
        scores[r] = {c: float(s) for c, s in score_fn(rs, pay, aud.cbar.get(r - 1)).items()}
    return ShadowFederation(seed=seed, source=source, hist=aud.hist, cbar=aud.cbar,
                            scores=scores)


def shadow_scale(info: InformationSet, shadow: ShadowFederation, onset: int,
                 first: int = 3) -> float:
    own = [c.norm for c in info.own_hist
           if first <= c.round < onset and np.isfinite(c.norm) and c.norm > EPS]
    meds = []
    for t in range(first, onset):
        live = [c.norm for j, h in shadow.hist.items() if j != info.victim for c in h
                if c.round == t and np.isfinite(c.norm) and c.norm > EPS]
        if live:
            meds.append(float(np.median(live)))
    if not own or not meds:
        return float("nan")
    den = float(np.median(meds))
    return float(np.median(own)) / den if den > EPS else float("nan")


def shadow_peers(shadow: ShadowFederation, r: int, seat: int, K: int,
                 lag_window: int = 5) -> dict:
    Cbar_prev = shadow.cbar.get(r - 1)
    hist = {j: [c for c in h if c.round <= r] for j, h in shadow.hist.items()}
    seat_past = [c for c in hist.get(seat, []) if c.round <= r - 1]
    peer_ids = sorted(j for j in hist if j != seat)

    def pool_entries(hj: List[ClientRound]) -> List[Entry]:
        out = []
        for lag in range(0, lag_window + 1):
            idx = len(hj) - 1 - lag
            if idx < 0:
                break
            if hj[idx].round != r - lag:
                raise ValueError("shadow history is not one record per round")
            e = _entry_from(hj[idx], K)
            if e is not None:
                out.append(e)
        return out

    peer_I: List[Optional[Entry]] = []
    peer_full: List[np.ndarray] = []
    pu1, pu2, pnorm, pstep, prep = [], [], [], [], []
    for j in peer_ids:
        hj = hist[j]
        cj = hj[-1]
        if cj.round != r:
            raise ValueError("shadow peer %d has no record for round %d" % (j, r))
        if cj.C.shape[0] != K:
            raise ValueError("the shadow federation has %d classes, not %d" % (cj.C.shape[0], K))
        peer_I.append(_entry_from(cj, K))
        peer_full.append(cj.C - Cbar_prev if Cbar_prev is not None else np.zeros_like(cj.C))
        pu1.append(cj.u1); pu2.append(cj.u2); pnorm.append(cj.norm)
        pstep.append(float(np.linalg.norm(cj.C - hj[-2].C)) if len(hj) >= 2 else float("nan"))
        best = float("nan")
        for lag in range(2, min(lag_window, len(hj) - 1) + 1):
            dd = float(np.linalg.norm(cj.C - hj[-1 - lag].C))
            if np.isfinite(dd) and (not np.isfinite(best) or dd < best):
                best = dd
        prep.append(best)

    pools = {j: pool_entries(hist[j]) for j in peer_ids}
    seat_own = []
    for lag in range(1, lag_window + 1):
        idx = len(seat_past) - lag
        if idx < 0:
            break
        e = _entry_from(seat_past[idx], K)
        if e is not None:
            seat_own.append(e)
    x_const = []
    for k, j in enumerate(peer_ids):
        ej = peer_I[k]
        if ej is None:
            x_const.append(float("nan"))
            continue
        best = float("nan")
        for j2 in peer_ids:
            if j2 != j:
                best = _max_cosine(ej, pools[j2], best)
        x_const.append(_max_cosine(ej, seat_own, best))
    return {"peer_ids": peer_ids, "peer_I": peer_I, "peer_full": peer_full,
            "u1": np.asarray(pu1), "u2": np.asarray(pu2), "norm": np.asarray(pnorm),
            "step": np.asarray(pstep), "rep_raw": np.asarray(prep),
            "x_const": np.asarray(x_const), "pool": [e for j in peer_ids for e in pools[j]]}


def shadow_view(info: InformationSet, shadow: ShadowFederation, rho: float,
                lag_window: int = 5) -> Optional[CohortView]:
    own = _own_part(info, lag_window)
    if own is None:
        return None
    r, K = info.round, own["K"]
    Cbar_prev = own["Cbar_prev"]
    sp = shadow_peers(shadow, r, info.victim, K, lag_window)

    def scaled(e: Optional[Entry]) -> Optional[Entry]:
        return None if e is None else Entry(I=rho * e.I, mask=e.mask, norm=rho * e.norm)

    pnorm = rho * sp["norm"]
    live = [v for v in pnorm if np.isfinite(v) and v > EPS]
    return CohortView(
        victim=info.victim, K=K, d=Cbar_prev.shape[1], Cbar_prev=Cbar_prev,
        t_own=own["t_own"], own_prev=own["own_prev"], own_prev2=own["own_prev2"],
        own_C_prev=own["own_C_prev"], own_C_lags=own["own_C_lags"],
        own_rep_lag_ok=own["own_rep_lag_ok"],
        peer_ids=[-(j + 1) for j in sp["peer_ids"]],
        peer_I=[scaled(e) for e in sp["peer_I"]],
        peer_u1=sp["u1"], peer_u2=sp["u2"], peer_norm=pnorm,
        peer_step=rho * sp["step"], peer_rep_raw=rho * sp["rep_raw"],
        peer_x_const=sp["x_const"], pool_I=[scaled(e) for e in sp["pool"]],
        peer_I_full=[rho * f for f in sp["peer_full"]],
        cohort_scale=float(np.median(live)) if live else float("nan"))


def shadow_peer_scores(shadow: ShadowFederation, r: int, seat: int) -> Dict[int, float]:
    return {-(j + 1): float(s) for j, s in shadow.scores.get(r, {}).items() if j != seat}


class SolverInput:
    def __init__(self, cbar: Dict[int, np.ndarray], scores: List[RoundScores]):
        self.cbar = cbar
        self.scores = scores


@contextlib.contextmanager
def injected_view(view: Optional[CohortView]):
    saved = (_aa.build_view, _ca.build_view)

    def fixed(*_args, **_kwargs):
        return view

    _aa.build_view = _ca.build_view = fixed
    try:
        yield
    finally:
        _aa.build_view, _ca.build_view = saved


def mimic_aim(peer_scores: Dict[int, float], margin: float = 0.0, structural_cut: bool = False
              ) -> float:
    vals = [v for v in peer_scores.values()
            if np.isfinite(v) and (not structural_cut or v < SCORE_STRUCTURAL)]
    if not vals:
        return 0.0
    aim = float(np.median(vals))
    if margin:
        mad = float(np.median(np.abs(np.asarray(vals) - aim)))
        aim = max(0.0, aim - margin * MAD_SCALE * mad)
    return aim


def quantise(C: np.ndarray) -> np.ndarray:
    return np.round(C * FIXED_POINT_SCALE) / FIXED_POINT_SCALE


class ExactOracle:
    def __init__(self, aud: InnovationAuditor, r: int, payloads: Dict[int, tuple], victim: int,
                 score_fn: Callable, limit: int, tau: float):
        self._aud, self._r, self._victim = aud, r, victim
        self._payloads = dict(payloads)
        self._score_fn = score_fn
        self.limit = int(limit)
        self.tau = float(tau)
        self.calls = 0

    def _evaluate(self, C: np.ndarray) -> float:
        if self.calls >= self.limit:
            raise RuntimeError("oracle budget of %d queries exceeded" % self.limit)
        if not np.array_equal(quantise(C), C):
            raise ValueError("oracle queries must lie on the 2^-16 payload grid")
        self.calls += 1
        aud = self._aud
        snap = {c: len(h) for c, h in aud.hist.items()}
        n_scores = len(aud.scores)
        pay = dict(self._payloads)
        pay[self._victim] = (C, pay[self._victim][1])
        try:
            rs = aud.observe_round(self._r, pay)
            s = float(self._score_fn(rs, pay))
        finally:
            for c, h in list(aud.hist.items()):
                del h[snap.get(c, 0):]
            del aud.scores[n_scores:]
        return s

    def score(self, C: np.ndarray) -> float:
        return self._evaluate(C)

    def flagged(self, C: np.ndarray) -> bool:
        s = self._evaluate(C)
        return bool(np.isfinite(s) and s > self.tau)


def search_basis(view: CohortView, free: np.ndarray, cohort: bool, rng: np.random.Generator,
                 n_random: int = 3) -> np.ndarray:
    cands: List[np.ndarray] = []
    if view.own_prev is not None:
        cands.append(view.own_prev.I)
    if view.own_prev2 is not None:
        cands.append(view.own_prev2.I)
    if view.own_prev is not None and view.own_prev2 is not None:
        m3 = view.t_own * view.own_prev.mask * view.own_prev2.mask
        ib3 = view.own_prev.I * m3[:, None]
        ic3 = view.own_prev2.I * m3[:, None]
        nc = float(np.linalg.norm(ic3))
        if nc > EPS:
            uc = ic3 / nc
            cands.append(ib3 - float((ib3 * uc).sum()) * uc)
    cands.append(view.Cbar_prev - view.own_C_prev)
    if view.own_C_lags:
        cands.append(view.own_C_prev - view.own_C_lags[0])
    if cohort and view.peer_I_full:
        PF = np.stack(view.peer_I_full)
        mu = PF.mean(0)
        cands.append(mu)
        cands.append(np.median(PF, axis=0))
        nn = np.linalg.norm(PF.reshape(len(PF), -1), axis=1)
        nm = float(np.linalg.norm(mu))
        if nm > EPS and (nn > EPS).any():
            cs = (PF.reshape(len(PF), -1) @ mu.reshape(-1)) / (np.maximum(nn, EPS) * nm)
            cands.append(PF[int(np.argmax(cs))])
    for _ in range(n_random):
        cands.append(rng.standard_normal(view.Cbar_prev.shape))
    return _gram_schmidt(cands, free)


def zero_order_refine(oracle: ExactOracle, C_cheap: np.ndarray, C0: np.ndarray,
                      basis: np.ndarray, free: np.ndarray, B: float, target: float,
                      rng: np.random.Generator, sigma0: float = 0.1, tol: float = 1e-2
                      ) -> Tuple[np.ndarray, float, float]:
    gap = lambda s: abs(s - target) if np.isfinite(s) else float("inf")
    x_best = C0
    s_best = oracle.score(C0)
    f_best = gap(s_best)
    m = basis.shape[0]
    sigma = sigma0 * B
    while oracle.calls < oracle.limit and f_best > tol and m > 0 and B > 0:
        z = rng.standard_normal(m)
        D = (x_best - C_cheap) + sigma * np.tensordot(z, basis, axes=1) / math.sqrt(m)
        D = D * free[:, None]
        n = float(np.linalg.norm(D))
        if n > B:
            D = D * (B / n)
        x = quantise(C_cheap + D)
        s = oracle.score(x)
        f = gap(s)
        if f < f_best:
            x_best, s_best, f_best = x, s, f
            sigma *= math.exp(1.0 / 3.0)
        else:
            sigma *= math.exp(-1.0 / 12.0)
    return x_best, s_best, f_best


def decision_ladder(oracle: ExactOracle, C_cheap: np.ndarray, C_pref: np.ndarray,
                    safe: Callable[[], np.ndarray]) -> Tuple[np.ndarray, int, bool]:
    Q = oracle.limit
    if not oracle.flagged(C_pref):
        return C_pref, 0, False
    if Q < 2:
        return C_pref, 0, True
    C_safe = safe()
    for k in range(1, Q):
        t = k / (Q - 1)
        x = quantise(C_cheap + (1.0 - t) * (C_pref - C_cheap) + t * (C_safe - C_cheap))
        if not oracle.flagged(x):
            return x, k, False
    return C_safe, Q - 1, True


class LimitedAccessAttacker:
    def __init__(self, defense: str, level: AccessLevel, cfg, seed: int, *, tau: float,
                 cal_median: float = float("nan"), onset: int = 12, n_pseudo: int = 9,
                 fb_step: float = 0.5, fb_decay: float = 0.8, es_sigma0: float = 0.1,
                 es_tol: float = 1e-2, shadow: Optional[ShadowFederation] = None):
        if defense not in ("P", "F"):
            raise ValueError(defense)
        if level.kind == "shadow" and (shadow is None or shadow.seed != level.shadow_seed):
            raise ValueError("level %s needs the shadow federation of seed %d"
                             % (level.name, level.shadow_seed))
        self.shadow = shadow if level.kind == "shadow" else None
        self.defense, self.level, self.cfg = defense, level, cfg
        self.solver = (WhiteBoxAttacker(cfg, seed=seed) if defense == "P"
                       else CombinedAttacker(cfg, seed=seed))
        self._aux_seed = 1_000_000 + seed
        self._aux = None
        self.es_rng = np.random.default_rng(2_000_000 + seed)
        self.tau, self.cal_median = float(tau), float(cal_median)
        self.onset, self.n_pseudo = onset, n_pseudo
        self.fb_step, self.fb_decay = fb_step, fb_decay
        self.es_sigma0, self.es_tol = es_sigma0, es_tol
        self.margin = 0.0

    def _aux_solver(self):
        if self._aux is None:
            import dataclasses
            if self.defense == "P":
                self._aux = WhiteBoxAttacker(dataclasses.replace(self.cfg, aim="min"),
                                             seed=self._aux_seed)
            else:
                self._aux = CombinedAttacker(self.cfg, seed=self._aux_seed)
        return self._aux

    def predict(self, view: CohortView, C: np.ndarray) -> float:
        cfg = self.cfg
        Ct = torch.as_tensor(C, dtype=torch.float64, device=cfg.device)
        with torch.no_grad():
            if self.defense == "P":
                sur = SuspicionSurrogate(view, cfg.mad_floor, cfg.min_cohort, cfg.device,
                                         g_two_sided=cfg.g_two_sided, use_cos=cfg.use_cos)
                return float(sur.forward(Ct)["suspicion"])
            sur = CombinedSurrogate(view, cfg.norm, cfg.rule, cfg.mad_floor, cfg.min_cohort,
                                    cfg.device)
            return float(sur.forward(Ct)["combined"])

    @staticmethod
    def _record(rec: dict) -> dict:
        out = {k: v for k, v in rec.items() if k != "pred_groups"}
        out["predicted"] = rec.get("predicted_suspicion", rec.get("predicted_combined",
                                                                  float("nan")))
        return out

    def _solve(self, solver, info: InformationSet, view: Optional[CohortView],
               C_honest: np.ndarray, margin: float, safe: bool) -> Tuple[np.ndarray, dict]:
        r, v = info.round, info.victim
        present = {v: info.present_own}
        with injected_view(view):
            if self.defense == "P":
                if self.level.kind == "shadow":
                    per = {j: {"suspicion": s}
                           for j, s in shadow_peer_scores(self.shadow, r, v).items()}
                elif self.level.sees_peers:
                    per = {j: {"suspicion": s} for j, s in info.peer_scores.items()}
                else:
                    per = {-1: {"suspicion": self.cal_median}}
                solver.cfg.aim_margin = 0.0 if safe else margin
                src = SolverInput(info.cbar, [RoundScores(round=r - info.delay, per_client=per)])
                Cn, rec = solver.craft(src, r, v, C_honest, present, None)
            else:
                if safe:
                    aim = 0.0
                elif self.level.kind == "shadow":
                    aim = mimic_aim(shadow_peer_scores(self.shadow, r, v), margin)
                elif self.level.sees_peers:
                    aim = mimic_aim(info.peer_scores, margin)
                else:
                    aim = self.cal_median
                Cn, rec = solver.craft(SolverInput(info.cbar, []), r, v, C_honest, present, None,
                                       aim=aim)
        return Cn, self._record(rec)

    def craft(self, aud: InnovationAuditor, score_hist: Dict[int, Dict[int, float]], r: int,
              victim: int, C_honest: np.ndarray, present_r: Dict[int, np.ndarray],
              peer_C_r: Optional[Dict[int, np.ndarray]] = None,
              oracle: Optional[ExactOracle] = None) -> Tuple[np.ndarray, dict]:
        lv = self.level
        if lv.kind == "current":
            if self.defense == "P":
                Cn, rec = self.solver.craft(aud, r, victim, C_honest, present_r, peer_C_r)
            else:
                prev = {c: s for c, s in score_hist.get(r - 1, {}).items() if c != victim}
                Cn, rec = self.solver.craft(aud, r, victim, C_honest, present_r, peer_C_r,
                                            aim=mimic_aim(prev))
            rec = self._record(rec)
            rec.update(oracle_calls=0, margin=0.0)
            return Cn, rec

        info = extract_information(aud, score_hist, r, victim, present_r[victim], lv)
        return self.craft_from_information(info, C_honest, oracle)

    def craft_from_information(self, info: InformationSet, C_honest: np.ndarray,
                               oracle: Optional[ExactOracle] = None
                               ) -> Tuple[np.ndarray, dict]:
        lv, cfg = self.level, self.cfg
        rho = float("nan")
        if lv.kind == "pseudo":
            view = pseudo_view(info, self.n_pseudo, self.onset, cfg.lag_window, lv.rescale)
        elif lv.kind == "shadow":
            rho = shadow_scale(info, self.shadow, self.onset)
            view = shadow_view(info, self.shadow, rho, cfg.lag_window)
        else:
            view = delayed_view(info, cfg.lag_window)
        margin = self.margin if lv.kind == "feedback" else 0.0
        Cn, rec = self._solve(self.solver, info, view, C_honest, margin, safe=False)
        rec.update(oracle_calls=0, margin=margin)
        if lv.kind == "shadow":
            rec["shadow_rho"] = rho
        if lv.kind not in ("query", "decision"):
            return Cn, rec
        if oracle is None or oracle.limit != lv.queries:
            raise ValueError("level %s needs an oracle with a budget of %d queries"
                             % (lv.name, lv.queries))
        if rec.get("status") != "ok":
            return Cn, rec

        r = info.round
        C_cheap = self.solver._cheap_objective(info.cbar[r - 1], C_honest)
        free = np.asarray(info.present_own, bool).astype(np.float64)
        rec["surrogate_choice_predicted"] = rec["predicted"]
        if lv.kind == "query":
            basis = search_basis(view, free, self.defense == "F", self.es_rng,
                                 cfg.n_random_basis)
            C, s, f = zero_order_refine(oracle, C_cheap, Cn, basis, free, rec["budget"],
                                        rec["aim"], self.es_rng, self.es_sigma0, self.es_tol)
            rec.update(oracle_score=s, oracle_gap=f, search_dim=int(basis.shape[0]))
        else:
            def safe() -> np.ndarray:
                Cs, rs = self._solve(self._aux_solver(), info, view, C_honest, 0.0, safe=True)
                rec["safe_predicted"] = rs["predicted"]
                return Cs
            C, rung, all_flagged = decision_ladder(oracle, C_cheap, Cn, safe)
            rec.update(rung=rung, all_flagged=all_flagged)
        if oracle.calls > lv.queries:
            raise AssertionError("oracle used %d > %d queries" % (oracle.calls, lv.queries))
        rec["oracle_calls"] = oracle.calls
        if not np.array_equal(C, Cn):
            rec["predicted"] = self.predict(view, C)
        return C, rec

    def observe_outcome(self, flagged: bool) -> None:
        if self.level.kind == "feedback":
            self.margin = self.margin + self.fb_step if flagged else self.margin * self.fb_decay
