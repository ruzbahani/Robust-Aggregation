from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-12
MAD_SCALE = 1.4826
SCORE_STRUCTURAL = 1.0e3
SCORE_STAT_MAX = 2.5e2

QUANT_STEP = 1.0 / (1 << 16)


def _frobenius_norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a))


def _unit_vector(a: np.ndarray) -> np.ndarray:
    n = _frobenius_norm(a)
    return a / n if n > EPS else a


def frobenius_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    na, nb = _frobenius_norm(a), _frobenius_norm(b)
    if na <= EPS or nb <= EPS:
        return float("nan")
    return float((a * b).sum() / (na * nb))


@dataclass
class ClientRound:
    round: int
    C: np.ndarray
    present: np.ndarray
    innovation: Optional[np.ndarray] = None
    innov_rows: Optional[np.ndarray] = None
    u1: float = float("nan")
    u2: float = float("nan")
    norm: float = float("nan")
    declared: Optional[np.ndarray] = None
    derived: Optional[np.ndarray] = None


@dataclass
class RoundScores:
    round: int
    per_client: Dict[int, Dict[str, float]] = field(default_factory=dict)


class InnovationAuditor:
    def __init__(self, lag_window: int = 5, min_cohort: int = 5, mad_floor: float = 1e-3,
                 mask_policy: str = "declared", carry_check: bool = False,
                 carry_tol: float = QUANT_STEP, monitor_support: bool = False,
                 monitor_churn: bool = False, monitor_support_drop: bool = False,
                 always_score_mask_free: bool = False):
        if mask_policy not in ("declared", "full", "derived"):
            raise ValueError(f"unknown mask_policy {mask_policy!r}")
        self.lag_window = lag_window
        self.min_cohort = min_cohort
        self.mad_floor = mad_floor
        self.mask_policy = mask_policy
        self.carry_check = carry_check
        self.carry_tol = carry_tol
        self.monitor_support = monitor_support
        self.monitor_churn = monitor_churn
        self.monitor_support_drop = monitor_support_drop
        self.always_score_mask_free = always_score_mask_free
        self.hist: Dict[int, List[ClientRound]] = {}
        self.cbar: Dict[int, np.ndarray] = {}
        self.scores: List[RoundScores] = []

    def set_aggregate(self, round_: int, Cbar: np.ndarray) -> None:
        self.cbar[round_] = np.asarray(Cbar, dtype=np.float64).copy()

    def observe_round(self, round_: int, payloads: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> RoundScores:
        Cbar_prev = self.cbar.get(round_ - 1)
        raw: Dict[int, Dict[str, float]] = {}
        units: Dict[int, np.ndarray] = {}

        for cid, (C, present) in payloads.items():
            C = np.asarray(C, dtype=np.float64)
            declared = np.asarray(present, dtype=bool)

            if Cbar_prev is not None:
                derived = np.abs(C - Cbar_prev).max(axis=1) > self.carry_tol
            else:
                derived = declared.copy()

            if self.mask_policy == "full":
                present = np.ones_like(declared)
            elif self.mask_policy == "derived":
                present = derived
            else:
                present = declared

            cr = ClientRound(round=round_, C=C, present=present,
                             declared=declared, derived=derived)
            h = self.hist.setdefault(cid, [])
            prev = h[-1] if h else None

            carry_viol = int((derived & ~declared).sum()) if Cbar_prev is not None else 0

            if Cbar_prev is not None and prev is not None:
                T = present & prev.present
                if T.sum() >= 2:
                    cr.innov_rows = np.where(T)[0]
                    cr.innovation = (C - Cbar_prev)[T]
                    cr.norm = _frobenius_norm(cr.innovation)
                    if prev.innovation is not None and prev.innov_rows is not None:
                        common = np.intersect1d(cr.innov_rows, prev.innov_rows)
                        if len(common) >= 2:
                            ia = cr.innovation[np.searchsorted(cr.innov_rows, common)]
                            ib = prev.innovation[np.searchsorted(prev.innov_rows, common)]
                            cr.u1 = frobenius_cosine(ia, ib)
                            prev2 = h[-2] if len(h) >= 2 else None
                            if (prev2 is not None and prev2.innovation is not None
                                    and prev2.innov_rows is not None):
                                common3 = np.intersect1d(common, prev2.innov_rows)
                                if len(common3) >= 2:
                                    ia3 = cr.innovation[np.searchsorted(cr.innov_rows, common3)]
                                    ib3 = prev.innovation[np.searchsorted(prev.innov_rows, common3)]
                                    ic3 = prev2.innovation[np.searchsorted(prev2.innov_rows, common3)]
                                    ub, uc = _unit_vector(ib3), _unit_vector(ic3)
                                    res_a = ia3 - (ia3 * ub).sum() * ub
                                    res_b = ib3 - (ib3 * uc).sum() * uc
                                    cr.u2 = frobenius_cosine(res_a, res_b)
            h.append(cr)
            if cr.innovation is not None and cr.norm > EPS:
                units[cid] = _unit_vector(cr.innovation)
            supp = float(derived.sum())
            churn = (float(np.logical_xor(derived, prev.derived).sum())
                     if (prev is not None and prev.derived is not None) else float("nan"))
            past = [float(c.derived.sum()) for c in h[-(self.lag_window + 1):-1]
                    if c.derived is not None]
            supp_drop = (math.log((supp + 1.0) / (float(np.median(past)) + 1.0))
                         if len(past) >= 2 else float("nan"))
            raw[cid] = {"u1": cr.u1, "u2": cr.u2, "norm": cr.norm,
                        "defined": cr.innovation is not None and np.isfinite(cr.norm),
                        "supp": supp, "churn": churn, "supp_drop": supp_drop,
                        "carry_viol": float(carry_viol),
                        "n_T": float(cr.innov_rows.size) if cr.innov_rows is not None else 0.0}

        live = [v["norm"] for v in raw.values()
                if v.get("defined") and np.isfinite(v["norm"]) and v["norm"] > EPS]
        cohort_scale = float(np.median(live)) if live else 0.0
        for cid, v in raw.items():
            n = v["norm"]
            if not v.get("defined"):
                if (self.mask_policy == "derived" and v.get("supp", float("nan")) == 0.0
                        and self.cbar.get(round_ - 1) is not None):
                    v["degenerate"] = True
                    v["undefined"] = False
                    continue
                v["degenerate"] = False
                v["undefined"] = True
                continue
            v["undefined"] = False
            v["degenerate"] = bool(cohort_scale > EPS
                                   and (not np.isfinite(n) or n <= 1e-6 * cohort_scale))

        norms = [v["norm"] for v in raw.values() if np.isfinite(v["norm"]) and v["norm"] > EPS]
        med_norm = float(np.median(norms)) if norms else float("nan")
        for cid, v in raw.items():
            v["g"] = (math.log(v["norm"] / med_norm)
                      if np.isfinite(v["norm"]) and v["norm"] > EPS and med_norm > EPS else float("nan"))

        for cid in raw:
            best = float("nan")
            ui = units.get(cid)
            if ui is not None:
                for other, hh in self.hist.items():
                    if other == cid:
                        continue
                    for lag in range(0, self.lag_window + 1):
                        idx = len(hh) - 1 - lag
                        if idx < 0:
                            break
                        o = hh[idx]
                        if o.innovation is None or o.innov_rows is None or o.norm <= EPS:
                            continue
                        common = np.intersect1d(self.hist[cid][-1].innov_rows, o.innov_rows)
                        if len(common) < 2:
                            continue
                        a = self.hist[cid][-1].innovation[np.searchsorted(self.hist[cid][-1].innov_rows, common)]
                        b = o.innovation[np.searchsorted(o.innov_rows, common)]
                        c = frobenius_cosine(a, b)
                        if np.isfinite(c) and (not np.isfinite(best) or c > best):
                            best = c
            raw[cid]["x"] = best

        steps = []
        for cid in raw:
            hh = self.hist.get(cid, [])
            if len(hh) >= 2:
                d = _frobenius_norm(hh[-1].C - hh[-2].C)
                if np.isfinite(d):
                    steps.append(d)
        cohort_step = float(np.median(steps)) if steps else 0.0
        for cid in raw:
            hh = self.hist.get(cid, [])
            best_rep = float("nan")
            if cohort_step > EPS and len(hh) >= 3:
                cur = hh[-1].C
                for lag in range(2, min(self.lag_window, len(hh) - 1) + 1):
                    d = _frobenius_norm(cur - hh[-1 - lag].C) / cohort_step
                    if np.isfinite(d) and (not np.isfinite(best_rep) or d < best_rep):
                        best_rep = d
            raw[cid]["rep"] = -best_rep if np.isfinite(best_rep) else float("nan")

        rs = RoundScores(round=round_, per_client=self._z_scores(raw))
        self.scores.append(rs)
        return rs

    def _z_scores(self, raw: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
        out = {cid: dict(v) for cid, v in raw.items()}
        keys = ["u1", "u2", "g", "x", "rep"]
        if self.monitor_support:
            keys += ["supp"]
        if self.monitor_churn:
            keys += ["churn"]
        if self.monitor_support_drop:
            keys += ["supp_drop"]
        for key in keys:
            vals = [v[key] for v in raw.values() if np.isfinite(v.get(key, float("nan")))]
            if len(vals) < self.min_cohort:
                for cid in out:
                    out[cid][f"z_{key}"] = float("nan")
                continue
            arr = np.asarray(vals, dtype=float)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            scale = MAD_SCALE * mad + self.mad_floor
            for cid in out:
                v = raw[cid].get(key, float("nan"))
                out[cid][f"z_{key}"] = (v - med) / scale if np.isfinite(v) else float("nan")
        MASK_FREE = ((("rep", +1.0),)
                     + ((("supp", -1.0),) if self.monitor_support else ())
                     + ((("churn", +1.0),) if self.monitor_churn else ())
                     + ((("supp_drop", -1.0),) if self.monitor_support_drop else ()))
        NEEDS_T = (("u1", None), ("u2", None), ("g", -1.0), ("x", +1.0))

        def collect(z, spec):
            parts = []
            for key, oriented in spec:
                zz = z.get(f"z_{key}", float("nan"))
                if np.isfinite(zz):
                    parts.append(abs(zz) if oriented is None else oriented * zz)
            return parts

        def statistical(parts):
            if not parts:
                return float("nan"), False
            raw = max(parts)
            return min(raw, SCORE_STAT_MAX), bool(raw > SCORE_STAT_MAX)

        for cid in out:
            z = out[cid]
            z["suspicion_clipped"] = False
            if self.carry_check and z.get("carry_viol", 0.0) > 0.0:
                z["suspicion"] = SCORE_STRUCTURAL
                continue
            if z.get("undefined"):
                if not self.always_score_mask_free:
                    z["suspicion"] = float("nan")
                    continue
                z["suspicion"], z["suspicion_clipped"] = statistical(collect(z, MASK_FREE))
                continue
            if z.get("degenerate"):
                z["suspicion"] = SCORE_STRUCTURAL
                continue
            z["suspicion"], z["suspicion_clipped"] = statistical(
                collect(z, NEEDS_T + MASK_FREE))
        return out
