from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

EPS = 1e-12
MAD_SCALE = 1.4826
SCORE_STRUCTURAL = 1.0e3
SCORE_FROZEN = 5.0e2
SCORE_STAT_MAX = 2.5e2


@dataclass
class ChainLink:
    round: int
    E: bytes
    proto_velocity: float
    probe_velocity: float
    label_stability: float
    proto_drift_ratio: float
    claimed_proto_drift: float
    measured_proto_drift: float
    n_samples: int
    accepted: bool
    drift_alignment: float = 0.0

    @property
    def coupling(self) -> float:
        if self.probe_velocity <= EPS:
            return float("nan")
        return self.proto_velocity / self.probe_velocity


@dataclass
class AuditVerdict:
    round: int
    coupling: float
    coupling_z: float
    record_residual: float
    frozen: bool
    flagged: bool
    reasons: List[str] = field(default_factory=list)
    n_history: int = 0
    alignment: float = 0.0
    alignment_z: float = float("nan")
    score: float = 0.0
    score_clipped: bool = False

    def to_dict(self) -> dict:
        d = dict(round=self.round, coupling=self.coupling, coupling_z=self.coupling_z,
                 record_residual=self.record_residual, frozen=self.frozen,
                 flagged=self.flagged, reasons=list(self.reasons), n_history=self.n_history,
                 alignment=self.alignment, alignment_z=self.alignment_z,
                 score=self.score, score_clipped=self.score_clipped)
        for k, v in d.items():
            if isinstance(v, float) and not math.isfinite(v):
                d[k] = None
        return d


def robust_z_score(x: float, history: List[float]) -> float:
    if len(history) < 2:
        return float("nan")
    h = np.asarray(history, dtype=np.float64)
    med = float(np.median(h))
    mad = float(np.median(np.abs(h - med)))
    scale = MAD_SCALE * mad
    if scale <= EPS:
        scale = float(h.std())
        if scale <= EPS:
            return 0.0 if abs(x - med) <= EPS else float("inf")
    return abs(x - med) / scale


class TrajectoryAuditor:
    def __init__(self, min_history: int = 6, z_threshold: float = 4.0, record_tol: float = 0.02):
        self.min_history = min_history
        self.z_threshold = z_threshold
        self.record_tol = record_tol
        self.chains: Dict[bytes, List[ChainLink]] = {}
        self.verdicts: Dict[bytes, List[AuditVerdict]] = {}

    def observe(self, G: bytes, link: ChainLink) -> AuditVerdict:
        chain = self.chains.setdefault(G, [])
        hist = [l.coupling for l in chain if math.isfinite(l.coupling) and l.coupling > 0]
        log_hist = [math.log(c) for c in hist]

        rho = link.coupling
        reasons: List[str] = []
        z = float("nan")

        frozen_encoder = (link.probe_velocity <= EPS) and link.round > 1
        nonfinite = not (math.isfinite(link.probe_velocity) and math.isfinite(link.proto_velocity))
        if nonfinite:
            reasons.append("nonfinite-behaviour")
        elif frozen_encoder:
            if link.proto_velocity > EPS:
                reasons.append("frozen-encoder-moving-prototypes")
            else:
                reasons.append("frozen-encoder")
        elif len(log_hist) >= self.min_history and math.isfinite(rho) and rho > 0:
            z = robust_z_score(math.log(rho), log_hist)
            if math.isfinite(z) and z > self.z_threshold:
                reasons.append("coupling" if math.log(rho) > float(np.median(log_hist)) else "coupling-inverse")

        align_hist = [l.drift_alignment for l in chain if math.isfinite(l.drift_alignment)]
        align_z = float("nan")
        if len(align_hist) >= self.min_history and math.isfinite(link.drift_alignment):
            med = float(np.median(align_hist))
            if link.drift_alignment < med:
                align_z = robust_z_score(link.drift_alignment, align_hist)
                if math.isfinite(align_z) and align_z > self.z_threshold:
                    reasons.append("drift-misalignment")

        denom = abs(link.measured_proto_drift) + EPS
        residual = abs(link.claimed_proto_drift - link.measured_proto_drift) / denom
        frozen_protos = link.measured_proto_drift <= EPS and link.round > 1

        if residual > self.record_tol:
            reasons.append("record-mismatch")
        if frozen_protos:
            reasons.append("frozen-prototypes")

        v = AuditVerdict(round=link.round, coupling=rho, coupling_z=z, record_residual=residual,
                         frozen=frozen_encoder or frozen_protos, flagged=bool(reasons),
                         reasons=reasons, n_history=len(log_hist),
                         alignment=link.drift_alignment, alignment_z=align_z,
                         score=self._score(z, reasons, align_z),
                         score_clipped=self._clipped(z, reasons, align_z))
        chain.append(link)
        self.verdicts.setdefault(G, []).append(v)
        return v

    @staticmethod
    def _score(z: float, reasons: List[str], align_z: float = float("nan")) -> float:
        if any(r in ("nonfinite-behaviour", "frozen-encoder-moving-prototypes",
                     "record-mismatch", "frozen-prototypes") for r in reasons):
            return SCORE_STRUCTURAL
        if "frozen-encoder" in reasons:
            return SCORE_FROZEN
        parts = [v for v in (z, align_z) if math.isfinite(v)]
        return min(max(parts), SCORE_STAT_MAX) if parts else 0.0

    @staticmethod
    def _clipped(z: float, reasons: List[str], align_z: float = float("nan")) -> bool:
        if any(r in ("nonfinite-behaviour", "frozen-encoder-moving-prototypes",
                     "record-mismatch", "frozen-prototypes", "frozen-encoder")
               for r in reasons):
            return False
        parts = [v for v in (z, align_z) if math.isfinite(v)]
        return bool(parts) and max(parts) > SCORE_STAT_MAX

    def couplings(self, G: bytes) -> List[float]:
        return [l.coupling for l in self.chains.get(G, []) if math.isfinite(l.coupling)]

    def scores(self, G: bytes) -> List[float]:
        return [v.coupling_z for v in self.verdicts.get(G, []) if math.isfinite(v.coupling_z)]

    def all_scores(self) -> List[float]:
        out: List[float] = []
        for G in self.verdicts:
            out.extend(self.scores(G))
        return out

    def first_flag(self, G: bytes, since_round: int = 0) -> Optional[int]:
        for v in self.verdicts.get(G, []):
            if v.flagged and v.round >= since_round:
                return v.round
        return None

    def summary(self) -> dict:
        return {
            "clients": len(self.chains),
            "links": sum(len(c) for c in self.chains.values()),
            "flagged_links": sum(1 for vs in self.verdicts.values() for v in vs if v.flagged),
            "scored_links": len(self.all_scores()),
        }


def link_from_verify(round_: int, E: bytes, probe_result, record, C: np.ndarray,
                     C_prev: Optional[np.ndarray], accepted: bool) -> ChainLink:
    if C_prev is None:
        measured = 0.0
    else:
        measured = float(np.linalg.norm(C - C_prev, axis=1).mean())
    return ChainLink(
        round=round_,
        E=E,
        proto_velocity=getattr(probe_result, "proto_velocity", 0.0) if probe_result else 0.0,
        probe_velocity=getattr(probe_result, "probe_velocity", 0.0) if probe_result else 0.0,
        label_stability=probe_result.label_stability if probe_result else 1.0,
        proto_drift_ratio=probe_result.proto_drift_ratio if probe_result else 0.0,
        claimed_proto_drift=float(record.proto_drift),
        measured_proto_drift=measured,
        n_samples=int(record.n_samples),
        accepted=accepted,
        drift_alignment=getattr(probe_result, "drift_alignment", 0.0) if probe_result else 0.0,
    )
