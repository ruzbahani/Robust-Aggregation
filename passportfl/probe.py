from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .identity import H, DOMAIN_PROBE, DOMAIN_SUBSET, canonical_vectors, dequantize


def probe_seed(G: bytes) -> bytes:
    return H(DOMAIN_PROBE + G)


def make_probes(G: bytes, m: int = 256, shape=(3, 32, 32)) -> torch.Tensor:
    seed_int = int.from_bytes(probe_seed(G)[:8], "little")
    rng = np.random.default_rng(seed_int)
    c, h, w = shape
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    probes = np.zeros((m, c, h, w), dtype=np.float32)
    for i in range(m):
        img = np.zeros((c, h, w), dtype=np.float32)
        for _ in range(4):
            fx, fy = rng.uniform(0.5, 4.0, size=2)
            ph = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.3, 1.0, size=(c, 1, 1))
            img += amp * np.sin(2 * np.pi * (fx * xx + fy * yy) + ph)[None]
        img += 0.3 * rng.standard_normal(size=(c, h, w))
        img = img / (np.abs(img).max() + 1e-8)
        probes[i] = img
    return torch.from_numpy(probes)


def leaf_hash(emb: np.ndarray) -> bytes:
    return H(b"\x00" + canonical_vectors(emb))


def _node_hash(a: bytes, b: bytes) -> bytes:
    return H(b"\x01" + a + b)


class MerkleTree:
    def __init__(self, leaves: List[bytes]):
        assert len(leaves) > 0
        self.levels = [list(leaves)]
        while len(self.levels[-1]) > 1:
            cur = self.levels[-1]
            if len(cur) % 2 == 1:
                cur = cur + [cur[-1]]
            self.levels.append([_node_hash(cur[i], cur[i + 1]) for i in range(0, len(cur), 2)])

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def proof(self, idx: int) -> List[Tuple[bytes, bool]]:
        out = []
        for level in self.levels[:-1]:
            lv = level if len(level) % 2 == 0 else level + [level[-1]]
            sib = idx ^ 1
            out.append((lv[sib], sib > idx))
            idx //= 2
        return out

    @staticmethod
    def verify(root: bytes, leaf: bytes, proof: List[Tuple[bytes, bool]]) -> bool:
        h = leaf
        for sib, right in proof:
            h = _node_hash(h, sib) if right else _node_hash(sib, h)
        return h == root


def select_subset(E_prev: bytes, m: int, t: int) -> List[int]:
    chosen: List[int] = []
    ctr = 0
    while len(chosen) < t:
        h = H(DOMAIN_SUBSET + E_prev + ctr.to_bytes(4, "little"))
        idx = int.from_bytes(h[:4], "little") % m
        if idx not in chosen:
            chosen.append(idx)
        ctr += 1
    return sorted(chosen)


@dataclass
class ProbeReveal:
    index: int
    embedding: np.ndarray
    proof: List[Tuple[bytes, bool]]


@torch.no_grad()
def embed_probes(model, probes: torch.Tensor, device="cpu", batch: int = 64) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(probes), batch):
        outs.append(model.features(probes[i:i + batch].to(device)).cpu().numpy())
    return dequantize(np.concatenate(outs, 0))


def nearest_prototype(emb: np.ndarray, C: np.ndarray) -> np.ndarray:
    d = ((emb[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    return d.argmin(-1)


def commit_and_reveal(embs: np.ndarray, E_prev: bytes, t: int):
    tree = MerkleTree([leaf_hash(e) for e in embs])
    idxs = select_subset(E_prev, len(embs), t)
    reveals = [ProbeReveal(i, embs[i], tree.proof(i)) for i in idxs]
    return tree.root, reveals


@dataclass
class ProbeCheckResult:
    merkle_ok: bool
    subset_ok: bool
    proto_ok: bool
    stability_ok: bool
    drift_ok: bool
    max_drift: float
    proto_drift_ratio: float
    label_stability: float
    proto_velocity: float = 0.0
    probe_velocity: float = 0.0
    n_velocity_probes: int = 0
    drift_alignment: float = 0.0

    @property
    def coupling(self) -> float:
        if self.probe_velocity <= 0.0:
            return float("nan")
        return self.proto_velocity / self.probe_velocity

    @property
    def ok(self) -> bool:
        return self.merkle_ok and self.subset_ok and self.proto_ok and self.stability_ok and self.drift_ok


def mean_separation(C: np.ndarray) -> float:
    K = len(C)
    d = np.linalg.norm(C[:, None] - C[None], axis=-1)
    return float(d[~np.eye(K, dtype=bool)].mean())


def prototype_drift_ratio(C: np.ndarray, C_prev: np.ndarray) -> float:
    sep = max(mean_separation(C), mean_separation(C_prev)) + 1e-12
    return float(np.linalg.norm(C - C_prev, axis=1).mean() / sep)


def check_reveals(F: bytes, reveals: List[ProbeReveal], C: np.ndarray, C_prev: Optional[np.ndarray],
                  C_prev_round: int, E_prev: bytes, m: int, t: int, history: Dict[int, Tuple[int, np.ndarray]],
                  round_: int, delta_per_round: float, max_proto_ratio: float = 1.0,
                  min_label_stability: float = 0.5, warmup_rounds: int = 5) -> ProbeCheckResult:
    expected = select_subset(E_prev, m, t)
    subset_ok = [r.index for r in reveals] == expected
    merkle_ok = all(MerkleTree.verify(F, leaf_hash(r.embedding), r.proof) for r in reveals)
    ratio, stability = 0.0, 1.0
    proto_velocity = 0.0
    if C_prev is not None:
        elapsed = max(1, round_ - C_prev_round)
        ratio = prototype_drift_ratio(C, C_prev) / elapsed
        proto_velocity = float(np.linalg.norm(C - C_prev, axis=1).mean()) / elapsed
        if history:
            stored = np.stack([e for (_, e) in history.values()])
            stability = float((nearest_prototype(stored, C) == nearest_prototype(stored, C_prev)).mean())
    vels, disps = [], []
    for r in reveals:
        if r.index in history:
            r0, e0 = history[r.index]
            el = max(1, round_ - r0)
            d = (r.embedding - e0) / el
            vels.append(float(np.linalg.norm(d)))
            disps.append(d)
    probe_velocity = float(np.mean(vels)) if vels else 0.0
    alignment = 0.0
    if disps and C_prev is not None:
        dz = np.mean(np.stack(disps), axis=0)
        dc = np.mean(C - C_prev, axis=0) / max(1, round_ - C_prev_round)
        nz, nc = np.linalg.norm(dz), np.linalg.norm(dc)
        if nz > 1e-12 and nc > 1e-12:
            alignment = float(np.dot(dz, dc) / (nz * nc))
    warm = round_ <= warmup_rounds
    proto_ok = warm or ratio <= max_proto_ratio
    stability_ok = warm or (stability >= min_label_stability)
    max_drift = 0.0
    drift_ok = True
    scale = mean_separation(C) + 1e-12
    for r in reveals:
        if r.index in history:
            r0, e0 = history[r.index]
            d = float(np.linalg.norm(r.embedding - e0)) / scale
            allowed = delta_per_round * max(1, round_ - r0)
            max_drift = max(max_drift, d)
            if d > allowed and not warm:
                drift_ok = False
    res = ProbeCheckResult(merkle_ok, subset_ok, proto_ok, stability_ok, drift_ok, max_drift, ratio, stability,
                           proto_velocity, probe_velocity, len(vels), alignment)
    if merkle_ok and subset_ok:
        for r in reveals:
            history[r.index] = (round_, r.embedding)
    return res
