from __future__ import annotations

from typing import Dict, Optional

import numpy as np

EPS = 1e-12


def _flatten(I) -> np.ndarray:
    return np.asarray(I, dtype=np.float64).ravel()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return 0.0
    return float(a @ b / (na * nb))


def fltrust_scores(innov: Dict[int, np.ndarray],
                   root_update: np.ndarray) -> Dict[int, float]:
    g0 = _flatten(root_update)
    out: Dict[int, float] = {}
    for i, I in innov.items():
        ts = max(0.0, _cosine(_flatten(I), g0))
        out[i] = -ts
    return out


def fltrust_weights(innov: Dict[int, np.ndarray],
                    root_update: np.ndarray) -> Dict[int, float]:
    g0 = _flatten(root_update)
    ts = {i: max(0.0, _cosine(_flatten(I), g0)) for i, I in innov.items()}
    tot = sum(ts.values())
    if tot <= EPS:
        return {i: 0.0 for i in ts}
    return {i: v / tot for i, v in ts.items()}


class FoolsGold:
    def __init__(self, k: float = 1.0, importance: float = 0.0):
        self.k = k
        self.importance = importance
        self.hist: Dict[int, np.ndarray] = {}

    def update(self, innov: Dict[int, np.ndarray]) -> Dict[int, float]:
        for i, I in innov.items():
            v = _flatten(I)
            if i in self.hist and self.hist[i].shape == v.shape:
                self.hist[i] = self.hist[i] + v
            else:
                self.hist[i] = v.copy()

        cids = sorted(innov)
        n = len(cids)
        if n < 2:
            return {i: float("nan") for i in cids}

        H = np.stack([self.hist[i] for i in cids])
        if 0.0 < self.importance <= 1.0 and H.shape[1] > 1:
            mag = np.abs(H.sum(axis=0))
            keep = max(1, int(round(self.importance * H.shape[1])))
            idx = np.argpartition(-mag, keep - 1)[:keep]
            H = H[:, idx]
        norms = np.linalg.norm(H, axis=1)
        safe = np.maximum(norms, EPS)
        Hn = H / safe[:, None]
        cs = Hn @ Hn.T
        np.fill_diagonal(cs, -1.0)
        cs[norms <= EPS, :] = 0.0
        cs[:, norms <= EPS] = 0.0
        np.fill_diagonal(cs, -1.0)

        v = cs.max(axis=1)

        cs_p = cs.copy()
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                if v[b] > EPS and v[a] < v[b]:
                    cs_p[a, b] = cs[a, b] * v[a] / v[b]

        alpha = 1.0 - cs_p.max(axis=1)
        alpha = np.clip(alpha, 0.0, None)
        mx = alpha.max()
        if mx > EPS:
            alpha = alpha / mx
        alpha = np.clip(alpha, EPS, 1.0 - EPS)

        alpha = self.k * (np.log(alpha / (1.0 - alpha)) + 0.5)
        alpha = np.clip(alpha, 0.0, 1.0)

        return {cids[a]: float(1.0 - alpha[a]) for a in range(n)}


def foolsgold_weights_from_scores(scores: Dict[int, float]) -> Dict[int, float]:
    return {i: 1.0 - s for i, s in scores.items()}
