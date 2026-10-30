from __future__ import annotations

from typing import Dict, List

import numpy as np

EPS = 1e-12


def _flatten(I: np.ndarray) -> np.ndarray:
    return np.asarray(I, dtype=np.float64).ravel()


def krum_scores(innov: Dict[int, np.ndarray], n_byzantine: int = 2) -> Dict[int, float]:
    cids = sorted(innov)
    n = len(cids)
    k = max(1, n - n_byzantine - 2)
    V = {i: _flatten(innov[i]) for i in cids}
    out: Dict[int, float] = {}
    for i in cids:
        d = sorted(float(np.sum((V[i] - V[j]) ** 2)) for j in cids if j != i)
        out[i] = float(np.sum(d[:k]))
    return out


def median_deviation(innov: Dict[int, np.ndarray]) -> Dict[int, float]:
    cids = sorted(innov)
    V = np.stack([_flatten(innov[i]) for i in cids])
    med = np.median(V, axis=0)
    return {i: float(np.linalg.norm(V[k] - med)) for k, i in enumerate(cids)}


def trimmed_mean_deviation(innov: Dict[int, np.ndarray], trim: float = 0.2) -> Dict[int, float]:
    cids = sorted(innov)
    V = np.stack([_flatten(innov[i]) for i in cids])
    n = len(cids)
    c = int(np.floor(trim * n))
    S = np.sort(V, axis=0)
    core = S[c:n - c] if n - 2 * c > 0 else S
    tm = core.mean(axis=0)
    return {i: float(np.linalg.norm(V[k] - tm)) for k, i in enumerate(cids)}


def bulyan_center(innov: Dict[int, np.ndarray], n_byzantine: int = 2) -> np.ndarray:
    cids = sorted(innov)
    n = len(cids)
    V = {i: _flatten(innov[i]) for i in cids}
    f = min(n_byzantine, max(0, (n - 3) // 4))

    pool = list(cids)
    selected: List[int] = []
    target = max(1, n - 2 * f)
    while pool and len(selected) < target:
        sub = {i: innov[i] for i in pool}
        if len(sub) < 3:
            selected.extend(pool)
            break
        s = krum_scores(sub, min(f, max(0, len(sub) - 3)))
        best = min(s, key=lambda k: s[k])
        selected.append(best)
        pool.remove(best)

    S = np.stack([V[i] for i in selected])
    m = len(selected)
    beta = max(1, m - 2 * f)

    med = np.median(S, axis=0)
    order = np.argsort(np.abs(S - med), axis=0)[:beta]
    agg = np.take_along_axis(S, order, axis=0).mean(axis=0)

    return agg


def bulyan_deviation(innov: Dict[int, np.ndarray], n_byzantine: int = 2) -> Dict[int, float]:
    agg = bulyan_center(innov, n_byzantine)
    return {i: float(np.linalg.norm(_flatten(innov[i]) - agg)) for i in sorted(innov)}


def norm_score(innov: Dict[int, np.ndarray]) -> Dict[int, float]:
    cids = sorted(innov)
    norms = {i: float(np.linalg.norm(_flatten(innov[i]))) for i in cids}
    med = float(np.median([v for v in norms.values() if v > EPS])) if norms else 0.0
    out = {}
    for i in cids:
        v = norms[i]
        out[i] = abs(np.log((v + EPS) / (med + EPS))) if med > EPS else float("nan")
    return out


def cosine_to_cohort(innov: Dict[int, np.ndarray]) -> Dict[int, float]:
    cids = sorted(innov)
    V = {i: _flatten(innov[i]) for i in cids}
    mean = np.mean(np.stack([V[i] for i in cids]), axis=0)
    nm = float(np.linalg.norm(mean))
    out = {}
    for i in cids:
        ni = float(np.linalg.norm(V[i]))
        out[i] = -float(V[i] @ mean / (ni * nm)) if ni > EPS and nm > EPS else float("nan")
    return out


def pairwise_max_cosine(innov: Dict[int, np.ndarray]) -> Dict[int, float]:
    cids = sorted(innov)
    V = {i: _flatten(innov[i]) for i in cids}
    N = {i: float(np.linalg.norm(V[i])) for i in cids}
    out = {}
    for i in cids:
        best = float("nan")
        for j in cids:
            if j == i or N[i] <= EPS or N[j] <= EPS:
                continue
            c = float(V[i] @ V[j] / (N[i] * N[j]))
            if not np.isfinite(best) or c > best:
                best = c
        out[i] = best
    return out


BASELINES = {
    "krum": krum_scores,
    "median_dev": median_deviation,
    "trimmed_mean_dev": trimmed_mean_deviation,
    "norm_bound": norm_score,
    "cosine_cohort": cosine_to_cohort,
    "foolsgold_maxcos": pairwise_max_cosine,
}


def score_round(innov: Dict[int, np.ndarray]) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {}
    for name, fn in BASELINES.items():
        try:
            out[name] = fn(innov)
        except Exception:
            out[name] = {i: float("nan") for i in innov}
    return out
