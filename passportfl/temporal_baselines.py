from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12
MAD_SCALE = 1.4826

MODES = ("causal", "burnin")

INNOV_STATS = (
    "ar1", "ar1_std", "ar1_two", "ar2", "ar2_std", "ar2_two",
    "ewma", "ewma_std", "ewma_two",
    "cusum", "ph",
    "maha_lw", "maha_pca",
    "iforest", "svdd",
)
PAYLOAD_STATS = ("par1", "par1_std", "par1_two", "par2", "par2_std", "par2_two",
                 "pewma", "pewma_std", "pewma_two")

BASE_STATS = INNOV_STATS + PAYLOAD_STATS


def stat_names() -> List[str]:
    return [f"tb_{s}_{m}" for m in MODES for s in BASE_STATS]


def family_of(name: str) -> str:
    core = name[3:] if name.startswith("tb_") else name
    for m in MODES:
        if core.endswith("_" + m):
            return core[: -(len(m) + 1)]
    return core


def _median(a) -> float:
    v = np.asarray(a, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def _robust_scale(a) -> float:
    v = np.asarray(a, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    med = float(np.median(v))
    return MAD_SCALE * float(np.median(np.abs(v - med)))


def _average_path_length(n: int) -> float:
    if n <= 1:
        return 0.0
    h = math.log(n - 1) + 0.5772156649015329
    return 2.0 * h - 2.0 * (n - 1) / n


def _build_isolation_tree(X: np.ndarray, idx: np.ndarray, rng: np.random.Generator,
                 depth: int, hlim: int):
    n = idx.size
    if depth >= hlim or n <= 1:
        return ("leaf", n, depth)
    sub = X[idx]
    lo = sub.min(axis=0)
    hi = sub.max(axis=0)
    span = hi - lo
    feats = np.flatnonzero(span > 1e-12)
    if feats.size == 0:
        return ("leaf", n, depth)
    f = int(feats[int(rng.integers(feats.size))])
    p = float(lo[f] + span[f] * rng.random())
    go_left = sub[:, f] < p
    left, right = idx[go_left], idx[~go_left]
    if left.size == 0 or right.size == 0:
        return ("leaf", n, depth)
    return (f, p,
            _build_isolation_tree(X, left, rng, depth + 1, hlim),
            _build_isolation_tree(X, right, rng, depth + 1, hlim))


def _isolation_path_length(node, q: np.ndarray) -> float:
    while node[0] != "leaf":
        node = node[2] if q[node[0]] < node[1] else node[3]
    return node[2] + _average_path_length(node[1])


def isolation_forest_fit(X: np.ndarray, n_trees: int, seed: int):
    n = X.shape[0]
    if n < 3:
        return None
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    hlim = int(math.ceil(math.log2(max(n, 2))))
    idx = np.arange(n)
    return ([_build_isolation_tree(X, idx, rng, 0, hlim) for _ in range(n_trees)], n)


def isolation_forest_score(model, q: np.ndarray) -> float:
    if model is None:
        return float("nan")
    trees, n = model
    c = _average_path_length(n)
    if c <= 0:
        return float("nan")
    e = float(np.mean([_isolation_path_length(t, q) for t in trees]))
    return float(2.0 ** (-e / c))


def _project_capped_simplex(y: np.ndarray, cap: float) -> np.ndarray:
    n = y.size
    if n * cap <= 1.0:
        return np.full(n, cap)
    bp = np.sort(np.concatenate([y, y - cap]))
    F = np.clip(y[None, :] - bp[:, None], 0.0, cap).sum(axis=1)
    if F[0] <= 1.0:
        theta = bp[0]
    elif F[-1] >= 1.0:
        theta = bp[-1]
    else:
        j = int(np.searchsorted(-F, -1.0))
        j = min(max(j, 1), len(bp) - 1)
        f0, f1 = F[j - 1], F[j]
        theta = bp[j] if abs(f0 - f1) <= EPS else \
            bp[j - 1] + (f0 - 1.0) * (bp[j] - bp[j - 1]) / (f0 - f1)
    return np.clip(y - theta, 0.0, cap)


def svdd_fit(K: np.ndarray, nu: float = 0.1, iters: int = 120):
    n = K.shape[0]
    if n < 3:
        return None
    cap = min(1.0, max(1.0 / n, 1.0 / (nu * n)))
    a = np.full(n, 1.0 / n)
    dg = np.diag(K).copy()
    L = max(float(np.abs(K).sum(axis=1).max()), EPS)
    step = 1.0 / (2.0 * L)
    for _ in range(iters):
        a = _project_capped_simplex(a - step * (2.0 * (K @ a) - dg), cap)
    return (a, float(a @ K @ a))


def svdd_score(model, k_x: np.ndarray, k_xx: float) -> float:
    if model is None:
        return float("nan")
    a, aKa = model
    return float(k_xx - 2.0 * float(a @ k_x) + aKa)


def _ar_fit(G: np.ndarray, fit: Sequence[int], order: int, rounds: Sequence[int]
            ) -> Optional[np.ndarray]:
    ts = np.array([t for t in fit if t - order >= 0
                   and all(rounds[t - j] == rounds[t] - j for j in range(order + 1))], dtype=int)
    if ts.size < order + 2:
        return None
    A = np.empty((order, order))
    b = np.empty(order)
    for k in range(order):
        b[k] = G[ts, ts - 1 - k].sum()
        for l in range(order):
            A[k, l] = G[ts - 1 - k, ts - 1 - l].sum()
    tr = float(np.trace(A))
    A = A + (1e-10 * tr / order if tr > 0 else 1e-12) * np.eye(order)
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def _ar_residuals(G: np.ndarray, ts: np.ndarray, a: np.ndarray) -> np.ndarray:
    order = len(a)
    if ts.size == 0:
        return np.zeros(0)
    r2 = G[ts, ts].astype(float)
    for k in range(order):
        r2 = r2 - 2.0 * a[k] * G[ts, ts - 1 - k]
    for k in range(order):
        for l in range(order):
            r2 = r2 + a[k] * a[l] * G[ts - 1 - k, ts - 1 - l]
    return np.sqrt(np.maximum(r2, 0.0))


def _ewma_weights(n_lags: int, lam: float) -> np.ndarray:
    w = np.array([lam ** k for k in range(n_lags)], dtype=float)
    s = w.sum()
    return w / s if s > EPS else w


def _ewma_residual(G: np.ndarray, t: int, lam: float, max_lags: int) -> float:
    n_lags = min(max_lags, t)
    if n_lags < 1:
        return float("nan")
    w = _ewma_weights(n_lags, lam)
    lags = np.arange(t - 1, t - 1 - n_lags, -1)
    r2 = float(G[t, t]) - 2.0 * float(w @ G[t, lags])
    r2 += float(w @ G[np.ix_(lags, lags)] @ w)
    return math.sqrt(max(r2, 0.0))


def _centered_gram(G: np.ndarray, fit: Sequence[int]) -> np.ndarray:
    idx = np.asarray(fit, dtype=int)
    Gf = G[np.ix_(idx, idx)]
    rm = Gf.mean(axis=1, keepdims=True)
    cm = Gf.mean(axis=0, keepdims=True)
    return Gf - rm - cm + Gf.mean()


def _project_new_point(G: np.ndarray, fit: Sequence[int], cur: int) -> Tuple[np.ndarray, float]:
    idx = np.asarray(fit, dtype=int)
    gc = G[cur, idx]
    Gf = G[np.ix_(idx, idx)]
    mgc = float(gc.mean())
    rowm = Gf.mean(axis=1)
    allm = float(Gf.mean())
    return gc - mgc - rowm + allm, max(float(G[cur, cur]) - 2.0 * mgc + allm, 0.0)


@dataclass
class _History:
    V: List[np.ndarray] = field(default_factory=list)
    rounds: List[int] = field(default_factory=list)
    G: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    ewma_cache: Dict[int, float] = field(default_factory=dict)

    def push(self, v: np.ndarray, r: int) -> int:
        n = len(self.V)
        newG = np.zeros((n + 1, n + 1))
        if n:
            g = np.fromiter((float(u @ v) for u in self.V), dtype=float, count=n)
            newG[:n, :n] = self.G
            newG[:n, n] = g
            newG[n, :n] = g
        newG[n, n] = float(v @ v)
        self.G = newG
        self.V.append(v)
        self.rounds.append(r)
        return n

    def fit_indices(self, cur: int, mode: str, burnin_end: int) -> List[int]:
        if mode == "burnin":
            return [t for t in range(cur) if self.rounds[t] < burnin_end]
        return list(range(cur))

    def consecutive(self, t: int, k: int) -> bool:
        if t - k < 0:
            return False
        return all(self.rounds[t - j] == self.rounds[t] - j for j in range(k + 1))

    def ewma(self, t: int, lam: float, max_lags: int) -> float:
        v = self.ewma_cache.get(t)
        if v is None:
            v = _ewma_residual(self.G, t, lam, max_lags)
            self.ewma_cache[t] = v
        return v


@dataclass
class _ClientState:
    innov: _History = field(default_factory=_History)
    payload: _History = field(default_factory=_History)
    cusum: Dict[str, List[float]] = field(default_factory=dict)
    ph: Dict[str, List[float]] = field(default_factory=dict)
    cache: Dict[tuple, object] = field(default_factory=dict)


class TemporalBaselineAuditor:
    def __init__(self, burnin_end: int = 12, ewma_lambda: float = 0.7, ewma_max_lags: int = 8,
                 pca_max: int = 8, if_trees: int = 64, svdd_nu: float = 0.1,
                 cusum_k: float = 0.5, ph_delta: float = 0.5, seed: int = 0):
        self.burnin_end = burnin_end
        self.ewma_lambda = ewma_lambda
        self.ewma_max_lags = ewma_max_lags
        self.pca_max = pca_max
        self.if_trees = if_trees
        self.svdd_nu = svdd_nu
        self.cusum_k = cusum_k
        self.ph_delta = ph_delta
        self.seed = seed
        self.state: Dict[int, _ClientState] = {}
        self.cbar: Dict[int, np.ndarray] = {}
        self._names = stat_names()

    def set_aggregate(self, round_: int, Cbar: np.ndarray) -> None:
        self.cbar[round_] = np.asarray(Cbar, dtype=np.float64)

    def observe_round(self, round_: int, payloads: Dict[int, Tuple[np.ndarray, np.ndarray]]
                      ) -> Dict[int, Dict[str, float]]:
        Cbar_prev = self.cbar.get(round_ - 1)
        out: Dict[int, Dict[str, float]] = {}
        for cid, (C, _present) in payloads.items():
            st = self.state.setdefault(cid, _ClientState())
            C = np.asarray(C, dtype=np.float64)
            row = {k: float("nan") for k in self._names}

            p_cur = st.payload.push(np.ascontiguousarray(C).ravel().copy(), round_)
            v_cur = None
            if Cbar_prev is not None:
                v_cur = st.innov.push(np.ascontiguousarray(C - Cbar_prev).ravel(), round_)

            for mode in MODES:
                if v_cur is not None:
                    self._innovation_stats(st, cid, v_cur, mode, row)
                self._ar_family(st.payload, p_cur, mode, row,
                                ("par1", "par2", "pewma"), f"_{mode}")
            out[cid] = row
        return out

    def _ar_family(self, sq: _History, cur: int, mode: str, row: Dict[str, float],
                   names: Tuple[str, str, str], tag: str) -> None:
        G = sq.G
        fit = sq.fit_indices(cur, mode, self.burnin_end)
        n_ar, n_ar2, n_ew = names
        for order, nm in ((1, n_ar), (2, n_ar2)):
            a = _ar_fit(G, fit, order, sq.rounds)
            if a is None or not sq.consecutive(cur, order):
                continue
            res = float(_ar_residuals(G, np.array([cur]), a)[0])
            row[f"tb_{nm}{tag}"] = res
            ts = np.array([t for t in fit if sq.consecutive(t, order)], dtype=int)
            scale = _median(_ar_residuals(G, ts, a))
            self._scaled(row, f"tb_{nm}", tag, res, scale)
        if sq.consecutive(cur, 1):
            res = sq.ewma(cur, self.ewma_lambda, self.ewma_max_lags)
            row[f"tb_{n_ew}{tag}"] = res
            scale = _median([sq.ewma(t, self.ewma_lambda, self.ewma_max_lags)
                             for t in fit if t >= 1])
            self._scaled(row, f"tb_{n_ew}", tag, res, scale)

    @staticmethod
    def _scaled(row: Dict[str, float], stem: str, tag: str, res: float, scale: float) -> None:
        if not (np.isfinite(scale) and scale > EPS and np.isfinite(res)):
            return
        row[f"{stem}_std{tag}"] = res / scale
        row[f"{stem}_two{tag}"] = abs(math.log(max(res, 1e-9 * scale) / scale))

    def _innovation_stats(self, st: _ClientState, cid: int, cur: int, mode: str,
                     row: Dict[str, float]) -> None:
        sq = st.innov
        G = sq.G
        fit = sq.fit_indices(cur, mode, self.burnin_end)
        tag = f"_{mode}"

        self._ar_family(sq, cur, mode, row, ("ar1", "ar2", "ewma"), tag)

        if fit:
            fit_i = np.asarray(fit, dtype=int)
            norms_fit = np.sqrt(np.maximum(G[fit_i, fit_i], 0.0))
            med = _median(norms_fit)
            sc = _robust_scale(norms_fit)
            if not (np.isfinite(sc) and sc > EPS):
                sc = abs(med) * 1e-3 + 1e-9
            x = (math.sqrt(max(G[cur, cur], 0.0)) - med) / sc
            cs = st.cusum.setdefault(mode, [0.0, 0.0])
            cs[0] = max(0.0, cs[0] + x - self.cusum_k)
            cs[1] = max(0.0, cs[1] - x - self.cusum_k)
            row[f"tb_cusum{tag}"] = max(cs[0], cs[1])
            ph = st.ph.setdefault(mode, [0.0, 0.0, 0.0, 0.0])
            ph[0] += x - self.ph_delta
            ph[1] = min(ph[1], ph[0])
            ph[2] += -x - self.ph_delta
            ph[3] = min(ph[3], ph[2])
            row[f"tb_ph{tag}"] = max(ph[0] - ph[1], ph[2] - ph[3])

        if len(fit) < 4:
            return
        sig = (mode, fit[0], fit[-1], len(fit))

        cached = st.cache.get(("pca",) + sig)
        if cached is None:
            Gc = _centered_gram(G, fit)
            lam, U = np.linalg.eigh(Gc)
            o = np.argsort(lam)[::-1]
            cached = (np.clip(lam[o], 0.0, None), U[:, o], Gc)
            st.cache[("pca",) + sig] = cached
        lam, U, Gc = cached
        n = len(fit)
        rank = int((lam > max(float(lam[0]), EPS) * 1e-9).sum())
        c, d2 = _project_new_point(G, fit, cur)

        if rank >= 1:
            lam_r = lam[:rank]
            ev = lam_r / n
            s = (U[:, :rank].T @ c) / np.sqrt(np.maximum(lam_r, EPS))
            q = max(d2 - float((s ** 2).sum()), 0.0)

            p = float(sq.V[cur].size)
            mu = (float(np.trace(Gc)) / n) / p
            S2 = float((Gc ** 2).sum()) / (n * n)
            dlw = max(S2 - p * mu * mu, 0.0) / p
            dg = np.diag(Gc)
            b2 = float((dg ** 2 - 2.0 * (Gc ** 2).sum(axis=1) / n + S2).sum()) / (n * n) / p
            b2 = max(min(b2, dlw), 0.0)
            alpha = float(min(max(b2 / dlw, 0.0), 1.0)) if dlw > EPS else 1.0
            floor = alpha * mu
            if floor > EPS:
                t2 = float((s ** 2 / ((1.0 - alpha) * ev + floor)).sum())
                row[f"tb_maha_lw{tag}"] = math.sqrt(max(t2 + q / floor, 0.0))

            m = min(self.pca_max, rank)
            ridge = 1e-6 * float(ev[0]) + EPS
            row[f"tb_maha_pca{tag}"] = math.sqrt(float((s[:m] ** 2 / (ev[:m] + ridge)).sum()))

            Xtr = U[:, :m] * np.sqrt(lam[:m])[None, :]
            qtr = np.maximum(dg - (Xtr ** 2).sum(axis=1), 0.0)
            Xtr = np.hstack([Xtr, np.sqrt(qtr)[:, None]])
            xq = np.concatenate([s[:m], [math.sqrt(q)]])
            ikey = ("if",) + sig
            model = st.cache.get(ikey)
            if model is None:
                model = isolation_forest_fit(Xtr, self.if_trees,
                                    seed=self.seed * 1000003 + cid * 9176 + n * 31 +
                                    (0 if mode == "causal" else 7))
                st.cache[ikey] = model
            row[f"tb_iforest{tag}"] = isolation_forest_score(model, xq)

        idx = np.asarray(fit, dtype=int)
        dgG = G[idx, idx]
        skey = ("svdd",) + sig
        cachedk = st.cache.get(skey)
        if cachedk is None:
            D2 = np.maximum(dgG[:, None] + dgG[None, :] - 2.0 * G[np.ix_(idx, idx)], 0.0)
            iu = np.triu_indices(len(idx), 1)
            med_d2 = float(np.median(D2[iu])) if iu[0].size else 1.0
            sig2 = med_d2 if med_d2 > EPS else 1.0
            cachedk = (svdd_fit(np.exp(-D2 / (2.0 * sig2)), self.svdd_nu), sig2)
            st.cache[skey] = cachedk
        model, sig2 = cachedk
        dx = np.maximum(float(G[cur, cur]) + dgG - 2.0 * G[cur, idx], 0.0)
        row[f"tb_svdd{tag}"] = svdd_score(model, np.exp(-dx / (2.0 * sig2)), 1.0)
