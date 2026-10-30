from __future__ import annotations

import hashlib
import hmac
import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .innovation import EPS, MAD_SCALE, SCORE_STRUCTURAL

FIXED_POINT_SCALE = 1 << 16

STAT_NAMES: Tuple[str, ...] = ("u1", "u2", "g", "x", "rep")
ORIENT: Dict[str, Optional[float]] = {"u1": None, "u2": None, "g": -1.0, "x": +1.0,
                                      "rep": +1.0}

CODE_OK, CODE_UNDEFINED, CODE_DEGENERATE = 0, 1, 2


class CommitReveal:
    @staticmethod
    def wire(C: np.ndarray) -> np.ndarray:
        return np.round(np.asarray(C, dtype=np.float64) * FIXED_POINT_SCALE).astype(np.int64)

    @staticmethod
    def commit(C: np.ndarray, present: np.ndarray, cid: int, round_: int,
               nonce: bytes = b"") -> str:
        h = hashlib.sha256()
        h.update(b"passportfl/commit/v1|")
        h.update(str(int(cid)).encode() + b"|" + str(int(round_)).encode() + b"|")
        h.update(np.asarray(present, dtype=np.uint8).tobytes())
        h.update(CommitReveal.wire(C).tobytes())
        h.update(nonce)
        return h.hexdigest()

    @staticmethod
    def verify(digest: str, C: np.ndarray, present: np.ndarray, cid: int, round_: int,
               nonce: bytes = b"") -> bool:
        return hmac.compare_digest(
            digest, CommitReveal.commit(C, present, cid, round_, nonce))

    @staticmethod
    def attacker_knowledge(commit_reveal: bool) -> str:
        return "peer_lagged" if commit_reveal else "peer_current"


@dataclass(frozen=True)
class Draw:
    sel: Tuple[bool, ...]
    w: Tuple[float, ...]
    rule: str
    tau_off: float

    @property
    def key(self) -> Tuple:
        return (self.sel, self.w, self.rule)

    def label(self) -> str:
        names = "+".join(n for n, s in zip(STAT_NAMES, self.sel) if s)
        wl = ",".join("%g" % x for x, s in zip(self.w, self.sel) if s)
        return "%s[%s](%s)%+g" % (self.rule, names, wl, self.tau_off)


@dataclass
class Mechanism:
    name: str
    subset_size: int = 5
    rule: str = "max"
    weight_levels: Tuple[float, ...] = (1.0,)
    tau_band: Tuple[float, ...] = (0.0,)
    max_weight_combos: int = 0

    def support(self) -> List[Draw]:
        out: List[Draw] = []
        idx = list(range(len(STAT_NAMES)))
        combos = list(itertools.combinations(idx, self.subset_size))
        for c in combos:
            sel = tuple(i in c for i in idx)
            wgrids = list(itertools.product(self.weight_levels, repeat=self.subset_size))
            if self.max_weight_combos and len(wgrids) > self.max_weight_combos:
                step = len(wgrids) / float(self.max_weight_combos)
                wgrids = [wgrids[int(k * step)] for k in range(self.max_weight_combos)]
            for wg in wgrids:
                w = [0.0] * len(STAT_NAMES)
                for pos, i in enumerate(c):
                    w[i] = float(wg[pos])
                for off in self.tau_band:
                    out.append(Draw(sel=sel, w=tuple(w), rule=self.rule, tau_off=float(off)))
        return out

    def describe(self) -> str:
        return ("%s: %d-of-%d %s, weights %s, tau band %s, |support| = %d"
                % (self.name, self.subset_size, len(STAT_NAMES), self.rule,
                   list(self.weight_levels), list(self.tau_band), len(self.support())))


def beacon_index(beacon_seed: int, round_: int, n: int) -> int:
    h = hashlib.sha256(b"passportfl/beacon/v1|%d|%d" % (int(beacon_seed), int(round_)))
    return int.from_bytes(h.digest()[:8], "big") % max(1, int(n))


class BeaconSchedule:
    def __init__(self, mech: Mechanism, beacon_seed: int = 0):
        self.mech = mech
        self.seed = beacon_seed
        self.draws = mech.support()

    def __len__(self) -> int:
        return len(self.draws)

    def draw_index(self, round_: int) -> int:
        return beacon_index(self.seed, round_, len(self.draws))

    def draw(self, round_: int) -> Draw:
        return self.draws[self.draw_index(round_)]


def oriented_from_z(v: Dict[str, float]) -> Tuple[np.ndarray, int]:
    if v.get("undefined"):
        return np.full(len(STAT_NAMES), np.nan), CODE_UNDEFINED
    if v.get("degenerate"):
        return np.full(len(STAT_NAMES), np.nan), CODE_DEGENERATE
    out = np.full(len(STAT_NAMES), np.nan)
    for k, name in enumerate(STAT_NAMES):
        z = v.get("z_" + name, float("nan"))
        if np.isfinite(z):
            o = ORIENT[name]
            out[k] = abs(z) if o is None else o * z
    return out, CODE_OK


def draw_matrices(draws: Sequence[Draw]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    W = np.asarray([d.w for d in draws], dtype=np.float64)
    M = np.asarray([d.sel for d in draws], dtype=bool)
    O = np.asarray([d.tau_off for d in draws], dtype=np.float64)
    return W, M, O


def score_under_draws(T: np.ndarray, code: np.ndarray, draws: Sequence[Draw]) -> np.ndarray:
    W, M, _ = draw_matrices(draws)
    D = len(draws)
    shp = T.shape[:-1]
    out = np.full(shp + (D,), np.nan)
    Tb = T.reshape(-1, T.shape[-1])
    cb = code.reshape(-1)
    ob = out.reshape(-1, D)
    rule = draws[0].rule
    if any(d.rule != rule for d in draws):
        raise ValueError("mixed fusion rules in one support are not supported")
    fin = np.isfinite(Tb)
    for j in range(D):
        sel = M[j] & fin
        vals = np.where(sel, Tb * W[j], np.nan)
        with np.errstate(invalid="ignore"):
            if rule == "max":
                any_ok = sel.any(axis=1)
                s = np.full(len(Tb), np.nan)
                if any_ok.any():
                    s[any_ok] = np.nanmax(vals[any_ok], axis=1)
            elif rule == "sum":
                wsum = (np.where(sel, W[j], 0.0)).sum(axis=1)
                s = np.where(wsum > EPS, np.nansum(np.where(sel, Tb * W[j], 0.0), axis=1)
                             / np.maximum(wsum, EPS), np.nan)
            else:
                raise ValueError(rule)
        s = np.where(cb == CODE_DEGENERATE, SCORE_STRUCTURAL, s)
        s = np.where(cb == CODE_UNDEFINED, np.nan, s)
        ob[:, j] = s
    return out


def fit_threshold(vals: np.ndarray, target: float) -> float:
    v = np.where(np.isfinite(vals), vals, -np.inf).ravel()
    n = v.size
    if n == 0:
        return float("inf")
    m = int(math.floor(target * n))
    s = np.sort(v)
    idx = n - 1 - m
    return float(s[idx]) if idx >= 0 else float("-inf")


def fit_per_draw_thresholds(cal_scores: np.ndarray, draws: Sequence[Draw],
                            target: float) -> np.ndarray:
    key_of: Dict[Tuple, List[int]] = {}
    for j, d in enumerate(draws):
        key_of.setdefault(d.key, []).append(j)
    tau = np.zeros(len(draws))
    for key, js in key_of.items():
        base = fit_threshold(cal_scores[:, js[0]], target)
        for j in js:
            tau[j] = base + draws[j].tau_off
    return tau


def flags_under_schedule(scores: np.ndarray, tau: np.ndarray, rounds: Sequence[int],
                         sched: BeaconSchedule) -> Tuple[np.ndarray, np.ndarray]:
    R, C, D = scores.shape
    idx = np.asarray([sched.draw_index(r) for r in rounds], dtype=int)
    s = scores[np.arange(R)[:, None], np.arange(C)[None, :], idx[:, None]]
    m = s - tau[idx][:, None]
    f = np.where(np.isfinite(m), m, -np.inf) > 0.0
    return m, f


try:
    import torch
except Exception:
    torch = None

BIG = 1.0e6


class DrawObjective:
    def __init__(self, avail: Sequence[bool], draws: Sequence[Draw], tau: np.ndarray,
                 mode: str, p: Optional[np.ndarray] = None, kappa: float = 0.25,
                 device: str = "cpu", dtype=None):
        if torch is None:
            raise RuntimeError("torch is required to run the adaptive attacker")
        dtype = dtype or torch.float64
        self.mode = mode
        self.kappa = kappa
        av = np.asarray(avail, dtype=bool)
        W, M, _ = draw_matrices(draws)
        keep = (M & av[None, :]).any(axis=1)
        if not keep.any():
            keep = np.ones(len(draws), bool)
        self.keep = keep
        self.rule = draws[0].rule
        Wk = W[keep] * (M[keep] & av[None, :])
        self.sel = torch.as_tensor((M[keep] & av[None, :]), device=device)
        self.W = torch.as_tensor(Wk, dtype=dtype, device=device)
        self.tau = torch.as_tensor(tau[keep], dtype=dtype, device=device)
        self.wsum = torch.as_tensor(np.maximum(Wk.sum(axis=1), EPS), dtype=dtype, device=device)
        pp = np.ones(keep.sum()) if p is None else np.asarray(p)[keep]
        self.p = torch.as_tensor(pp / pp.sum(), dtype=dtype, device=device)
        self.av = torch.as_tensor(av, dtype=dtype, device=device)
        self.dtype, self.device = dtype, device

    def _terms(self, o: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        parts = []
        for name in STAT_NAMES:
            v = o.get(name)
            parts.append(v if v is not None
                         else torch.zeros((), dtype=self.dtype, device=self.device))
        return torch.stack(parts)

    def margins(self, o: Dict[str, "torch.Tensor"], temp: Optional[float] = None) -> "torch.Tensor":
        t = self._terms(o) * self.av
        if self.rule == "sum":
            s = (self.W * t[None, :]).sum(dim=1) / self.wsum
        else:
            v = torch.where(self.sel, self.W * t[None, :],
                            torch.full_like(self.W, -BIG))
            s = (torch.logsumexp(temp * v, dim=1) / temp if temp else v.max(dim=1).values)
        return s - self.tau

    def exact(self, o: Dict[str, "torch.Tensor"]) -> float:
        with torch.no_grad():
            m = self.margins(o, None)
            if self.mode == "worst":
                return float(m.max())
            if self.mode == "expected":
                return float((self.p * m).sum())
            return float((self.p * torch.sigmoid(m / self.kappa)).sum())

    def smooth(self, o: Dict[str, "torch.Tensor"], temp: float) -> "torch.Tensor":
        m = self.margins(o, temp)
        if self.mode == "worst":
            return torch.logsumexp(temp * m, dim=0) / temp
        if self.mode == "expected":
            return (self.p * m).sum()
        return (self.p * torch.sigmoid(m / self.kappa)).sum()


class RandomisedAttacker:
    def __init__(self, cfg, draws: Sequence[Draw], tau: np.ndarray, mode: str,
                 seed: int = 0, kappa: float = 0.25):
        from .adaptive_attack import WhiteBoxAttacker
        self.inner = WhiteBoxAttacker(cfg, seed=seed)
        self.cfg = cfg
        self.draws = list(draws)
        self.tau = np.asarray(tau, dtype=np.float64)
        self.mode = mode
        self.kappa = kappa
        self.log: List[dict] = []

    def _cheap_objective(self, Cbar_prev, C_honest):
        return self.inner._cheap_objective(Cbar_prev, C_honest)

    def _optimise(self, sur, Ct, free, B, steps, lr0, init, basis, obj: DrawObjective):
        cfg = self.cfg
        p = init.clone().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr0)
        best_d = (torch.einsum("m,mkd->kd", init, basis) if basis is not None
                  else init).detach().clone()
        best_s = float("inf")
        for it in range(steps):
            frac = it / max(1, steps - 1)
            for grp in opt.param_groups:
                grp["lr"] = lr0 * (0.5 * (1 + math.cos(math.pi * frac)))
            temp = cfg.temp0 * (cfg.temp1 / cfg.temp0) ** frac
            opt.zero_grad(set_to_none=True)
            delta = (torch.einsum("m,mkd->kd", p, basis) if basis is not None
                     else p * free[:, None])
            o = sur.forward(Ct + delta)
            if o["_terms"].numel() == 0:
                break
            s = obj.exact(o)
            if s < best_s:
                best_s, best_d = s, delta.detach().clone()
            loss = obj.smooth(o, temp)
            if not torch.isfinite(loss):
                break
            loss.backward()
            opt.step()
            with torch.no_grad():
                if basis is None:
                    p.mul_(free[:, None])
                n = float(torch.linalg.norm(p))
                if n > B > 0:
                    p.mul_(B / n)
        return best_d, best_s

    def craft(self, aud, r: int, victim: int, C_honest: np.ndarray,
              present_r: Dict[int, np.ndarray],
              peer_C_r: Optional[Dict[int, np.ndarray]]) -> Tuple[np.ndarray, dict]:
        from .adaptive_attack import (FIXED_POINT_SCALE as FPS, SuspicionSurrogate,
                                      _gram_schmidt, build_view)
        cfg = self.cfg
        Cbar_prev = aud.cbar.get(r - 1)
        if Cbar_prev is None:
            return C_honest.copy(), {"round": r, "status": "no_aggregate",
                                     "predicted_objective": float("nan")}
        C_cheap = self._cheap_objective(Cbar_prev, C_honest)
        view = build_view(aud, r, victim, present_r, peer_C_r, cfg.knowledge, cfg.lag_window)
        if view is None:
            return C_cheap, {"round": r, "status": "undefined_innovation",
                             "predicted_objective": float("nan")}
        scale = view.cohort_scale
        if not np.isfinite(scale) or scale <= EPS:
            return C_cheap, {"round": r, "status": "no_cohort_scale",
                             "predicted_objective": float("nan")}
        B = cfg.beta * scale

        sur = SuspicionSurrogate(view, cfg.mad_floor, cfg.min_cohort, cfg.device,
                                 g_two_sided=cfg.g_two_sided, use_cos=cfg.use_cos)
        dev, dt = cfg.device, torch.float64
        Ct = torch.as_tensor(C_cheap, dtype=dt, device=dev)
        free_np = np.asarray(present_r[victim], bool).astype(np.float64)
        free = torch.as_tensor(free_np, dtype=dt, device=dev)
        n_free = max(1.0, float(free_np.sum()) * view.d)

        with torch.no_grad():
            o0 = sur.forward(Ct)
        avail = [bool(n in o0) for n in STAT_NAMES]
        obj = DrawObjective(avail, self.draws, self.tau, self.mode, device=dev, dtype=dt,
                            kappa=self.kappa)
        base_s = obj.exact(o0)
        best_delta = torch.zeros_like(Ct)
        best_s = base_s if np.isfinite(base_s) else float("inf")

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
        for _ in range(cfg.n_random_basis):
            cands.append(self.inner.rng.standard_normal(view.Cbar_prev.shape))
        Fn = _gram_schmidt(cands, free_np)
        if Fn.shape[0] > 0:
            Fb = torch.as_tensor(Fn, dtype=dt, device=dev)
            m = Fn.shape[0]
            inits = [torch.zeros(m, dtype=dt, device=dev)]
            if view.own_prev is not None:
                tgt = (Cbar_prev + scale * view.own_prev.I / max(view.own_prev.norm, EPS)) - C_cheap
                tgt = tgt * free_np[:, None]
                a0 = torch.as_tensor(Fn.reshape(m, -1) @ tgt.reshape(-1), dtype=dt, device=dev)
                na = float(torch.linalg.norm(a0))
                if na > B > 0:
                    a0 = a0 * (B / na)
                inits.append(a0)
            for _ in range(max(0, cfg.restarts - len(inits))):
                a = torch.as_tensor(self.inner.rng.standard_normal(m), dtype=dt, device=dev)
                a = a / torch.linalg.norm(a) * (B * float(self.inner.rng.uniform(0.3, 1.0)))
                inits.append(a)
            for a0 in inits:
                d, s = self._optimise(sur, Ct, free, B, cfg.basis_steps, cfg.lr_basis * B, a0, Fb, obj)
                if s < best_s:
                    best_s, best_delta = s, d

        if cfg.full_steps > 0:
            lr = cfg.lr_full * B / math.sqrt(n_free)
            d, s = self._optimise(sur, Ct, free, B, cfg.full_steps, lr, best_delta.clone(), None, obj)
            if s < best_s:
                best_s, best_delta = s, d

        with torch.no_grad():
            Cn = (Ct + best_delta).detach().cpu().numpy()
            if cfg.quantise:
                Cn = np.round(Cn * FPS) / FPS
            oq = sur.forward(torch.as_tensor(Cn, dtype=dt, device=dev))
            s_q = obj.exact(oq)
            mq = obj.margins(oq, None).detach().cpu().numpy()

        info = {"round": r, "status": "ok", "predicted_objective": s_q,
                "predicted_worst_margin": float(np.max(mq)),
                "predicted_mean_margin": float(np.mean(mq)),
                "cheap_objective": base_s, "budget": B, "cohort_scale": scale,
                "basis_dim": int(Fn.shape[0]), "n_draws_live": int(obj.keep.sum())}
        self.log.append(info)
        return Cn, info


def _self_test() -> bool:
    ok = True
    rng = np.random.default_rng(0)
    C = rng.normal(size=(8, 6))
    p = np.ones(8, bool)
    dg = CommitReveal.commit(C, p, 3, 7)
    ok &= CommitReveal.verify(dg, C, p, 3, 7)
    C2 = C.copy()
    C2[0, 0] += 1.0 / FIXED_POINT_SCALE
    ok &= not CommitReveal.verify(dg, C2, p, 3, 7)
    ok &= not CommitReveal.verify(dg, C, p, 4, 7)
    print("  commitment binds payload, client and round: %s" % ok)

    m = Mechanism("sub3", subset_size=3)
    sup = m.support()
    ok &= len(sup) == 10
    sched = BeaconSchedule(m, beacon_seed=11)
    hist = [sched.draw_index(r) for r in range(1, 4001)]
    cnt = np.bincount(hist, minlength=len(sup))
    print("  3-of-5 support = %d draws, 4000 rounds land %d..%d times per draw"
          % (len(sup), cnt.min(), cnt.max()))
    ok &= cnt.min() > 300
    ok &= [sched.draw_index(r) for r in range(1, 50)] == hist[:49]

    z = {"z_u1": -2.0, "z_u2": 0.5, "z_g": -3.0, "z_x": 1.0, "z_rep": -0.2}
    T, code = oriented_from_z(z)
    det = Mechanism("det", subset_size=5).support()
    s = score_under_draws(T[None, :], np.asarray([code]), det)[0, 0]
    ref = max(abs(-2.0), abs(0.5), 3.0, 1.0, -0.2)
    print("  5-of-5 max reproduces published suspicion: %.6f vs %.6f" % (s, ref))
    ok &= abs(s - ref) < 1e-12
    return bool(ok)


if __name__ == "__main__":
    import sys
    print("unpredictable.py self-test")
    sys.exit(0 if _self_test() else 1)
