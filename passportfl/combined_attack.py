from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .innovation import EPS, MAD_SCALE, SCORE_STRUCTURAL, InnovationAuditor
from .adaptive_attack import (DEN_FLOOR, FIXED_POINT_SCALE, CohortView, SuspicionSurrogate,
                              _gram_schmidt, build_view)

BASE_STATS = ["krum", "median_dev", "trimmed_mean_dev", "norm_bound",
              "cosine_cohort", "foolsgold_maxcos"]
COMB_STATS = ["suspicion"] + BASE_STATS
RULES = ("max", "sum", "two")
MIN_COHORT = 5


def cohort_z_score(vals: Dict[int, float], mad_floor: float,
             min_cohort: int = MIN_COHORT) -> Dict[int, float]:
    fin = [v for v in vals.values() if np.isfinite(v)]
    if len(fin) < min_cohort:
        return {c: float("nan") for c in vals}
    arr = np.asarray(fin, float)
    med = float(np.median(arr))
    scale = MAD_SCALE * float(np.median(np.abs(arr - med))) + mad_floor
    return {c: ((v - med) / scale if np.isfinite(v) and scale > 0 else float("nan"))
            for c, v in vals.items()}


def combine_one(z: Dict[str, float], rule: str) -> float:
    vals = [v for v in (z.get(k, float("nan")) for k in COMB_STATS) if np.isfinite(v)]
    if not vals:
        return float("nan")
    if rule == "max":
        return float(max(vals))
    if rule == "sum":
        return float(np.mean(vals))
    if rule == "two":
        if len(vals) < 2:
            return float(max(vals))
        s = sorted(vals, reverse=True)
        return float(s[1])
    raise ValueError(rule)


def apply_norm(raw: float, norm: Sequence[float]) -> float:
    c, s = float(norm[0]), float(norm[1])
    return (raw - c) / s if np.isfinite(raw) and s > 0 else float("nan")


def _column_median(V: torch.Tensor) -> torch.Tensor:
    s, _ = torch.sort(V, dim=0)
    n = V.shape[0]
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _median_1d(v: torch.Tensor) -> torch.Tensor:
    s, _ = torch.sort(v)
    n = s.numel()
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


class OrderStats:
    def __init__(self, peers: torch.Tensor, trim: float = 0.2):
        n = peers.shape[0] + 1
        self.n = n
        self.P, _ = torch.sort(peers, dim=0)
        zero = torch.zeros((1,) + tuple(peers.shape[1:]), dtype=peers.dtype,
                           device=peers.device)
        self.PC = torch.cat([zero, torch.cumsum(self.P, dim=0)])
        c = int(math.floor(trim * n))
        self.lo, self.hi = c, n - c - 1
        self.a, self.b = (n // 2 - 1, n // 2) if n % 2 == 0 else (n // 2, n // 2)
        self.core_n = float(self.hi - self.lo + 1)

    def _order(self, idx: int, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        P = self.P
        hi = P[idx] if idx < P.shape[0] else P[-1]
        lo = P[idx - 1] if idx - 1 >= 0 else P[0]
        return torch.where(r > idx, hi, torch.where(r == idx, x, lo))

    def median(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.a == self.b:
            return self._order(self.a, x, r)
        return 0.5 * (self._order(self.a, x, r) + self._order(self.b, x, r))

    def trimmed_mean(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        lo, hi, m = self.lo, self.hi, self.P.shape[0]
        iA = torch.clamp(torch.minimum(r - 1, torch.full_like(r, hi)) + 1, min=lo)
        sA = torch.gather(self.PC, 0, iA.unsqueeze(0)).squeeze(0) - self.PC[lo]
        iB = torch.clamp(torch.clamp(r, min=max(lo - 1, 0)), max=hi)
        sB = self.PC[hi] - torch.gather(self.PC, 0, iB.unsqueeze(0)).squeeze(0)
        inside = ((r >= lo) & (r <= hi)).to(x.dtype)
        return (sA + sB + x * inside) / self.core_n

    def rank(self, x: torch.Tensor) -> torch.Tensor:
        return (self.P < x).sum(0)


def baseline_stats_torch(V: torch.Tensor, n_byzantine: int = 2, trim: float = 0.2,
                         order: Optional[OrderStats] = None) -> Dict[str, torch.Tensor]:
    n, _ = V.shape
    out: Dict[str, torch.Tensor] = {}
    sq = (V * V).sum(1)
    G = V @ V.T
    D2 = torch.clamp(sq[:, None] + sq[None, :] - 2.0 * G, min=0.0)
    big = torch.full((n,), float("inf"), dtype=V.dtype, device=V.device)
    D2f = D2 + torch.diag(big)
    k = max(1, n - n_byzantine - 2)
    vals, _ = torch.topk(D2f, k, dim=1, largest=False)
    out["krum"] = vals.sum(1)

    if order is None:
        med = _column_median(V)
        c = int(math.floor(trim * n))
        S, _ = torch.sort(V, dim=0)
        core = S[c:n - c] if n - 2 * c > 0 else S
        tm = core.mean(0)
    else:
        x = V[0]
        r = order.rank(x)
        med = order.median(x, r)
        tm = order.trimmed_mean(x, r)
    out["median_dev"] = torch.sqrt(torch.clamp(((V - med) ** 2).sum(1), min=DEN_FLOOR))
    out["trimmed_mean_dev"] = torch.sqrt(torch.clamp(((V - tm) ** 2).sum(1), min=DEN_FLOOR))

    nrm = torch.sqrt(torch.clamp(sq, min=DEN_FLOOR))
    mednrm = _median_1d(nrm)
    out["norm_bound"] = torch.abs(torch.log((nrm + EPS) / (mednrm + EPS)))

    mean = V.mean(0)
    nm = torch.sqrt(torch.clamp((mean * mean).sum(), min=DEN_FLOOR))
    out["cosine_cohort"] = -(V @ mean) / (nrm * nm)

    Cos = G / (nrm[:, None] * nrm[None, :])
    Cos = Cos - torch.diag(big)
    out["foolsgold_maxcos"] = Cos.max(dim=1).values
    return out


class CombinedSurrogate:
    def __init__(self, view: CohortView, norm: Dict[str, Sequence[float]], rule: str = "max",
                 mad_floor: float = 1e-2, min_cohort: int = MIN_COHORT, device: str = "cpu",
                 dtype=torch.float64, n_byzantine: int = 2, trim: float = 0.2):
        self.v = view
        self.rule = rule
        self.norm = {k: (float(norm[k][0]), float(norm[k][1])) for k in COMB_STATS}
        self.mad_floor = mad_floor
        self.min_cohort = min_cohort
        self.dev, self.dt = device, dtype
        self.n_byz, self.trim = n_byzantine, trim
        self.sus = SuspicionSurrogate(view, mad_floor, min_cohort, device, dtype,
                                      g_two_sided=False, use_cos=False)
        T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype, device=device)
        self.Cbar = T(view.Cbar_prev)
        PF = np.stack([p.reshape(-1) for p in view.peer_I_full]) if view.peer_I_full else None
        self.peersV = T(PF) if PF is not None else None
        self.n_clients = (len(view.peer_I_full) + 1) if view.peer_I_full else 1
        self.order = (OrderStats(self.peersV, trim) if self.peersV is not None
                      and self.peersV.shape[0] + 1 >= 3 else None)

    def _z_own(self, own: torch.Tensor, others: torch.Tensor) -> Optional[torch.Tensor]:
        if 1 + others.numel() < self.min_cohort:
            return None
        v = torch.cat([own.reshape(1), others.reshape(-1)])
        med = _median_1d(v)
        mad = _median_1d(torch.abs(v - med))
        return (own - med) / (MAD_SCALE * mad + self.mad_floor)

    def forward(self, C: torch.Tensor) -> Dict[str, object]:
        o = self.sus.forward(C)
        atoms: List[torch.Tensor] = []
        groups: Dict[str, torch.Tensor] = {}
        cs, ss = self.norm["suspicion"]
        if o["_terms"].numel() > 0:
            t = (o["_terms"] - cs) / ss
            atoms.extend(list(t.unbind()))
            groups["suspicion"] = torch.max(t)

        if self.peersV is not None:
            own = (C - self.Cbar).reshape(1, -1)
            V = torch.cat([own, self.peersV], dim=0)
            b = baseline_stats_torch(V, self.n_byz, self.trim, self.order)
            for k in BASE_STATS:
                vals = b[k]
                z = self._z_own(vals[0], vals[1:])
                if z is None:
                    continue
                ck, sk = self.norm[k]
                zn = (z - ck) / sk
                atoms.append(zn)
                groups[k] = zn

        if not atoms:
            nan = torch.as_tensor(float("nan"), dtype=self.dt, device=self.dev)
            return {"atoms": torch.zeros(0, dtype=self.dt, device=self.dev),
                    "groups": {}, "gvec": torch.zeros(0, dtype=self.dt, device=self.dev),
                    "combined": nan, "raw": o}
        A = torch.stack(atoms)
        gnames = [k for k in COMB_STATS if k in groups]
        Gv = torch.stack([groups[k] for k in gnames])
        if self.rule == "max":
            comb = torch.max(A)
        elif self.rule == "sum":
            comb = Gv.mean()
        elif self.rule == "two":
            comb = torch.topk(Gv, 2).values[1] if Gv.numel() >= 2 else Gv.max()
        else:
            raise ValueError(self.rule)
        return {"atoms": A, "groups": groups, "gvec": Gv, "gnames": gnames,
                "combined": comb, "raw": o}


@dataclass
class CombinedAttackConfig:
    phi: float = 0.0
    beta: float = 1.0
    rule: str = "max"
    knowledge: str = "peer_current"
    aim: str = "mimic"
    basis_steps: int = 120
    full_steps: int = 250
    restarts: int = 3
    warm_families: bool = True
    lr_basis: float = 0.08
    lr_full: float = 0.015
    temp0: float = 4.0
    temp1: float = 60.0
    mad_floor: float = 1e-2
    lag_window: int = 5
    min_cohort: int = MIN_COHORT
    n_random_basis: int = 3
    device: str = "cpu"
    quantise: bool = True
    schedule: str = "flat"
    norm: Dict[str, Sequence[float]] = field(default_factory=dict)


def schedule_factor(kind: str, idx: int, n: int) -> float:
    if kind == "flat" or n <= 1:
        return 1.0
    if kind == "alt":
        return 1.6 if idx % 2 == 0 else 0.4
    if kind == "ramp":
        return 2.0 * idx / (n - 1)
    if kind == "front":
        return 2.0 * (n - 1 - idx) / (n - 1)
    raise ValueError(kind)


class CombinedAttacker:
    def __init__(self, cfg: CombinedAttackConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.log: List[dict] = []

    def _cheap_objective(self, Cbar_prev: np.ndarray, C_honest: np.ndarray) -> np.ndarray:
        return Cbar_prev + self.cfg.phi * (C_honest - Cbar_prev)

    def _loss(self, o: Dict[str, object], temp: float, aim: float, family: str
              ) -> Optional[torch.Tensor]:
        A: torch.Tensor = o["atoms"]
        if A.numel() == 0:
            return None
        groups: Dict[str, torch.Tensor] = o["groups"]
        if family == "temporal":
            if "suspicion" not in groups:
                return None
            sub = [g for k, g in groups.items() if k == "suspicion"]
            smooth = torch.stack(sub).max()
            return smooth if aim == 0.0 else (smooth - aim) ** 2
        if family == "cross":
            sub = [g for k, g in groups.items() if k != "suspicion"]
            if not sub:
                return None
            st = torch.stack(sub)
            smooth = torch.logsumexp(temp * st, dim=0) / temp
            return smooth if aim == 0.0 else (smooth - aim) ** 2
        rule = self.cfg.rule
        if rule == "max":
            smooth = torch.logsumexp(temp * A, dim=0) / temp
        elif rule == "sum":
            Gv: torch.Tensor = o["gvec"]
            smooth = Gv.mean()
        elif rule == "two":
            Gv = o["gvec"]
            if Gv.numel() >= 2:
                smooth = torch.topk(Gv, 2).values[1]
            else:
                smooth = Gv.max()
        else:
            raise ValueError(rule)
        return smooth if aim == 0.0 else (smooth - aim) ** 2

    def _optimise(self, sur: CombinedSurrogate, Ct: torch.Tensor, free: torch.Tensor, B: float,
             steps: int, lr0: float, init: torch.Tensor, basis: Optional[torch.Tensor],
             aim: float, family: str = "combined") -> Tuple[torch.Tensor, float]:
        cfg = self.cfg
        p = init.clone().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr0)
        best_d = (torch.einsum("m,mkd->kd", init, basis) if basis is not None
                  else init * free[:, None]).detach().clone()
        best_s = float("inf")
        for it in range(steps):
            frac = it / max(1, steps - 1)
            for grp in opt.param_groups:
                grp["lr"] = lr0 * (0.5 * (1 + math.cos(math.pi * frac)))
            temp = cfg.temp0 * (cfg.temp1 / cfg.temp0) ** frac
            opt.zero_grad(set_to_none=True)
            delta = torch.einsum("m,mkd->kd", p, basis) if basis is not None else p * free[:, None]
            o = sur.forward(Ct + delta)
            comb = o["combined"]
            if not torch.isfinite(comb):
                break
            s = float(comb.detach())
            obj = abs(s - aim)
            if obj < best_s:
                best_s, best_d = obj, delta.detach().clone()
            loss = self._loss(o, temp, aim, family)
            if loss is None or not torch.isfinite(loss):
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

    def craft(self, aud: InnovationAuditor, r: int, victim: int, C_honest: np.ndarray,
              present_r: Dict[int, np.ndarray], peer_C_r: Optional[Dict[int, np.ndarray]],
              aim: float = 0.0, beta_mult: float = 1.0) -> Tuple[np.ndarray, dict]:
        cfg = self.cfg
        Cbar_prev = aud.cbar.get(r - 1)
        if Cbar_prev is None:
            return C_honest.copy(), {"round": r, "status": "no_aggregate",
                                     "predicted_combined": float("nan")}
        C_cheap = self._cheap_objective(Cbar_prev, C_honest)
        view = build_view(aud, r, victim, present_r, peer_C_r, cfg.knowledge, cfg.lag_window)
        if view is None:
            return C_cheap, {"round": r, "status": "undefined_innovation",
                             "predicted_combined": float("nan")}
        scale = view.cohort_scale
        if not np.isfinite(scale) or scale <= EPS:
            return C_cheap, {"round": r, "status": "no_cohort_scale",
                             "predicted_combined": float("nan")}
        B = cfg.beta * beta_mult * scale

        sur = CombinedSurrogate(view, cfg.norm, cfg.rule, cfg.mad_floor, cfg.min_cohort,
                                cfg.device)
        dev, dt = cfg.device, torch.float64
        Ct = torch.as_tensor(C_cheap, dtype=dt, device=dev)
        free_np = np.asarray(present_r[victim], bool).astype(np.float64)
        free = torch.as_tensor(free_np, dtype=dt, device=dev)
        n_free = max(1.0, float(free_np.sum()) * view.d)

        with torch.no_grad():
            base_s = float(sur.forward(Ct)["combined"])
        best_delta = torch.zeros_like(Ct)
        best_s = abs(base_s - aim) if np.isfinite(base_s) else float("inf")

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
        if view.peer_I_full:
            PF = np.stack(view.peer_I_full)
            cands.append(PF.mean(0))
            cands.append(np.median(PF, axis=0))
            mu = PF.mean(0)
            nn = np.linalg.norm(PF.reshape(len(PF), -1), axis=1)
            nm = float(np.linalg.norm(mu))
            if nm > EPS and (nn > EPS).any():
                cs = (PF.reshape(len(PF), -1) @ mu.reshape(-1)) / (np.maximum(nn, EPS) * nm)
                cands.append(PF[int(np.argmax(cs))])
        for _ in range(cfg.n_random_basis):
            cands.append(self.rng.standard_normal(view.Cbar_prev.shape))
        Fn = _gram_schmidt(cands, free_np)

        info_extra = {}
        if Fn.shape[0] > 0:
            Fb = torch.as_tensor(Fn, dtype=dt, device=dev)
            m = Fn.shape[0]
            proj = lambda tgt: torch.as_tensor(
                Fn.reshape(m, -1) @ (tgt * free_np[:, None]).reshape(-1), dtype=dt, device=dev)
            clipB = lambda a: (a * (B / float(torch.linalg.norm(a)))
                               if float(torch.linalg.norm(a)) > B > 0 else a)

            inits: List[torch.Tensor] = [torch.zeros(m, dtype=dt, device=dev)]
            if view.own_prev is not None:
                tgt = (Cbar_prev + scale * view.own_prev.I / max(view.own_prev.norm, EPS)) - C_cheap
                inits.append(clipB(proj(tgt)))
            if view.peer_I_full:
                mu = np.stack(view.peer_I_full).mean(0)
                nm = float(np.linalg.norm(mu))
                if nm > EPS:
                    tgt = (Cbar_prev + scale * mu / nm) - C_cheap
                    inits.append(clipB(proj(tgt)))
            for _ in range(max(0, cfg.restarts)):
                a = torch.as_tensor(self.rng.standard_normal(m), dtype=dt, device=dev)
                a = a / torch.linalg.norm(a) * (B * float(self.rng.uniform(0.3, 1.0)))
                inits.append(a)

            if cfg.warm_families:
                warm: List[torch.Tensor] = []
                for fam in ("temporal", "cross"):
                    d, _ = self._optimise(sur, Ct, free, B, max(40, cfg.basis_steps // 2),
                                     cfg.lr_basis * B, inits[0].clone(), Fb, 0.0, fam)
                    a = proj(d.detach().cpu().numpy())
                    warm.append(clipB(a))
                if len(warm) == 2:
                    warm.append(clipB(0.5 * (warm[0] + warm[1])))
                inits.extend(warm)

            for a0 in inits:
                d, s = self._optimise(sur, Ct, free, B, cfg.basis_steps, cfg.lr_basis * B,
                                 a0, Fb, aim)
                if s < best_s:
                    best_s, best_delta = s, d
            info_extra["basis_dim"] = int(m)
            info_extra["n_inits"] = len(inits)

        if cfg.full_steps > 0:
            lr = cfg.lr_full * B / math.sqrt(n_free)
            d, s = self._optimise(sur, Ct, free, B, cfg.full_steps, lr, best_delta.clone(),
                             None, aim)
            if s < best_s:
                best_s, best_delta = s, d

        with torch.no_grad():
            Cn = (Ct + best_delta).detach().cpu().numpy()
            if cfg.quantise:
                Cn = np.round(Cn * FIXED_POINT_SCALE) / FIXED_POINT_SCALE
            oq = sur.forward(torch.as_tensor(Cn, dtype=dt, device=dev))
            s_q = float(oq["combined"])
            gq = {k: float(v) for k, v in oq["groups"].items()}

        info = {"round": r, "status": "ok", "predicted_combined": s_q,
                "objective_gap": best_s, "aim": aim, "cheap_combined": base_s,
                "budget": B, "cohort_scale": scale, "pred_groups": gq}
        info.update(info_extra)
        self.log.append(info)
        return Cn, info
