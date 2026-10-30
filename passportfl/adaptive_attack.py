from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .innovation import EPS, MAD_SCALE, SCORE_STRUCTURAL, InnovationAuditor

FIXED_POINT_SCALE = 1 << 16
DEN_FLOOR = 1e-24


@dataclass
class Entry:
    I: np.ndarray
    mask: np.ndarray
    norm: float


def _entry_from(cr, K: int) -> Optional[Entry]:
    if cr is None or cr.innovation is None or cr.innov_rows is None:
        return None
    n = float(np.linalg.norm(cr.innovation))
    if not np.isfinite(n) or n <= EPS:
        return None
    I = np.zeros((K, cr.innovation.shape[1]), dtype=np.float64)
    I[cr.innov_rows] = cr.innovation
    m = np.zeros(K, dtype=np.float64)
    m[cr.innov_rows] = 1.0
    return Entry(I=I, mask=m, norm=n)


def _masked_cosine(A: np.ndarray, ma: np.ndarray, B: np.ndarray, mb: np.ndarray) -> float:
    common = ma * mb
    if common.sum() < 2:
        return float("nan")
    inner = float((A * B).sum())
    na2 = float((mb * (A * A).sum(1)).sum())
    nb2 = float((ma * (B * B).sum(1)).sum())
    if na2 <= EPS * EPS or nb2 <= EPS * EPS:
        return float("nan")
    return inner / math.sqrt(na2 * nb2)


@dataclass
class CohortView:
    victim: int
    K: int
    d: int
    Cbar_prev: np.ndarray
    t_own: np.ndarray
    own_prev: Optional[Entry]
    own_prev2: Optional[Entry]
    own_C_prev: np.ndarray
    own_C_lags: List[np.ndarray]
    own_rep_lag_ok: bool
    peer_ids: List[int] = field(default_factory=list)
    peer_I: List[Entry] = field(default_factory=list)
    peer_u1: np.ndarray = field(default_factory=lambda: np.zeros(0))
    peer_u2: np.ndarray = field(default_factory=lambda: np.zeros(0))
    peer_norm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    peer_step: np.ndarray = field(default_factory=lambda: np.zeros(0))
    peer_rep_raw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    peer_x_const: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pool_I: List[Entry] = field(default_factory=list)
    cohort_scale: float = float("nan")
    peer_I_full: List[np.ndarray] = field(default_factory=list)


def _own_raw_history(aud: InnovationAuditor, cid: int, K: int):
    h = aud.hist.get(cid, [])
    prev = h[-1] if h else None
    prev2 = h[-2] if len(h) >= 2 else None
    return _entry_from(prev, K), _entry_from(prev2, K), prev


def _peer_round_stats(aud: InnovationAuditor, j: int, C: np.ndarray, present: np.ndarray,
                      Cbar_prev: np.ndarray, K: int, lag_window: int):
    h = aud.hist.get(j, [])
    prev = h[-1] if h else None
    out = {"entry": None, "u1": float("nan"), "u2": float("nan"), "norm": float("nan"),
           "step": float("nan"), "rep_raw": float("nan")}
    if prev is None:
        return out
    out["step"] = float(np.linalg.norm(C - prev.C))
    L = len(h)
    if L >= 2:
        best = float("nan")
        for lag in range(2, min(lag_window, L) + 1):
            dd = float(np.linalg.norm(C - h[-lag].C))
            if np.isfinite(dd) and (not np.isfinite(best) or dd < best):
                best = dd
        out["rep_raw"] = best
    T = np.asarray(present, bool) & prev.present
    if T.sum() < 2:
        return out
    rows = np.where(T)[0]
    Ifull = np.zeros((K, C.shape[1]), dtype=np.float64)
    Ifull[rows] = (C - Cbar_prev)[rows]
    m = np.zeros(K, dtype=np.float64)
    m[rows] = 1.0
    nrm = float(np.linalg.norm(Ifull))
    out["norm"] = nrm
    out["entry"] = Entry(I=Ifull, mask=m, norm=nrm) if nrm > EPS else None
    pe = _entry_from(prev, K)
    if pe is not None:
        out["u1"] = _masked_cosine(Ifull, m, pe.I, pe.mask)
        pe2 = _entry_from(h[-2], K) if L >= 2 else None
        if pe2 is not None:
            m3 = m * pe.mask * pe2.mask
            if m3.sum() >= 2:
                ia3 = Ifull * m3[:, None]
                ib3 = pe.I * m3[:, None]
                ic3 = pe2.I * m3[:, None]
                nb, nc = np.linalg.norm(ib3), np.linalg.norm(ic3)
                if nb > EPS and nc > EPS:
                    ub, uc = ib3 / nb, ic3 / nc
                    res_a = ia3 - float((ia3 * ub).sum()) * ub
                    res_b = ib3 - float((ib3 * uc).sum()) * uc
                    na, nr = np.linalg.norm(res_a), np.linalg.norm(res_b)
                    if na > EPS and nr > EPS:
                        out["u2"] = float((res_a * res_b).sum() / (na * nr))
    return out


def build_view(aud: InnovationAuditor, r: int, victim: int, present_r: Dict[int, np.ndarray],
               peer_C_r: Optional[Dict[int, np.ndarray]], knowledge: str,
               lag_window: int = 5) -> Optional[CohortView]:
    Cbar_prev = aud.cbar.get(r - 1)
    if Cbar_prev is None:
        return None
    K, d = Cbar_prev.shape
    own_prev, own_prev2, own_prev_cr = _own_raw_history(aud, victim, K)
    if own_prev_cr is None:
        return None
    T = np.asarray(present_r[victim], bool) & own_prev_cr.present
    if T.sum() < 2:
        return None
    t_own = np.zeros(K, dtype=np.float64)
    t_own[np.where(T)[0]] = 1.0

    h_own = aud.hist[victim]
    L_own = len(h_own)
    own_C_lags = [h_own[-lag].C for lag in range(2, min(lag_window, L_own) + 1)]

    peer_ids = [j for j in sorted(aud.hist) if j != victim]
    peer_I: List[Entry] = []
    peer_full: List[np.ndarray] = []
    pu1, pu2, pnorm, pstep, prep = [], [], [], [], []

    if knowledge == "peer_current":
        if peer_C_r is None:
            raise ValueError("peer_current knowledge needs the peers' round-r payloads")
        for j in peer_ids:
            st = _peer_round_stats(aud, j, peer_C_r[j], present_r[j], Cbar_prev, K, lag_window)
            peer_I.append(st["entry"])
            peer_full.append(peer_C_r[j] - Cbar_prev)
            pu1.append(st["u1"]); pu2.append(st["u2"]); pnorm.append(st["norm"])
            pstep.append(st["step"]); prep.append(st["rep_raw"])
    elif knowledge == "peer_lagged":
        Cbar_prev2 = aud.cbar.get(r - 2)
        for j in peer_ids:
            hj = aud.hist[j]
            cj = hj[-1]
            peer_I.append(_entry_from(cj, K))
            peer_full.append(cj.C - Cbar_prev2 if Cbar_prev2 is not None
                             else np.zeros_like(Cbar_prev))
            pu1.append(cj.u1); pu2.append(cj.u2); pnorm.append(cj.norm)
            pstep.append(float(np.linalg.norm(cj.C - hj[-2].C)) if len(hj) >= 2 else float("nan"))
            best = float("nan")
            for lag in range(2, min(lag_window, len(hj) - 1) + 1):
                dd = float(np.linalg.norm(cj.C - hj[-1 - lag].C))
                if np.isfinite(dd) and (not np.isfinite(best) or dd < best):
                    best = dd
            prep.append(best)
    else:
        raise ValueError(knowledge)

    pool: List[Entry] = []
    lag0 = (knowledge == "peer_current")
    for k, j in enumerate(peer_ids):
        if lag0 and peer_I[k] is not None:
            pool.append(peer_I[k])
        hj = aud.hist[j]
        for lag in range(1, lag_window + 1):
            idx = len(hj) - lag
            if idx < 0:
                break
            e = _entry_from(hj[idx], K)
            if e is not None:
                pool.append(e)

    x_const = []
    for k, j in enumerate(peer_ids):
        ej = peer_I[k]
        if ej is None:
            x_const.append(float("nan"))
            continue
        best = float("nan")
        for k2, j2 in enumerate(peer_ids):
            if j2 == j:
                continue
            cands = []
            if lag0 and peer_I[k2] is not None:
                cands.append(peer_I[k2])
            hj2 = aud.hist[j2]
            for lag in range(1, lag_window + 1):
                idx = len(hj2) - lag
                if idx < 0:
                    break
                e = _entry_from(hj2[idx], K)
                if e is not None:
                    cands.append(e)
            for e in cands:
                c = _masked_cosine(ej.I, ej.mask, e.I, e.mask)
                if np.isfinite(c) and (not np.isfinite(best) or c > best):
                    best = c
        for lag in range(1, lag_window + 1):
            idx = L_own - lag
            if idx < 0:
                break
            e = _entry_from(h_own[idx], K)
            if e is None:
                continue
            c = _masked_cosine(ej.I, ej.mask, e.I, e.mask)
            if np.isfinite(c) and (not np.isfinite(best) or c > best):
                best = c
        x_const.append(best)

    live = [v for v in pnorm if np.isfinite(v) and v > EPS]
    return CohortView(
        victim=victim, K=K, d=d, Cbar_prev=Cbar_prev, t_own=t_own,
        own_prev=own_prev, own_prev2=own_prev2,
        own_C_prev=own_prev_cr.C, own_C_lags=own_C_lags, own_rep_lag_ok=(L_own >= 2),
        peer_ids=peer_ids, peer_I=peer_I,
        peer_u1=np.asarray(pu1), peer_u2=np.asarray(pu2), peer_norm=np.asarray(pnorm),
        peer_step=np.asarray(pstep), peer_rep_raw=np.asarray(prep),
        peer_x_const=np.asarray(x_const), pool_I=pool, peer_I_full=peer_full,
        cohort_scale=float(np.median(live)) if live else float("nan"))


class SuspicionSurrogate:
    def __init__(self, view: CohortView, mad_floor: float = 1e-2, min_cohort: int = 5,
                 device: str = "cpu", dtype=torch.float64, g_two_sided: bool = False,
                 use_cos: bool = False):
        self.v = view
        self.mad_floor = mad_floor
        self.min_cohort = min_cohort
        self.dev = device
        self.dt = dtype
        self.g_two_sided = g_two_sided
        self.use_cos = use_cos
        T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype, device=device)
        K = view.K
        self.Cbar = T(view.Cbar_prev)
        self.t = T(view.t_own)
        self.C_prev = T(view.own_C_prev)
        self.C_lags = T(np.stack(view.own_C_lags)) if view.own_C_lags else None

        self.u1_ok = False
        if view.own_prev is not None and float((view.t_own * view.own_prev.mask).sum()) >= 2:
            self.u1_ok = True
            self.prevI = T(view.own_prev.I)
            self.prev_m = T(view.own_prev.mask)
            self.u1_nb2 = T(float((view.t_own * (view.own_prev.I ** 2).sum(1)).sum()))

        self.u2_ok = False
        if view.own_prev is not None and view.own_prev2 is not None:
            m3 = view.t_own * view.own_prev.mask * view.own_prev2.mask
            if m3.sum() >= 2:
                ib3 = view.own_prev.I * m3[:, None]
                ic3 = view.own_prev2.I * m3[:, None]
                nb, nc = float(np.linalg.norm(ib3)), float(np.linalg.norm(ic3))
                if nb > EPS and nc > EPS:
                    ub, uc = ib3 / nb, ic3 / nc
                    res_b = ib3 - float((ib3 * uc).sum()) * uc
                    nrb = float(np.linalg.norm(res_b))
                    if nrb > EPS:
                        self.u2_ok = True
                        self.m3 = T(m3)
                        self.ub = T(ub)
                        self.res_b = T(res_b)
                        self.u2_nrb = T(nrb)

        pn = view.peer_norm
        live = np.isfinite(pn) & (pn > EPS)
        self.peer_norm_live = T(pn[live])
        self.log_peer_norm_live = T(np.log(pn[live]))
        ps = view.peer_step
        self.peer_step_fin = T(ps[np.isfinite(ps)])
        pr = view.peer_rep_raw
        self.peer_rep_fin = T(pr[np.isfinite(pr)])

        pool = [e for e in view.pool_I if e is not None]
        self.pool_ok = False
        if pool:
            Ip = np.stack([e.I for e in pool])
            Mp = np.stack([e.mask for e in pool])
            Rp = np.stack([(e.I ** 2).sum(1) for e in pool])
            cnt = Mp @ view.t_own
            nb2 = Rp @ view.t_own
            keep = (cnt >= 2) & (nb2 > EPS * EPS)
            if keep.any():
                self.pool_ok = True
                self.pool_I = T(Ip[keep])
                self.pool_m = T(Mp[keep])
                self.pool_nb2 = T(nb2[keep])

        pe = [(k, e) for k, e in enumerate(view.peer_I) if e is not None]
        base_all = view.peer_x_const
        self.peer_x_const_only = T(np.zeros(0))
        self.px_ok = False
        if pe:
            Ij = np.stack([e.I for _, e in pe])
            Mj = np.stack([e.mask for _, e in pe])
            Rj = np.stack([(e.I ** 2).sum(1) for _, e in pe])
            idx = np.asarray([k for k, _ in pe])
            cnt = Mj @ view.t_own
            nb2 = Rj @ view.t_own
            movable = (cnt >= 2) & (nb2 > EPS * EPS)
            base = base_all[idx]
            if movable.any():
                self.px_ok = True
                self.pxI = T(Ij[movable])
                self.pxm = T(Mj[movable])
                self.px_nb2 = T(nb2[movable])
                b = base[movable]
                self.px_base = T(np.where(np.isfinite(b), b, -np.inf))
            fixed = base[~movable]
            fixed = fixed[np.isfinite(fixed)]
            self.peer_x_const_only = T(fixed)

        self.pf = None
        if use_cos and view.peer_I_full:
            PF = np.stack(view.peer_I_full)
            nn = np.linalg.norm(PF.reshape(len(PF), -1), axis=1)
            keep = nn > EPS
            if keep.any():
                self.pf = T(PF[keep])
                self.pf_norm = T(nn[keep])
                self.pf_sum = T(PF.sum(0))
                self.n_cohort = float(len(PF) + 1)

        u1o = view.peer_u1[np.isfinite(view.peer_u1)]
        u2o = view.peer_u2[np.isfinite(view.peer_u2)]
        self.u1_oth = T(u1o)
        self.u2_oth = T(u2o)

    @staticmethod
    def _median(x: torch.Tensor) -> torch.Tensor:
        s, _ = torch.sort(x)
        n = s.numel()
        if n % 2 == 1:
            return s[n // 2]
        return 0.5 * (s[n // 2 - 1] + s[n // 2])

    def _z_own(self, own: torch.Tensor, others: torch.Tensor) -> Optional[torch.Tensor]:
        if 1 + others.numel() < self.min_cohort:
            return None
        v = torch.cat([own.reshape(1), others.reshape(-1)])
        med = self._median(v)
        mad = self._median(torch.abs(v - med))
        return (own - med) / (MAD_SCALE * mad + self.mad_floor)

    def forward(self, C: torch.Tensor) -> Dict[str, torch.Tensor]:
        t = self.t
        A = (C - self.Cbar) * t[:, None]
        a_rn2 = (A * A).sum(1)
        norm = torch.sqrt(torch.clamp(a_rn2.sum(), min=DEN_FLOOR))
        terms: Dict[str, torch.Tensor] = {}
        raw: Dict[str, torch.Tensor] = {}

        if self.u1_ok:
            inner = (A * self.prevI).sum()
            na2 = (self.prev_m * a_rn2).sum()
            u1 = inner / torch.sqrt(torch.clamp(na2 * self.u1_nb2, min=DEN_FLOOR))
            raw["u1"] = u1
            z = self._z_own(u1, self.u1_oth)
            if z is not None:
                terms["u1"] = torch.abs(z)

        if self.u2_ok:
            Am = A * self.m3[:, None]
            ra = Am - (Am * self.ub).sum() * self.ub
            nra = torch.sqrt(torch.clamp((ra * ra).sum(), min=DEN_FLOOR))
            u2 = (ra * self.res_b).sum() / (nra * self.u2_nrb)
            raw["u2"] = u2
            z = self._z_own(u2, self.u2_oth)
            if z is not None:
                terms["u2"] = torch.abs(z)

        med_norm = self._median(torch.cat([norm.reshape(1), self.peer_norm_live]))
        log_med = torch.log(torch.clamp(med_norm, min=DEN_FLOOR))
        g = torch.log(norm) - log_med
        raw["g"] = g
        peers_g = self.log_peer_norm_live - log_med
        z = self._z_own(g, peers_g)
        if z is not None:
            terms["g"] = torch.abs(z) if self.g_two_sided else -z

        if self.pool_ok:
            inner = (self.pool_I * A).sum(dim=(1, 2))
            na2 = self.pool_m @ a_rn2
            cos = inner / torch.sqrt(torch.clamp(na2 * self.pool_nb2, min=DEN_FLOOR))
            x = torch.max(cos)
            raw["x"] = x
            if self.px_ok:
                innq = (self.pxI * A).sum(dim=(1, 2))
                na2q = self.pxm @ a_rn2
                cosq = innq / torch.sqrt(torch.clamp(na2q * self.px_nb2, min=DEN_FLOOR))
                px = torch.maximum(cosq, self.px_base)
                peers_x = torch.cat([px, self.peer_x_const_only])
            else:
                peers_x = self.peer_x_const_only
            z = self._z_own(x, peers_x)
            if z is not None:
                terms["x"] = z

        step_own = torch.sqrt(torch.clamp(((C - self.C_prev) ** 2).sum(), min=DEN_FLOOR))
        cohort_step = self._median(torch.cat([step_own.reshape(1), self.peer_step_fin]))
        if self.C_lags is not None:
            ds = torch.sqrt(torch.clamp(((C.unsqueeze(0) - self.C_lags) ** 2).sum(dim=(1, 2)),
                                        min=DEN_FLOOR))
            rep = -torch.min(ds) / cohort_step
            raw["rep"] = rep
            peers_rep = -self.peer_rep_fin / cohort_step
            z = self._z_own(rep, peers_rep)
            if z is not None:
                terms["rep"] = z

        if self.pf is not None:
            Af = C - self.Cbar
            M = (self.pf_sum + Af) / self.n_cohort
            nM = torch.sqrt(torch.clamp((M * M).sum(), min=DEN_FLOOR))
            nA = torch.sqrt(torch.clamp((Af * Af).sum(), min=DEN_FLOOR))
            c_own = -(Af * M).sum() / (nA * nM)
            c_peers = -torch.einsum("pkd,kd->p", self.pf, M) / (self.pf_norm * nM)
            raw["cos"] = c_own
            z = self._z_own(c_own, c_peers)
            if z is not None:
                terms["cos"] = z

        nan = torch.as_tensor(float("nan"), dtype=self.dt, device=self.dev)
        out: Dict[str, torch.Tensor] = {"norm": norm}
        out.update(terms)
        for k in ("u1", "u2", "g", "x", "rep", "cos"):
            out["_raw_" + k] = raw.get(k, nan)
        if terms:
            st = torch.stack(list(terms.values()))
            out["_terms"] = st
            out["_term_names"] = list(terms)
            out["suspicion"] = torch.max(st)
        else:
            out["_terms"] = torch.zeros(0, dtype=self.dt, device=self.dev)
            out["_term_names"] = []
            out["suspicion"] = nan
        return out


@dataclass
class AttackConfig:
    phi: float = 0.0
    beta: float = 1.0
    knowledge: str = "peer_current"
    basis_steps: int = 150
    full_steps: int = 200
    restarts: int = 4
    lr_basis: float = 0.08
    lr_full: float = 0.015
    temp0: float = 4.0
    temp1: float = 60.0
    mad_floor: float = 1e-2
    lag_window: int = 5
    min_cohort: int = 5
    n_random_basis: int = 3
    device: str = "cpu"
    quantise: bool = True
    g_two_sided: bool = False
    use_cos: bool = False
    aim: str = "min"
    aim_margin: float = 0.0


def _gram_schmidt(cands: List[np.ndarray], free: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    out: List[np.ndarray] = []
    for c in cands:
        v = np.asarray(c, dtype=np.float64) * free[:, None]
        for u in out:
            v = v - float((v * u).sum()) * u
        n = float(np.linalg.norm(v))
        if n > tol * max(1.0, float(np.linalg.norm(c))):
            out.append(v / n)
    return np.stack(out) if out else np.zeros((0,) + cands[0].shape)


class WhiteBoxAttacker:
    def __init__(self, cfg: AttackConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.log: List[dict] = []

    def _cheap_objective(self, Cbar_prev: np.ndarray, C_honest: np.ndarray) -> np.ndarray:
        return Cbar_prev + self.cfg.phi * (C_honest - Cbar_prev)

    def _optimise(self, sur: "SuspicionSurrogate", Ct: torch.Tensor, free: torch.Tensor,
             B: float, steps: int, lr0: float, init: torch.Tensor,
             basis: Optional[torch.Tensor], aim: float = 0.0) -> Tuple[torch.Tensor, float]:
        cfg = self.cfg
        p = init.clone().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr0)
        best_d = (torch.einsum("m,mkd->kd", init, basis) if basis is not None else init).detach().clone()
        best_s = float("inf")
        for it in range(steps):
            frac = it / max(1, steps - 1)
            for grp in opt.param_groups:
                grp["lr"] = lr0 * (0.5 * (1 + math.cos(math.pi * frac)))
            temp = cfg.temp0 * (cfg.temp1 / cfg.temp0) ** frac
            opt.zero_grad(set_to_none=True)
            delta = torch.einsum("m,mkd->kd", p, basis) if basis is not None else p * free[:, None]
            o = sur.forward(Ct + delta)
            terms = o["_terms"]
            if terms.numel() == 0:
                break
            s = float(o["suspicion"].detach())
            obj = abs(s - aim)
            if obj < best_s:
                best_s, best_d = obj, delta.detach().clone()
            smooth = torch.logsumexp(temp * terms, dim=0) / temp
            loss = smooth if aim == 0.0 else (smooth - aim) ** 2
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

    def craft(self, aud: InnovationAuditor, r: int, victim: int,
              C_honest: np.ndarray, present_r: Dict[int, np.ndarray],
              peer_C_r: Optional[Dict[int, np.ndarray]]) -> Tuple[np.ndarray, dict]:
        cfg = self.cfg
        Cbar_prev = aud.cbar.get(r - 1)
        if Cbar_prev is None:
            return C_honest.copy(), {"round": r, "status": "no_aggregate",
                                     "predicted_suspicion": float("nan")}
        C_cheap = self._cheap_objective(Cbar_prev, C_honest)
        view = build_view(aud, r, victim, present_r, peer_C_r, cfg.knowledge, cfg.lag_window)
        if view is None:
            return C_cheap, {"round": r, "status": "undefined_innovation",
                             "predicted_suspicion": float("nan")}
        scale = view.cohort_scale
        if not np.isfinite(scale) or scale <= EPS:
            return C_cheap, {"round": r, "status": "no_cohort_scale",
                             "predicted_suspicion": float("nan")}
        B = cfg.beta * scale

        sur = SuspicionSurrogate(view, cfg.mad_floor, cfg.min_cohort, cfg.device,
                                 g_two_sided=cfg.g_two_sided, use_cos=cfg.use_cos)
        dev, dt = cfg.device, torch.float64
        Ct = torch.as_tensor(C_cheap, dtype=dt, device=dev)
        free_np = np.asarray(present_r[victim], bool).astype(np.float64)
        free = torch.as_tensor(free_np, dtype=dt, device=dev)
        n_free = max(1.0, float(free_np.sum()) * view.d)

        aim = 0.0
        if cfg.aim == "mimic" and aud.scores:
            vals = [v.get("suspicion", float("nan")) for c, v in aud.scores[-1].per_client.items()
                    if c != victim]
            vals = [v for v in vals if np.isfinite(v) and v < SCORE_STRUCTURAL]
            if vals:
                aim = float(np.median(vals))
                if cfg.aim_margin:
                    mad = float(np.median(np.abs(np.asarray(vals) - aim)))
                    aim = max(0.0, aim - cfg.aim_margin * MAD_SCALE * mad)

        with torch.no_grad():
            base_s = float(sur.forward(Ct)["suspicion"])
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
        for _ in range(cfg.n_random_basis):
            cands.append(self.rng.standard_normal(view.Cbar_prev.shape))
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
                a = torch.as_tensor(self.rng.standard_normal(m), dtype=dt, device=dev)
                a = a / torch.linalg.norm(a) * (B * float(self.rng.uniform(0.3, 1.0)))
                inits.append(a)
            for a0 in inits:
                d, s = self._optimise(sur, Ct, free, B, cfg.basis_steps, cfg.lr_basis * B, a0, Fb, aim)
                if s < best_s:
                    best_s, best_delta = s, d

        if cfg.full_steps > 0:
            lr = cfg.lr_full * B / math.sqrt(n_free)
            d, s = self._optimise(sur, Ct, free, B, cfg.full_steps, lr, best_delta.clone(), None, aim)
            if s < best_s:
                best_s, best_delta = s, d

        with torch.no_grad():
            Cn = (Ct + best_delta).detach().cpu().numpy()
            if cfg.quantise:
                Cn = np.round(Cn * FIXED_POINT_SCALE) / FIXED_POINT_SCALE
            s_q = float(sur.forward(torch.as_tensor(Cn, dtype=dt, device=dev))["suspicion"])

        info = {"round": r, "status": "ok", "predicted_suspicion": s_q,
                "objective_gap": best_s, "aim": aim, "cheap_suspicion": base_s,
                "budget": B, "cohort_scale": scale, "basis_dim": int(Fn.shape[0])}
        self.log.append(info)
        return Cn, info
