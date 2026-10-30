from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12
QUANT_STEP = 1.0 / (1 << 16)


def _carry(arch: dict, cid: int, r: int) -> np.ndarray:
    cb = arch["cbar"]
    if (r - 1) in cb:
        return cb[r - 1].copy()
    return arch["C"][cid][r].copy()


def _row_norms(A: np.ndarray) -> np.ndarray:
    return np.linalg.norm(A, axis=1)


def _true_support(arch: dict, cid: int, r: int) -> np.ndarray:
    return np.asarray(arch["present"][cid][r], dtype=bool)


def cohort_innovation_norm(arch: dict, cid: int, r: int) -> float:
    if (r - 1) not in arch["cbar"] or (r - 2) not in arch["cbar"]:
        return float("nan")
    base = arch["cbar"][r - 2]
    ns = []
    for j in arch["C"]:
        if j == cid or (r - 1) not in arch["C"][j]:
            continue
        m = np.asarray(arch["present"][j][r - 1], dtype=bool)
        ns.append(float(np.linalg.norm((arch["C"][j][r - 1] - base)[m])))
    return float(np.median(ns)) if ns else float("nan")


def select_rows(arch: dict, cid: int, r: int, j: int, rule: str,
                rng: np.random.Generator, state: dict) -> np.ndarray:
    order = rank_rows(arch, cid, r, rule, rng)
    return np.sort(order[:j]) if j < len(order) else np.sort(order)


def rank_rows(arch: dict, cid: int, r: int, rule: str,
              rng: np.random.Generator) -> np.ndarray:
    S = np.where(_true_support(arch, cid, r))[0]
    if rule == "first":
        return S
    if rule == "random":
        return rng.permutation(S)

    base = _carry(arch, cid, r)
    I = (arch["C"][cid][r] - base)[S]
    if rule == "largest":
        score = _row_norms(I)
    elif rule == "stable":
        if (r - 2) in arch["cbar"] and (r - 1) in arch["C"][cid]:
            Ip = (arch["C"][cid][r - 1] - arch["cbar"][r - 2])[S]
            na, nb = _row_norms(I), _row_norms(Ip)
            good = (na > EPS) & (nb > EPS)
            score = np.where(good, (I * Ip).sum(1) / np.maximum(na * nb, EPS), -np.inf)
        else:
            score = _row_norms(I)
    else:
        raise ValueError(f"unknown selection rule {rule!r}")
    return S[np.argsort(-score)]


def craft_rows(arch: dict, cid: int, r: int, rows: np.ndarray, craft: str,
               rng: np.random.Generator, published: Optional[Dict[int, np.ndarray]],
               shrink_frac: float = 0.1) -> np.ndarray:
    base = _carry(arch, cid, r)
    if craft == "echo":
        return base[rows]
    if craft == "honest":
        return arch["C"][cid][r][rows]
    if craft == "shrink":
        return base[rows] + shrink_frac * (arch["C"][cid][r][rows] - base[rows])
    if craft == "scaled":
        I = arch["C"][cid][r][rows] - base[rows]
        n = float(np.linalg.norm(I))
        tgt = cohort_innovation_norm(arch, cid, r)
        if not np.isfinite(tgt) or n <= EPS:
            return arch["C"][cid][r][rows]
        return base[rows] + (tgt / n) * I
    if craft == "ar2":
        hist = sorted(k for k in (published or {}) if k < r)
        if len(hist) < 3:
            return arch["C"][cid][r][rows]
        c1, c2, c3 = published[hist[-1]], published[hist[-2]], published[hist[-3]]
        ext = c1 + (c1 - c2) + 0.25 * ((c1 - c2) - (c2 - c3))
        return ext[rows]
    raise ValueError(f"unknown craft {craft!r}")


@dataclass
class MaskAttack:
    name: str
    craft: str = "honest"
    mode: str = "shrink"
    j: int = 2
    rule: str = "stable"
    n_blocks: int = 2
    reselect: bool = False
    shrink_frac: float = 0.1
    ramp_rounds: int = 12

    def declared(self, arch: dict, cid: int, r: int, rng: np.random.Generator,
                 state: dict) -> np.ndarray:
        S = _true_support(arch, cid, r)
        K = S.shape[0]
        m = np.zeros(K, dtype=bool)
        if self.mode == "none":
            return S.copy()
        if self.mode == "churn":
            idx = np.where(S)[0]
            blocks = np.array_split(idx, max(2, self.n_blocks))
            b = blocks[state.setdefault("churn_i", 0) % len(blocks)]
            state["churn_i"] = state.get("churn_i", 0) + 1
            if self.j and self.j < len(b):
                b = b[:self.j]
            m[b] = True
            return m
        if self.mode == "ramp":
            t = state.get("ramp_t", 0)
            state["ramp_t"] = t + 1
            order = state.get("ramp_order")
            if order is None:
                order = rank_rows(arch, cid, r, self.rule, rng)
                state["ramp_order"] = order
            n0 = len(order)
            frac = min(1.0, t / max(1, self.ramp_rounds))
            size = int(round(n0 * (float(self.j) / max(1, n0)) ** frac))
            m[order[:max(self.j, min(n0, size))]] = True
            return m
        rows = state.get("rows")
        if rows is None or self.reselect:
            rows = select_rows(arch, cid, r, self.j, self.rule, rng, state)
            state["rows"] = rows
        m[rows] = True
        return m

    def payload(self, arch: dict, cid: int, r: int, rng: np.random.Generator,
                state: dict) -> Tuple[np.ndarray, np.ndarray]:
        mask = self.declared(arch, cid, r, rng, state)
        rows = np.where(mask)[0]
        C = _carry(arch, cid, r)
        if len(rows):
            C[rows] = craft_rows(arch, cid, r, rows, self.craft, rng,
                                 state.get("published"), self.shrink_frac)
        return C, mask


def training_saving(counts: np.ndarray, declared: np.ndarray, craft: str,
                    shrink_frac: float = 0.1) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    tot = counts.sum()
    if tot <= 0:
        return float("nan")
    kept = counts[np.asarray(declared, dtype=bool)].sum()
    if craft == "echo":
        return 1.0
    if craft == "shrink":
        return 1.0 - shrink_frac * kept / tot
    return 1.0 - kept / tot


def build_registry(j_grid: Sequence[int] = (100, 50, 20, 10, 5, 3, 2, 1),
                   crafts: Sequence[str] = ("echo", "shrink", "honest", "scaled"),
                   rules: Sequence[str] = ("stable",)) -> Dict[str, MaskAttack]:
    reg: Dict[str, MaskAttack] = {}
    for craft in crafts:
        for rule in rules:
            for j in j_grid:
                tag = "" if len(rules) == 1 else f"_{rule}"
                reg[f"shrink{tag}_{craft}_j{j}"] = MaskAttack(
                    name=f"shrink{tag}_{craft}_j{j}", craft=craft, mode="shrink",
                    j=j, rule=rule)
        reg[f"shrink_first_{craft}_j5"] = MaskAttack(
            name=f"shrink_first_{craft}_j5", craft=craft, mode="shrink", j=5, rule="first")
        reg[f"churn_{craft}"] = MaskAttack(
            name=f"churn_{craft}", craft=craft, mode="churn", j=0, n_blocks=2)
        reg[f"churn3_{craft}"] = MaskAttack(
            name=f"churn3_{craft}", craft=craft, mode="churn", j=0, n_blocks=3)
        reg[f"nomask_{craft}"] = MaskAttack(
            name=f"nomask_{craft}", craft=craft, mode="none")
        for j, rr in ((5, 12), (5, 20), (2, 20)):
            reg[f"ramp{rr}_{craft}_j{j}"] = MaskAttack(
                name=f"ramp{rr}_{craft}_j{j}", craft=craft, mode="ramp", j=j,
                ramp_rounds=rr)
    return reg
