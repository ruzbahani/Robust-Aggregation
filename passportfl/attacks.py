from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from .fedproto import Client
from .protocol import Contribution


@dataclass
class AttackConfig:
    kind: str = "none"
    start_round: int = 10
    strength: float = 1.0
    target_cid: Optional[int] = None


class AttackerClient(Client):
    def __init__(self, *args, attack: AttackConfig, **kw):
        super().__init__(*args, **kw)
        self.attack = attack
        self.peer_C: Optional[np.ndarray] = None
        self.attack_log: List[Dict] = []

    def _active(self) -> bool:
        return self.attack.kind != "none" and (self.holder.round + 1) >= self.attack.start_round

    def _blend(self, C_own: np.ndarray, C_foreign: np.ndarray, alpha: float) -> np.ndarray:
        return (1.0 - alpha) * C_own + alpha * C_foreign

    def round_step(self) -> Contribution:
        kind = self.attack.kind
        if not self._active():
            return super().round_step()

        if kind == "free_rider":
            C_own, counts, intra, inter = self.compute_prototypes()
            C = C_own if self.global_C is None else self._blend(
                C_own, self.global_C.cpu().numpy(), self.attack.strength)
            stats = {"train_loss": 0.0, "local_acc": 0.0}

        elif kind in ("substitution", "adaptive_blend"):
            stats = self.local_train()
            C_own, counts, intra, inter = self.compute_prototypes()
            C = C_own if self.peer_C is None else self._blend(C_own, self.peer_C, self.attack.strength)

        elif kind == "co_move":
            stats = self.local_train()
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.dim() > 1:
                        p.add_(torch.randn_like(p) * self.attack.strength * p.std())
            C_own, counts, intra, inter = self.compute_prototypes()
            C = C_own if self.peer_C is None else self._blend(C_own, self.peer_C, 1.0)

        elif kind == "model_swap":
            stats = self.local_train()
            if (self.holder.round + 1) == self.attack.start_round:
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.dim() > 1 and torch.rand(1).item() < self.attack.strength:
                            torch.nn.init.kaiming_normal_(p)
            C_own, counts, intra, inter = self.compute_prototypes()
            C = C_own

        elif kind == "record_fabrication":
            stats = self.local_train()
            C, counts, intra, inter = self.compute_prototypes()
            self._last_counts = counts * self.attack.strength
            contrib = self.holder.issue(C, dict(n_samples=int(self.n_samples * self.attack.strength),
                                                train_loss=stats["train_loss"], local_acc=stats["local_acc"],
                                                intra_compactness=intra, inter_separation=inter))
            self.calibrate()
            return contrib

        else:
            raise ValueError(f"unknown attack kind {kind!r}")

        self._last_counts = counts
        contrib = self.holder.issue(C, dict(n_samples=self.n_samples, train_loss=stats["train_loss"],
                                            local_acc=stats["local_acc"], intra_compactness=intra,
                                            inter_separation=inter))
        self.calibrate()
        return contrib


def attach_peer_prototypes(clients: List[Client]) -> None:
    honest = [c for c in clients if not isinstance(c, AttackerClient) or c.attack.kind == "none"]
    if not honest:
        return
    cache: Dict[int, np.ndarray] = {}
    for c in clients:
        if isinstance(c, AttackerClient) and c.attack.kind in ("substitution", "adaptive_blend", "co_move"):
            victim = honest[c.cid % len(honest)]
            if victim.cid not in cache:
                cache[victim.cid] = victim.compute_prototypes()[0]
            c.peer_C = cache[victim.cid]
