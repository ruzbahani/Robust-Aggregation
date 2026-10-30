from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fedproto import FedConfig, Server
from .protocol import Contribution

LAMBDA_ARMS = {"lam1": 1.0, "lam10": 10.0, "lam01": 0.1}
FEDTGP_PRIMARY_ARM = "lam1"
FEDTGP_DEFAULT_LAM = LAMBDA_ARMS[FEDTGP_PRIMARY_ARM]


@dataclass
class TGPConfig:
    server_epochs: int = 100
    server_lr: float = 0.01
    server_batch: int = 10
    margin_threshold: float = 100.0
    hidden_dim: Optional[int] = None
    seed: int = 0
    device: str = "cpu"


class TrainableGlobalPrototypes(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.embedings = nn.Embedding(num_classes, feature_dim)
        self.middle = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        return self.fc(self.middle(self.embedings(class_ids)))


def _seed_from(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def cluster_centres(payloads: Sequence[Tuple[np.ndarray, np.ndarray]], K: int):
    d = payloads[0][0].shape[1]
    num = np.zeros((K, d))
    cnt = np.zeros(K)
    for C, pres in payloads:
        pres = np.asarray(pres, bool)
        num[pres] += np.asarray(C, np.float64)[pres]
        cnt[pres] += 1
    has = cnt > 0
    Q = np.zeros((K, d))
    Q[has] = num[has] / cnt[has, None]
    return Q, has


def class_gaps(Q: np.ndarray, has: np.ndarray) -> Dict[str, object]:
    K = len(Q)
    gap = np.full(K, 1e9)
    idx = np.flatnonzero(has)
    if len(idx) >= 2:
        D = np.sqrt(np.maximum(((Q[idx, None, :] - Q[None, idx, :]) ** 2).sum(-1), 0.0))
        np.fill_diagonal(D, np.inf)
        gap[idx] = D.min(axis=1)
    min_gap = float(gap.min())
    gap[gap > 1e8] = min_gap
    return {"gap": gap, "min_gap": min_gap, "max_gap": float(gap.max())}


class TGPTrainer:
    def __init__(self, num_classes: int, feature_dim: int, cfg: TGPConfig):
        self.K, self.d, self.cfg = num_classes, feature_dim, cfg
        hidden = cfg.hidden_dim or feature_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_seed_from("fedtgp-init", cfg.seed))
            self.model = TrainableGlobalPrototypes(num_classes, feature_dim, hidden)
        self.model.to(cfg.device)
        self.ids = torch.arange(num_classes, device=cfg.device)

    def state(self) -> Dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy().copy() for k, v in self.model.state_dict().items()}

    def load_state(self, st: Dict[str, np.ndarray]) -> None:
        self.model.load_state_dict({k: torch.as_tensor(v) for k, v in st.items()})
        self.model.to(self.cfg.device)

    @torch.no_grad()
    def prototypes(self) -> np.ndarray:
        self.model.eval()
        return self.model(self.ids).double().cpu().numpy()

    def fit_round(self, payloads: Sequence[Tuple[np.ndarray, np.ndarray]], round_: int) -> dict:
        t0 = time.time()
        cfg, dev = self.cfg, self.cfg.device
        Q, has = cluster_centres(payloads, self.K)
        gaps = class_gaps(Q, has)
        margin = min(gaps["max_gap"], cfg.margin_threshold)
        X, Y = [], []
        for C, pres in payloads:
            ks = np.flatnonzero(np.asarray(pres, bool))
            X.append(np.asarray(C, np.float32)[ks])
            Y.append(ks)
        X = torch.from_numpy(np.concatenate(X)).to(dev)
        Y = torch.from_numpy(np.concatenate(Y).astype(np.int64)).to(dev)
        n = len(Y)
        gen = torch.Generator().manual_seed(_seed_from("fedtgp-shuffle", cfg.seed, round_))
        opt = torch.optim.SGD(self.model.parameters(), lr=cfg.server_lr)
        self.model.train()
        last_epoch_losses: List[float] = []
        loss = torch.zeros(())
        for e in range(cfg.server_epochs):
            perm = torch.randperm(n, generator=gen).to(dev)
            ep = []
            for b in range(0, n, cfg.server_batch):
                idx = perm[b:b + cfg.server_batch]
                proto, y = X[idx], Y[idx]
                gen_p = self.model(self.ids)
                fs = (proto ** 2).sum(1, keepdim=True)
                cs = (gen_p ** 2).sum(1, keepdim=True)
                dist = torch.sqrt((fs - 2 * proto @ gen_p.T + cs.T).clamp_min(1e-12))
                dist = dist + F.one_hot(y, self.K).to(dist.dtype) * margin
                loss = F.cross_entropy(-dist, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if e == cfg.server_epochs - 1:
                    ep.append(float(loss.detach()))
            if e == cfg.server_epochs - 1:
                last_epoch_losses = ep
        P = self.prototypes()
        diff = P - Q
        return {"round": int(round_), "P_hat": P, "Q": Q, "Q_has": has,
                "margin": float(margin), "max_gap": gaps["max_gap"], "min_gap": gaps["min_gap"],
                "n_pairs": int(n), "loss_last": float(loss.detach()),
                "loss_last_epoch_mean": float(np.mean(last_epoch_losses)) if last_epoch_losses else float("nan"),
                "p_minus_q_fro": float(np.linalg.norm(diff[has])),
                "p_minus_q_row_mean": float(np.linalg.norm(diff[has], axis=1).mean()) if has.any() else float("nan"),
                "seconds": time.time() - t0}


class TGPServer(Server):
    def __init__(self, cfg: FedConfig, tgp_cfg: Optional[TGPConfig] = None):
        super().__init__(cfg)
        self.tgp_cfg = tgp_cfg or TGPConfig(device=cfg.device,
                                            seed=cfg.experiment_seed or 0)
        self.trainer = TGPTrainer(cfg.num_classes, cfg.embed_dim, self.tgp_cfg)
        self.initial_state = self.trainer.state()
        self.tgp_log: List[dict] = []
        self.cbar_fedproto: Optional[np.ndarray] = None
        self.qbar: Optional[np.ndarray] = None
        self.qbar_has: Optional[np.ndarray] = None
        self.last_state: Dict[str, np.ndarray] = self.initial_state

    def aggregate(self, contribs: List[Contribution], counts: List[np.ndarray]) -> np.ndarray:
        prev_global = self.global_C
        n0 = len(self.log)
        C_fp = super().aggregate(contribs, counts)
        self.cbar_fedproto = C_fp.copy()
        accepted = [e["accepted"] for e in self.log[n0:]]
        payloads = [(c.C, np.asarray(cnt) > 0) for c, cnt, ok in zip(contribs, counts, accepted) if ok]
        rnd = contribs[0].round if contribs else -1
        if not payloads:
            P = prev_global if prev_global is not None else self.trainer.prototypes()
            self.global_C = P
            self.qbar, self.qbar_has = P.copy(), np.zeros(len(P), bool)
            self.tgp_log.append({"round": rnd, "skipped": True})
            return P
        info = self.trainer.fit_round(payloads, rnd)
        P = info.pop("P_hat")
        Q, has = info.pop("Q"), info.pop("Q_has")
        self.qbar = np.where(has[:, None], Q, P)
        self.qbar_has = has
        info["p_minus_cbar_fedproto_fro"] = float(np.linalg.norm(P - C_fp))
        info["q_minus_cbar_fedproto_fro"] = float(np.linalg.norm((Q - C_fp)[has]))
        info["classes_without_holder"] = int((~has).sum())
        self.tgp_log.append(info)
        self.last_state = self.trainer.state()
        self.global_C = P
        return P

    def config_dict(self) -> dict:
        d = asdict(self.tgp_cfg)
        d["hidden_dim"] = d["hidden_dim"] or self.cfg.embed_dim
        return d
