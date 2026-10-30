from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .audit import TrajectoryAuditor, link_from_verify
from .passport import PassportResNet18
from .protocol import Contribution, PassportHolder, Receiver, VerifyResult
from .reproducibility import experiment_entropy


@dataclass
class FedConfig:
    num_classes: int = 100
    embed_dim: int = 128
    width: int = 64
    rounds: int = 50
    local_epochs: int = 2
    batch_size: int = 64
    lr: float = 0.01
    lam: float = 1.0
    penalty: str = "mse"
    alpha_dyn: float = 0.0
    calib_steps: int = 20
    calib_lr: float = 0.001
    m_probes: int = 256
    t_reveal: int = 32
    delta_per_round: float = 1.0
    max_proto_ratio: float = 1.0
    min_label_stability: float = 0.5
    warmup_rounds: int = 5
    audit_min_history: int = 6
    audit_z_threshold: float = 4.0
    grad_clip: float = 5.0
    scheme: str = "ML-DSA-44"
    device: str = "cpu"
    experiment_seed: Optional[int] = None


class Client:
    def __init__(self, cid: int, model: PassportResNet18, train_ds: Dataset, cfg: FedConfig, entropy: Optional[bytes] = None):
        self.cid = cid
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.train_ds = train_ds
        self.loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
        external_key = None
        if cfg.experiment_seed is not None:
            if entropy is None:
                entropy = experiment_entropy(cfg.experiment_seed, cid, "federation-v1")
            external_key = experiment_entropy(cfg.experiment_seed, cid, "external-key-v1")
        self.holder = PassportHolder(self.model, cfg.scheme, cfg.m_probes, cfg.t_reveal,
                                     cfg.device, entropy, external_key=external_key)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
        self.global_C: Optional[torch.Tensor] = None
        self.n_samples = len(train_ds)

    def _loss(self, x, y):
        logits, z = self.model(x)
        loss = F.cross_entropy(logits, y)
        if self.global_C is not None:
            if self.cfg.penalty == "euclid":
                loss = loss + self.cfg.lam * (z - self.global_C[y]).norm(dim=1).mean()
            else:
                loss = loss + self.cfg.lam * F.mse_loss(z, self.global_C[y])
        return loss, logits

    def local_train(self, epochs: Optional[int] = None,
                    max_steps: Optional[int] = None) -> Dict[str, float]:
        self.model.train()
        epochs = self.cfg.local_epochs if epochs is None else epochs
        tot, n, correct, skipped, steps = 0.0, 0, 0, 0, 0
        for _ in range(epochs):
            for x, y in self.loader:
                if max_steps is not None and steps >= max_steps:
                    break
                steps += 1
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)
                loss, logits = self._loss(x, y)
                if not torch.isfinite(loss):
                    self.opt.zero_grad(set_to_none=True)
                    skipped += 1
                    continue
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()
                tot += loss.item() * len(y)
                n += len(y)
                correct += (logits.argmax(1) == y).sum().item()
        self.skipped_batches = skipped
        return {"train_loss": tot / max(n, 1), "local_acc": correct / max(n, 1)}

    def calibrate(self):
        if self.cfg.alpha_dyn == 0.0 or self.cfg.calib_steps == 0:
            return
        opt = torch.optim.SGD(self.model.parameters(), lr=self.cfg.calib_lr, momentum=0.9)
        self.model.train()
        it = iter(self.loader)
        for _ in range(self.cfg.calib_steps):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(self.loader)
                x, y = next(it)
            x, y = x.to(self.cfg.device), y.to(self.cfg.device)
            loss, _ = self._loss(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def compute_prototypes(self, dataset: Optional[Dataset] = None):
        self.model.eval()
        K, d = self.cfg.num_classes, self.cfg.embed_dim
        sums = torch.zeros(K, d, device=self.cfg.device)
        counts = torch.zeros(K, device=self.cfg.device)
        sq = torch.zeros(K, device=self.cfg.device)
        zs, ys = [], []
        for x, y in DataLoader(self.train_ds if dataset is None else dataset, batch_size=256):
            x, y = x.to(self.cfg.device), y.to(self.cfg.device)
            z = self.model.features(x)
            sums.index_add_(0, y, z)
            counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float))
            zs.append(z); ys.append(y)
        present = counts > 0
        C = torch.zeros(K, d, device=self.cfg.device)
        C[present] = sums[present] / counts[present, None]
        if self.global_C is not None:
            C[~present] = self.global_C[~present]
        z, y = torch.cat(zs), torch.cat(ys)
        intra = (z - C[y]).norm(dim=1).mean().item()
        Cp = C[present]
        if Cp.shape[0] > 1:
            pd = torch.cdist(Cp, Cp)
            inter = pd[~torch.eye(len(Cp), dtype=bool, device=pd.device)].mean().item()
        else:
            inter = 0.0
        return C.cpu().numpy(), counts.cpu().numpy(), intra, inter

    def round_step(self) -> Contribution:
        stats = self.local_train()
        C, counts, intra, inter = self.compute_prototypes()
        self._last_counts = counts
        contrib = self.holder.issue(C, dict(n_samples=self.n_samples, train_loss=stats["train_loss"],
                                            local_acc=stats["local_acc"], intra_compactness=intra,
                                            inter_separation=inter))
        self.calibrate()
        return contrib

    def receive_global(self, C: np.ndarray):
        self.global_C = torch.from_numpy(C).float().to(self.cfg.device)

    @torch.no_grad()
    def evaluate(self, test_ds: Dataset, use_prototypes: bool = True) -> Dict[str, float]:
        self.model.eval()
        correct_cls, correct_proto, n = 0, 0, 0
        C = self.global_C
        for x, y in DataLoader(test_ds, batch_size=256):
            x, y = x.to(self.cfg.device), y.to(self.cfg.device)
            logits, z = self.model(x)
            correct_cls += (logits.argmax(1) == y).sum().item()
            if use_prototypes and C is not None:
                correct_proto += (torch.cdist(z, C).argmin(1) == y).sum().item()
            n += len(y)
        out = {"acc_classifier": correct_cls / n}
        if use_prototypes and C is not None:
            out["acc_prototype"] = correct_proto / n
        return out


class Server:
    def __init__(self, cfg: FedConfig):
        self.cfg = cfg
        self.receiver = Receiver(cfg.m_probes, cfg.t_reveal, cfg.delta_per_round, cfg.max_proto_ratio,
                                 cfg.min_label_stability, cfg.warmup_rounds)
        self.auditor = TrajectoryAuditor(cfg.audit_min_history, cfg.audit_z_threshold)
        self.global_C: Optional[np.ndarray] = None
        self.log: List[Dict] = []
        self.degenerate_rounds: List[int] = []

    def aggregate(self, contribs: List[Contribution], counts: List[np.ndarray]) -> np.ndarray:
        K, d = self.cfg.num_classes, self.cfg.embed_dim
        num = np.zeros((K, d)); den = np.zeros(K)
        accepted = []
        for c, cnt in zip(contribs, counts):
            st = self.receiver.peers.get(c.client_G)
            C_prev = st.C_prev if st is not None else None
            res: VerifyResult = self.receiver.verify(c)
            verdict = self.auditor.observe(
                c.client_G, link_from_verify(c.round, c.E_new, res.probe, c.record, c.C, C_prev, res.accepted))
            self.log.append({"round": c.round, "G": c.client_G.hex()[:8], "accepted": res.accepted, "reason": res.reason,
                             "wire_bytes": c.wire_size(),
                             "proto_ratio": res.probe.proto_drift_ratio if res.probe else None,
                             "stability": res.probe.label_stability if res.probe else None,
                             "max_drift": res.probe.max_drift if res.probe else None,
                             "proto_velocity": res.probe.proto_velocity if res.probe else None,
                             "probe_velocity": res.probe.probe_velocity if res.probe else None,
                             "audit": verdict.to_dict()})
            if not res.accepted:
                continue
            accepted.append(c)
            num += c.C * cnt[:, None]
            den += cnt
        C = np.zeros((K, d))
        has = den > 0
        C[has] = num[has] / den[has, None]
        if self.global_C is not None:
            C[~has] = self.global_C[~has]
        if not accepted:
            rnd = contribs[0].round if contribs else -1
            print(f"[WARN] round {rnd}: 0/{len(contribs)} contributions accepted; "
                  f"global prototypes frozen at the previous round", file=sys.stderr)
            self.degenerate_rounds.append(rnd)
        self.global_C = C
        return C
