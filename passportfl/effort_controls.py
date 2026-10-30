from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
from torch.optim.swa_utils import update_bn
from torch.utils.data import DataLoader, Dataset, TensorDataset

BN_STATISTICS = ("running_mean", "running_var", "num_batches_tracked")
RECALIBRATION_BATCH = 256


def is_bn_statistic(key: str) -> bool:
    return key.rsplit(".", 1)[-1] in BN_STATISTICS


def hybrid_state(parameters: Dict[str, torch.Tensor], statistics: Dict[str, torch.Tensor]) -> dict:
    if parameters.keys() != statistics.keys():
        raise ValueError("state dicts of different architectures")
    return {k: (statistics[k] if is_bn_statistic(k) else parameters[k]) for k in parameters}


def batchnorm_summary(model: torch.nn.Module) -> dict:
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    return {"modules": len(bns),
            "momenta": sorted({m.momentum for m in bns}, key=lambda v: -1 if v is None else v),
            "non_affine": sum(not m.affine for m in bns),
            "statistic_buffers": sum(is_bn_statistic(k) for k in model.state_dict())}


def cached_dataset(ds: Dataset) -> TensorDataset:
    xs, ys = zip(*(ds[i] for i in range(len(ds))))
    return TensorDataset(torch.stack(xs), torch.as_tensor(ys))


@torch.no_grad()
def recalibrated_state(client, state: Dict[str, torch.Tensor], dataset: Dataset,
                       batch_size: int = RECALIBRATION_BATCH) -> dict:
    client.model.load_state_dict(state)
    update_bn(DataLoader(dataset, batch_size=batch_size), client.model, device=client.cfg.device)
    out = copy.deepcopy(client.model.state_dict())
    if any(not torch.equal(out[k], state[k]) for k in state if not is_bn_statistic(k)):
        raise AssertionError("recalibration changed a parameter")
    return out


def forward_only(client, max_passes: Optional[int] = None, epochs: Optional[int] = None) -> Dict[str, float]:
    client.model.train()
    epochs = client.cfg.local_epochs if epochs is None else epochs
    tot, n, correct, skipped, passes = 0.0, 0, 0, 0, 0
    with torch.no_grad():
        for _ in range(epochs):
            for x, y in client.loader:
                if max_passes is not None and passes >= max_passes:
                    break
                passes += 1
                x, y = x.to(client.cfg.device), y.to(client.cfg.device)
                loss, logits = client._loss(x, y)
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                tot += loss.item() * len(y)
                n += len(y)
                correct += (logits.argmax(1) == y).sum().item()
    client.skipped_batches = skipped
    client.forward_passes = passes
    return {"train_loss": tot / max(n, 1), "local_acc": correct / max(n, 1)}
