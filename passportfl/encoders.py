from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import identity as idm
from .passport import PassportResNet18, PassportSpec, build_model


class PlainEncoderMixin:
    family: str = "plain"

    def passport_spec(self) -> PassportSpec:
        return PassportSpec([])

    def install_passport(self, tensors: Dict[str, torch.Tensor]):
        self.passport_tensors = {}

    def install_dynamic(self, E: Optional[bytes]):
        return None

    def genesis(self, entropy: Optional[bytes] = None) -> Dict[str, torch.Tensor]:
        self.passport_tensors = {}
        return {}


class PlainResNet18(PassportResNet18):
    def __init__(self, num_classes: int = 100, embed_dim: int = 128, width: int = 64):
        super().__init__(num_classes, embed_dim, width, passport_stages=(), alpha=0.0)
        self.family = "PlainResNet18-w%d" % width


class CNN4(PlainEncoderMixin, nn.Module):
    family = "CNN4"

    def __init__(self, num_classes: int = 100, embed_dim: int = 128, hidden: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, hidden)
        self.embed = nn.Linear(hidden, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.passport_tensors: Dict[str, torch.Tensor] = {}

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = F.max_pool2d(F.relu(self.conv1(x)), 2)
        out = F.max_pool2d(F.relu(self.conv2(out)), 2)
        out = F.relu(self.fc1(out.flatten(1)))
        return self.embed(out)

    def forward(self, x: torch.Tensor):
        z = self.features(x)
        return self.classifier(z), z


class InvertedResidual(nn.Module):
    def __init__(self, cin: int, cout: int, expansion: int, stride: int):
        super().__init__()
        hidden = cin * expansion
        layers: List[nn.Module] = []
        if expansion != 1:
            layers += [nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
                   nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
                   nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.block = nn.Sequential(*layers)
        self.use_res = stride == 1 and cin == cout

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out


class MobileNetV2(PlainEncoderMixin, nn.Module):
    family = "MobileNetV2-cifar"
    CFG = ((1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
           (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1))

    def __init__(self, num_classes: int = 100, embed_dim: int = 128, last_channels: int = 1280):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32),
                                  nn.ReLU6(inplace=True))
        blocks: List[nn.Module] = []
        cin = 32
        for t, c, n, s in self.CFG:
            for i in range(n):
                blocks.append(InvertedResidual(cin, c, t, s if i == 0 else 1))
                cin = c
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Conv2d(cin, last_channels, 1, bias=False),
                                  nn.BatchNorm2d(last_channels), nn.ReLU6(inplace=True))
        self.embed = nn.Linear(last_channels, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.passport_tensors: Dict[str, torch.Tensor] = {}

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.blocks(self.stem(x)))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.embed(out)

    def forward(self, x: torch.Tensor):
        z = self.features(x)
        return self.classifier(z), z


ENCODERS = {
    "resnet18_w64": lambda K, d: PlainResNet18(K, d, 64),
    "resnet18_w32": lambda K, d: PlainResNet18(K, d, 32),
    "cnn4": lambda K, d: CNN4(K, d),
    "mobilenetv2": lambda K, d: MobileNetV2(K, d),
}

GROUPS = {
    "resnet18": ("resnet18",),
    "hetero4": ("resnet18_w64", "resnet18_w32", "cnn4", "mobilenetv2"),
}


def group_assignment(group: str, clients: int) -> List[str]:
    g = GROUPS[group]
    return [g[i % len(g)] for i in range(clients)]


def build_group_model(group: str, cid: int, num_classes: int = 100, embed_dim: int = 128,
                      width: int = 64, passport: bool = True) -> nn.Module:
    if group == "resnet18":
        return build_model(num_classes, embed_dim, width, passport, 0.0)
    if passport:
        raise ValueError("architecture group %r has no passport variant; use --no-passport" % group)
    name = group_assignment(group, cid + 1)[cid]
    return ENCODERS[name](num_classes, embed_dim)


def architecture_descriptor(model) -> bytes:
    spec = model.passport_spec()
    d = {"family": getattr(model, "family", "PassportResNet18"), "embed_dim": model.embed_dim,
         "num_classes": model.num_classes, "passport_in_channels": spec.in_channels}
    return idm.canonical_json(d)


def rebind_identity(holder) -> None:
    from .probe import MerkleTree, embed_probes, leaf_hash, make_probes
    if holder.round != 0:
        raise RuntimeError("rebind_identity must run before the first round")
    holder.arch = architecture_descriptor(holder.model)
    pre_G = idm.H(b"preG|" + holder.arch + b"|" + holder.pk)
    holder.probes = make_probes(pre_G, holder.m).to(holder.device)
    embs0 = embed_probes(holder.model, holder.probes, holder.device)
    holder.F0 = MerkleTree([leaf_hash(e) for e in embs0]).root
    holder.G = idm.genesis_id(holder.arch, holder.pk, holder.F0)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def self_check(num_classes: int = 100, embed_dim: int = 128) -> Dict[str, dict]:
    out = {}
    x = torch.randn(4, 3, 32, 32)
    for name, make in ENCODERS.items():
        m = make(num_classes, embed_dim).eval()
        with torch.no_grad():
            logits, z = m(x)
            z2 = m.features(x)
        assert logits.shape == (4, num_classes) and z.shape == (4, embed_dim), name
        assert torch.allclose(z, z2), name
        assert m.passport_spec().in_channels == [] and m.genesis() == {}, name
        assert m.passport_tensors == {} and m.install_dynamic(b"x" * 32) is None, name
        out[name] = {"family": m.family, "params": n_params(m),
                     "descriptor": architecture_descriptor(m).decode()}
    ref = build_model(num_classes, embed_dim, 64, False, 0.0)
    assert architecture_descriptor(ref) == idm.architecture_descriptor(ref)
    descs = [v["descriptor"] for v in out.values()]
    assert len(set(descs)) == len(descs), "descriptors must differ across encoders"
    for k, v in out.items():
        print("%-14s %-22s %10d params" % (k, v["family"], v["params"]))
    return out
