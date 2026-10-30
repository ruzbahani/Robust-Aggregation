from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

PASSPORT_SPATIAL = 8
DYN_BITS = 256


@dataclass
class PassportSpec:
    in_channels: List[int]

    def n_layers(self) -> int:
        return len(self.in_channels)


def make_passport_tensors(spec: PassportSpec, entropy: Optional[bytes] = None) -> Dict[str, torch.Tensor]:
    if entropy is None:
        entropy = os.urandom(64)
    tensors: Dict[str, torch.Tensor] = {}
    counter = 0
    for li, cin in enumerate(spec.in_channels):
        for kind in ("gamma", "beta"):
            n = cin * PASSPORT_SPATIAL * PASSPORT_SPATIAL
            buf = bytearray()
            while len(buf) < n * 4:
                buf += hashlib.sha256(entropy + counter.to_bytes(8, "little")).digest()
                counter += 1
            u32 = torch.frombuffer(bytearray(buf[: n * 4]), dtype=torch.int32).clone()
            vals = ((u32 & 0xFFFFFF).to(torch.float32) / float(1 << 23)) - 1.0
            t = vals.view(1, cin, PASSPORT_SPATIAL, PASSPORT_SPATIAL)
            tensors[f"p{li}_{kind}"] = t.contiguous()
    return tensors


def passport_bytes(tensors: Dict[str, torch.Tensor]) -> bytes:
    out = bytearray()
    for name in sorted(tensors.keys()):
        t = tensors[name].detach().cpu().to(torch.float32).contiguous()
        out += name.encode("utf-8") + b"\x00"
        out += t.numpy().astype("<f4").tobytes()
    return bytes(out)


def _bits_to_signs(h: bytes, nbits: int = DYN_BITS) -> torch.Tensor:
    bits = []
    for b in h:
        for k in range(8):
            bits.append(1.0 if (b >> k) & 1 else -1.0)
            if len(bits) == nbits:
                return torch.tensor(bits, dtype=torch.float32)
    while len(bits) < nbits:
        bits.append(-1.0)
    return torch.tensor(bits, dtype=torch.float32)


class PassportNorm(nn.Module):
    def __init__(self, conv: nn.Conv2d, layer_index: int, alpha: float = 0.0):
        super().__init__()
        self.conv = conv
        self.layer_index = layer_index
        self.bn = nn.BatchNorm2d(conv.out_channels, affine=False)
        self.alpha = alpha
        g = torch.Generator().manual_seed(0x5EED + layer_index)
        P = torch.randn(conv.out_channels, DYN_BITS, generator=g) / (DYN_BITS ** 0.5)
        self.register_buffer("P_dyn", P, persistent=False)
        self.register_buffer("z_dyn", torch.zeros(DYN_BITS), persistent=False)
        self.register_buffer("theta_gamma", torch.zeros(1), persistent=False)
        self.register_buffer("theta_beta", torch.zeros(1), persistent=False)

    def install_passport(self, theta_gamma: torch.Tensor, theta_beta: torch.Tensor):
        self.theta_gamma = theta_gamma.detach().to(self.bn.running_mean.device)
        self.theta_beta = theta_beta.detach().to(self.bn.running_mean.device)

    def install_dynamic(self, E: Optional[bytes]):
        if E is None:
            self.z_dyn = torch.zeros(DYN_BITS, device=self.P_dyn.device)
        else:
            self.z_dyn = _bits_to_signs(hashlib.sha256(E).digest()).to(self.P_dyn.device)

    @staticmethod
    def _unit_rms(v: torch.Tensor) -> torch.Tensor:
        return v / (v.pow(2).mean().sqrt() + 1e-8)

    def _affine(self):
        w = self.conv.weight
        pad = w.shape[-1] // 2
        gamma = self._unit_rms(F.conv2d(self.theta_gamma, w, stride=1, padding=pad).mean(dim=(0, 2, 3)))
        beta = self._unit_rms(F.conv2d(self.theta_beta, w, stride=1, padding=pad).mean(dim=(0, 2, 3)))
        if self.alpha != 0.0:
            gamma = gamma + self.alpha * torch.tanh(self.P_dyn @ self.z_dyn)
        return gamma, beta

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        gamma, beta = self._affine()
        h = self.bn(h)
        return gamma.view(1, -1, 1, 1) * h + beta.view(1, -1, 1, 1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, passport_layers: List[PassportNorm], use_passport: bool, alpha: float):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        if use_passport:
            self.bn1 = PassportNorm(self.conv1, len(passport_layers), alpha)
            passport_layers.append(self.bn1)
            self.bn2 = PassportNorm(self.conv2, len(passport_layers), alpha)
            passport_layers.append(self.bn2)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class PassportResNet18(nn.Module):
    def __init__(self, num_classes: int = 100, embed_dim: int = 128, width: int = 64,
                 passport_stages=(2, 3, 4), alpha: float = 0.0):
        super().__init__()
        self.passport_layers: List[PassportNorm] = []
        self.conv1 = nn.Conv2d(3, width, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.in_planes = width
        self.layer1 = self._make_layer(width, 1, 1 in passport_stages, alpha)
        self.layer2 = self._make_layer(width * 2, 2, 2 in passport_stages, alpha)
        self.layer3 = self._make_layer(width * 4, 2, 3 in passport_stages, alpha)
        self.layer4 = self._make_layer(width * 8, 2, 4 in passport_stages, alpha)
        self.embed = nn.Linear(width * 8, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.passport_tensors: Dict[str, torch.Tensor] = {}

    def _make_layer(self, planes, stride, use_passport, alpha):
        layers = [BasicBlock(self.in_planes, planes, stride, self.passport_layers, use_passport, alpha),
                  BasicBlock(planes, planes, 1, self.passport_layers, use_passport, alpha)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def passport_spec(self) -> PassportSpec:
        return PassportSpec([pl.conv.in_channels for pl in self.passport_layers])

    def install_passport(self, tensors: Dict[str, torch.Tensor]):
        self.passport_tensors = {k: v.clone() for k, v in tensors.items()}
        for li, pl in enumerate(self.passport_layers):
            pl.install_passport(tensors[f"p{li}_gamma"], tensors[f"p{li}_beta"])

    def install_dynamic(self, E: Optional[bytes]):
        for pl in self.passport_layers:
            pl.install_dynamic(E)

    def genesis(self, entropy: Optional[bytes] = None) -> Dict[str, torch.Tensor]:
        t = make_passport_tensors(self.passport_spec(), entropy)
        self.install_passport(t)
        return t

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.embed(out)

    def forward(self, x: torch.Tensor):
        z = self.features(x)
        return self.classifier(z), z


def build_model(num_classes=100, embed_dim=128, width=64, passport=True, alpha=0.0) -> PassportResNet18:
    stages = (2, 3, 4) if passport else ()
    return PassportResNet18(num_classes, embed_dim, width, stages, alpha)
