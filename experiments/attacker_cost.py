from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from passportfl.adaptive_attack import AttackConfig, WhiteBoxAttacker
from passportfl.data import dirichlet_partition, load_cifar100, load_synthetic
from passportfl.fedproto import Client, FedConfig
from passportfl.innovation import InnovationAuditor
from passportfl.passport import build_model

ARCH = os.path.join(os.path.dirname(__file__), "..", "results", "archives")


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor()


def hardware(device: str) -> dict:
    from passportfl.reproducibility import environment
    return {"cpu": cpu_name(), "logical_cores": os.cpu_count(),
            "torch_threads": torch.get_num_threads(), "honest_device": device,
            "attacker_device": AttackConfig().device, **environment()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=None)
    ap.add_argument("--rounds", type=int, default=3, help="rounds to time on each side")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--attack-start", type=int, default=12)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--basis-steps", type=int, default=80)
    ap.add_argument("--full-steps", type=int, default=80)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--mad-floor", type=float, default=0.01)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/attacker_cost.json")
    a = ap.parse_args()
    if a.archive is None:
        from protocol_eval import discover_archives
        available = discover_archives(ARCH, "arch_cifar100_s")
        if 0 not in available:
            raise FileNotFoundError("The timing experiment needs the seed-0 innovation archive.")
        a.archive = available[0]
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    z = np.load(a.archive, allow_pickle=True)
    C, cbar, present = z["C"].item(), z["cbar"].item(), z["present"].item()
    cfg = AttackConfig(beta=a.beta, aim="mimic", knowledge="peer_current",
                       basis_steps=a.basis_steps, full_steps=a.full_steps,
                       restarts=a.restarts, mad_floor=a.mad_floor)
    atk = WhiteBoxAttacker(cfg, seed=7)
    aud = InnovationAuditor(mad_floor=a.mad_floor)
    victim = 0

    craft_s = []
    for r in range(1, a.attack_start + a.rounds):
        if (r - 1) in cbar:
            aud.set_aggregate(r - 1, cbar[r - 1])
        present_r = {cid: present[cid][r] for cid in range(a.clients)}
        payloads = {cid: (C[cid][r], present_r[cid]) for cid in range(a.clients)}
        if r >= a.attack_start:
            peer_C = {j: C[j][r] for j in range(a.clients) if j != victim}
            t0 = time.perf_counter()
            atk.craft(aud, r, victim, C[victim][r], present_r, peer_C)
            craft_s.append(time.perf_counter() - t0)
        aud.observe_round(r, payloads)
    print("attacker craft, %d rounds: median %.3f s  (min %.3f, max %.3f)"
          % (len(craft_s), statistics.median(craft_s), min(craft_s), max(craft_s)))

    fcfg = FedConfig(num_classes=a.num_classes, embed_dim=a.embed_dim, width=a.width,
                     rounds=a.rounds, local_epochs=a.local_epochs, device=a.device,
                     scheme="Ed25519", m_probes=256, t_reveal=32)
    if a.synthetic:
        train, _ = load_synthetic(1200, 300, a.num_classes, a.seed)
    else:
        train, _ = load_cifar100()
    parts = dirichlet_partition(train, a.clients, 0.5, a.num_classes, a.seed)
    client = Client(0, build_model(a.num_classes, a.embed_dim, a.width, True, 0.0),
                    parts[0], fcfg)
    train_s = []
    for _ in range(a.rounds):
        t0 = time.perf_counter()
        client.local_train()
        train_s.append(time.perf_counter() - t0)
        print("  honest local_train round: %.2f s" % train_s[-1], flush=True)

    med_a, med_h = statistics.median(craft_s), statistics.median(train_s)
    res = {
        "device": a.device,
        "local_epochs": a.local_epochs,
        "examples_in_partition": len(parts[0]),
        "solver": {"basis_steps": a.basis_steps, "full_steps": a.full_steps,
                   "restarts": a.restarts, "beta": a.beta},
        "attacker_craft_seconds": {"median": med_a, "min": min(craft_s),
                                   "max": max(craft_s), "n": len(craft_s)},
        "honest_local_train_seconds": {"median": med_h, "min": min(train_s),
                                       "max": max(train_s), "n": len(train_s)},
        "honest_over_attacker": med_h / med_a,
        "hardware": hardware(a.device),
    }
    print("\nhonest local training: median %.2f s per round" % med_h)
    print("attacker at phi=0:     median %.3f s per round" % med_a)
    print("honest / attacker:     %.1fx" % (med_h / med_a))
    print("\nphi=0 is not free, and it is not close to the thing it replaces.")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
