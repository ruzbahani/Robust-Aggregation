from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import Subset

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from passportfl.baselines import BASELINES
from passportfl.reference_baselines import FoolsGold, fltrust_scores
from passportfl.data import labels_of, load_dataset
from passportfl.fedproto import Client, FedConfig
from passportfl.passport import build_model

sys.path.insert(0, os.path.join(HERE, "experiments"))
from innovation_eval import ATTACKS, roc_auc

ARCHIVES = os.path.join(HERE, "results", "archives")
RESULTS = os.path.join(HERE, "results")


def root_indices(train, size: int, num_classes: int, seed: int) -> list:
    y = labels_of(train)
    rng = np.random.default_rng(seed)
    per = max(1, size // num_classes)
    idx = []
    for k in range(num_classes):
        pool = np.flatnonzero(y == k)
        if len(pool):
            idx.extend(rng.choice(pool, size=min(per, len(pool)), replace=False).tolist())
    rng.shuffle(idx)
    return idx[:size] if size < len(idx) else idx


def train_root_model(arch_path: str, size: int, seed: int, device: str,
               dataset: str = "cifar100", cache: str | None = None) -> np.ndarray:
    if cache and os.path.exists(cache):
        z = np.load(cache)
        return z["g0"]

    z = np.load(arch_path, allow_pickle=True)
    cbar = z["cbar"].item()
    rounds = sorted(cbar)

    torch.manual_seed(seed)
    np.random.seed(seed)
    train, _, num_classes = load_dataset(dataset)
    root = Subset(train, root_indices(train, size, num_classes, seed))
    cfg = FedConfig(num_classes=num_classes, device=device, experiment_seed=seed)
    model = build_model(num_classes, cfg.embed_dim, cfg.width, passport=True, alpha=0.0)
    server_client = Client(-1, model, root, cfg)

    g0 = {}
    t0 = time.time()
    for r in rounds:
        if (r - 1) not in cbar:
            continue
        server_client.receive_global(cbar[r - 1])
        server_client.local_train()
        C0, _, _, _ = server_client.compute_prototypes()
        g0[r] = C0 - cbar[r - 1]
    print("   root model: %d rounds on %d examples in %.1fs"
          % (len(g0), len(root), time.time() - t0))

    rs = sorted(g0)
    out = np.stack([g0[r] for r in rs])
    if cache:
        np.savez_compressed(cache, g0=out, rounds=np.array(rs))
    return out


def innovations(arch, victim, attack, r, rng, clients, attacked: bool):
    cbar_prev = arch["cbar"][r - 1]
    out = {}
    for cid in range(clients):
        C = arch["C"][cid][r]
        if victim is not None and cid == victim and attack is not None and attacked:
            C = ATTACKS[attack](arch, cid, r, rng)
        arch["published"][cid][r] = C
        out[cid] = np.asarray(C, dtype=np.float64) - cbar_prev
    return out


def evaluate_seed(arch_path: str, g0: np.ndarray, attack_start: int, eval_from: int,
                  clients: int, seed: int) -> dict:
    z = np.load(arch_path, allow_pickle=True)
    arch = {"C": z["C"].item(), "present": z["present"].item(), "cbar": z["cbar"].item()}
    rounds = [r for r in sorted(arch["cbar"]) if (r - 1) in arch["cbar"]]
    gmap = {r: g0[i] for i, r in enumerate(rounds[:len(g0)])}

    FG_VARIANTS = {"foolsgold": 0.0, "foolsgold_imp10": 0.10}

    out = {}
    for attack in ATTACKS:
        pos_ft, neg_ft = [], []
        pos_fg = {k: [] for k in FG_VARIANTS}
        neg_fg = {k: [] for k in FG_VARIANTS}
        for victim in range(clients):
            rng = np.random.default_rng(1000 * seed + victim)
            fg = {k: FoolsGold(importance=imp) for k, imp in FG_VARIANTS.items()}
            arch["published"] = {i: {} for i in range(clients)}
            for r in rounds:
                attacked = r >= attack_start
                I = innovations(arch, victim, attack, r, rng, clients, attacked)
                s_fg = {k: f.update(I) for k, f in fg.items()}
                if r not in gmap or r < eval_from:
                    continue
                s_ft = fltrust_scores(I, gmap[r])
                for cid in range(clients):
                    a = s_ft.get(cid)
                    if cid == victim and attacked:
                        if np.isfinite(a):
                            pos_ft.append(a)
                        for k in FG_VARIANTS:
                            if np.isfinite(s_fg[k].get(cid, float("nan"))):
                                pos_fg[k].append(s_fg[k][cid])
                    elif cid != victim:
                        if np.isfinite(a):
                            neg_ft.append(a)
                        for k in FG_VARIANTS:
                            if np.isfinite(s_fg[k].get(cid, float("nan"))):
                                neg_fg[k].append(s_fg[k][cid])
        out[attack] = {
            "fltrust": roc_auc(pos_ft, neg_ft),
            "n_pos": len(pos_ft), "n_neg": len(neg_ft),
            **{k: roc_auc(pos_fg[k], neg_fg[k]) for k in FG_VARIANTS},
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--root-size", type=int, default=100)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--attack-start", type=int, default=12)
    ap.add_argument("--eval-from", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--config", default="cifar100_d05",
                    choices=["cifar100_d05", "cifar100_d01", "cifar10"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    PATTERNS = {
        "cifar100_d05": (["arch_cifar100_s%d_c10_r35_s%d.npz",
                          "arch_cifar100_s%d_c10_r35_d0.5_s%d.npz"], "cifar100"),
        "cifar100_d01": (["arch_cifar100_d01_s%d_c10_r35_d0.1_s%d.npz",
                          "arch_cifar100_d01_s%d_c10_r35_s%d.npz"], "cifar100"),
        "cifar10": (["arch_cifar10_s%d_c10_r35_s%d.npz",
                     "arch_cifar10_s%d_c10_r35_d0.5_s%d.npz"], "cifar10"),
    }
    pats, dataset = PATTERNS[args.config]
    out_path = args.out or os.path.join(
        RESULTS, "faithful_baselines_%s.json" % args.config)

    per_seed = {}
    for s in args.seeds:
        cands = [os.path.join(ARCHIVES, p % (s, s)) for p in pats]
        arch_path = next((p for p in cands if os.path.exists(p)), None)
        if not arch_path:
            print("seed %d: no archive for %s" % (s, args.config))
            continue
        print("seed %d: %s" % (s, os.path.basename(arch_path)))
        cache = os.path.join(ARCHIVES, "fltrust_root_%s_r%d_s%d.npz"
                             % (args.config, args.root_size, s))
        g0 = train_root_model(arch_path, args.root_size, s, args.device,
                        dataset=dataset, cache=cache)
        per_seed[s] = evaluate_seed(arch_path, g0, args.attack_start, args.eval_from,
                                    args.clients, s)

    print()
    print("Detection AUC, published algorithms, mean over %d seeds." % len(per_seed))
    print("%-14s %12s %12s %14s"
          % ("attack", "FLTrust", "FoolsGold", "FG importance"))
    print("-" * 56)
    summary = {}
    for attack in ATTACKS:
        ft = [per_seed[s][attack]["fltrust"] for s in per_seed
              if np.isfinite(per_seed[s][attack]["fltrust"])]
        summary[attack] = {
            "fltrust_mean": float(np.mean(ft)) if ft else None,
            "fltrust_std": float(np.std(ft)) if ft else None,
        }
        cells = ["%.3f" % summary[attack]["fltrust_mean"] if ft else "n/a"]
        for k in ("foolsgold", "foolsgold_imp10"):
            v = [per_seed[s][attack][k] for s in per_seed
                 if k in per_seed[s][attack] and np.isfinite(per_seed[s][attack][k])]
            summary[attack]["%s_mean" % k] = float(np.mean(v)) if v else None
            summary[attack]["%s_std" % k] = float(np.std(v)) if v else None
            cells.append("%.3f" % np.mean(v) if v else "n/a")
        print("%-14s %12s %12s %14s" % (attack, cells[0], cells[1], cells[2]))

    payload = {
        "config": args.config, "root_size": args.root_size, "seeds": list(per_seed),
        "attack_start": args.attack_start, "eval_from": args.eval_from,
        "summary": summary, "per_seed": {str(k): v for k, v in per_seed.items()},
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nwrote %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
