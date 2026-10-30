from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from passportfl.baselines import score_round
from passportfl.data import dirichlet_partition, load_dataset
from passportfl.effort_controls import batchnorm_summary, forward_only
from passportfl.fedproto import Client, FedConfig, Server
from passportfl.innovation import SCORE_STRUCTURAL, InnovationAuditor
from passportfl.passport import build_model
from passportfl.reproducibility import environment

TAU = 3.7859094579543044
MAD_FLOOR = 0.01
MODES = ("steps", "forward", "zero")
MODE_ARGS = ("mode", "forward_fraction", "archive_out")


def roc_auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


class ReducedClient(Client):
    def __init__(self, *a, start: int, max_steps: int, mode: str = "steps", **k):
        super().__init__(*a, **k)
        self.start, self.max_steps, self.round = start, max_steps, 0
        self.mode = mode
        self.steps_taken = []
        self.forward_taken = []
        self.published_stats = []
        self._last_stats = None

    def local_round(self):
        if self.round < self.start:
            stats = self.local_train()
            self._last_stats = dict(stats)
            self.steps_taken.append(self.full_steps)
            self.forward_taken.append(self.full_steps)
        elif self.mode == "forward":
            stats = forward_only(self, max_passes=self.max_steps)
            self.steps_taken.append(0)
            self.forward_taken.append(self.forward_passes)
        else:
            stats = dict(self._last_stats)
            self.steps_taken.append(0)
            self.forward_taken.append(0)
        return stats

    def round_step(self):
        self.round += 1
        if self.mode == "steps":
            cap = self.max_steps if self.round >= self.start else None
            stats = self.local_train(max_steps=cap)
            self.steps_taken.append(cap if cap is not None else self.full_steps)
        else:
            stats = self.local_round()
            self.published_stats.append(dict(stats))
        C, counts, intra, inter = self.compute_prototypes()
        self._last_counts = counts
        contrib = self.holder.issue(C, dict(n_samples=self.n_samples, train_loss=stats["train_loss"],
                                            local_acc=stats["local_acc"], intra_compactness=intra,
                                            inter_separation=inter))
        self.calibrate()
        return contrib

    @property
    def full_steps(self) -> int:
        return len(self.loader) * self.cfg.local_epochs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=None,
                    help="fraction of the victim's local optimizer steps actually run "
                         "(mode steps only; default 0.1)")
    ap.add_argument("--victim", type=int, default=0)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=35)
    ap.add_argument("--attack-start", type=int, default=12)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--dirichlet", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--mode", choices=MODES, default="steps",
                    help="steps: capped optimizer steps; forward: train-mode forward passes "
                         "only; zero: no local computation")
    ap.add_argument("--forward-fraction", type=float, default=1.0,
                    help="mode forward: fraction of the round's forward passes run")
    ap.add_argument("--archive-out", default="",
                    help="archive path (default under results/archives in mode steps, "
                         "results/archives_rev2 otherwise)")
    a = ap.parse_args()
    if a.mode == "steps":
        a.fraction = 0.1 if a.fraction is None else a.fraction
    elif a.fraction is not None:
        ap.error("--fraction applies to --mode steps; use --forward-fraction")
    if a.mode == "zero" and a.attack_start < 2:
        ap.error("--mode zero repeats the metadata of an honest round; --attack-start must be >= 2")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    train, test, nc = load_dataset(a.dataset)
    cfg = FedConfig(experiment_seed=a.seed, num_classes=nc, embed_dim=a.embed_dim, width=a.width, rounds=a.rounds,
                    local_epochs=a.local_epochs, warmup_rounds=10 ** 6,
                    audit_z_threshold=10 ** 6, scheme="Ed25519", device=a.device)
    parts = dirichlet_partition(train, a.clients, a.dirichlet, nc, a.seed)

    clients = []
    for i, p in enumerate(parts):
        m = build_model(nc, a.embed_dim, a.width, True, 0.0)
        if i == a.victim and a.mode == "steps":
            full = math.ceil(len(p) / cfg.batch_size) * a.local_epochs
            cap = max(1, int(round(a.fraction * full)))
            clients.append(ReducedClient(i, m, p, cfg, start=a.attack_start, max_steps=cap))
            print("victim %d: %d full steps per round, capped at %d from round %d (%.0f%%)"
                  % (i, full, cap, a.attack_start, 100 * cap / full), flush=True)
        elif i == a.victim:
            full = math.ceil(len(p) / cfg.batch_size) * a.local_epochs
            cap = max(1, int(round(a.forward_fraction * full))) if a.mode == "forward" else 0
            clients.append(ReducedClient(i, m, p, cfg, start=a.attack_start, max_steps=cap, mode=a.mode))
            print("victim %d: %d full steps per round; from round %d no optimizer step and %d "
                  "train-mode forward passes (%s)" % (i, full, a.attack_start, cap, a.mode), flush=True)
        else:
            clients.append(Client(i, m, p, cfg))
    server = Server(cfg)
    aud = InnovationAuditor(mad_floor=MAD_FLOOR)

    pos = {"suspicion": []}
    neg = {"suspicion": []}
    flags_pos = flags_neg = n_pos = n_neg = 0
    norm_ratio = []
    arch = {"C": {c.cid: {} for c in clients}, "present": {c.cid: {} for c in clients},
            "cbar": {}}
    Cbar = None
    t0 = time.time()
    for r in range(1, a.rounds + 1):
        contribs, counts = [], []
        for c in clients:
            contribs.append(c.round_step())
            counts.append(c._last_counts)
        payloads = {c.cid: (contribs[c.cid].C.copy(), np.asarray(counts[c.cid]) > 0)
                    for c in clients}
        for cid, (Cc, pr) in payloads.items():
            arch["C"][cid][r] = Cc.copy()
            arch["present"][cid][r] = pr.copy()
        if Cbar is not None:
            aud.set_aggregate(r - 1, Cbar)
        rs = aud.observe_round(r, payloads)
        base = score_round({i: payloads[i][0] - Cbar for i in payloads}) if Cbar is not None else {}
        if r >= a.attack_start and Cbar is not None:
            norms = {i: float(np.linalg.norm(payloads[i][0] - Cbar)) for i in payloads}
            others = np.median([v for i, v in norms.items() if i != a.victim])
            norm_ratio.append(norms[a.victim] / max(others, 1e-12))
            for i in payloads:
                s = float(rs.per_client[i].get("suspicion", float("nan")))
                (pos if i == a.victim else neg)["suspicion"].append(s)
                if np.isfinite(s):
                    if i == a.victim:
                        n_pos += 1; flags_pos += int(s > TAU)
                    else:
                        n_neg += 1; flags_neg += int(s > TAU)
                for rule, vals in base.items():
                    (pos if i == a.victim else neg).setdefault(rule, []).append(float(vals.get(i, float("nan"))))
        Cbar_new = server.aggregate(contribs, counts)
        arch["cbar"][r] = Cbar_new.copy()
        Cbar = Cbar_new
        for c in clients:
            c.receive_global(Cbar)
        print(json.dumps({"round": r, "elapsed_s": round(time.time() - t0, 1),
                          "victim_suspicion": (pos["suspicion"][-1] if r >= a.attack_start and Cbar is not None and pos["suspicion"] else None)}), flush=True)

    config = vars(a)
    if a.mode == "steps":
        config = {k: v for k, v in config.items() if k not in MODE_ARGS or (k == "archive_out" and v)}
    out = {
        "config": config,
        "victim_steps_per_round": clients[a.victim].steps_taken,
        "victim_full_steps": clients[a.victim].full_steps,
        "auc": {k: roc_auc(pos[k], neg[k]) for k in pos},
        "tpr_client_round": flags_pos / max(n_pos, 1),
        "honest_fpr_client_round": flags_neg / max(n_neg, 1),
        "n_pos": n_pos, "n_neg": n_neg, "tau": TAU,
        "victim_innovation_over_cohort_median": {
            "mean": float(np.mean(norm_ratio)), "median": float(np.median(norm_ratio)),
            "min": float(np.min(norm_ratio)), "max": float(np.max(norm_ratio)),
            "per_round": [float(x) for x in norm_ratio]},
    }
    if a.mode == "steps":
        tag = "f%g" % a.fraction
    else:
        v = clients[a.victim]
        tag = ("fwd" if a.forward_fraction == 1.0 else "fwd%g" % a.forward_fraction) if a.mode == "forward" else "zero"
        out.update({
            "mode": a.mode, "schema": "reduced-effort-mode-v1", "environment": environment(),
            "elapsed_s": time.time() - t0,
            "victim_optimizer_steps_per_round": v.steps_taken,
            "victim_forward_passes_per_round": v.forward_taken,
            "victim_published_train_loss": [s["train_loss"] for s in v.published_stats],
            "victim_published_local_acc": [s["local_acc"] for s in v.published_stats],
            "batchnorm": batchnorm_summary(v.model)
        })
    arch_dir = os.path.join(HERE, "results", "archives" if a.mode == "steps" else "archives_rev2")
    arch_path = a.archive_out or os.path.join(arch_dir, "arch_reduced_%s_s%d.npz" % (tag, a.seed))
    os.makedirs(os.path.dirname(os.path.abspath(arch_path)), exist_ok=True)
    np.savez_compressed(arch_path, C=arch["C"], present=arch["present"], cbar=arch["cbar"])
    path = a.out or os.path.join(HERE, "results", "reduced_effort_%s_s%d.json" % (tag, a.seed))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    if a.mode == "steps":
        print("\nfraction %g seed %d:" % (a.fraction, a.seed))
    else:
        print("\nmode %s seed %d:" % (tag, a.seed))
    for k, v in out["auc"].items():
        print("   %-18s AUC %.3f" % (k, v))
    print("   auditor TPR at frozen tau %.3f   honest FPR %.4f" % (out["tpr_client_round"], out["honest_fpr_client_round"]))
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
