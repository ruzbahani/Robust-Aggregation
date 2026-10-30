from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from passportfl.data import dirichlet_partition, labels_of, load_cifar100
from passportfl.passport import build_model, make_passport_tensors
from passportfl.reproducibility import environment

from negative_results import DEFAULT_ENTROPY, jsonable
from passport_equivalence_eval import rel, sha256_file


@torch.no_grad()
def accuracy(model, ds, C, device, bs=256) -> dict:
    model.eval()
    pc, pp, ys = [], [], []
    for x, y in DataLoader(ds, batch_size=bs):
        x = x.to(device)
        logits, z = model(x)
        pc.append(logits.argmax(1).cpu())
        pp.append(torch.cdist(z, C).argmin(1).cpu())
        ys.append(y)
    pc, pp, ys = torch.cat(pc), torch.cat(pp), torch.cat(ys)
    return {"acc_classifier": float((pc == ys).double().mean()), "acc_prototype": float((pp == ys).double().mean()),
            "n": int(len(ys)), "pred_classifier": pc}


def shard_of(ck: dict, a, train) -> tuple:
    if "train_indices" in ck:
        return list(ck["train_indices"]), [], "stored in the checkpoint"
    m = re.search(r"_client(\d+)\.pt$", os.path.basename(a.checkpoint))
    cid = a.client if a.client is not None else (int(m.group(1)) if m else 0)
    parts = dirichlet_partition(train, a.clients, a.dirichlet, 100, a.seed)
    others = [list(p.indices) for i, p in enumerate(parts) if i != cid]
    return list(parts[cid].indices), others, ("dirichlet_partition(train, %d, %g, 100, %d), client %d"
                                              % (a.clients, a.dirichlet, a.seed, cid))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(REPO, "checkpoints", "cifar100_client0.pt"))
    ap.add_argument("--data-root", default=os.path.join(REPO, "data"))
    ap.add_argument("--clients", type=int, default=10, help="federation size of the archived run")
    ap.add_argument("--dirichlet", type=float, default=0.5, help="Dirichlet concentration of the archived run")
    ap.add_argument("--seed", type=int, default=0, help="partition seed of the archived run")
    ap.add_argument("--client", type=int, default=None, help="client index; default: from the file name")
    ap.add_argument("--attack-frac", type=float, default=0.2, help="fraction of the shard the adversary has")
    ap.add_argument("--attack-steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--entropy", default=DEFAULT_ENTROPY, help="hex seed of the random passport")
    ap.add_argument("--torch-seed", type=int, default=0, help="seed of the loader order and augmentation")
    ap.add_argument("--check-samples", type=int, default=2000,
                    help="images per shard for the partition check (0 skips the check)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=0, help="first N test images only (smoke tests); 0 = all")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "ambiguity_attack_archived.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    np.random.seed(0)
    t_all = time.time()
    dev = a.device

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    K, D, W = cfg.get("num_classes", 100), cfg.get("embed_dim", 128), cfg.get("width", 64)
    alpha = float(ck.get("alpha_dyn", 0.0))
    model = build_model(K, D, W, passport=True, alpha=alpha)
    model.load_state_dict(ck["state"])
    model = model.to(dev)
    model.install_passport({k: v.float() for k, v in ck["passport"].items()})
    model.install_dynamic(ck.get("E") if alpha else None)
    C = ck["global_C"].detach().float().to(dev)

    train, test = load_cifar100(root=a.data_root)
    train_plain, _ = load_cifar100(root=a.data_root, augment=False)
    if a.n_test:
        test = Subset(test, range(a.n_test))
    xs, ys = zip(*DataLoader(test, batch_size=1000))
    test = TensorDataset(torch.cat(xs), torch.cat(ys))
    idx, others, provenance = shard_of(ck, a, train)
    y_train = labels_of(train)
    present = np.unique(y_train[idx])

    genuine = accuracy(model, test, C, dev)
    out = {"genuine": {k: genuine[k] for k in ("acc_classifier", "acc_prototype", "n")}}

    if a.check_samples and others:
        rng = np.random.default_rng(0)
        own = accuracy(model, Subset(train_plain, rng.permutation(idx)[:a.check_samples].tolist()), C, dev)
        rest = np.concatenate(others)
        oth = accuracy(model, Subset(train_plain, rng.permutation(rest)[:a.check_samples].tolist()), C, dev)
        pred_in = float(np.isin(genuine["pred_classifier"].numpy(), present).mean())
        out["partition_check"] = {
            "own_shard_acc_classifier": own["acc_classifier"], "own_shard_acc_prototype": own["acc_prototype"],
            "other_shards_acc_classifier": oth["acc_classifier"], "other_shards_acc_prototype": oth["acc_prototype"],
            "n_per_side": own["n"],
            "test_predictions_on_shard_classes": pred_in,
            "test_labels_on_shard_classes": float(np.isin(labels_of(test), present).mean())}

    spec = model.passport_spec()
    rand_t = make_passport_tensors(spec, bytes.fromhex(a.entropy))
    adv = copy.deepcopy(model)
    adv.install_passport({k: v.to(dev) for k, v in rand_t.items()})
    adv.install_dynamic(None)
    n_adv = int(len(idx) * a.attack_frac)
    adv_ds = Subset(train, idx[:n_adv])
    torch.manual_seed(a.torch_seed)
    loader = DataLoader(adv_ds, batch_size=64, shuffle=True)
    opt = torch.optim.SGD(adv.parameters(), lr=0.01, momentum=0.9)
    curve = []

    def record(step):
        ev = accuracy(adv, test, C, dev)
        curve.append({"step": step, "acc_prototype": ev["acc_prototype"], "acc_classifier": ev["acc_classifier"],
                      "seconds": time.time() - t_all})
        print("  step %5d  acc %.4f / %.4f  (%.0f s)" % (step, ev["acc_classifier"], ev["acc_prototype"],
                                                        time.time() - t_all), flush=True)

    record(0)
    step, t_train = 0, 0.0
    while step < a.attack_steps:
        for x, y in loader:
            t0 = time.time()
            x, y = x.to(dev), y.to(dev)
            adv.train()
            logits, z = adv(x)
            loss = F.cross_entropy(logits, y) + F.mse_loss(z, C[y])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            t_train += time.time() - t0
            if step % a.eval_every == 0 or step == a.attack_steps:
                record(step)
            if step >= a.attack_steps:
                break

    best = max(curve, key=lambda r: r["acc_prototype"])
    out.update({
        "attack": {"adv_samples": n_adv, "steps": a.attack_steps, "curve": curve,
                   "final_acc_prototype": curve[-1]["acc_prototype"],
                   "final_acc_classifier": curve[-1]["acc_classifier"],
                   "best_acc_prototype": best["acc_prototype"], "best_step": best["step"],
                   "final_over_genuine_prototype": curve[-1]["acc_prototype"] / genuine["acc_prototype"],
                   "train_seconds": t_train},
        "meta": {"environment": environment(), "device": dev, "threads": a.threads,
                 "deterministic": dev == "cpu",
                 "checkpoint": rel(a.checkpoint), "checkpoint_sha256": sha256_file(a.checkpoint),
                 "alpha_dyn_checkpoint": alpha, "shard": provenance, "shard_size": len(idx),
                 "shard_classes": int(len(present)), "entropy": a.entropy, "torch_seed": a.torch_seed,
                 "attack_frac": a.attack_frac, "eval_every": a.eval_every, "n_test": genuine["n"],
                 "optimizer": "SGD lr 0.01 momentum 0.9, batch 64, loss CE + MSE(z, C[y]) (binding_eval.py)",
                 "random_passport": "make_passport_tensors(spec, entropy)", "seconds": time.time() - t_all},
    })
    out = {k: out[k] for k in ("meta", "genuine", "partition_check", "attack") if k in out}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=jsonable)
    print("\nwrote %s  (%.1f s)" % (a.out, out["meta"]["seconds"]))
    print(json.dumps({k: out[k] for k in ("genuine", "partition_check") if k in out}, indent=1))
    print("final", out["attack"]["final_acc_prototype"], out["attack"]["final_acc_classifier"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
