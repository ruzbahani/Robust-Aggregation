import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
from torch.utils.data import Subset

from passportfl.data import load_dataset, labels_of
from passportfl.fedproto import Client, FedConfig, Server
from passportfl.passport import build_model

DEFAULT_LADDER = (2, 3, 5, 8, 12, 20, 35, 60, 80, 100)


def support_partition(ds, n_clients, sizes, num_classes, seed):
    y = labels_of(ds)
    rng = np.random.default_rng(seed)
    sizes = list(sizes)
    assert len(sizes) == n_clients
    assert max(sizes) <= num_classes

    held = [set() for _ in range(n_clients)]
    order = rng.permutation(num_classes)
    for k in order:
        cand = [i for i in range(n_clients) if len(held[i]) < sizes[i]]
        if not cand:
            break
        i = max(cand, key=lambda i: (sizes[i] - len(held[i]), rng.random()))
        held[i].add(int(k))
    for i in range(n_clients):
        while len(held[i]) < sizes[i]:
            k = int(rng.integers(num_classes))
            held[i].add(k)
    uncovered = [k for k in range(num_classes) if not any(k in h for h in held)]
    if uncovered:
        if sum(sizes) < num_classes:
            raise SystemExit(
                "infeasible support partition: %d clients holding %s classes provide %d "
                "slots for %d classes, so %d classes cannot be covered. Raise "
                "--classes-per-client to at least %d, or raise --clients."
                % (n_clients, sizes[0] if len(set(sizes)) == 1 else sizes, sum(sizes),
                   num_classes, len(uncovered),
                   -(-num_classes // n_clients)))
        big = int(np.argmax(sizes))
        held[big].update(uncovered)

    idx_per_client = [[] for _ in range(n_clients)]
    for k in range(num_classes):
        holders = [i for i in range(n_clients) if k in held[i]]
        if not holders:
            continue
        idx_k = np.where(y == k)[0]
        rng.shuffle(idx_k)
        for c, part in zip(holders, np.array_split(idx_k, len(holders))):
            idx_per_client[c].extend(part.tolist())
    return ([Subset(ds, sorted(i)) for i in idx_per_client],
            [sorted(h) for h in held])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=35)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--partition", default="mixed", choices=["mixed", "shard"])
    ap.add_argument("--classes-per-client", type=int, default=5)
    ap.add_argument("--ladder", default=",".join(str(s) for s in DEFAULT_LADDER))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="maskmix")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    train, test, nc = load_dataset(a.dataset)
    cfg = FedConfig(experiment_seed=a.seed, num_classes=nc, embed_dim=a.embed_dim, width=a.width, rounds=a.rounds,
                    local_epochs=a.local_epochs, warmup_rounds=10 ** 6,
                    audit_z_threshold=10 ** 6, scheme="Ed25519", device=a.device)

    if a.partition == "shard":
        sizes = [a.classes_per_client] * a.clients
    else:
        sizes = [int(x) for x in a.ladder.split(",")]
        if len(sizes) != a.clients:
            raise SystemExit("--ladder must have exactly --clients entries")
    parts, held = support_partition(train, a.clients, sizes, nc, a.seed)
    print(json.dumps({"support_sizes": [len(h) for h in held],
                      "samples": [len(p) for p in parts]}))

    clients = [Client(i, build_model(nc, a.embed_dim, a.width, True, 0.0), p, cfg)
               for i, p in enumerate(parts)]
    server = Server(cfg)

    arch = {"C": {i: {} for i in range(a.clients)},
            "present": {i: {} for i in range(a.clients)}, "cbar": {}}
    t0 = time.time()
    for r in range(1, a.rounds + 1):
        contribs, counts = [], []
        for c in clients:
            contribs.append(c.round_step()); counts.append(c._last_counts)
        for c, cnt in zip(clients, counts):
            arch["C"][c.cid][r] = contribs[c.cid].C.copy()
            arch["present"][c.cid][r] = (np.asarray(cnt) > 0)
        C = server.aggregate(contribs, counts)
        arch["cbar"][r] = C.copy()
        for c in clients:
            c.receive_global(C)
        if r % 5 == 0 or r == 1:
            print(json.dumps({"round": r, "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "archives")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"arch_{a.tag}_c{a.clients}_r{a.rounds}_s{a.seed}.npz")
    np.savez_compressed(path, C=arch["C"], present=arch["present"], cbar=arch["cbar"])
    print("archived to", path)


if __name__ == "__main__":
    main()
