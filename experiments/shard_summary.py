from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)

from passportfl.data import dirichlet_partition, labels_of, load_cifar100
from passportfl.reproducibility import environment

FEDERATIONS = {
    "archived_fedproto_cifar100": (10, 0.5, 0, "train_federation.py, results/fedproto_cifar100.json config"),
    "binding_eval": (2, 1.0, 0, "binding_eval.py: dirichlet_partition(train, 2, 1.0, 100, seed 0)"),
}


def summary(y: np.ndarray, idx: list, num_classes: int, min_images: int) -> dict:
    counts = np.bincount(y[idx], minlength=num_classes)
    return {"size": int(len(idx)), "classes_present": int((counts > 0).sum()),
            "classes_with_min_images": int((counts >= min_images).sum()),
            "max_class_share": float(counts.max() / max(1, len(idx))),
            "per_class_counts": counts.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(REPO, "data"))
    ap.add_argument("--min-images", type=int, default=20, help="threshold of classes_with_min_images")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "shard_summary.json"))
    a = ap.parse_args()
    t0 = time.time()
    train, _ = load_cifar100(root=a.data_root, augment=False)
    y = labels_of(train)
    K = 100
    out = {"meta": {"environment": environment(), "dataset": "cifar100", "train_size": int(len(y)),
                    "num_classes": K, "min_images": a.min_images}}
    for name, (n, alpha, seed, src) in FEDERATIONS.items():
        parts = dirichlet_partition(train, n, alpha, K, seed)
        clients = [summary(y, list(p.indices), K, a.min_images) for p in parts]
        out[name] = {"clients": n, "dirichlet": alpha, "seed": seed, "source": src,
                     "client0": {k: v for k, v in clients[0].items()},
                     "per_client": [{k: v for k, v in c.items() if k != "per_class_counts"} for c in clients],
                     "per_client_class_counts": [c["per_class_counts"] for c in clients]}
        c0 = clients[0]
        print("%-28s client 0: %5d images, %3d classes, %3d with >= %d images"
              % (name, c0["size"], c0["classes_present"], c0["classes_with_min_images"], a.min_images))
    out["meta"]["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
