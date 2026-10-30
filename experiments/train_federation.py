import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from passportfl.data import load_cifar100, load_synthetic, dirichlet_partition
from passportfl.fedproto import Client, FedConfig, Server
from passportfl.passport import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--dirichlet", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--alpha-dyn", type=float, default=0.0)
    ap.add_argument("--calib-steps", type=int, default=20)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--m-probes", type=int, default=256)
    ap.add_argument("--t-reveal", type=int, default=32)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--max-proto-ratio", type=float, default=1.0)
    ap.add_argument("--min-label-stability", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=5,
                    help="behavioural checks are logged but not enforced for this many rounds; "
                         "set > --rounds to log the honest distribution without enforcing")
    ap.add_argument("--scheme", default="ML-DSA-44")
    ap.add_argument("--no-passport", action="store_true", help="baseline: plain BatchNorm ResNet, identity layer still runs")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-models", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cfg = FedConfig(experiment_seed=a.seed, num_classes=a.num_classes, embed_dim=a.embed_dim, width=a.width, rounds=a.rounds,
                    local_epochs=a.local_epochs, lr=a.lr, lam=a.lam, alpha_dyn=a.alpha_dyn, calib_steps=a.calib_steps,
                    m_probes=a.m_probes, t_reveal=a.t_reveal, delta_per_round=a.delta,
                    max_proto_ratio=a.max_proto_ratio, min_label_stability=a.min_label_stability,
                    warmup_rounds=a.warmup, scheme=a.scheme, device=a.device)

    if a.synthetic:
        train, test = load_synthetic(600 * a.clients, 300, a.num_classes, a.seed)
    else:
        train, test = load_cifar100()
    parts = dirichlet_partition(train, a.clients, a.dirichlet, a.num_classes, a.seed)

    clients = []
    for i, p in enumerate(parts):
        m = build_model(a.num_classes, a.embed_dim, a.width, passport=not a.no_passport, alpha=a.alpha_dyn)
        clients.append(Client(i, m, p, cfg))
    server = Server(cfg)

    hist = []
    for r in range(1, a.rounds + 1):
        t0 = time.time()
        contribs, counts = [], []
        for c in clients:
            contribs.append(c.round_step()); counts.append(c._last_counts)
        t_issue = time.time() - t0
        t1 = time.time()
        C = server.aggregate(contribs, counts)
        t_verify = time.time() - t1
        for c in clients:
            c.receive_global(C)
        accs = [c.evaluate(test) for c in clients]
        row = {"round": r,
               "acc_classifier": float(np.mean([x["acc_classifier"] for x in accs])),
               "acc_prototype": float(np.mean([x["acc_prototype"] for x in accs])),
               "accepted": sum(1 for l in server.log if l["round"] == r and l["accepted"]),
               "wire_bytes_mean": float(np.mean([l["wire_bytes"] for l in server.log if l["round"] == r])),
               "t_issue_s": t_issue, "t_verify_s": t_verify}
        hist.append(row)
        print(json.dumps(row))

    os.makedirs("results", exist_ok=True)
    tag = a.tag or ("synthetic" if a.synthetic else "cifar100")
    with open(f"results/fedproto_{tag}.json", "w") as f:
        json.dump({"config": vars(a), "history": hist, "verify_log": server.log}, f, indent=1)
    if a.save_models:
        os.makedirs("checkpoints", exist_ok=True)
        for c in clients:
            torch.save({"state": c.model.state_dict(), "passport": c.model.passport_tensors, "E": c.holder.E,
                        "G": c.holder.G, "global_C": c.global_C.cpu()}, f"checkpoints/{tag}_client{c.cid}.pt")


if __name__ == "__main__":
    main()
