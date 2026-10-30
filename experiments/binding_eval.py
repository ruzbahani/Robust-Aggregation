import argparse, copy, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from passportfl.data import load_cifar100, load_synthetic, dirichlet_partition
from passportfl.fedproto import Client, FedConfig, Server
from passportfl.passport import build_model, make_passport_tensors
from passportfl.reproducibility import experiment_entropy

REVISION_OPTIONS = ("random_theta_seed", "save_checkpoint", "out")


def evaluate(client, test):
    return client.evaluate(test)["acc_prototype"]


def with_passport(client, tensors, E, test):
    m = client.model
    saved_t, saved_E = m.passport_tensors, client.holder.E
    m.install_passport(tensors); m.install_dynamic(E)
    acc = evaluate(client, test)
    m.install_passport(saved_t); m.install_dynamic(saved_E)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--alpha-dyn", type=float, default=0.0)
    ap.add_argument("--calib-steps", type=int, default=20)
    ap.add_argument("--attack-frac", type=float, default=0.2, help="fraction of local data the adversary has")
    ap.add_argument("--attack-steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--max-proto-ratio", type=float, default=1.0)
    ap.add_argument("--min-label-stability", type=float, default=0.5)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-theta-seed", type=int, default=None,
                    help="derive the random passport (static control and ambiguity attack) and the "
                         "random-E control from this public seed instead of os.urandom")
    ap.add_argument("--save-checkpoint", default=None,
                    help="save client A, its chain-head history and B's passport here after training")
    ap.add_argument("--out", default="results/binding_eval.json")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    cfg = FedConfig(experiment_seed=a.seed, num_classes=a.num_classes, embed_dim=a.embed_dim, width=a.width, rounds=a.rounds,
                    local_epochs=a.local_epochs, alpha_dyn=a.alpha_dyn, calib_steps=a.calib_steps,
                    m_probes=64 if a.synthetic else 256, t_reveal=16 if a.synthetic else 32, device=a.device,
                    scheme="Ed25519", warmup_rounds=a.warmup, delta_per_round=a.delta,
                    max_proto_ratio=a.max_proto_ratio, min_label_stability=a.min_label_stability)
    if a.synthetic:
        train, test = load_synthetic(1200, 300, a.num_classes, a.seed)
    else:
        train, test = load_cifar100()
    parts = dirichlet_partition(train, 2, 1.0, a.num_classes, a.seed)
    clients = [Client(i, build_model(a.num_classes, a.embed_dim, a.width, True, a.alpha_dyn), p, cfg) for i, p in enumerate(parts)]
    server = Server(cfg)

    E_history = {0: [], 1: []}
    for r in range(a.rounds):
        contribs = [c.round_step() for c in clients]
        for c in clients:
            E_history[c.cid].append(c.holder.E)
        C = server.aggregate(contribs, [c._last_counts for c in clients])
        for c in clients:
            c.receive_global(C)
    A, B = clients
    config = dict(vars(a))
    for k in REVISION_OPTIONS:
        if config[k] == ap.get_default(k):
            del config[k]
    out = {"config": config}
    if a.save_checkpoint:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_checkpoint)), exist_ok=True)
        torch.save({"state": A.model.state_dict(), "passport": A.model.passport_tensors,
                    "E": A.holder.E, "G": A.holder.G, "global_C": A.global_C.detach().cpu(),
                    "alpha_dyn": a.alpha_dyn, "E_history": list(E_history[0]),
                    "foreign_passports": {"client1": B.model.passport_tensors},
                    "train_indices": [int(i) for i in A.train_ds.indices], "scheme": cfg.scheme,
                    "m_probes": cfg.m_probes, "t_reveal": cfg.t_reveal,
                    "config": dict(vars(a)), "source": "experiments/binding_eval.py"}, a.save_checkpoint)

    spec = A.model.passport_spec()
    if a.random_theta_seed is None:
        rand_t = make_passport_tensors(spec)
    else:
        rand_t = make_passport_tensors(spec, experiment_entropy(a.random_theta_seed, 0, "binding-random-theta"))
    zero_t = {k: torch.zeros_like(v) for k, v in A.model.passport_tensors.items()}
    E_A = A.holder.E
    out["static"] = {
        "correct": with_passport(A, A.model.passport_tensors, E_A, test),
        "random_theta": with_passport(A, rand_t, E_A, test),
        "zero_theta": with_passport(A, zero_t, E_A, test),
        "foreign_theta_B": with_passport(A, B.model.passport_tensors, E_A, test),
    }

    if a.alpha_dyn > 0:
        hist = E_history[0]
        out["version"] = {
            "E_installed_calibrated": with_passport(A, A.model.passport_tensors, hist[-1], test),
            "E_trained_on": with_passport(A, A.model.passport_tensors, hist[-2], test) if len(hist) > 1 else None,
            "E_minus_2": with_passport(A, A.model.passport_tensors, hist[-3], test) if len(hist) > 2 else None,
            "E_minus_5": with_passport(A, A.model.passport_tensors, hist[-6], test) if len(hist) > 5 else None,
            "E_first": with_passport(A, A.model.passport_tensors, hist[0], test),
            "random_E": with_passport(A, A.model.passport_tensors,
                                      os.urandom(32) if a.random_theta_seed is None else
                                      experiment_entropy(a.random_theta_seed, 0, "binding-random-E")[:32], test),
            "no_E": with_passport(A, A.model.passport_tensors, None, test),
        }

    adv = copy.deepcopy(A.model)
    adv.install_passport(rand_t); adv.install_dynamic(None)
    n_adv = int(len(A.train_ds) * a.attack_frac)
    adv_ds = torch.utils.data.Subset(A.train_ds, list(range(n_adv)))
    loader = DataLoader(adv_ds, batch_size=64, shuffle=True)
    opt = torch.optim.SGD(adv.parameters(), lr=0.01, momentum=0.9)
    Cg = A.global_C
    curve = []

    def adv_acc():
        adv.eval(); correct = n = 0
        with torch.no_grad():
            for x, y in DataLoader(test, batch_size=256):
                x, y = x.to(a.device), y.to(a.device)
                z = adv.features(x); correct += (torch.cdist(z, Cg).argmin(1) == y).sum().item(); n += len(y)
        return correct / n

    curve.append((0, adv_acc()))
    step = 0
    while step < a.attack_steps:
        for x, y in loader:
            x, y = x.to(a.device), y.to(a.device)
            adv.train(); logits, z = adv(x)
            loss = F.cross_entropy(logits, y) + F.mse_loss(z, Cg[y])
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            if step % 50 == 0 or step == a.attack_steps:
                curve.append((step, adv_acc()))
            if step >= a.attack_steps:
                break
    out["ambiguity_attack"] = {"adv_samples": n_adv, "curve": curve}

    CA, _, _, _ = A.compute_prototypes()
    CB, _, _, _ = B.compute_prototypes()
    saved = (B.holder.round, B.holder.E, B.holder.prev_C)
    rec = dict(n_samples=1, train_loss=0.0, local_acc=0.0, intra_compactness=0.0, inter_separation=0.0)
    stolen = B.holder.issue(CA, rec)
    res_stolen = copy.deepcopy(server.receiver).verify(stolen)
    B.holder.round, B.holder.E, B.holder.prev_C = saved; B.model.install_dynamic(saved[1])
    honest = B.holder.issue(CB, rec)
    res_honest = copy.deepcopy(server.receiver).verify(honest)
    honest_ratios = [l["proto_ratio"] for l in server.log if l["proto_ratio"] is not None and l["round"] > 1]
    out["substitution"] = {
        "stolen_accepted": res_stolen.accepted, "stolen_reason": res_stolen.reason,
        "honest_reason": res_honest.reason,
        "stolen_label_stability": res_stolen.probe.label_stability if res_stolen.probe else None,
        "honest_accepted": res_honest.accepted,
        "honest_label_stability": res_honest.probe.label_stability if res_honest.probe else None,
        "stolen_proto_ratio": res_stolen.probe.proto_drift_ratio if res_stolen.probe else None,
        "honest_proto_ratio": res_honest.probe.proto_drift_ratio if res_honest.probe else None,
        "honest_proto_ratio_history": {"mean": float(np.mean(honest_ratios)), "max": float(np.max(honest_ratios))} if honest_ratios else None,
    }

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
