import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from passportfl.data import dirichlet_partition, load_dataset, load_synthetic
from passportfl.fedproto import Client, FedConfig, Server
from passportfl.baselines import BASELINES, score_round
from passportfl.innovation import InnovationAuditor
from passportfl.passport import build_model
from passportfl.encoders import GROUPS, build_group_model, group_assignment, rebind_identity
from passportfl.fedtgp import FEDTGP_DEFAULT_LAM, TGPConfig, TGPServer

EXTENSION_DEFAULTS = {"arch_group": "resnet18", "server": "fedproto", "lam": None, "penalty": None,
                      "archive_dir": "results/archives", "tgp_epochs": 100, "tgp_lr": 0.01,
                      "tgp_batch": 10, "tgp_tau": 100.0, "tgp_hidden": None,
                      "tgp_device": None, "tgp_threads": None, "run_info": False}


def attack_global_echo(arch, cid, r, rng):
    return arch["cbar"][r - 1].copy() if (r - 1) in arch["cbar"] else arch["C"][cid][r].copy()


def attack_peer_copy(arch, cid, r, rng):
    others = [j for j in arch["C"] if j != cid]
    return arch["C"][others[cid % len(others)]][r].copy()


def attack_lagged_self(arch, cid, r, rng):
    pub = arch["published"][cid]
    src = max(min(pub), r - 3)
    return pub[src].copy()


def attack_ar2_replay(arch, cid, r, rng):
    rounds = sorted(k for k in arch["C"][cid] if k < r)
    if len(rounds) < 3:
        return arch["C"][cid][r].copy()
    c1, c2, c3 = (arch["C"][cid][rounds[-1]], arch["C"][cid][rounds[-2]], arch["C"][cid][rounds[-3]])
    return c1 + (c1 - c2) + 0.25 * ((c1 - c2) - (c2 - c3))


def attack_noise(arch, cid, r, rng):
    base = arch["cbar"][r - 1] if (r - 1) in arch["cbar"] else arch["C"][cid][r]
    scale = np.linalg.norm(arch["C"][cid][r] - base) / max(1.0, np.sqrt(base.size))
    return base + rng.normal(0, scale, size=base.shape)


def attack_shrink(arch, cid, r, rng):
    base = arch["cbar"][r - 1] if (r - 1) in arch["cbar"] else arch["C"][cid][r]
    return base + 0.1 * (arch["C"][cid][r] - base)


ATTACKS = {
    "global_echo": attack_global_echo,
    "peer_copy": attack_peer_copy,
    "lagged_self": attack_lagged_self,
    "ar2_replay": attack_ar2_replay,
    "noise": attack_noise,
    "shrink": attack_shrink,
}


def roc_auc(pos, neg):
    pos = np.asarray([p for p in pos if np.isfinite(p)], float)
    neg = np.asarray([n for n in neg if np.isfinite(n)], float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt)); np.add.at(s, inv, ranks); ranks = (s / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def replay(arch, attack_name, attack_start, eval_from, clients, rounds, seed):
    rng = np.random.default_rng(seed)
    fn = ATTACKS[attack_name]
    OURS = ("u1", "u2", "g", "x", "rep", "suspicion")
    pos = {k: [] for k in tuple(OURS) + tuple(BASELINES)}
    neg = {k: [] for k in pos}
    for victim in range(clients):
        aud = InnovationAuditor()
        arch["published"] = {i: {} for i in range(clients)}
        for r in range(1, rounds + 1):
            if (r - 1) in arch["cbar"]:
                aud.set_aggregate(r - 1, arch["cbar"][r - 1])
            payloads = {}
            for cid in range(clients):
                C = arch["C"][cid][r]
                if cid == victim and r >= attack_start:
                    C = fn(arch, cid, r, rng)
                arch["published"][cid][r] = C
                payloads[cid] = (C, arch["present"][cid][r])
            rs = aud.observe_round(r, payloads)

            base = {}
            Cbar_prev = arch["cbar"].get(r - 1)
            if Cbar_prev is not None:
                innov = {cid: (payloads[cid][0] - Cbar_prev) for cid in payloads}
                if len(innov) >= 3:
                    base = score_round(innov)

            if r < max(eval_from, attack_start):
                continue
            for cid, v in rs.per_client.items():
                tgt = pos if cid == victim else neg
                for k in pos:
                    if k in base:
                        val = base[k].get(cid, float("nan"))
                    else:
                        val = v.get(k, float("nan"))
                    if np.isfinite(val):
                        tgt[k].append(float(val))
    return pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--dirichlet", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--attack-start", type=int, default=12)
    ap.add_argument("--eval-from", type=int, default=12)
    ap.add_argument("--dataset", default="cifar100", choices=["cifar100", "cifar10"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-cache", action="store_true", help="force a fresh FL run")
    ap.add_argument("--no-passport", action="store_true",
                    help="plain BatchNorm encoder: does the inversion depend on the passport "
                         "architecture, or only on prototype exchange?")
    ap.add_argument("--arch-group", default="resnet18", choices=sorted(GROUPS),
                    help="encoder of each client (passportfl/encoders.py); hetero4 assigns "
                         "client i the (i mod 4)-th of resnet18_w64, resnet18_w32, cnn4, "
                         "mobilenetv2 and needs --no-passport")
    ap.add_argument("--server", default="fedproto", choices=["fedproto", "fedtgp"],
                    help="fedproto: count-weighted mean; fedtgp: trainable global prototypes "
                         "(passportfl/fedtgp.py)")
    ap.add_argument("--lam", type=float, default=None,
                    help="client coupling weight lambda of lambda*MSE(z, global[y]); unset: "
                         "FedConfig's 1.0 for fedproto, %g (the primary arm of "
                         "fedtgp.LAMBDA_ARMS) for fedtgp" % FEDTGP_DEFAULT_LAM)
    ap.add_argument("--penalty", default=None, choices=["mse", "euclid"],
                    help="client coupling penalty: mse = (lam/d)||z - global[y]||^2 (the code of "
                         "FedProto and FedTGP; the default), euclid = lam ||z - global[y]|| (the "
                         "form stated in FedTGP's paper)")
    ap.add_argument("--archive-dir", default="results/archives",
                    help="archive cache directory; use one outside results/archives for "
                         "configurations that released analyses must not glob")
    ap.add_argument("--tgp-epochs", type=int, default=100, help="FedTGP server epochs S")
    ap.add_argument("--tgp-lr", type=float, default=0.01, help="FedTGP server SGD learning rate")
    ap.add_argument("--tgp-batch", type=int, default=10, help="FedTGP server batch size")
    ap.add_argument("--tgp-tau", type=float, default=100.0, help="FedTGP margin threshold tau")
    ap.add_argument("--tgp-hidden", type=int, default=None,
                    help="FedTGP hidden width; unset: the embedding dimension")
    ap.add_argument("--tgp-device", default=None,
                    help="device of the FedTGP server; unset: --device. The server's steps are "
                         "tiny, so on a shared GPU a single CPU thread can be faster")
    ap.add_argument("--tgp-threads", type=int, default=None,
                    help="torch CPU threads of the process (set it with --tgp-device cpu; "
                         "1 is fastest for the server and makes offline re-fits exact)")
    ap.add_argument("--run-info", action="store_true",
                    help="also write results/innovation_run_<tag>.json (always written for a "
                         "non-default --arch-group or --server)")
    a = ap.parse_args()
    extended = (a.arch_group != "resnet18" or a.server != "fedproto" or a.lam is not None
                or a.penalty is not None)
    if a.arch_group != "resnet18" and not a.no_passport:
        ap.error("--arch-group %s has no passport layers; add --no-passport" % a.arch_group)
    if extended and (a.tag or a.dataset).startswith("cifar100_s"):
        ap.error("tags starting with cifar100_s are reserved for the released configuration")
    if extended and os.path.normpath(os.path.abspath(a.archive_dir)) == os.path.normpath(
            os.path.abspath("results/archives")):
        ap.error("released analyses glob results/archives/arch_*.npz; give a separate "
                 "--archive-dir (e.g. results/archives_rev2)")

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cfg = FedConfig(experiment_seed=a.seed, num_classes=a.num_classes, embed_dim=a.embed_dim, width=a.width, rounds=a.rounds,
                    local_epochs=a.local_epochs, warmup_rounds=10 ** 6, audit_z_threshold=10 ** 6,
                    scheme="Ed25519", device=a.device)
    lam = a.lam if a.lam is not None else (FEDTGP_DEFAULT_LAM if a.server == "fedtgp" else None)
    if lam is not None:
        cfg.lam = lam
    if a.penalty is not None:
        cfg.penalty = a.penalty
    if a.synthetic:
        train, test = load_synthetic(600 * a.clients, 300, a.num_classes, a.seed)
    else:
        train, test, nc = load_dataset(a.dataset)
        a.num_classes = nc
        cfg.num_classes = nc
    parts = dirichlet_partition(train, a.clients, a.dirichlet, a.num_classes, a.seed)
    if a.arch_group == "resnet18":
        clients = [Client(i, build_model(a.num_classes, a.embed_dim, a.width,
                                         not a.no_passport, 0.0), p, cfg)
                   for i, p in enumerate(parts)]
    else:
        clients = [Client(i, build_group_model(a.arch_group, i, a.num_classes, a.embed_dim,
                                               a.width, not a.no_passport), p, cfg)
                   for i, p in enumerate(parts)]
        for c in clients:
            rebind_identity(c.holder)
    if a.server == "fedproto":
        server = Server(cfg)
    else:
        if a.tgp_threads:
            torch.set_num_threads(a.tgp_threads)
        server = TGPServer(cfg, TGPConfig(server_epochs=a.tgp_epochs, server_lr=a.tgp_lr,
                                          server_batch=a.tgp_batch, margin_threshold=a.tgp_tau,
                                          hidden_dim=a.tgp_hidden, seed=a.seed,
                                          device=a.tgp_device or a.device))

    os.makedirs(a.archive_dir, exist_ok=True)
    cache = f"{a.archive_dir}/arch_{a.tag or a.dataset}_c{a.clients}_r{a.rounds}_d{a.dirichlet}_s{a.seed}.npz"
    alternate_cache = f"{a.archive_dir}/arch_{a.tag or a.dataset}_c{a.clients}_r{a.rounds}_s{a.seed}.npz"
    if not os.path.exists(cache) and a.dirichlet == 0.5 and os.path.exists(alternate_cache):
        cache = alternate_cache
    if os.path.exists(cache) and not a.no_cache:
        z = np.load(cache, allow_pickle=True)
        if extended:
            check_cached_meta(z, run_meta(a, cfg, server), cache)
        arch = {"C": z["C"].item(), "present": z["present"].item(), "cbar": z["cbar"].item()}
        print(f"loaded archive from {cache}")
        return write_results(a, arch)

    arch = {"C": {i: {} for i in range(a.clients)}, "present": {i: {} for i in range(a.clients)}, "cbar": {}}
    tgp = a.server == "fedtgp"
    if tgp:
        arch.update(qbar={}, qbar_has={}, cbar_fedproto={}, tgp_log={},
                    tgp_state={0: server.initial_state})
    round_s = []
    t0 = time.time()
    for r in range(1, a.rounds + 1):
        tr = time.time()
        contribs, counts = [], []
        for c in clients:
            contribs.append(c.round_step()); counts.append(c._last_counts)
        for c, cnt in zip(clients, counts):
            arch["C"][c.cid][r] = contribs[c.cid].C.copy()
            arch["present"][c.cid][r] = (np.asarray(cnt) > 0)
        C = server.aggregate(contribs, counts)
        arch["cbar"][r] = C.copy()
        if tgp:
            arch["qbar"][r] = server.qbar.copy()
            arch["qbar_has"][r] = server.qbar_has.copy()
            arch["cbar_fedproto"][r] = server.cbar_fedproto.copy()
            arch["tgp_log"][r] = dict(server.tgp_log[-1])
            arch["tgp_state"][r] = server.last_state
        for c in clients:
            c.receive_global(C)
        round_s.append(time.time() - tr)
        print(json.dumps({"round": r, "elapsed_s": round(time.time() - t0, 1)}))

    if not tgp and not extended:
        np.savez_compressed(cache, C=arch["C"], present=arch["present"], cbar=arch["cbar"])
    elif not tgp:
        np.savez_compressed(cache, C=arch["C"], present=arch["present"], cbar=arch["cbar"],
                            meta=json.dumps(run_meta(a, cfg, server)))
    else:
        np.savez_compressed(cache, C=arch["C"], present=arch["present"], cbar=arch["cbar"],
                            qbar=arch["qbar"], qbar_has=arch["qbar_has"],
                            cbar_fedproto=arch["cbar_fedproto"], tgp_log=arch["tgp_log"],
                            tgp_state=arch["tgp_state"],
                            meta=json.dumps(run_meta(a, cfg, server)))
    print(f"archived to {cache}")
    if extended or a.run_info:
        write_run_info(a, cfg, server, clients, test, cache, round_s, time.time() - t0)
    return write_results(a, arch)


def run_meta(a, cfg, server) -> dict:
    names = (group_assignment(a.arch_group, a.clients) if a.arch_group != "resnet18"
             else ["resnet18_w%d_%s" % (a.width, "plain" if a.no_passport else "passport")] * a.clients)
    meta = {"arch_group": a.arch_group, "encoders": names, "server": a.server, "lam": cfg.lam,
            "seed": a.seed, "clients": a.clients, "rounds": a.rounds, "dirichlet": a.dirichlet}
    if cfg.penalty != "mse":
        meta["penalty"] = cfg.penalty
    if isinstance(server, TGPServer):
        meta["tgp"] = server.config_dict()
    return meta


def check_cached_meta(z, want: dict, cache: str) -> None:
    if "meta" not in z.files:
        raise SystemExit(f"{cache} stores no configuration (meta), so it cannot be checked "
                         "against this command; rerun with --no-cache or remove the archive")
    have = json.loads(str(z["meta"]))
    drop = lambda m: {k: ({kk: vv for kk, vv in v.items() if kk != "device"}
                          if isinstance(v, dict) else v) for k, v in m.items()}
    have, want = drop(have), drop(json.loads(json.dumps(want)))
    diff = sorted(k for k in set(have) | set(want) if have.get(k) != want.get(k))
    if diff:
        raise SystemExit(f"{cache} was produced with a different configuration ("
                         + "; ".join(f"{k}: archived {have.get(k)!r}, requested {want.get(k)!r}"
                                     for k in diff)
                         + "); rerun with --no-cache or another --tag")


def write_run_info(a, cfg, server, clients, test, cache, round_s, total_s):
    from passportfl.encoders import n_params
    from passportfl.reproducibility import environment
    meta = run_meta(a, cfg, server)
    acc = []
    for c, name in zip(clients, meta["encoders"]):
        ev = c.evaluate(test)
        acc.append({"client": c.cid, "encoder": name, "params": n_params(c.model),
                    "n_train": c.n_samples, **ev})
    out = {"meta": {"command": "python experiments/innovation_eval.py " + " ".join(sys.argv[1:]),
                    "environment": environment(), "archive": os.path.basename(cache)},
           "config": vars(a), "federation": meta,
           "timing": {"total_s": round(total_s, 1), "round_s": [round(x, 2) for x in round_s]},
           "accuracy_final_round": acc,
           "degenerate_rounds": list(server.degenerate_rounds)}
    if isinstance(server, TGPServer):
        out["tgp_log"] = server.tgp_log
    tag = a.tag or ("synthetic" if a.synthetic else a.dataset)
    os.makedirs("results", exist_ok=True)
    with open(f"results/innovation_run_{tag}.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("final-round accuracy (classifier / prototype): " + "  ".join(
        "%s %.3f/%.3f" % (x["encoder"], x["acc_classifier"], x.get("acc_prototype", float("nan")))
        for x in acc))


def write_results(a, arch):
    results = {}
    for name in ATTACKS:
        pos, neg = replay(arch, name, a.attack_start, a.eval_from, a.clients, a.rounds, a.seed)
        results[name] = {
            k: {"auc": roc_auc(pos[k], neg[k]),
                "pos_mean": float(np.mean(pos[k])) if pos[k] else None,
                "neg_mean": float(np.mean(neg[k])) if neg[k] else None,
                "n_pos": len(pos[k]), "n_neg": len(neg[k])}
            for k in pos
        }
        print(f"[{name}] " + "  ".join(
            f"{k} AUC={results[name][k]['auc']:.3f}" for k in ("suspicion","u2","rep")))

    config = {k: v for k, v in vars(a).items()
              if k not in EXTENSION_DEFAULTS or v != EXTENSION_DEFAULTS[k]}
    out = {"config": config, "results": results}
    os.makedirs("results", exist_ok=True)
    tag = a.tag or ("synthetic" if a.synthetic else a.dataset)
    with open(f"results/innovation_{tag}.json", "w") as f:
        json.dump(out, f, indent=1, default=float)

    print("\n" + "=" * 94)
    COLS = ["suspicion","u1","u2","g","x","rep"] + list(BASELINES)
    print("%-14s" % "attack" + "".join("%17s" % c for c in COLS))
    print("=" * (14 + 17 * len(COLS)))
    for name, v in results.items():
        print("%-14s" % name + "".join("%17.3f" % v[c]["auc"] for c in COLS))


if __name__ == "__main__":
    main()
