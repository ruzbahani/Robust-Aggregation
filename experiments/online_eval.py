from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from torch.utils.data import TensorDataset

from passportfl.baselines import BASELINES, score_round
from passportfl.data import (CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD,
                             dirichlet_partition, load_dataset, load_synthetic)
from passportfl.fedproto import Client, FedConfig, Server
from passportfl.identity import dequantize
from passportfl.innovation import InnovationAuditor
from passportfl.passport import build_model


def attack_payload(kind, *, cbar_prev, C_own, own_true, own_pub, peer_pub_now, r, cid,
                   rng, strength):
    if kind == "global_echo":
        return cbar_prev.copy() if cbar_prev is not None else C_own.copy()

    if kind == "shrink":
        base = cbar_prev if cbar_prev is not None else C_own
        return base + strength * (C_own - base)

    if kind == "lead":
        base = cbar_prev if cbar_prev is not None else C_own
        return C_own + strength * (C_own - base)

    if kind == "ar2_replay":
        rounds = sorted(k for k in own_true if k < r)
        if len(rounds) < 3:
            return C_own.copy()
        c1, c2, c3 = own_true[rounds[-1]], own_true[rounds[-2]], own_true[rounds[-3]]
        return c1 + (c1 - c2) + 0.25 * ((c1 - c2) - (c2 - c3))

    if kind == "lagged_self":
        rounds = sorted(own_pub)
        if not rounds:
            return C_own.copy()
        src = max(min(rounds), r - 3)
        while src not in own_pub and src < r:
            src += 1
        return own_pub[src].copy()

    if kind == "peer_copy":
        others = sorted(peer_pub_now)
        if not others:
            return C_own.copy()
        return peer_pub_now[others[cid % len(others)]].copy()

    if kind == "noise":
        base = cbar_prev if cbar_prev is not None else C_own
        scale = float(np.linalg.norm(C_own - base)) / max(1.0, np.sqrt(base.size))
        return base + rng.normal(0, scale, size=base.shape)

    if kind == "none":
        return C_own.copy()

    raise ValueError(f"unknown attack kind {kind!r}")


SCENARIOS = {
    "honest":          {},
    "global_echo":     {7: ("global_echo", 1.0)},
    "shrink":          {7: ("shrink", 0.1)},
    "ar2_replay":      {7: ("ar2_replay", 1.0)},
    "peer_copy":       {7: ("peer_copy", 1.0)},
    "collude3_shrink": {5: ("shrink", 0.1), 6: ("shrink", 0.1), 7: ("shrink", 0.1)},
    "collude3_lead":   {5: ("lead", 3.0), 6: ("lead", 3.0), 7: ("shrink", 0.1)},
}


def load_archive_lazily(dataset: str, root: str = "./data"):
    import torchvision
    cls, mean, std, nc = ((torchvision.datasets.CIFAR100, CIFAR100_MEAN, CIFAR100_STD, 100)
                          if dataset == "cifar100"
                          else (torchvision.datasets.CIFAR10, CIFAR10_MEAN, CIFAR10_STD, 10))
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    out = []
    for train in (True, False):
        ds = cls(root, train=train, download=True, transform=None)
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - m) / s
        y = torch.as_tensor(ds.targets, dtype=torch.long)
        td = TensorDataset(x.contiguous(), y)
        td.targets = y.numpy()
        out.append(td)
    return out[0], out[1], nc


def roc_auc(pos, neg):
    pos = np.asarray([p for p in pos if np.isfinite(p)], float)
    neg = np.asarray([n for n in neg if np.isfinite(n)], float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt))
    np.add.at(s, inv, ranks)
    ranks = (s / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


STAT_KEYS = ("suspicion", "u1", "u2", "g", "x", "rep",
             "z_u1", "z_u2", "z_g", "z_x", "z_rep", "norm")


def score_one_round(auditor, r, pub, present, cbar_prev, attackers, rows):
    if cbar_prev is not None:
        auditor.set_aggregate(r - 1, cbar_prev)
    payloads = {cid: (pub[cid], present[cid]) for cid in sorted(pub)}
    rs = auditor.observe_round(r, payloads)

    base = {}
    if cbar_prev is not None and len(payloads) >= 3:
        innov = {cid: (pub[cid] - cbar_prev) for cid in payloads}
        base = score_round(innov)

    for cid, v in rs.per_client.items():
        row = {"round": r, "cid": int(cid), "attacker": bool(cid in attackers)}
        for k in STAT_KEYS:
            val = v.get(k, float("nan"))
            row[k] = float(val) if np.isfinite(val) else None
        row["degenerate"] = bool(v.get("degenerate", False))
        row["undefined"] = bool(v.get("undefined", False))
        for bname, bvals in base.items():
            val = bvals.get(cid, float("nan"))
            row[bname] = float(val) if np.isfinite(val) else None
        rows.append(row)
    return rs


def build_federation(a, cfg, parts, num_classes):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    clients = []
    for i, p in enumerate(parts):
        h = hashlib.sha256(f"passportfl|online|seed{a.seed}|client{i}".encode()).digest()
        entropy = h + hashlib.sha256(h).digest()
        model = build_model(num_classes, a.embed_dim, a.width, True, 0.0)
        clients.append(Client(i, model, p, cfg, entropy=entropy))
    return clients, Server(cfg)


def snapshot_parameters(clients, server, auditor, arch):
    return {
        "clients": [{
            "model": {k: v.detach().cpu().clone() for k, v in c.model.state_dict().items()},
            "opt": copy.deepcopy(c.opt.state_dict()),
            "holder_round": c.holder.round,
            "holder_E": c.holder.E,
            "holder_prev_C": None if c.holder.prev_C is None else c.holder.prev_C.copy(),
            "global_C": None if c.global_C is None else c.global_C.detach().cpu().clone(),
        } for c in clients],
        "server": copy.deepcopy(server),
        "auditor": copy.deepcopy(auditor),
        "arch": copy.deepcopy(arch),
        "torch_rng": torch.get_rng_state().clone(),
        "cuda_rng": ([s.clone() for s in torch.cuda.get_rng_state_all()]
                     if torch.cuda.is_available() else None),
        "np_rng": np.random.get_state(),
    }


def restore_parameters(snap, clients):
    for c, s in zip(clients, snap["clients"]):
        c.model.load_state_dict(s["model"])
        c.opt.load_state_dict(copy.deepcopy(s["opt"]))
        c.holder.round = s["holder_round"]
        c.holder.E = s["holder_E"]
        c.holder.prev_C = None if s["holder_prev_C"] is None else s["holder_prev_C"].copy()
        c.global_C = None if s["global_C"] is None else s["global_C"].to(c.cfg.device)
        c.model.install_dynamic(s["holder_E"])
    torch.set_rng_state(snap["torch_rng"].clone())
    if snap["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all([s.clone() for s in snap["cuda_rng"]])
    np.random.set_state(snap["np_rng"])
    return (copy.deepcopy(snap["server"]), copy.deepcopy(snap["auditor"]),
            copy.deepcopy(snap["arch"]))


def new_archive(n):
    return {"true": {i: {} for i in range(n)}, "pub": {i: {} for i in range(n)},
            "present": {i: {} for i in range(n)}, "cbar": {}, "counts": {i: {} for i in range(n)},
            "loss": {i: {} for i in range(n)}}


def run_rounds(clients, server, auditor, arch, r0, r1, roles, a, rows, rng, label=""):
    n = len(clients)
    t0 = time.time()
    for r in range(r0, r1 + 1):
        cbar_prev = server.global_C.copy() if server.global_C is not None else None

        true_C, counts = {}, {}
        for c in clients:
            stats = c.local_train()
            C, cnt, intra, inter = c.compute_prototypes()
            true_C[c.cid] = np.asarray(C, dtype=np.float64)
            counts[c.cid] = cnt
            arch["true"][c.cid][r] = true_C[c.cid].copy()
            arch["counts"][c.cid][r] = cnt.astype(np.float32).copy()
            arch["loss"][c.cid][r] = {"train_loss": stats["train_loss"],
                                      "local_acc": stats["local_acc"],
                                      "intra": intra, "inter": inter}
            c._round_meta = (stats, intra, inter)

        active = sorted(roles) if r >= a.attack_start else []
        pub_now = {}
        for c in clients:
            if c.cid not in active:
                pub_now[c.cid] = true_C[c.cid]
        for cid in active:
            kind, strength = roles[cid]
            pub_now[cid] = attack_payload(
                kind, cbar_prev=cbar_prev, C_own=true_C[cid],
                own_true={k: v for k, v in arch["true"][cid].items() if k < r},
                own_pub=dict(arch["pub"][cid]),
                peer_pub_now={j: pub_now[j] for j in pub_now}, r=r, cid=cid,
                rng=rng, strength=strength)

        contribs = []
        for c in clients:
            stats, intra, inter = c._round_meta
            contribs.append(c.holder.issue(
                pub_now[c.cid], dict(n_samples=c.n_samples, train_loss=stats["train_loss"],
                                     local_acc=stats["local_acc"], intra_compactness=intra,
                                     inter_separation=inter)))
            c.calibrate()
        for c in clients:
            arch["pub"][c.cid][r] = np.asarray(contribs[c.cid].C, dtype=np.float64).copy()
            arch["present"][c.cid][r] = (counts[c.cid] > 0)

        Cbar = server.aggregate(contribs, [counts[c.cid] for c in clients])
        arch["cbar"][r] = np.asarray(Cbar, dtype=np.float64).copy()

        score_one_round(auditor, r,
                        {c.cid: arch["pub"][c.cid][r] for c in clients},
                        {c.cid: arch["present"][c.cid][r] for c in clients},
                        cbar_prev, set(active), rows)

        for c in clients:
            c.receive_global(Cbar)

        print(json.dumps({"scenario": label, "round": r,
                          "elapsed_s": round(time.time() - t0, 1)}), flush=True)
    return arch


def replay_open_loop(honest_arch, roles, a, seed):
    rng = np.random.default_rng(seed)
    aud = InnovationAuditor()
    rows = []
    pub_hist = {i: {} for i in range(a.clients)}
    for r in range(1, a.rounds + 1):
        cbar_prev = honest_arch["cbar"].get(r - 1)
        active = sorted(roles) if r >= a.attack_start else []
        pub_now = {}
        for cid in range(a.clients):
            if cid not in active:
                pub_now[cid] = honest_arch["pub"][cid][r]
        for cid in active:
            kind, strength = roles[cid]
            pub_now[cid] = dequantize(attack_payload(
                kind, cbar_prev=cbar_prev,
                C_own=honest_arch["true"][cid][r],
                own_true={k: v for k, v in honest_arch["true"][cid].items() if k < r},
                own_pub=dict(pub_hist[cid]),
                peer_pub_now={j: pub_now[j] for j in pub_now}, r=r, cid=cid,
                rng=rng, strength=strength))
        for cid in range(a.clients):
            pub_hist[cid][r] = pub_now[cid]
        present = {cid: honest_arch["present"][cid][r] for cid in range(a.clients)}
        score_one_round(aud, r, pub_now, present, cbar_prev, set(active), rows)
    return rows


def summarise_scores(rows, a, tag):
    ev = [x for x in rows if x["round"] >= a.eval_from]
    pos = [x for x in ev if x["attacker"]]
    neg = [x for x in ev if not x["attacker"]]
    out = {"tag": tag, "n_pos": len(pos), "n_neg": len(neg), "auc": {}, "pos_mean": {},
           "neg_mean": {}, "neg_median": {}, "neg_p95": {}}
    keys = list(STAT_KEYS) + list(BASELINES)
    for k in keys:
        p = [x[k] for x in pos if x.get(k) is not None]
        q = [x[k] for x in neg if x.get(k) is not None]
        out["auc"][k] = roc_auc(p, q)
        out["pos_mean"][k] = float(np.mean(p)) if p else None
        out["neg_mean"][k] = float(np.mean(q)) if q else None
        out["neg_median"][k] = float(np.median(q)) if q else None
        out["neg_p95"][k] = float(np.percentile(q, 95)) if q else None
    out["pos_degenerate_frac"] = float(np.mean([x["degenerate"] for x in pos])) if pos else None
    out["neg_degenerate_frac"] = float(np.mean([x["degenerate"] for x in neg])) if neg else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=28)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--dirichlet", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--attack-start", type=int, default=10)
    ap.add_argument("--eval-from", type=int, default=12)
    ap.add_argument("--dataset", default="cifar100", choices=["cifar100", "cifar10"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--fast-data", action="store_true",
                    help="in-memory pre-normalised tensors, augmentation off (see load_archive_lazily)")
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenarios", default="honest,global_echo,shrink,ar2_replay,collude3_shrink")
    ap.add_argument("--tag", default="")
    ap.add_argument("--eval-utility", action="store_true", default=True)
    ap.add_argument("--no-eval-utility", dest="eval_utility", action="store_false")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if a.torch_threads:
        torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    cfg = FedConfig(num_classes=a.num_classes, embed_dim=a.embed_dim, width=a.width,
                    rounds=a.rounds, local_epochs=a.local_epochs, warmup_rounds=10 ** 6,
                    audit_z_threshold=10 ** 6, scheme="Ed25519", device=a.device)
    if a.synthetic:
        train, test = load_synthetic(600 * a.clients, 300, a.num_classes, a.seed)
        nc = a.num_classes
    elif a.fast_data:
        train, test, nc = load_archive_lazily(a.dataset)
        a.num_classes = nc
        cfg.num_classes = nc
    else:
        train, test, nc = load_dataset(a.dataset)
        a.num_classes = nc
        cfg.num_classes = nc
    parts = dirichlet_partition(train, a.clients, a.dirichlet, nc, a.seed)

    clients, server = build_federation(a, cfg, parts, nc)
    weights = np.asarray([c.n_samples for c in clients], dtype=float)
    weights = weights / weights.sum()

    scen_names = [s for s in a.scenarios.split(",") if s]
    assert "honest" in scen_names and scen_names[0] == "honest", \
        "the honest control must run first: it is the open-loop archive"

    auditor = InnovationAuditor()
    arch = new_archive(a.clients)
    prefix_rows = []
    rng = np.random.default_rng(a.seed)
    t_start = time.time()
    if a.attack_start > 1:
        run_rounds(clients, server, auditor, arch, 1, a.attack_start - 1, {},
                   a, prefix_rows, rng, label="prefix")
    snap = snapshot_parameters(clients, server, auditor, arch)
    print(f"[prefix] rounds 1..{a.attack_start - 1} done in {time.time() - t_start:.0f}s", flush=True)

    results = {"config": vars(a), "client_weights": weights.tolist(),
               "scenarios": {}, "openloop": {}, "utility": {}}
    archives = {}

    honest_arch = None

    def scenario_spec(name):
        return {cid % a.clients: v for cid, v in SCENARIOS[name].items()}

    os.makedirs(a.outdir, exist_ok=True)
    tag = a.tag or f"{a.dataset}_s{a.seed}"
    path = os.path.join(a.outdir, f"online_{tag}.json")

    def drift_for(name):
        attackers = set(scenario_spec(name))
        per_round = []
        for r in range(a.attack_start, a.rounds + 1):
            h_cb = honest_arch["cbar"][r]
            a_cb = archives[name]["cbar"][r]
            d_cb = a_cb - h_cb
            atk_dir = np.zeros_like(h_cb)
            for cid in attackers:
                atk_dir += (archives[name]["pub"][cid][r] - h_cb)
            na, nd = np.linalg.norm(atk_dir), np.linalg.norm(d_cb)
            cos = float((atk_dir * d_cb).sum() / (na * nd)) if na > 1e-12 and nd > 1e-12 else float("nan")
            rel = []
            for cid in range(a.clients):
                if cid in attackers:
                    continue
                hp, ap_ = honest_arch["pub"][cid][r], archives[name]["pub"][cid][r]
                den = np.linalg.norm(hp)
                if den > 1e-12:
                    rel.append(float(np.linalg.norm(ap_ - hp) / den))
            per_round.append({
                "round": r,
                "cbar_rel_shift": float(nd / (np.linalg.norm(h_cb) + 1e-12)),
                "cos_shift_vs_attacker": cos,
                "honest_payload_rel_shift_mean": float(np.mean(rel)) if rel else None,
            })
        return per_round

    def save():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, default=float)

    results["drift"] = {}
    for name in scen_names:
        roles = scenario_spec(name)
        attackers = sorted(roles)
        s_server, s_auditor, s_arch = restore_parameters(snap, clients)
        rows = list(prefix_rows)
        rng = np.random.default_rng(a.seed + 1000)
        t0 = time.time()
        run_rounds(clients, s_server, s_auditor, s_arch, a.attack_start, a.rounds,
                   roles, a, rows, rng, label=name)
        el = time.time() - t0
        results["scenarios"][name] = {
            "attackers": attackers,
            "roles": {str(k): list(v) for k, v in roles.items()},
            "attacker_weight": float(weights[attackers].sum()) if attackers else 0.0,
            "elapsed_s": round(el, 1),
            "closed_loop": summarise_scores(rows, a, f"closed:{name}"),
            "rows": rows,
            "degenerate_rounds": s_server.degenerate_rounds,
        }
        archives[name] = s_arch
        if name == "honest":
            honest_arch = s_arch
        if a.eval_utility:
            accs = [c.evaluate(test) for c in clients]
            results["utility"][name] = {
                "acc_classifier_mean": float(np.mean([x["acc_classifier"] for x in accs])),
                "acc_prototype_mean": float(np.mean([x.get("acc_prototype", np.nan) for x in accs])),
                "per_client": accs,
            }
        o_rows = replay_open_loop(honest_arch, roles, a, a.seed)
        results["openloop"][name] = {"summary": summarise_scores(o_rows, a, f"open:{name}"),
                                     "rows": o_rows}
        if name != "honest":
            results["drift"][name] = drift_for(name)
        save()
        print(f"[{name}] suspicion AUC  open={results['openloop'][name]['summary']['auc']['suspicion']:.3f}"
              f"  closed={results['scenarios'][name]['closed_loop']['auc']['suspicion']:.3f}"
              f"  ({el:.0f}s)", flush=True)

    print(f"wrote {path}")
    os.makedirs(os.path.join(a.outdir, "archives"), exist_ok=True)
    npz = os.path.join(a.outdir, "archives", f"online_{tag}.npz")

    def _small(v):
        if isinstance(v, dict):
            return {k: _small(x) for k, x in v.items()}
        return np.asarray(v, dtype=np.float32) if np.asarray(v).dtype.kind == "f" else v

    np.savez_compressed(npz, **{f"{n}__{k}": np.asarray(_small(v), dtype=object)
                                for n, ar in archives.items()
                                for k, v in ar.items() if k in ("cbar", "pub", "true", "present")})
    print(f"wrote {npz}")

    print("\n" + "=" * 78)
    print("%-18s %10s %10s %10s %10s" % ("scenario", "AUC_open", "AUC_closed", "delta", "atk_wt"))
    print("=" * 78)
    for name in scen_names:
        if name == "honest":
            continue
        o = results["openloop"][name]["summary"]["auc"]["suspicion"]
        c = results["scenarios"][name]["closed_loop"]["auc"]["suspicion"]
        print("%-18s %10.3f %10.3f %10.3f %10.3f"
              % (name, o, c, c - o, results["scenarios"][name]["attacker_weight"]))


if __name__ == "__main__":
    main()
