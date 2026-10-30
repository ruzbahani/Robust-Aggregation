import argparse, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from passportfl.baselines import BASELINES, score_round
from passportfl.innovation import InnovationAuditor

OURS = ("u1", "u2", "g", "x", "rep", "suspicion")


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


def load_archive(path):
    z = np.load(path, allow_pickle=True)
    return {"C": z["C"].item(), "present": z["present"].item(), "cbar": z["cbar"].item()}


def make_attack(kind, strength):
    def fn(arch, pub, cid, r, rng):
        base = arch["cbar"].get(r - 1)
        honest = arch["C"][cid][r]
        if kind == "shrink":
            return honest if base is None else base + strength * (honest - base)
        if kind == "global_echo":
            return honest if base is None else base + (1.0 - strength) * (honest - base)
        if kind == "peer_copy":
            others = [j for j in arch["C"] if j != cid]
            peer = arch["C"][others[cid % len(others)]][r]
            return (1.0 - strength) * honest + strength * peer
        if kind == "lagged_self":
            src = max(min(pub[cid]), r - 3)
            return (1.0 - strength) * honest + strength * pub[cid][src]
        if kind == "ar2_replay":
            rounds = sorted(k for k in pub[cid] if k < r)
            if len(rounds) < 3:
                return honest
            c1, c2, c3 = pub[cid][rounds[-1]], pub[cid][rounds[-2]], pub[cid][rounds[-3]]
            forged = c1 + (c1 - c2) + 0.25 * ((c1 - c2) - (c2 - c3))
            return (1.0 - strength) * honest + strength * forged
        raise ValueError(kind)
    return fn


def evaluate(arch, kind, strength, n_attackers, clients, rounds, attack_start, eval_from, seed):
    rng = np.random.default_rng(seed)
    fn = make_attack(kind, strength)
    attackers = set(range(clients - n_attackers, clients))
    aud = InnovationAuditor()
    pub = {i: {} for i in range(clients)}
    pos = {k: [] for k in tuple(OURS) + tuple(BASELINES)}
    neg = {k: [] for k in pos}

    for r in range(1, rounds + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        payloads = {}
        for cid in range(clients):
            C = arch["C"][cid][r]
            if cid in attackers and r >= attack_start:
                C = fn(arch, pub, cid, r, rng)
            pub[cid][r] = C
            payloads[cid] = (C, arch["present"][cid][r])
        rs = aud.observe_round(r, payloads)

        base = {}
        Cbar_prev = arch["cbar"].get(r - 1)
        if Cbar_prev is not None:
            base = score_round({cid: payloads[cid][0] - Cbar_prev for cid in payloads})

        if r < max(eval_from, attack_start):
            continue
        for cid, v in rs.per_client.items():
            tgt = pos if cid in attackers else neg
            for k in pos:
                val = base[k].get(cid, float("nan")) if k in base else v.get(k, float("nan"))
                if np.isfinite(val):
                    tgt[k].append(float(val))
    return {k: roc_auc(pos[k], neg[k]) for k in pos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="")
    ap.add_argument("--glob", default="")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=35)
    ap.add_argument("--attack-start", type=int, default=12)
    ap.add_argument("--eval-from", type=int, default=12)
    ap.add_argument("--out", default="results/sweeps.json")
    a = ap.parse_args()

    if a.archive:
        paths = [a.archive]
    elif a.glob:
        paths = sorted(glob.glob(a.glob))
    else:
        from protocol_eval import discover_archives
        available = discover_archives("results/archives", "arch_cifar100_s")
        wanted = [int(s) for s in a.seeds.split(",")]
        missing = sorted(set(wanted) - set(available))
        if missing:
            raise FileNotFoundError(f"Missing collusion-sweep seeds: {missing}")
        paths = [available[s] for s in wanted]
    if not paths:
        raise FileNotFoundError(f"No archives matched {a.glob}")
    print(f"archives: {len(paths)}")
    archives = [load_archive(p) for p in paths]

    out = {"archives": paths, "fraction_sweep": {}, "strength_sweep": {}}

    print("\n=== attacker fraction (attack = peer_copy, full strength) ===")
    keys = ["suspicion", "krum", "median_dev", "norm_bound", "foolsgold_maxcos"]
    print("%-10s" % "n_attack" + "".join("%16s" % k for k in keys))
    for n_att in (1, 2, 3, 4, 5):
        runs = [evaluate(ar, "peer_copy", 1.0, n_att, a.clients, a.rounds,
                         a.attack_start, a.eval_from, i) for i, ar in enumerate(archives)]
        row = {k: float(np.nanmean([r[k] for r in runs])) for k in runs[0]}
        out["fraction_sweep"][str(n_att)] = row
        print("%-10d" % n_att + "".join("%16.3f" % row[k] for k in keys))

    for kind in ("shrink", "peer_copy", "ar2_replay"):
        print(f"\n=== strength sweep: {kind} (1 attacker) ===")
        print("%-10s" % "strength" + "".join("%16s" % k for k in keys))
        out["strength_sweep"][kind] = {}
        for s in (0.1, 0.25, 0.5, 0.75, 1.0):
            runs = [evaluate(ar, kind, s, 1, a.clients, a.rounds,
                             a.attack_start, a.eval_from, i) for i, ar in enumerate(archives)]
            row = {k: float(np.nanmean([r[k] for r in runs])) for k in runs[0]}
            out["strength_sweep"][kind][str(s)] = row
            print("%-10.2f" % s + "".join("%16.3f" % row[k] for k in keys))

    os.makedirs("results", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
