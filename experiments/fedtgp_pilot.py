from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

import fedtgp_attacks as FA
import protocol_eval as P
from passportfl.baselines import bulyan_deviation, krum_scores, median_deviation, trimmed_mean_deviation
from passportfl.fedtgp import TGPConfig, TGPTrainer
from passportfl.innovation import InnovationAuditor

ATTACK_START = 12
SCOPE_START = 12
RULES = {"krum": lambda innov: krum_scores(innov, n_byzantine=2),
         "median_dev": median_deviation,
         "trimmed_mean_dev": trimmed_mean_deviation,
         "bulyan_dev": lambda innov: bulyan_deviation(innov, n_byzantine=2)}


def fit_server(arch: dict, rounds: int, cfg: TGPConfig) -> dict:
    clients = len(arch["C"])
    K, d = arch["cbar"][1].shape
    tr = TGPTrainer(K, d, cfg)
    P_hat, Q, HAS, diag = {}, {}, {}, []
    for r in range(1, rounds + 1):
        payloads = [(arch["C"][i][r], arch["present"][i][r]) for i in range(clients)]
        info = tr.fit_round(payloads, r)
        P_hat[r] = info.pop("P_hat")
        q, has = info.pop("Q"), info.pop("Q_has")
        Q[r] = np.where(has[:, None], q, P_hat[r])
        HAS[r] = has
        info["p_minus_cbar_fro"] = float(np.linalg.norm(P_hat[r] - arch["cbar"][r]))
        info["q_minus_cbar_fro"] = float(np.linalg.norm((q - arch["cbar"][r])[has]))
        diag.append(info)
        print("round %2d  margin %.3f  ||P-Q|| %.2f  ||P-cbar|| %.2f  ||Q-cbar|| %.2f  loss %.4f  %.1fs"
              % (r, info["margin"], info["p_minus_q_fro"], info["p_minus_cbar_fro"],
                 info["q_minus_cbar_fro"], info["loss_last_epoch_mean"], info["seconds"]), flush=True)
    return {"P_hat": P_hat, "Q": Q, "has": HAS, "diag": diag}


def geometry(arch: dict, bcast: dict, Q: dict, rounds: int) -> dict:
    return FA.innovation_geometry(arch, bcast, Q, rounds, SCOPE_START)


def replay_variant(arch: dict, variant: str, seed: int, rounds: int) -> dict:
    fn = FA.VARIANTS[variant]
    clients = len(arch["C"])
    names = list(RULES) + ["suspicion"]
    pos = {k: [] for k in names}
    neg = {k: [] for k in names}
    ranks = {k: [] for k in RULES}
    for victim in range(clients):
        work = dict(arch)
        work["published"] = {i: {} for i in range(clients)}
        rng = np.random.default_rng([seed, victim + 1, FA.VARIANT_ORDER.index(variant) + 1])
        aud = InnovationAuditor()
        for r in range(1, rounds + 1):
            if (r - 1) in work["cbar"]:
                aud.set_aggregate(r - 1, work["cbar"][r - 1])
            payloads = {}
            for cid in range(clients):
                C = work["C"][cid][r]
                if cid == victim and r >= ATTACK_START:
                    C = fn(work, cid, r, rng)
                work["published"][cid][r] = C
                payloads[cid] = (C, work["present"][cid][r])
            rs = aud.observe_round(r, payloads)
            if r < SCOPE_START or (r - 1) not in work["cbar"]:
                continue
            innov = {cid: (payloads[cid][0] - work["cbar"][r - 1]).ravel() for cid in payloads}
            for k, f in RULES.items():
                s = f(innov)
                arr = np.array([s[i] for i in range(clients)])
                pos[k].append(float(arr[victim]))
                neg[k].extend(float(arr[j]) for j in range(clients) if j != victim)
                ranks[k].append(FA.attacker_rank(arr, victim))
            for cid, v in rs.per_client.items():
                val = float(v.get("suspicion", float("nan")))
                if np.isfinite(val):
                    (pos if cid == victim else neg)["suspicion"].append(val)
    out = {}
    for k in names:
        e = {"auc": P.roc_auc(np.asarray(pos[k]), np.asarray(neg[k])),
             "n_pos": len(pos[k]), "n_neg": len(neg[k])}
        if k in ranks:
            e.update(FA.rank_summary(ranks[k]))
        out[k] = e
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.path.join(HERE, "results", "archives",
                                                      "arch_nopp_s0_c10_r35_d0.5_s0.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=35)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tgp-epochs", type=int, default=100)
    ap.add_argument("--tgp-lr", type=float, default=0.01)
    ap.add_argument("--tgp-batch", type=int, default=10)
    ap.add_argument("--tgp-tau", type=float, default=100.0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "fedtgp_pilot.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()

    arch = FA.load_archive_ext(a.archive)
    clients, rounds = P.archive_shape(arch)
    rounds = min(rounds, a.rounds)
    cfg = TGPConfig(server_epochs=a.tgp_epochs, server_lr=a.tgp_lr, server_batch=a.tgp_batch,
                    margin_threshold=a.tgp_tau, seed=a.seed, device=a.device)
    fit = fit_server(arch, rounds, cfg)

    fp = {"C": arch["C"], "present": arch["present"], "cbar": arch["cbar"], "qbar": arch["qbar"]}
    tg = {"C": arch["C"], "present": arch["present"], "cbar": fit["P_hat"], "qbar": fit["Q"]}
    q_check = max(float(np.abs(fit["Q"][r] - arch["qbar"][r])[fit["has"][r]].max())
                  for r in range(1, rounds + 1))

    out = {"meta": {"command": "python experiments/fedtgp_pilot.py " + " ".join(sys.argv[1:]),
                    "archive": os.path.basename(a.archive), "rounds_fitted": rounds,
                    "attack_start": ATTACK_START, "scope_start": SCOPE_START,
                    "tgp": vars(cfg), "qbar_fit_vs_recomputed_max_abs": q_check},
           "server": fit["diag"],
           "geometry": {"fedproto_cbar": geometry(fp, fp["cbar"], fp["qbar"], rounds),
                        "fedtgp_refit": geometry(tg, tg["cbar"], tg["qbar"], rounds)},
           "variants": {}}
    for name, ar in (("fedproto_cbar", fp), ("fedtgp_refit", tg)):
        out["variants"][name] = {}
        for v in FA.VARIANT_ORDER:
            res = replay_variant(ar, v, a.seed, rounds)
            out["variants"][name][v] = res
            print("%-14s %-20s " % (name, v) + "  ".join(
                "%s AUC %.3f rank %.2f r1 %.2f" % (k, e["auc"], e["rank_mean"], e["rank1_frac"])
                if "rank_mean" in e else "%s AUC %.3f" % (k, e["auc"])
                for k, e in res.items()), flush=True)
    from passportfl.reproducibility import environment
    out["meta"]["environment"] = environment()
    out["meta"]["wall_s"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("wrote %s (%.0fs)" % (a.out, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
