from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

import fedtgp_attacks as FA
import protocol_eval as P

ATTACK_START = 12
SCOPE_START = 12
COHORTS = ("fedtgp_echo", "fedtgp_offset_echo", "fedtgp_recent_offset_echo", "fedtgp_affine_echo",
           "leave_one_out")


def published_history(arch: dict, variant: str, victim: int, upto: int) -> dict:
    clients = len(arch["C"])
    work = dict(arch)
    work["published"] = {i: {} for i in range(clients)}
    rng = np.random.default_rng(0)
    fn = FA.VARIANTS[variant]
    for r in range(1, upto + 1):
        for cid in range(clients):
            C = arch["C"][cid][r]
            if cid == victim and r >= ATTACK_START:
                C = fn(work, cid, r, rng)
            work["published"][cid][r] = C
    return work["published"][victim]


def refit_round(task: dict) -> dict:
    import torch
    from passportfl.fedtgp import TGPConfig, TGPTrainer
    torch.set_num_threads(task["threads"])
    t0 = time.time()
    z = np.load(task["path"], allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    state = z["tgp_state"].item()
    arch = FA.load_archive_ext(task["path"])
    r = task["round"]
    clients = len(arch["C"])
    K, d = arch["cbar"][1].shape
    t = meta["tgp"]
    cfg = TGPConfig(server_epochs=t["server_epochs"], server_lr=t["server_lr"],
                    server_batch=t["server_batch"], margin_threshold=t["margin_threshold"],
                    hidden_dim=t["hidden_dim"], seed=t["seed"], device=task["device"])
    tr = TGPTrainer(K, d, cfg)
    base = [(arch["C"][i][r], arch["present"][i][r]) for i in range(clients)]

    def fit(payloads):
        tr.load_state(state[r - 1])
        return tr.fit_round(payloads, r)

    clean = fit(base)
    P_prev = arch["cbar"][r - 1]
    R_clean = clean["P_hat"] - P_prev
    rc2 = float((R_clean * R_clean).sum())
    out = {"seed": task["seed"], "round": r,
           "clean_vs_archived_max_abs": float(np.abs(clean["P_hat"] - arch["cbar"][r]).max()),
           "clean_step_norm": float(np.sqrt(rc2)), "clean_margin": clean["margin"], "cohorts": {}}
    for v in task["seats"]:
        for c in task.get("cohorts", COHORTS):
            if c == "leave_one_out":
                pl = [base[i] for i in range(clients) if i != v]
            else:
                hist = published_history(arch, c, v, r)
                pl = list(base)
                pl[v] = (hist[r], arch["present"][v][r])
            res = fit(pl)
            R = res["P_hat"] - P_prev
            out["cohorts"]["%s|%d" % (c, v)] = {
                "shift_own": float(np.linalg.norm(R - R_clean)) / np.sqrt(rc2),
                "progress_own": float((R * R_clean).sum()) / rc2,
                "broadcast_change": float(np.linalg.norm(res["P_hat"] - clean["P_hat"])),
                "margin": res["margin"]}
    out["seconds"] = time.time() - t0
    return out


def quart(x) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"median": None, "q25": None, "q75": None, "n": 0}
    q25, med, q75 = np.percentile(x, [25, 50, 75])
    return {"median": float(med), "q25": float(q25), "q75": float(q75), "n": int(x.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", default=os.path.join(HERE, "results", "archives_rev2"))
    ap.add_argument("--archive-prefix", default="arch_fedtgp_lam1_s",
                    help="one FedTGP lambda arm (passportfl/fedtgp.py, LAMBDA_ARMS)")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--seats", default="0,1")
    ap.add_argument("--cohorts", default=",".join(COHORTS),
                    help="attacked cohorts: variants of fedtgp_attacks.VARIANTS and leave_one_out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "fedtgp_live_impact.json"))
    a = ap.parse_args()
    t0 = time.time()
    ints = lambda s: [int(x) for x in s.split(",") if x.strip()]
    paths = P.discover_archives(a.archives, a.archive_prefix)
    seeds = [s for s in ints(a.seeds) if s in paths]
    if not seeds:
        raise SystemExit("no archive with prefix %s in %s" % (a.archive_prefix, a.archives))
    seats = ints(a.seats)
    cohorts = [c.strip() for c in a.cohorts.split(",") if c.strip()]
    bad = [c for c in cohorts if c != "leave_one_out" and c not in FA.VARIANTS]
    if bad:
        raise SystemExit("unknown cohort(s) %s" % bad)
    if "leave_one_out" not in cohorts:
        cohorts.append("leave_one_out")
    arch_meta = {}
    for s in seeds:
        z = np.load(paths[s], allow_pickle=True)
        if "meta" not in z.files or "tgp_state" not in z.files:
            raise SystemExit("%s is not a FedTGP archive (no meta/tgp_state)" % paths[s])
        arch_meta[str(s)] = json.loads(str(z["meta"]))
    confs = {FA.config_key(m) for m in arch_meta.values()}
    if len(confs) > 1:
        raise SystemExit("archives of different configurations under prefix %s: %s"
                         % (a.archive_prefix, sorted(confs)))
    tasks = []
    for s in seeds:
        _, rounds = P.archive_shape(FA.load_archive_ext(paths[s]))
        for r in range(SCOPE_START, rounds + 1):
            tasks.append({"path": paths[s], "seed": s, "round": r, "seats": seats,
                          "cohorts": cohorts, "device": a.device, "threads": a.threads})
    workers = max(1, min(4, a.workers))
    if workers <= 1:
        res = [refit_round(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(refit_round, tasks))
    summary = {}
    for c in cohorts:
        rows = [x["cohorts"]["%s|%d" % (c, v)] for x in res for v in seats]
        loo = [x["cohorts"]["leave_one_out|%d" % v]["shift_own"] for x in res for v in seats]
        ent = {m: quart([q[m] for q in rows]) for m in ("shift_own", "progress_own", "broadcast_change")}
        if c != "leave_one_out":
            ent["shift_over_loo"] = quart([q["shift_own"] / l if l > 0 else np.nan
                                           for q, l in zip(rows, loo)])
        summary[c] = ent
        print("%-20s shift_own %.3f  progress_own %.3f  shift/LOO %s"
              % (c, ent["shift_own"]["median"], ent["progress_own"]["median"],
                 "%.3f" % ent["shift_over_loo"]["median"] if "shift_over_loo" in ent else "-"))
    from passportfl.reproducibility import environment
    out = {"meta": {"command": "python experiments/fedtgp_live_impact.py " + " ".join(sys.argv[1:]),
                    "archives": {str(s): os.path.basename(paths[s]) for s in seeds},
                    "archive_meta": arch_meta,
                    "seats": seats, "attack_start": ATTACK_START, "scope_start": SCOPE_START,
                    "clean_vs_archived_max_abs": max(x["clean_vs_archived_max_abs"] for x in res),
                    "compute_s_total": round(sum(x["seconds"] for x in res), 1),
                    "wall_s": round(time.time() - t0, 1), "environment": environment()},
           "summary": summary, "per_round": res}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("clean re-fit vs archived broadcast: max |diff| %.3g" % out["meta"]["clean_vs_archived_max_abs"])
    print("wrote %s (%.0fs)" % (a.out, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
