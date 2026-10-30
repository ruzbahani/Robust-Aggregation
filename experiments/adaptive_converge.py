from __future__ import annotations

import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

from passportfl.innovation import InnovationAuditor
from passportfl.adaptive_attack import AttackConfig, WhiteBoxAttacker, FIXED_POINT_SCALE

ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "results", "archives")

EFFORTS = [
    ("E0_r1_b20_f0",    dict(restarts=1, basis_steps=20,  full_steps=0)),
    ("E1_r3_b80_f80",   dict(restarts=3, basis_steps=80,  full_steps=80)),
    ("E2_r4_b150_f200", dict(restarts=4, basis_steps=150, full_steps=200)),
    ("E3_r6_b300_f400", dict(restarts=6, basis_steps=300, full_steps=400)),
    ("E4_r8_b600_f1000", dict(restarts=8, basis_steps=600, full_steps=1000)),
]


def load_archive(path):
    z = np.load(path, allow_pickle=True)
    return {"C": z["C"].item(), "present": z["present"].item(), "cbar": z["cbar"].item()}


def run_one_configuration(task):
    torch.set_num_threads(1)
    arch = load_archive(task["path"])
    clients = len(arch["C"])
    rounds = max(max(v) for v in arch["C"].values())
    victim = task["victim"]
    mad_floor = task["mad_floor"]
    cfg = AttackConfig(phi=task["phi"], beta=task["beta"], knowledge=task["knowledge"],
                       aim=task["aim"], mad_floor=mad_floor, **task["effort"])
    atk = WhiteBoxAttacker(cfg, seed=1000 * task["seed"] + victim)
    aud = InnovationAuditor(mad_floor=mad_floor)
    objs, susp, bused, preds = [], [], [], []
    t0 = time.time()
    for r in range(1, rounds + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        present_r = {cid: arch["present"][cid][r] for cid in range(clients)}
        payloads = {cid: (arch["C"][cid][r], present_r[cid]) for cid in range(clients)}
        if r >= task["attack_start"]:
            peer_C = ({j: arch["C"][j][r] for j in range(clients) if j != victim}
                      if cfg.knowledge == "peer_current" else None)
            Cp, info = atk.craft(aud, r, victim, arch["C"][victim][r], present_r, peer_C)
            payloads[victim] = (Cp, present_r[victim])
            if info.get("status") == "ok":
                objs.append(float(info["objective_gap"]))
                preds.append(float(info["predicted_suspicion"]))
                Cb = aud.cbar.get(r - 1)
                Cc = atk._cheap_objective(Cb, arch["C"][victim][r])
                bused.append(float(np.linalg.norm(Cp - Cc)) / info["budget"])
        rs = aud.observe_round(r, payloads)
        if r >= task["attack_start"]:
            susp.append(float(rs.per_client[victim].get("suspicion", float("nan"))))
    m = lambda v: float(np.nanmean(v)) if v else None
    return {"effort": task["name"], "seed": task["seed"], "victim": victim,
            "phi": task["phi"], "beta": task["beta"],
            "obj_mean": m(objs), "obj_max": float(np.nanmax(objs)) if objs else None,
            "susp_mean": m(susp), "susp_max": float(np.nanmax(susp)) if susp else None,
            "bused_mean": m(bused), "n": len(objs), "secs": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", default=ARCHIVE_DIR)
    ap.add_argument("--prefix", default="arch_cifar100_s")
    ap.add_argument("--seeds", default="0,3")
    ap.add_argument("--victims", default="0,4,7")
    ap.add_argument("--phi", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--aim", default="mimic")
    ap.add_argument("--knowledge", default="peer_current")
    ap.add_argument("--attack-start", type=int, default=12)
    ap.add_argument("--mad-floor", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "results", "adaptive_converge.json"))
    a = ap.parse_args()
    import re
    paths = {}
    for fn in sorted(os.listdir(a.archives)):
        if fn.startswith(a.prefix) and fn.endswith(".npz"):
            m = re.search(r"_s(\d+)\.npz$", fn)
            if m:
                paths[int(m.group(1))] = os.path.join(a.archives, fn)
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    victims = [int(x) for x in a.victims.split(",") if x.strip()]
    tasks = []
    for name, eff in EFFORTS:
        for s in seeds:
            for v in victims:
                tasks.append({"name": name, "effort": eff, "path": paths[s], "seed": s,
                              "victim": v, "phi": a.phi, "beta": a.beta, "aim": a.aim,
                              "knowledge": a.knowledge, "attack_start": a.attack_start,
                              "mad_floor": a.mad_floor})
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(run_one_configuration, tasks, chunksize=1)):
            rows.append(r)
            print("  %d/%d %.1fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    agg = {}
    for r in rows:
        agg.setdefault(r["effort"], []).append(r)
    out = {"config": vars(a), "rows": rows, "summary": {}}
    print("\n%-20s %9s %9s %9s %9s %8s %8s" %
          ("effort", "obj_mean", "obj_max", "susp_mean", "susp_max", "Bused", "secs/pass"))
    for name, _ in EFFORTS:
        rs = agg.get(name, [])
        if not rs:
            continue
        f = lambda k: float(np.nanmean([x[k] for x in rs if x[k] is not None]))
        out["summary"][name] = {k: f(k) for k in
                                ("obj_mean", "obj_max", "susp_mean", "susp_max",
                                 "bused_mean", "secs")}
        print("%-20s %9.4f %9.4f %9.4f %9.4f %8.3f %8.1f" %
              (name, f("obj_mean"), f("obj_max"), f("susp_mean"), f("susp_max"),
               f("bused_mean"), f("secs")))
    with open(os.path.abspath(a.out), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\nwrote %s (%.1fs)" % (os.path.abspath(a.out), time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
