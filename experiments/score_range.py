from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

import protocol_eval as P
from passportfl import innovation as inn
from passportfl.reproducibility import environment

RESULTS = os.path.join(HERE, "results")
ARCHIVES = os.path.join(RESULTS, "archives")
CLIP = inn.SCORE_STAT_MAX
STRUCTURAL = inn.SCORE_STRUCTURAL
ATTACK_START = 12
FLOORS = (1e-3, 1e-2)


def archive_for(seed: int) -> str:
    hits = sorted(glob.glob(os.path.join(ARCHIVES, "arch_cifar100_s%d_c10_r35*_s%d.npz" % (seed, seed))))
    if not hits:
        raise FileNotFoundError("archive of seed %d" % seed)
    return hits[0]


def replay(arch: dict, attack, seed: int, floor: float) -> dict:
    inn.SCORE_STAT_MAX = math.inf
    clients, rounds = P.archive_shape(arch)
    work = dict(arch)
    rng = np.random.default_rng(seed)
    fn = P.ATTACKS[attack] if attack else None
    stats = {"victim": [], "other": []}
    degenerate = {"victim": 0, "other": 0}
    victims = range(clients) if fn else [None]
    for victim in victims:
        aud = inn.InnovationAuditor(mad_floor=floor)
        work["published"] = {i: {} for i in range(clients)}
        for r in range(1, rounds + 1):
            if (r - 1) in arch["cbar"]:
                aud.set_aggregate(r - 1, arch["cbar"][r - 1])
            payloads = {}
            for cid in range(clients):
                C = arch["C"][cid][r]
                if fn is not None and cid == victim and r >= ATTACK_START:
                    C = fn(work, cid, r, rng)
                work["published"][cid][r] = C
                payloads[cid] = (C, arch["present"][cid][r])
            rs = aud.observe_round(r, payloads)
            if r < ATTACK_START:
                continue
            for cid, v in rs.per_client.items():
                role = "victim" if cid == victim else "other"
                if v.get("degenerate"):
                    degenerate[role] += 1
                    continue
                s = v.get("suspicion", float("nan"))
                if np.isfinite(s):
                    stats[role].append(float(s))
    return {"stats": stats, "degenerate": degenerate}


def task(job):
    seed, attack, floor = job
    z = np.load(archive_for(seed), allow_pickle=True)
    arch = {"C": z["C"].item(), "present": z["present"].item(), "cbar": z["cbar"].item()}
    out = replay(arch, attack, seed, floor)
    summ = {}
    for role, xs in out["stats"].items():
        xs = np.asarray(xs)
        summ[role] = {"n": int(xs.size), "max": float(xs.max()) if xs.size else None,
                      "at_or_above_clip": int((xs >= CLIP).sum()),
                      "degenerate": out["degenerate"][role]}
    return {"seed": seed, "attack": attack or "none", "floor": floor, "roles": summ}


def closed_loop() -> dict:
    files = sorted(glob.glob(os.path.join(RESULTS, "utility_impact*.json")) +
                   glob.glob(os.path.join(RESULTS, "utility_robust_*_s*.json")))
    out = {}
    for p in files:
        d = json.load(open(p, encoding="utf-8"))
        if "scenarios" not in d:
            continue
        per = {}
        for name, sc in d["scenarios"].items():
            attackers = {str(a) for a in (sc.get("attackers") or [])}
            n = hi = 0
            roles = set()
            mx = -math.inf
            for row in sc.get("rows", []):
                deg = row.get("degenerate", {}) or {}
                for cid, v in (row.get("suspicion_raw", {}) or {}).items():
                    if v is None or not math.isfinite(float(v)) or deg.get(cid):
                        continue
                    v = float(v)
                    n += 1
                    mx = max(mx, v)
                    if v >= CLIP:
                        hi += 1
                        roles.add("attacker" if cid in attackers else "honest")
            if n:
                per[name] = {"n": n, "max": mx, "at_or_above_clip": hi, "roles_at_clip": sorted(roles)}
        out[os.path.basename(p)] = per
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(RESULTS, "score_range.json"))
    a = ap.parse_args()
    t0 = time.time()
    seeds = [int(s) for s in a.seeds.split(",")]
    jobs = [(s, atk, f) for f in FLOORS for s in seeds for atk in (None,) + tuple(P.ATTACK_ORDER)]
    with ProcessPoolExecutor(max_workers=min(4, a.workers)) as ex:
        rows = list(ex.map(task, jobs))
    summary = {}
    for f in FLOORS:
        sel = [r for r in rows if r["floor"] == f]
        honest = [r["roles"]["other"]["max"] for r in sel if r["attack"] == "none"]
        bystander = [r["roles"]["other"]["max"] for r in sel if r["attack"] != "none"]
        victim = [r["roles"]["victim"]["max"] for r in sel if r["attack"] != "none" and r["roles"]["victim"]["max"] is not None]
        summary[str(f)] = {
            "max_unattacked": max(honest), "max_nonvictim_in_attack_replays": max(bystander),
            "max_victim": max(victim),
            "max_test_seeds_3_4": max((r["roles"][k]["max"] for r in sel if r["seed"] in (3, 4)
                                       for k in ("victim", "other") if r["roles"][k]["max"] is not None),
                                      default=None),
            "at_or_above_clip": sum(r["roles"][k]["at_or_above_clip"] for r in sel for k in ("victim", "other")),
            "degenerate_unattacked": sum(r["roles"]["other"]["degenerate"] for r in sel if r["attack"] == "none"),
            "degenerate_nonvictim": sum(r["roles"]["other"]["degenerate"] for r in sel if r["attack"] != "none"),
            "degenerate_victim_by_attack": {atk: sum(r["roles"]["victim"]["degenerate"] for r in sel if r["attack"] == atk)
                                            for atk in P.ATTACK_ORDER},
        }
    res = {"config": {"seeds": seeds, "floors": list(FLOORS), "attacks": list(P.ATTACK_ORDER),
                      "attack_start": ATTACK_START, "clip": CLIP, "structural": STRUCTURAL,
                      "command": "python experiments/score_range.py --workers %d" % a.workers},
           "summary_offline": summary, "offline": rows, "closed_loop": closed_loop(),
           "environment": environment(), "wall_s": round(time.time() - t0, 1)}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(summary, indent=1))
    print("wrote %s (%.0f s)" % (a.out, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
