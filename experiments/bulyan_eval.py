from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

import protocol_eval as P
from passportfl.baselines import bulyan_deviation

ARCH = os.path.join(HERE, "results", "archives")
CONFIGS = {
    "cifar100_d05": "arch_cifar100_s[0-4]_c10_r35*.npz",
    "cifar100_d01": "arch_cifar100_d01_s*_c10_r35*.npz",
    "cifar10": "arch_cifar10_s*_c10_r35*.npz",
    "late70": "arch_late70_s*_c10_r70*.npz",
    "cohort20": "arch_cohort20_s*_c20_r35*.npz",
    "cohort40": "arch_cohort40_s*_c40_r35*.npz",
    "nopassport": "arch_nopp_s*_c10_r35*.npz",
}
EXTRA_CONFIGS = {
    "hetero_fedproto": "arch_hetero_s*_c10_r35*.npz",
    "hetero_fedtgp_lam1": "arch_fedtgp_lam1_s*_c10_r35*.npz",
    "hetero_fedtgp_lam10": "arch_fedtgp_lam10_s*_c10_r35*.npz",
    "hetero_fedtgp_lam01": "arch_fedtgp_lam01_s*_c10_r35*.npz",
}
RELEASED_PER_SEED = os.path.join(HERE, "results", "bulyan_per_seed.json")
EXTRA_ARCH = os.path.join(HERE, "results", "archives_rev2")
F_BYZANTINE = 2
ATTACK_START = 12
EVAL_FROM = 12


def roc_auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def score_one_archive(path: str, seed: int, attack: str, victim: int):
    base = P.load_archive(path)
    clients, rounds = P.archive_shape(base)
    arch = dict(base)
    arch["published"] = {i: {} for i in range(clients)}
    fn = P.ATTACKS[attack]
    rng = np.random.default_rng([int(seed), int(victim) + 1, P.ATTACK_ORDER.index(attack) + 1])
    pos, neg = [], []
    for r in range(1, rounds + 1):
        Cbar_prev = arch["cbar"].get(r - 1)
        payloads = {}
        for cid in range(clients):
            C = arch["C"][cid][r]
            if cid == victim and r >= ATTACK_START:
                C = fn(arch, cid, r, rng)
            payloads[cid] = C
            arch["published"][cid][r] = C
        if Cbar_prev is None or r < EVAL_FROM:
            continue
        innov = {cid: (payloads[cid] - Cbar_prev).ravel() for cid in payloads}
        s = bulyan_deviation(innov, n_byzantine=F_BYZANTINE)
        for cid, v in s.items():
            (pos if cid == victim else neg).append(float(v))
    return pos, neg


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for f in (os.path.abspath(__file__), P.__file__,
              os.path.join(HERE, "passportfl", "baselines.py")):
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def shown_path(p: str) -> str:
    try:
        return os.path.relpath(p, HERE).replace(os.sep, "/")
    except ValueError:
        return os.path.abspath(p)


def score_archive_attack(task):
    p, seed, attack = task
    clients, _ = P.archive_shape(P.load_archive(p))
    pos, neg = [], []
    for victim in range(clients):
        pp, nn = score_one_archive(p, seed, attack, victim)
        pos += pp
        neg += nn
    return pos, neg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "bulyan_eval.json"))
    ap.add_argument("--per-seed-out", default="",
                    help="also write per-seed AUCs and their mean over seeds to this file")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help="comma-separated subset of: " + ", ".join(CONFIGS)
                         + "; revision configurations, read from --extra-archives: "
                         + ", ".join(EXTRA_CONFIGS))
    ap.add_argument("--extra-archives", default=EXTRA_ARCH)
    a = ap.parse_args()
    t0 = time.time()
    cfgs = [c.strip() for c in a.configs.split(",") if c.strip()]
    unknown = [c for c in cfgs if c not in CONFIGS and c not in EXTRA_CONFIGS]
    if unknown:
        raise SystemExit("unknown configuration(s): %s" % unknown)
    if set(cfgs) & set(EXTRA_CONFIGS) and os.path.abspath(a.out) == os.path.abspath(
            ap.get_default("out")):
        raise SystemExit("revision configurations go to a separate --out, not %s"
                         % ap.get_default("out"))
    if set(cfgs) & set(EXTRA_CONFIGS) and a.per_seed_out and os.path.abspath(
            a.per_seed_out) == os.path.abspath(RELEASED_PER_SEED):
        raise SystemExit("revision configurations go to a separate --per-seed-out, not %s"
                         % RELEASED_PER_SEED)

    plan, tasks = {}, []
    for cfg in cfgs:
        if cfg in CONFIGS:
            paths = sorted(glob.glob(os.path.join(ARCH, CONFIGS[cfg])))
        else:
            paths = sorted(glob.glob(os.path.join(a.extra_archives, EXTRA_CONFIGS[cfg])))
        plan[cfg] = paths
        for attack in P.ATTACK_ORDER:
            for p in paths:
                seed = int(p.rsplit("_s", 1)[-1].split(".")[0])
                tasks.append((p, seed, attack))
    if a.workers <= 1:
        sched = tasks
        done = [score_archive_attack(tk) for tk in sched]
    else:
        sched = sorted(tasks, key=lambda tk: -int(re.search(r"_c(\d+)_r",
                                                            os.path.basename(tk[0])).group(1)))
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            done = list(ex.map(score_archive_attack, sched))
    lists = dict(zip(sched, done))

    out, per_seed = {}, {}
    for cfg in cfgs:
        paths = plan[cfg]
        if not paths:
            print("%-14s no archives" % cfg)
            continue
        per, seeds = {}, {}
        for attack in P.ATTACK_ORDER:
            pos, neg = [], []
            seeds[attack] = {}
            for p in paths:
                seed = int(p.rsplit("_s", 1)[-1].split(".")[0])
                pp, nn = lists[(p, seed, attack)]
                pos += pp
                neg += nn
                seeds[attack][seed] = roc_auc(pp, nn)
            per[attack] = roc_auc(pos, neg)
        out[cfg] = per
        per_seed[cfg] = seeds
        print("%-14s " % cfg + "  ".join("%s=%.3f" % (k, v) for k, v in per.items()), flush=True)

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote %s" % a.out)

    if a.per_seed_out:
        from passportfl.reproducibility import environment
        summary = {}
        for cfg, seeds in per_seed.items():
            summary[cfg] = {}
            for attack, by_seed in seeds.items():
                v = [by_seed[s] for s in sorted(by_seed)]
                summary[cfg][attack] = {
                    "auc_mean": float(np.mean(v)), "auc_sd_ddof0": float(np.std(v)),
                    "auc_sd_ddof1": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                    "auc_min": float(min(v)), "auc_max": float(max(v)),
                    "per_seed": {str(s): by_seed[s] for s in sorted(by_seed)},
                    "pooled_auc": out[cfg][attack]}
            print("%-14s per-seed mean " % cfg + "  ".join(
                "%s=%.3f" % (k, summary[cfg][k]["auc_mean"]) for k in P.ATTACK_ORDER),
                flush=True)
        payload = {
            "meta": {
                "command": "python experiments/bulyan_eval.py --per-seed-out "
                           + shown_path(a.per_seed_out)
                           + ("" if cfgs == list(CONFIGS) else " --configs " + ",".join(cfgs)),
                "archives": {cfg: [os.path.basename(p) for p in paths]
                             for cfg, paths in plan.items()},
                "code_sha256": code_fingerprint(),
                "environment": environment(),
                "wall_time_s": round(time.time() - t0, 1),
            },
            "configs": summary,
        }
        with open(a.per_seed_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        print("wrote %s" % a.per_seed_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
