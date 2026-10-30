from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from passportfl.innovation import SCORE_STRUCTURAL, InnovationAuditor
from passportfl.reproducibility import environment

SCHEMA = "reduced-effort-drivers-v3"
MAD_FLOOR = 0.01
MAG_FIRST = 3
CONTROL_RESULT = "innovation_cifar100_s%d.json"
CONTROL_KEYS = ("clients", "rounds", "attack_start", "local_epochs", "dirichlet", "width", "embed_dim", "seed")
STATS = (("u1", None), ("u2", None), ("g", -1.0), ("x", +1.0), ("rep", +1.0))
NAMES = tuple(k for k, _ in STATS)
ATTACKERS = ("f0.1", "f0.5", "fwd", "zero")
TOL = 1e-12


def roc_auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def oriented(z: dict) -> dict:
    out = {}
    for key, sign in STATS:
        v = z.get("z_" + key, float("nan"))
        out[key] = float(abs(v) if sign is None else sign * v) if np.isfinite(v) else float("nan")
    return out


def finite_median(xs):
    v = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    return float(np.median(v)) if len(v) else None


def archive_path(results: str, tag: str, seed: int) -> str:
    folder = "archives" if re.fullmatch(r"f[0-9.]+", tag) else "archives_rev2"
    return os.path.join(results, folder, "arch_reduced_%s_s%d.npz" % (tag, seed))


def control_archive(results: str, cfg: dict) -> str:
    stem = "arch_%s_c%d_r%d" % (cfg.get("tag") or cfg["dataset"], cfg["clients"], cfg["rounds"])
    path = os.path.join(results, "archives", "%s_d%s_s%d.npz" % (stem, cfg["dirichlet"], cfg["seed"]))
    alternate = os.path.join(results, "archives", "%s_s%d.npz" % (stem, cfg["seed"]))
    return alternate if not os.path.exists(path) and cfg["dirichlet"] == 0.5 and os.path.exists(alternate) else path


def check_control(control: dict, ctrl_arch: str, run: dict, arch: str) -> list:
    keys = [k for k in CONTROL_KEYS if k in control["config"] and k in run["config"]]
    if not {"seed", "rounds", "attack_start"} <= set(keys):
        raise ValueError("%s: control and run do not both record seed, rounds and attack start" % arch)
    diff = {k: (control["config"][k], run["config"][k]) for k in keys if control["config"][k] != run["config"][k]}
    dataset, tag = run["config"].get("dataset"), control["config"].get("tag")
    if dataset is not None and tag is not None and tag.split("_")[0] != dataset:
        diff["dataset"] = (tag, dataset)
    if diff:
        raise ValueError("%s: the control differs from the run in %r" % (arch, diff))
    Pc = np.load(ctrl_arch, allow_pickle=True)["present"].item()
    Pr = np.load(arch, allow_pickle=True)["present"].item()
    rounds = range(1, run["config"]["rounds"] + 1)
    if sorted(Pc) != sorted(Pr) or any(not np.array_equal(np.asarray(Pc[c][r], bool), np.asarray(Pr[c][r], bool))
                                       for c in Pr for r in rounds):
        raise ValueError("%s: declared classes differ from the control %s (not the same partition)" % (arch, ctrl_arch))
    return keys


def replay(arch: str, run: dict, check: bool = True) -> dict:
    z = np.load(arch, allow_pickle=True)
    C, P, cbar = z["C"].item(), z["present"].item(), z["cbar"].item()
    cfg = run["config"]
    victim, start, rounds, tau = cfg["victim"], cfg["attack_start"], cfg["rounds"], run["tau"]
    aud = InnovationAuditor(mad_floor=MAD_FLOOR)
    vic, hon = [], []
    gser = {cid: {} for cid in C}
    Cbar = None
    for r in range(1, rounds + 1):
        payloads = {cid: (C[cid][r], P[cid][r]) for cid in C}
        if Cbar is not None:
            aud.set_aggregate(r - 1, Cbar)
        rs = aud.observe_round(r, payloads)
        if Cbar is not None:
            for cid in C:
                g = rs.per_client[cid].get("g", float("nan"))
                if np.isfinite(g):
                    gser[cid][r] = float(g)
        if r >= start and Cbar is not None:
            delta = {cid: np.asarray(C[cid][r], np.float64) - np.asarray(C[cid][r - 1], np.float64) for cid in C}
            step = {cid: float(np.linalg.norm(delta[cid])) for cid in C}
            s_r = float(np.median(list(step.values())))
            for cid in C:
                both = np.asarray(P[cid][r], bool) & np.asarray(P[cid][r - 1], bool)
                sq = float((delta[cid] ** 2).sum())
                absent_share = float((delta[cid][~both] ** 2).sum()) / sq if sq > 0 else None
                n_absent = int((~np.asarray(P[cid][r], bool)).sum())
                zc = rs.per_client[cid]
                o = oriented(zc)
                parts = [v for v in o.values() if np.isfinite(v)]
                sus = float(zc.get("suspicion", float("nan")))
                rec = {"round": r, "suspicion": sus, "oriented": o,
                       "structural": bool(np.isfinite(sus) and sus >= SCORE_STRUCTURAL),
                       "undefined": bool(zc.get("undefined", False)),
                       "replay_dist": (-float(zc["rep"]) if np.isfinite(zc.get("rep", float("nan"))) else None),
                       "step": step[cid] / s_r if s_r > 0 else None,
                       "u1": float(zc["u1"]) if np.isfinite(zc.get("u1", float("nan"))) else None,
                       "absent_share": absent_share, "n_absent": n_absent}
                if np.isfinite(sus) and not rec["structural"] and not zc.get("suspicion_clipped") and not rec["undefined"]:
                    if abs(max(parts) - sus) > TOL * max(1.0, abs(sus)):
                        raise AssertionError("round %d client %d: score %r is not the largest part %r" % (r, cid, sus, max(parts)))
                (vic if cid == victim else hon).append(rec)
        Cbar = cbar[r]

    att = [v for v in vic if np.isfinite(v["suspicion"])]
    flagged = [v for v in att if v["suspicion"] > tau]
    neg = [h["suspicion"] for h in hon]
    honest_flags = sum(1 for s in neg if np.isfinite(s) and s > tau)
    n_neg = sum(1 for s in neg if np.isfinite(s))
    replayed = {"auc": roc_auc([v["suspicion"] for v in vic], neg), "tpr": len(flagged) / max(len(att), 1),
                "fpr": honest_flags / max(n_neg, 1)}
    if check:
        recorded = {"auc": run["auc"]["suspicion"], "tpr": run["tpr_client_round"], "fpr": run["honest_fpr_client_round"]}
        for k in replayed:
            if abs(replayed[k] - recorded[k]) > TOL:
                raise AssertionError("%s: replayed %s %r differs from the run's %r" % (arch, k, replayed[k], recorded[k]))

    def top(v):
        if v["structural"]:
            return "structural"
        fin = {k: x for k, x in v["oriented"].items() if np.isfinite(x)}
        return max(fin, key=fin.get) if fin else "none"

    argmax_flagged = {k: 0 for k in NAMES}
    for v in flagged:
        t = top(v)
        argmax_flagged[t] = argmax_flagged.get(t, 0) + 1

    cids = sorted(C)
    base = {c: finite_median([g for r, g in gser[c].items() if MAG_FIRST <= r < start]) for c in cids}
    late = {c: finite_median([g for r, g in gser[c].items() if r >= start]) for c in cids}
    change = {c: (late[c] - base[c]) if base[c] is not None and late[c] is not None else None for c in cids}
    others = [c for c in cids if c != victim]
    honest_change = [change[c] for c in others if change[c] is not None]

    def rank(vals):
        if vals[victim] is None:
            return None
        return 1 + sum(1 for c in others if vals[c] is not None and vals[c] < vals[victim])

    magnitude = {"baseline_rounds": [MAG_FIRST, start - 1], "attack_rounds": [start, rounds],
                 "n_clients": len(cids),
                 "victim_g_baseline": base[victim], "victim_g_attack": late[victim],
                 "victim_change": change[victim],
                 "victim_ratio_baseline": float(np.exp(base[victim])) if base[victim] is not None else None,
                 "victim_ratio_attack": float(np.exp(late[victim])) if late[victim] is not None else None,
                 "victim_change_rank": rank(change), "victim_attack_rank": rank(late),
                 "honest_change": honest_change,
                 "honest_change_min": min(honest_change) if honest_change else None,
                 "honest_change_max": max(honest_change) if honest_change else None,
                 "honest_attack_g": [late[c] for c in others]}
    out = {"victim": victim, "attack_start": start, "rounds": rounds, "tau": tau,
           "n_attacked": len(att), "n_flagged": len(flagged), "flagged_rounds": [v["round"] for v in flagged],
           "unflagged_rounds": [v["round"] for v in att if v["suspicion"] <= tau],
           "honest_flags": honest_flags, "n_honest": n_neg, "replayed": replayed,
           "argmax_flagged": argmax_flagged,
           "argmax_attacked": {k: sum(1 for v in att if top(v) == k) for k in NAMES},
           "exceeds_tau_alone": {k: sum(1 for v in att if v["oriented"][k] > tau) for k in NAMES},
           "flagged_without": {k: sum(1 for v in att if max([x for j, x in v["oriented"].items()
                                                              if j != k and np.isfinite(x)] or [-np.inf]) > tau)
                               for k in NAMES},
           "flagged_by_only": {k: sum(1 for v in flagged if v["oriented"][k] > tau
                                      and all(not (x > tau) for j, x in v["oriented"].items() if j != k))
                               for k in NAMES},
           "median_oriented_z": {k: finite_median([v["oriented"][k] for v in att]) for k in NAMES},
           "honest_median_oriented_z": {k: finite_median([h["oriented"][k] for h in hon]) for k in NAMES},
           "auc_single": {k: roc_auc([v["oriented"][k] for v in vic], [h["oriented"][k] for h in hon])
                          for k in NAMES},
           "raw": {"replay_dist_over_cohort_step": {"victim": finite_median([v["replay_dist"] for v in att]),
                                                    "honest": finite_median([h["replay_dist"] for h in hon])},
                   "step_over_cohort_step": {"victim": finite_median([v["step"] for v in att]),
                                             "honest": finite_median([h["step"] for h in hon])},
                   "u1": {"victim": finite_median([v["u1"] for v in att]),
                          "honest": finite_median([h["u1"] for h in hon])},
                   "absent_row_share_of_step": {"victim": finite_median([v["absent_share"] for v in att]),
                                                "honest": finite_median([h["absent_share"] for h in hon])},
                   "absent_rows": {"victim": finite_median([v["n_absent"] for v in att]),
                                   "honest": finite_median([h["n_absent"] for h in hon])}},
           "magnitude": magnitude,
           "per_round": [{"round": v["round"], "suspicion": v["suspicion"], "top": top(v),
                          "oriented": v["oriented"], "replay_dist": v["replay_dist"], "step": v["step"], "u1": v["u1"],
                          "absent_share": v["absent_share"], "n_absent": v["n_absent"]}
                         for v in vic]}
    return out


def ms(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    return [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, int(len(v))]


def summarize(runs: dict) -> dict:
    seeds = sorted(runs)
    R = [runs[s] for s in seeds]
    total = lambda f: int(sum(f(r) for r in R))
    return {"seeds": seeds, "n": len(seeds),
            "n_attacked": total(lambda r: r["n_attacked"]), "n_flagged": total(lambda r: r["n_flagged"]),
            "n_honest": total(lambda r: r["n_honest"]), "honest_flags": total(lambda r: r["honest_flags"]),
            "argmax_flagged": {k: total(lambda r, k=k: r["argmax_flagged"].get(k, 0)) for k in NAMES},
            "exceeds_tau_alone": {k: total(lambda r, k=k: r["exceeds_tau_alone"][k]) for k in NAMES},
            "flagged_without": {k: total(lambda r, k=k: r["flagged_without"][k]) for k in NAMES},
            "flagged_by_only": {k: total(lambda r, k=k: r["flagged_by_only"][k]) for k in NAMES},
            "per_seed_flagged": [r["n_flagged"] for r in R],
            "per_seed_flagged_without_rep": [r["flagged_without"]["rep"] for r in R],
            "per_seed_argmax_rep": [r["argmax_flagged"].get("rep", 0) for r in R],
            "median_oriented_z": {k: ms([r["median_oriented_z"][k] for r in R]) for k in NAMES},
            "per_seed_median_oriented_z": {k: [r["median_oriented_z"][k] for r in R] for k in NAMES},
            "honest_median_oriented_z": {k: ms([r["honest_median_oriented_z"][k] for r in R]) for k in NAMES},
            "auc_single": {k: ms([r["auc_single"][k] for r in R]) for k in NAMES},
            "raw": {q: {w: ms([r["raw"][q][w] for r in R]) for w in ("victim", "honest")} for q in R[0]["raw"]},
            "per_seed_raw": {q: {w: [r["raw"][q][w] for r in R] for w in ("victim", "honest")} for q in R[0]["raw"]},
            "magnitude": magnitude_summary([r["magnitude"] for r in R])}


def magnitude_summary(M: list) -> dict:
    vc = [m["victim_change"] for m in M]
    hmin = [m["honest_change_min"] for m in M if m["honest_change_min"] is not None]
    hmax = [m["honest_change_max"] for m in M if m["honest_change_max"] is not None]
    return {"victim_change": ms(vc), "per_seed_victim_change": vc,
            "n_victim_falls": sum(1 for v in vc if v is not None and v < 0),
            "per_seed_victim_change_rank": [m["victim_change_rank"] for m in M],
            "n_victim_largest_fall": sum(1 for m in M if m["victim_change_rank"] == 1),
            "per_seed_victim_attack_rank": [m["victim_attack_rank"] for m in M],
            "n_victim_smallest_attack": sum(1 for m in M if m["victim_attack_rank"] == 1),
            "honest_change_range": [min(hmin), max(hmax)] if hmin and hmax else None,
            "per_seed_victim_ratio": {"baseline": [m["victim_ratio_baseline"] for m in M],
                                      "attack": [m["victim_ratio_attack"] for m in M]}}


def control(a, runs: dict, sources: dict) -> dict:
    ctrl, keys = {}, None
    for s in a.seeds:
        jp = os.path.join(a.results, CONTROL_RESULT % s)
        with open(jp, encoding="utf-8") as fh:
            cres = json.load(fh)
        arch = control_archive(a.results, cres["config"])
        same = {}
        for tag in a.attackers:
            jr = os.path.join(a.results, "reduced_effort_%s_s%d.json" % (tag, s))
            with open(jr, encoding="utf-8") as fh:
                run = json.load(fh)
            keys = check_control(cres, arch, run, archive_path(a.results, tag, s))
            same[tag] = (run["config"]["victim"], run["config"]["attack_start"], run["config"]["rounds"], run["tau"])
        if len(set(same.values())) != 1:
            raise ValueError("seed %d: the runs differ in victim, attack start, rounds or threshold: %r" % (s, same))
        victim, start, rounds, tau = next(iter(same.values()))
        ctrl[s] = replay(arch, {"config": {"victim": victim, "attack_start": start, "rounds": rounds}, "tau": tau},
                         check=False)
        sources["control_s%d" % s] = {"result": os.path.relpath(jp, a.results).replace(os.sep, "/"),
                                      "archive": os.path.relpath(arch, a.results).replace(os.sep, "/")}
    comparison = {}
    for tag in a.attackers:
        vc = [runs[tag][s]["magnitude"]["victim_change"] for s in a.seeds]
        cc = [ctrl[s]["magnitude"]["victim_change"] for s in a.seeds]
        comparison[tag] = {"per_seed_victim_change": vc, "per_seed_control_change": cc,
                           "per_seed_difference": [v - c for v, c in zip(vc, cc)],
                           "n_victim_below_control": sum(1 for v, c in zip(vc, cc) if v < c),
                           "per_seed_auc_g": [runs[tag][s]["auc_single"]["g"] for s in a.seeds],
                           "per_seed_control_auc_g": [ctrl[s]["auc_single"]["g"] for s in a.seeds]}
    return {"compared_config_keys": keys, "declared_classes_equal_to_every_run": True,
            "summary": summarize(ctrl), "runs": {"s%d" % s: ctrl[s] for s in a.seeds}, "comparison": comparison}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--attackers", nargs="+", default=list(ATTACKERS),
                    help="run tags: f<fraction> (step-capped), fwd (forward passes only), zero")
    ap.add_argument("--results", default=os.path.join(HERE, "results"),
                    help="directory of the per-seed result files and of the archive folders")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "reduced_effort_drivers.json"))
    ap.add_argument("--no-control", action="store_true",
                    help="omit the attack-free control (client 0 of the released honest federations)")
    a = ap.parse_args()
    t0 = time.time()
    runs, sources = {}, {}
    for tag in a.attackers:
        for s in a.seeds:
            jp, arch = os.path.join(a.results, "reduced_effort_%s_s%d.json" % (tag, s)), archive_path(a.results, tag, s)
            with open(jp, encoding="utf-8") as fh:
                run = json.load(fh)
            if run["config"]["seed"] != s:
                raise ValueError("%s: seed %r" % (jp, run["config"]["seed"]))
            runs.setdefault(tag, {})[s] = replay(arch, run)
            sources["%s_s%d" % (tag, s)] = {"result": os.path.relpath(jp, a.results).replace(os.sep, "/"),
                                            "archive": os.path.relpath(arch, a.results).replace(os.sep, "/")}
            print("%-5s seed %d: flagged %2d/%d, argmax %s, without rep %d, median z_rep %.2f" % (
                tag, s, runs[tag][s]["n_flagged"], runs[tag][s]["n_attacked"],
                {k: v for k, v in runs[tag][s]["argmax_flagged"].items() if v},
                runs[tag][s]["flagged_without"]["rep"], runs[tag][s]["median_oriented_z"]["rep"]), flush=True)
    out = {"schema": SCHEMA,
           "config": {"seeds": a.seeds, "attackers": a.attackers, "mad_floor": MAD_FLOOR,
                      "magnitude_baseline_first_round": MAG_FIRST,
                      "statistics": {k: ("two-sided |z|" if o is None else "one-sided low (-z)" if o < 0
                                         else "one-sided high (+z)") for k, o in STATS}},
           "meta": {"environment": environment(), "sources": sources},
           "summary": {tag: summarize(runs[tag]) for tag in a.attackers},
           "runs": {"%s_s%d" % (tag, s): runs[tag][s] for tag in a.attackers for s in sorted(runs[tag])}}
    if not a.no_control:
        out["control"] = control(a, runs, sources)
    out["meta"]["seconds"] = time.time() - t0
    print("\n%-5s %8s %22s %12s %10s %42s" % ("tag", "flagged", "argmax among flagged", "without rep",
                                              "z_rep", "AUC alone u1 / u2 / g / x / rep"))
    for tag in a.attackers:
        v = out["summary"][tag]
        print("%-5s %4d/%-3d %22s %12s %10s %42s" % (
            tag, v["n_flagged"], v["n_attacked"], ",".join("%s %d" % (k, n) for k, n in v["argmax_flagged"].items() if n),
            "/".join(map(str, v["per_seed_flagged_without_rep"])),
            "%.2f" % v["median_oriented_z"]["rep"][0],
            " / ".join("%.3f" % v["auc_single"][k][0] for k in NAMES)))
    print("\nmagnitude: change of median g from rounds %d..attack_start-1 to the attacked rounds" % MAG_FIRST)
    for tag in a.attackers:
        m = out["summary"][tag]["magnitude"]
        print("%-5s victim %s; falls in %d/%d, largest fall in %d (ranks %s), smallest when attacked in %d (ranks %s); "
              "honest range %s" % (tag, " ".join("%+.3f" % x for x in m["per_seed_victim_change"]),
                                   m["n_victim_falls"], len(m["per_seed_victim_change"]), m["n_victim_largest_fall"],
                                   m["per_seed_victim_change_rank"], m["n_victim_smallest_attack"],
                                   m["per_seed_victim_attack_rank"],
                                   "%+.3f..%+.3f" % tuple(m["honest_change_range"])))
    if "control" in out:
        c = out["control"]["summary"]
        print("control (client 0 honest): flagged %d/%d, change %s, ranks %s, AUC of g %s; victim below control in %s" % (
            c["n_flagged"], c["n_attacked"], " ".join("%+.3f" % x for x in c["magnitude"]["per_seed_victim_change"]),
            c["magnitude"]["per_seed_victim_change_rank"],
            " ".join("%.3f" % out["control"]["runs"]["s%d" % s]["auc_single"]["g"] for s in c["seeds"]),
            {t: v["n_victim_below_control"] for t, v in out["control"]["comparison"].items()}))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
