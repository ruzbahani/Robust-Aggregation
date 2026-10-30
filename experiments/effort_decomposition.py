from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import Subset

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from passportfl.data import dirichlet_partition, load_dataset
from passportfl.effort_controls import (RECALIBRATION_BATCH, batchnorm_summary,
                                        cached_dataset, forward_only, hybrid_state,
                                        recalibrated_state)
from passportfl.fedproto import Client, FedConfig, Server
from passportfl.passport import build_model
from passportfl.reproducibility import experiment_entropy, environment

FRACTIONS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
REPORT_FROM = 12
BN_SCHEMA = "fixed-readout-bn-split-v1"
SPLIT_PARTS = ("move", "par", "bn", "int", "par_rev", "bn_rev", "shap_par", "shap_bn",
               "recal", "noise", "fwd")


def frobenius_norm(a):
    return float(np.linalg.norm(np.asarray(a, np.float64)))


def frobenius_cosine(a, b):
    na, nb = frobenius_norm(a), frobenius_norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return float("nan")
    return float(np.sum(np.asarray(a, np.float64) * np.asarray(b, np.float64)) / (na * nb))


def snapshot_parameters(c: Client) -> dict:
    return {"model": copy.deepcopy(c.model.state_dict()),
            "opt": copy.deepcopy(c.opt.state_dict()),
            "rng": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "np": np.random.get_state()}


def restore_parameters(c: Client, s: dict) -> None:
    c.model.load_state_dict(s["model"])
    c.opt.load_state_dict(s["opt"])
    torch.set_rng_state(s["rng"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["np"])


def full_steps(c: Client) -> int:
    return len(c.loader) * c.cfg.local_epochs


def probe_victim(c: Client, Cbar_prev: np.ndarray, caps: list, measurement_ds) -> tuple:
    snap = snapshot_parameters(c)
    C_pre, cnt_pre, _, _ = c.compute_prototypes(dataset=measurement_ds)
    pr = cnt_pre > 0
    posts = {}
    for cap in caps:
        restore_parameters(c, snap)
        c.local_train(max_steps=cap)
        Cc, _, _, _ = c.compute_prototypes(dataset=measurement_ds)
        posts[cap] = Cc
        if cap == 0 and not all(torch.equal(v, c.model.state_dict()[k])
                                for k, v in snap["model"].items()):
            raise AssertionError("zero-step control changed model state")
    restore_parameters(c, snap)
    contrib = c.round_step()
    full = full_steps(c)
    rng_after = torch.get_rng_state()
    posts[full], _, _, _ = c.compute_prototypes(dataset=measurement_ds)
    torch.set_rng_state(rng_after)

    off = (C_pre - Cbar_prev)[pr]
    mv_full = (posts[full] - C_pre)[pr]
    rec = {"present_rows": int(pr.sum()), "offset": frobenius_norm(off), "full_steps": full,
           "innov_full": frobenius_norm((posts[full] - Cbar_prev)[pr]),
           "published_vs_fixed_readout": frobenius_norm((np.asarray(contrib.C) - posts[full])[pr]),
           "by_steps": {}}
    for cap in sorted(posts):
        mv = (posts[cap] - C_pre)[pr]
        inn = (posts[cap] - Cbar_prev)[pr]
        rec["by_steps"][str(cap)] = {
            "move": frobenius_norm(mv), "innov": frobenius_norm(inn),
            "cos_move_offset": None if cap == 0 else frobenius_cosine(mv, off),
            "cos_move_fullmove": None if cap == 0 else frobenius_cosine(mv, mv_full),
        }
    return contrib, rec


def restore_exact(c: Client, s: dict) -> None:
    c.model.load_state_dict(s["model"])
    c.opt.load_state_dict(copy.deepcopy(s["opt"]))
    torch.set_rng_state(s["rng"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["np"])


def random_state() -> dict:
    return {"rng": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "np": np.random.get_state()}


def set_random_state(s: dict) -> None:
    torch.set_rng_state(s["rng"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["np"])


def same_tensors(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def same_optimizer_state(a: dict, b: dict) -> bool:
    if a["param_groups"] != b["param_groups"] or a["state"].keys() != b["state"].keys():
        return False
    return all(same_tensors({k: v for k, v in a["state"][i].items() if torch.is_tensor(v)},
                            {k: v for k, v in b["state"][i].items() if torch.is_tensor(v)})
               for i in a["state"])


def last_batch_size(c: Client, steps: int):
    if steps == 0:
        return None
    nb, bs = len(c.loader), c.cfg.batch_size
    return bs if (steps - 1) % nb < nb - 1 else len(c.train_ds) - bs * (nb - 1)


def split_readouts(c: Client, snap: dict, steps: int, measurement_ds) -> dict:
    state_k = copy.deepcopy(c.model.state_dict())
    theta0 = snap["model"]
    out = {"total": c.compute_prototypes(dataset=measurement_ds)[0]}
    c.model.load_state_dict(hybrid_state(state_k, theta0))
    out["par"] = c.compute_prototypes(dataset=measurement_ds)[0]
    c.model.load_state_dict(hybrid_state(theta0, state_k))
    out["bn"] = c.compute_prototypes(dataset=measurement_ds)[0]
    recalibrated_state(c, state_k, measurement_ds)
    out["recal"] = c.compute_prototypes(dataset=measurement_ds)[0]
    restore_exact(c, snap)
    forward_only(c, max_passes=steps)
    out["fwd"] = c.compute_prototypes(dataset=measurement_ds)[0]
    return out


def movement_parts(reads: dict, C_pre: np.ndarray, recal_pre: np.ndarray, pr: np.ndarray) -> dict:
    f = lambda m, ref: (np.asarray(m, np.float64) - np.asarray(ref, np.float64))[pr]
    D, par, bn = f(reads["total"], C_pre), f(reads["par"], C_pre), f(reads["bn"], C_pre)
    recal = f(reads["recal"], recal_pre)
    return {"move": D, "par": par, "bn": bn, "int": D - par - bn,
            "par_rev": D - bn, "bn_rev": D - par,
            "shap_par": 0.5 * (par + D - bn), "shap_bn": 0.5 * (bn + D - par),
            "recal": recal, "noise": D - recal, "fwd": f(reads["fwd"], C_pre)}


def finite_or_none(x):
    return None if x is None or not np.isfinite(x) else float(x)


def probe_victim_bn(c: Client, Cbar_prev: np.ndarray, caps: list, measurement_ds) -> tuple:
    snap = snapshot_parameters(c)
    C_pre, cnt_pre, _, _ = c.compute_prototypes(dataset=measurement_ds)
    pr = cnt_pre > 0
    recalibrated_state(c, snap["model"], measurement_ds)
    recal_pre = c.compute_prototypes(dataset=measurement_ds)[0]
    reads = {}
    for cap in caps:
        restore_exact(c, snap)
        c.local_train(max_steps=cap)
        if cap == 0 and not same_tensors(snap["model"], c.model.state_dict()):
            raise AssertionError("zero-step control changed model state")
        reads[cap] = split_readouts(c, snap, cap, measurement_ds)
        if cap == 0 and not same_tensors(snap["model"], c.model.state_dict()):
            raise AssertionError("zero forward passes changed model state")
    restore_exact(c, snap)
    contrib = c.round_step()
    full = full_steps(c)
    rng_after, skipped = random_state(), c.skipped_batches
    post = {"model": copy.deepcopy(c.model.state_dict()), "opt": copy.deepcopy(c.opt.state_dict())}
    reads[full] = split_readouts(c, snap, full, measurement_ds)
    c.model.load_state_dict(post["model"])
    c.opt.load_state_dict(copy.deepcopy(post["opt"]))
    set_random_state(rng_after)
    c.skipped_batches = skipped
    if not (same_tensors(post["model"], c.model.state_dict())
            and same_optimizer_state(post["opt"], c.opt.state_dict())):
        raise AssertionError("post-round state not reinstated")

    B = (np.asarray(C_pre, np.float64) - Cbar_prev)[pr]
    B_recal = (np.asarray(recal_pre, np.float64) - Cbar_prev)[pr]
    parts = {cap: movement_parts(r, C_pre, recal_pre, pr) for cap, r in reads.items()}
    fp = parts[full]
    rec = {"present_rows": int(pr.sum()), "offset": frobenius_norm(B),
           "offset_recal": frobenius_norm(B_recal), "full_steps": full,
           "innov_full": frobenius_norm(B + fp["move"]),
           "published_vs_fixed_readout": frobenius_norm((np.asarray(contrib.C) - reads[full]["total"])[pr]),
           "by_steps": {}}
    for cap in sorted(parts):
        p, zero = parts[cap], cap == 0
        D2 = float(np.sum(p["move"] ** 2))
        rec["by_steps"][str(cap)] = {
            "move": frobenius_norm(p["move"]), "innov": frobenius_norm(B + p["move"]),
            "cos_move_offset": None if zero else finite_or_none(frobenius_cosine(p["move"], B)),
            "cos_move_fullmove": None if zero else finite_or_none(frobenius_cosine(p["move"], fp["move"])),
            "last_batch_size": last_batch_size(c, cap),
            "norm": {k: frobenius_norm(v) for k, v in p.items()},
            "cos_full": {k: None if zero else finite_or_none(frobenius_cosine(v, fp[k])) for k, v in p.items()},
            "cos_move": {k: None if zero else finite_or_none(frobenius_cosine(v, p["move"]))
                         for k, v in p.items() if k != "move"},
            "cos_bn_fwd": None if zero else finite_or_none(frobenius_cosine(p["bn"], p["fwd"])),
            "share": {k: None if zero or D2 <= 1e-24 else float(np.sum(p["shap_" + k] * p["move"]) / D2)
                      for k in ("par", "bn")},
            "innov_nobn": frobenius_norm(B + p["par"]),
            "innov_shap_par": frobenius_norm(B + p["shap_par"]),
            "innov_fwd": frobenius_norm(B + p["fwd"]),
            "innov_recal": frobenius_norm(B_recal + p["recal"]),
        }
    return contrib, rec


def run_seed(a, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, test, nc = load_dataset(a.dataset)
    measurement_train, _, _ = load_dataset(a.dataset, augment=False)
    cfg = FedConfig(num_classes=nc, embed_dim=a.embed_dim, width=a.width, rounds=a.rounds,
                    local_epochs=a.local_epochs, warmup_rounds=10 ** 6,
                    audit_z_threshold=10 ** 6, scheme="Ed25519", device=a.device)
    parts = dirichlet_partition(train, a.clients, a.dirichlet, nc, seed)
    clients = [Client(i, build_model(nc, a.embed_dim, a.width, True, 0.0), p, cfg,
                      entropy=experiment_entropy(seed, i, "effort-decomposition-v2"))
               for i, p in enumerate(parts)]
    server = Server(cfg)
    victim = clients[a.victim]
    full = full_steps(victim)
    measurement_ds = Subset(measurement_train, parts[a.victim].indices)
    if a.bn_split:
        measurement_ds = cached_dataset(measurement_ds)
    probe = probe_victim_bn if a.bn_split else probe_victim
    caps = sorted({0 if f == 0 else max(1, int(round(f * full))) for f in a.fractions})
    caps = [c for c in caps if c < full]
    print("seed %d: victim %d has %d full steps per round; probing at %s"
          % (seed, a.victim, full, caps), flush=True)

    rounds_out = {}
    Cbar = None
    t0 = time.time()
    for r in range(1, a.rounds + 1):
        contribs, counts = [], []
        rec = None
        for c in clients:
            if c.cid == a.victim and Cbar is not None and r >= a.report_from:
                contrib, rec = probe(c, Cbar, caps, measurement_ds)
            else:
                contrib = c.round_step()
            contribs.append(contrib)
            counts.append(c._last_counts)
        if Cbar is not None and rec is not None:
            others = []
            for c in clients:
                if c.cid == a.victim:
                    continue
                pr = np.asarray(counts[c.cid]) > 0
                others.append(frobenius_norm((np.asarray(contribs[c.cid].C) - Cbar)[pr]))
            rec["cohort_median_innov"] = float(np.median(others))
            rounds_out[str(r)] = rec
            f10 = str(max(1, int(round(0.1 * full))))
            line = {"round": r, "elapsed_s": round(time.time() - t0, 1),
                    "offset/innov_full": round(rec["offset"] / max(rec["innov_full"], 1e-12), 3),
                    "move(0.1)/move(full)": round(rec["by_steps"][f10]["move"] /
                                                  max(rec["by_steps"][str(full)]["move"], 1e-12), 3),
                    "innov(0.1)/cohort_med": round(rec["by_steps"][f10]["innov"] /
                                                   max(rec["cohort_median_innov"], 1e-12), 3)}
            if a.bn_split:
                mf = max(rec["by_steps"][str(full)]["move"], 1e-12)
                for key in ("par", "bn", "recal", "fwd"):
                    line["%s(0.1)/move(full)" % key] = round(rec["by_steps"][f10]["norm"][key] / mf, 3)
                line["fwd(full)/move(full)"] = round(rec["by_steps"][str(full)]["norm"]["fwd"] / mf, 3)
            print(json.dumps(line), flush=True)
        Cbar = server.aggregate(contribs, counts)
        for c in clients:
            c.receive_global(Cbar)
    if a.bn_split:
        return {"config": vars(a), "seed": seed, "victim_full_steps": full, "caps": caps,
                "schema": BN_SCHEMA, "environment": environment(), "elapsed_s": time.time() - t0,
                "batchnorm": batchnorm_summary(victim.model),
                "parts": list(SPLIT_PARTS),
                "identity_entropy": {str(i): experiment_entropy(seed, i, "effort-decomposition-v2").hex()
                                     for i in range(a.clients)},
                "rounds": rounds_out}
    config = {k: v for k, v in vars(a).items() if k != "bn_split"}
    return {"config": config, "seed": seed, "victim_full_steps": full, "caps": caps,
            "schema": "fixed-readout-v2", "environment": environment(),
            "identity_entropy": {str(i): experiment_entropy(seed, i, "effort-decomposition-v2").hex()
                                 for i in range(a.clients)},
            "rounds": rounds_out}


def summarize(files: list, out_path: str) -> dict:
    per_seed = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        full = d["victim_full_steps"]
        if d.get("schema") != "fixed-readout-v2":
            raise ValueError(f"{f}: stale augmented-readout result; rerun the experiment")
        rows = [v for r, v in d["rounds"].items() if int(r) >= d["config"]["report_from"]]
        if not rows:
            raise ValueError(f"{f}: no measured rounds")
        s = {"seed": d["seed"], "full_steps": full, "n_rounds": len(rows), "by_steps": {}}
        s["offset_over_innov_full"] = float(np.median([v["offset"] / v["innov_full"] for v in rows]))
        s["innov_full_over_cohort_median"] = float(np.median([v["innov_full"] / v["cohort_median_innov"] for v in rows]))
        for cap in d["caps"] + [full]:
            k = str(cap)
            s["by_steps"][k] = {
                "fraction_of_steps": cap / full,
                "move_over_move_full": float(np.median([v["by_steps"][k]["move"] / v["by_steps"][str(full)]["move"] for v in rows])),
                "innov_over_innov_full": float(np.median([v["by_steps"][k]["innov"] / v["innov_full"] for v in rows])),
                "innov_over_cohort_median": float(np.median([v["by_steps"][k]["innov"] / v["cohort_median_innov"] for v in rows])),
                "move_over_offset": float(np.median([v["by_steps"][k]["move"] / v["offset"] for v in rows])),
                "cos_move_offset": (None if cap == 0 else float(np.nanmedian([v["by_steps"][k]["cos_move_offset"] for v in rows]))),
                "cos_move_fullmove": (None if cap == 0 else float(np.nanmedian([v["by_steps"][k]["cos_move_fullmove"] for v in rows]))),
            }
        per_seed.append(s)
    labels = {}
    for s in per_seed:
        for k, v in s["by_steps"].items():
            lab = "full" if int(k) == s["full_steps"] else "%g" % round(v["fraction_of_steps"], 2)
            labels.setdefault(lab, []).append(v)
    summary = {"schema": "fixed-readout-v2", "report_from_round": d["config"]["report_from"],
               "source_files": [os.path.basename(f) for f in files],
               "seeds": [s["seed"] for s in per_seed],
               "offset_over_innov_full": mean_and_std([s["offset_over_innov_full"] for s in per_seed]),
               "innov_full_over_cohort_median": mean_and_std([s["innov_full_over_cohort_median"] for s in per_seed]),
               "by_fraction": {}}
    for lab, vs in labels.items():
        summary["by_fraction"][lab] = {key: (None if vs[0][key] is None else mean_and_std([v[key] for v in vs]))
                                       for key in vs[0] if key != "fraction_of_steps"}
        summary["by_fraction"][lab]["fraction_of_steps"] = float(np.mean([v["fraction_of_steps"] for v in vs]))
    summary["per_seed"] = per_seed
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("\nmedian over rounds >= %d, mean (sd) over %d seeds" % (summary["report_from_round"], len(per_seed)))
    print("offset / innov(full)            %.3f (%.3f)" % tuple(summary["offset_over_innov_full"]))
    print("innov(full) / cohort median     %.3f (%.3f)" % tuple(summary["innov_full_over_cohort_median"]))
    print("%-8s %7s %12s %12s %12s %12s %10s %10s" % ("steps", "frac", "move/full", "innov/full",
                                                      "innov/cohort", "move/offset", "cos(m,off)", "cos(m,mf)"))
    for lab in sorted(summary["by_fraction"], key=lambda x: summary["by_fraction"][x]["fraction_of_steps"]):
        v = summary["by_fraction"][lab]
        print("%-8s %7.3f %12s %12s %12s %12s %10s %10s" % (
            lab, v["fraction_of_steps"], format_measurement(v["move_over_move_full"]), format_measurement(v["innov_over_innov_full"]),
            format_measurement(v["innov_over_cohort_median"]), format_measurement(v["move_over_offset"]),
            format_measurement(v["cos_move_offset"]), format_measurement(v["cos_move_fullmove"])))
    print("wrote %s" % out_path)
    return summary


def summarize_bn(files: list, out_path: str) -> dict:
    per_seed = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("schema") != BN_SCHEMA:
            raise ValueError(f"{f}: not a parameter/statistics split result")
        full = d["victim_full_steps"]
        fk = str(full)
        rows = [v for r, v in d["rounds"].items() if int(r) >= d["config"]["report_from"]]
        if not rows:
            raise ValueError(f"{f}: no measured rounds")

        def med(fn):
            xs = [fn(v) for v in rows]
            xs = np.asarray([np.nan if x is None else x for x in xs], float)
            return None if np.all(np.isnan(xs)) else float(np.nanmedian(xs))

        s = {"seed": d["seed"], "full_steps": full, "n_rounds": len(rows), "elapsed_s": d.get("elapsed_s"),
             "offset_over_innov_full": med(lambda v: v["offset"] / v["innov_full"]),
             "offset_recal_over_innov_recal_full": med(lambda v: v["offset_recal"] / v["by_steps"][fk]["innov_recal"]),
             "innov_full_over_cohort_median": med(lambda v: v["innov_full"] / v["cohort_median_innov"]),
             "published_vs_fixed_readout_over_move_full": med(lambda v: v["published_vs_fixed_readout"] / v["by_steps"][fk]["move"]),
             "by_steps": {}}
        if 0 in d["caps"]:
            s["zero_step_max_part_over_move_full"] = float(max(max(v["by_steps"]["0"]["norm"].values())
                                                               / v["by_steps"][fk]["move"] for v in rows))
            s["zero_step_part_over_move_full"] = {
                p: float(max(v["by_steps"]["0"]["norm"][p] / v["by_steps"][fk]["move"] for v in rows))
                for p in SPLIT_PARTS}
        for cap in d["caps"] + [full]:
            k = str(cap)
            e = {"fraction_of_steps": cap / full, "last_batch_size": rows[0]["by_steps"][k]["last_batch_size"]}
            for p in SPLIT_PARTS:
                e[p + "_over_move_full"] = med(lambda v, p=p: v["by_steps"][k]["norm"][p] / v["by_steps"][fk]["move"])
            e["recal_over_recal_full"] = med(lambda v: v["by_steps"][k]["norm"]["recal"] / v["by_steps"][fk]["norm"]["recal"])
            e["fwd_over_fwd_full"] = med(lambda v: v["by_steps"][k]["norm"]["fwd"] / v["by_steps"][fk]["norm"]["fwd"])
            for p in SPLIT_PARTS:
                e["cos_%s_full" % p] = med(lambda v, p=p: v["by_steps"][k]["cos_full"][p])
            for p in SPLIT_PARTS[1:]:
                e["cos_%s_move" % p] = med(lambda v, p=p: v["by_steps"][k]["cos_move"][p])
            e["cos_move_offset"] = med(lambda v: v["by_steps"][k]["cos_move_offset"])
            e["cos_bn_fwd"] = med(lambda v: v["by_steps"][k]["cos_bn_fwd"])
            e["share_par"] = med(lambda v: v["by_steps"][k]["share"]["par"])
            e["share_bn"] = med(lambda v: v["by_steps"][k]["share"]["bn"])
            e["move_over_offset"] = med(lambda v: v["by_steps"][k]["move"] / v["offset"])
            e["innov_over_innov_full"] = med(lambda v: v["by_steps"][k]["innov"] / v["innov_full"])
            e["innov_over_cohort_median"] = med(lambda v: v["by_steps"][k]["innov"] / v["cohort_median_innov"])
            e["innov_nobn_over_innov_full"] = med(lambda v: v["by_steps"][k]["innov_nobn"] / v["innov_full"])
            e["innov_nobn_over_innov_nobn_full"] = med(lambda v: v["by_steps"][k]["innov_nobn"] / v["by_steps"][fk]["innov_nobn"])
            e["innov_shap_par_over_innov_full"] = med(lambda v: v["by_steps"][k]["innov_shap_par"] / v["innov_full"])
            e["innov_fwd_over_innov_full"] = med(lambda v: v["by_steps"][k]["innov_fwd"] / v["innov_full"])
            e["innov_recal_over_innov_recal_full"] = med(lambda v: v["by_steps"][k]["innov_recal"] / v["by_steps"][fk]["innov_recal"])
            s["by_steps"][k] = e
        per_seed.append(s)
    labels = {}
    for s in per_seed:
        for k, v in s["by_steps"].items():
            lab = "full" if int(k) == s["full_steps"] else "%g" % round(v["fraction_of_steps"], 2)
            labels.setdefault(lab, []).append(v)
    top = ("offset_over_innov_full", "offset_recal_over_innov_recal_full", "innov_full_over_cohort_median",
           "published_vs_fixed_readout_over_move_full")
    summary = {"schema": BN_SCHEMA, "report_from_round": d["config"]["report_from"],
               "batchnorm": d["batchnorm"], "parts": d["parts"],
               "source_files": [os.path.basename(f) for f in files],
               "seeds": [s["seed"] for s in per_seed]}
    for key in top:
        summary[key] = mean_and_std([s[key] for s in per_seed])
    if all("zero_step_max_part_over_move_full" in s for s in per_seed):
        summary["zero_step_max_part_over_move_full"] = max(s["zero_step_max_part_over_move_full"] for s in per_seed)
        summary["zero_step_part_over_move_full"] = {p: max(s["zero_step_part_over_move_full"][p] for s in per_seed)
                                                    for p in SPLIT_PARTS}
    summary["by_fraction"] = {}
    for lab, vs in labels.items():
        row = {}
        for key in vs[0]:
            if key in ("fraction_of_steps", "last_batch_size"):
                continue
            xs = [v[key] for v in vs]
            row[key] = None if any(x is None for x in xs) else mean_and_std(xs)
        row["fraction_of_steps"] = float(np.mean([v["fraction_of_steps"] for v in vs]))
        row["last_batch_size"] = [v["last_batch_size"] for v in vs]
        summary["by_fraction"][lab] = row
    summary["per_seed"] = per_seed
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("\nmedian over rounds >= %d, mean (sd) over %d seeds; norms relative to the full-round total movement"
          % (summary["report_from_round"], len(per_seed)))
    for key in top:
        print("%-44s %s" % (key, format_measurement(summary[key])))
    cols = (("move/full", "move_over_move_full"), ("par/full", "par_over_move_full"),
            ("bn/full", "bn_over_move_full"), ("int/full", "int_over_move_full"),
            ("share par", "share_par"), ("recal/full", "recal_over_move_full"),
            ("noise/full", "noise_over_move_full"), ("fwd/full", "fwd_over_move_full"),
            ("cos(m,mf)", "cos_move_full"), ("cos(par,f)", "cos_par_full"),
            ("cos(rec,f)", "cos_recal_full"), ("innov/full", "innov_over_innov_full"),
            ("noBN inn", "innov_nobn_over_innov_full"), ("fwd inn", "innov_fwd_over_innov_full"))
    print("%-6s %6s" % ("steps", "frac") + "".join(" %12s" % c[0] for c in cols))
    for lab in sorted(summary["by_fraction"], key=lambda x: summary["by_fraction"][x]["fraction_of_steps"]):
        v = summary["by_fraction"][lab]
        print("%-6s %6.3f" % (lab, v["fraction_of_steps"]) + "".join(" %12s" % format_measurement(v[c[1]]) for c in cols))
    print("wrote %s" % out_path)
    return summary


def mean_and_std(xs):
    xs = np.asarray(xs, float)
    return [float(np.mean(xs)), float(np.std(xs))]


def format_measurement(ms):
    if ms is None:
        return "undefined"
    return "%.3f±%.3f" % (ms[0], ms[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--victim", type=int, default=0)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=35)
    ap.add_argument("--report-from", type=int, default=REPORT_FROM)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--dirichlet", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--bn-split", action="store_true",
                    help="split movement into parameter and BatchNorm-statistics parts; "
                         "writes effort_decomposition_bn_s<seed>.json and effort_decomposition_bn.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out_dir, exist_ok=True)
    stem = "effort_decomposition_bn" if a.bn_split else "effort_decomposition"
    files = []
    for s in a.seeds:
        path = os.path.join(a.out_dir, "%s_s%d.json" % (stem, s))
        if not a.summarize_only:
            out = run_seed(a, s)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1)
            print("wrote %s" % path, flush=True)
        files.append(path)
    (summarize_bn if a.bn_split else summarize)(files, os.path.join(a.out_dir, stem + ".json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
