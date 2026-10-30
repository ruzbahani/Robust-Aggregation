from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from passportfl import identity as idm
from passportfl.data import load_cifar100, load_synthetic
from passportfl.passport import build_model, make_passport_tensors, passport_bytes
from passportfl.reproducibility import environment

from negative_results import (DEFAULT_ENTROPY, DIST_MATCHED, cosine, entry_stats, forge,
                              jsonable, passport_map, seed_bytes)

EPS_IMPL = 1e-8
UNIFORM_SCALES = (0.5, 2.0, 4.0, 10.0, 1000.0)
AFFINE_SCALES = (1e-3, 1e-2, 0.1, 0.5, 2.0, 4.0, 10.0, 1000.0)
NULL_SCALES = (0.0, 1.0, 5.0, 50.0)
NET_DRAWS = ((0.1, 10.0),) * 5 + ((1e-3, 1e3),)
AFFINE_DRAWS = 200
ALPHA_EXTRA = 0.5
SECTIONS = ("family", "affine", "floor", "rescale", "nullspace", "forward", "eps0", "dynamic", "wellformed", "linkage")
LATTICE = 2.0 ** 23
KS_C01 = 1.6276
Z_C01 = 2.5758
EXTRA_SCREENS = ("serial_z_lag2_width", "serial_z_lag2_height", "serial_z_lag2_channel", "serial_z_lag3_width",
                 "serial_z_lag3_height", "serial_z_lag3_channel", "chanvar_z")
RANKMATCH_ITERS = 50
SHAPE_BETA = 0.005
SHAPE_BETAS_REPORTED = (0.005, 0.01, 0.015)
SHAPE_SEED = 123
SCREEN_SWEEP = 10
LINK_NULL = 20
UNLINK_BETAS = (0.005, 0.097, 0.149)
UNLINK_SECANT_FROM = ((0.097, 36.101146139581296), (0.149, 4.0215689424651435))
UNLINK_SECANT_BETA = (UNLINK_SECANT_FROM[1][0] - UNLINK_SECANT_FROM[1][1] * (UNLINK_SECANT_FROM[1][0] - UNLINK_SECANT_FROM[0][0])
                      / (UNLINK_SECANT_FROM[1][1] - UNLINK_SECANT_FROM[0][1]))
UNLINK_SEED_POOL = 40
LINK_ALIGN_NULL = 5


def rel(path: str) -> str:
    try:
        return os.path.relpath(os.path.abspath(path), REPO).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_subject(path: str, data_root: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    K, D, W = cfg.get("num_classes", 100), cfg.get("embed_dim", 128), cfg.get("width", 64)
    alpha = float(ck.get("alpha_dyn", 0.0))
    model = build_model(K, D, W, passport=True, alpha=alpha)
    model.load_state_dict(ck["state"])
    model = model.double().eval()
    theta = {k: v.detach().cpu().double() for k, v in ck["passport"].items()}
    if cfg.get("synthetic"):
        _, test = load_synthetic(1200, 300, K, cfg.get("seed", 0))
        dataset = "synthetic"
    else:
        _, test = load_cifar100(root=data_root)
        dataset = "cifar100"
    xs, ys = [], []
    for x, y in DataLoader(test, batch_size=1000):
        xs.append(x)
        ys.append(y)
    foreign = {}
    for name, t in ck.get("foreign_passports", {}).items():
        foreign[name] = {k: v.detach().cpu().double() for k, v in t.items()}
    return {"ck": ck, "model": model, "theta": theta, "C": ck["global_C"].detach().cpu().double(),
            "E": ck.get("E"), "alpha": alpha, "K": K, "D": D, "W": W, "dataset": dataset,
            "xs": torch.cat(xs), "ys": torch.cat(ys), "foreign": foreign, "config": cfg}


def unit_rms_with(eps: float):
    def f(v):
        return v / (v.pow(2).mean().sqrt() + eps) if eps else v / v.pow(2).mean().sqrt()
    return f


def matrix_affine(pl, A: torch.Tensor, eps: float):
    n = unit_rms_with(eps)

    def _affine():
        gamma = n(A @ pl.theta_gamma.reshape(-1))
        beta = n(A @ pl.theta_beta.reshape(-1))
        if pl.alpha != 0.0:
            gamma = gamma + pl.alpha * torch.tanh(pl.P_dyn @ pl.z_dyn)
        return gamma, beta
    return _affine


def configure(model, theta: dict, eps: float = EPS_IMPL, alpha: float = 0.0, E=None,
              route: str = "conv", mats=None):
    dt = next(model.parameters()).dtype
    model.install_passport({k: v.to(dt) for k, v in theta.items()})
    for li, pl in enumerate(model.passport_layers):
        pl.__dict__.pop("_unit_rms", None)
        pl.__dict__.pop("_affine", None)
        pl.alpha = alpha
        pl.install_dynamic(E)
        pl.z_dyn = pl.z_dyn.to(dt)
        if eps != EPS_IMPL:
            pl._unit_rms = unit_rms_with(eps)
        if route == "matrix":
            pl._affine = matrix_affine(pl, mats[li], eps)


@torch.no_grad()
def affine_of(model) -> list:
    return [tuple(t.detach().clone() for t in pl._affine()) for pl in model.passport_layers]


@torch.no_grad()
def run(model, S: dict, bs: int) -> dict:
    model.eval()
    dt = next(model.parameters()).dtype
    C = S["C"].to(dt)
    logits, pp = [], []
    t0 = time.time()
    for i in range(0, len(S["xs"]), bs):
        lg, z = model(S["xs"][i:i + bs].to(dt))
        logits.append(lg)
        pp.append(torch.cdist(z, C, compute_mode="donot_use_mm_for_euclid_dist").argmin(1))
    logits = torch.cat(logits)
    pc, pp = logits.argmax(1), torch.cat(pp)
    return {"logits": logits, "pred_classifier": pc, "pred_prototype": pp,
            "acc_classifier": float((pc == S["ys"]).double().mean()),
            "acc_prototype": float((pp == S["ys"]).double().mean()),
            "affine": affine_of(model), "seconds": time.time() - t0}


def compare(name: str, ev: dict, ref: dict, ref_name: str, theta=None, theta_ref=None, **extra) -> dict:
    dl = (ev["logits"] - ref["logits"]).abs()
    dg = max(float((g - g0).abs().max()) for (g, _), (g0, _) in zip(ev["affine"], ref["affine"]))
    db = max(float((b - b0).abs().max()) for (_, b), (_, b0) in zip(ev["affine"], ref["affine"]))
    row = {"name": name, "reference": ref_name,
           "acc_classifier": ev["acc_classifier"], "acc_prototype": ev["acc_prototype"],
           "d_acc_classifier_pp": 100.0 * (ev["acc_classifier"] - ref["acc_classifier"]),
           "d_acc_prototype_pp": 100.0 * (ev["acc_prototype"] - ref["acc_prototype"]),
           "max_abs_dlogit": float(dl.max()),
           "max_rel_dlogit": float(dl.max() / ref["logits"].abs().max()),
           "logits_bitwise_identical": bool(torch.equal(ev["logits"], ref["logits"])),
           "n_classifier_predictions_changed": int((ev["pred_classifier"] != ref["pred_classifier"]).sum()),
           "n_prototype_predictions_changed": int((ev["pred_prototype"] != ref["pred_prototype"]).sum()),
           "n_test": int(len(ev["pred_classifier"])),
           "max_abs_dgamma": dg, "max_abs_dbeta": db, "pass_seconds": ev["seconds"]}
    if theta is not None and theta_ref is not None:
        cos, nrm = cosine(theta_ref, theta)
        row.update(cos_theta_prime_theta=None if math.isnan(cos) else cos, norm_ratio=nrm)
    row.update(extra)
    return row


def key_of(theta: dict) -> dict:
    tb = passport_bytes(theta)
    return {"theta_sha256": idm.H(tb).hex(),
            "pk_Ed25519": idm.derive_signer(tb, "Ed25519").public_key.hex()}


def scaled(theta: dict, a: dict) -> dict:
    return {k: v * float(a[k]) for k, v in theta.items()}


def as_float32(theta: dict) -> dict:
    return {k: v.float().double() for k, v in theta.items()}


def per_branch_draw(keys, lo: float, hi: float, rng) -> dict:
    return {k: float(10.0 ** rng.uniform(math.log10(lo), math.log10(hi))) for k in keys}


def branch_rms(model, theta: dict) -> dict:
    out = {}
    for li, pl in enumerate(model.passport_layers):
        w = pl.conv.weight.detach().double()
        for kind in ("gamma", "beta"):
            y = Fn.conv2d(theta[f"p{li}_{kind}"].double(), w, padding=w.shape[-1] // 2).mean((0, 2, 3))
            out[f"p{li}_{kind}"] = {"rms": float(y.pow(2).mean().sqrt()), "max_abs": float(y.abs().max()),
                                    "cout": int(y.numel())}
    return out


def eps_bound(a: float, rms: float, eps: float = EPS_IMPL) -> float:
    return abs(a - 1.0) * eps / (a * rms + eps)


def section_family(S: dict) -> dict:
    model, theta = S["model"], S["theta"]
    t0 = time.time()
    layers, entries, rank, worst_cond = [], 0, 0, 0.0
    for li, pl in enumerate(model.passport_layers):
        A = passport_map(pl)
        s = torch.linalg.svdvals(A)
        tol = float(s.max()) * max(A.shape) * torch.finfo(torch.float64).eps
        r = int((s > tol).sum())
        N = A.shape[1]
        cond = float(s.max() / s[r - 1])
        worst_cond = max(worst_cond, cond)
        layers.append({"layer": li, "cin": pl.conv.in_channels, "cout": pl.conv.out_channels,
                       "entries_per_branch": N, "rank_per_branch": r, "full_row_rank": r == A.shape[0],
                       "nullspace_dim_per_branch": N - r, "cond_number": cond})
        entries += 2 * N
        rank += 2 * r
        del A, s
    n_branches = 2 * len(model.passport_layers)
    rms = branch_rms(model, theta)
    rmin_key = min(rms, key=lambda k: rms[k]["rms"])
    rmin = rms[rmin_key]["rms"]
    bounds = {}
    for a in UNIFORM_SCALES:
        worst = max(rms, key=lambda k: eps_bound(a, rms[k]["rms"]))
        bounds["%g" % a] = {"max_rel_bound": eps_bound(a, rms[worst]["rms"]), "branch": worst,
                            "max_abs_affine_bound": max(eps_bound(a, v["rms"]) * v["max_abs"] / (v["rms"] + EPS_IMPL)
                                                        for v in rms.values())}
    null = entries - rank
    return {
        "n_passport_layers": len(model.passport_layers), "n_branches": n_branches,
        "passport_entries": entries, "numeric_rank_total": rank, "nullspace_dim": null,
        "nullspace_fraction": null / entries,
        "family_dim_eps0": null + n_branches, "family_fraction_eps0": (null + n_branches) / entries,
        "family_constraints_eps0": rank - n_branches,
        "family_dim_eps_impl_exact": null, "family_fraction_eps_impl_exact": null / entries,
        "worst_cond_number": worst_cond,
        "branch_rms": rms, "rms_min": rmin, "rms_min_branch": rmin_key,
        "rms_max": max(v["rms"] for v in rms.values()),
        "eps": EPS_IMPL,
        "eps_bound_at_uniform_scales": bounds,
        "eps_bound_max_over_uniform_scales": max(b["max_rel_bound"] for b in bounds.values()),
        "eps_bound_sup_a_ge_half": EPS_IMPL / rmin,
        "per_layer": layers, "seconds": time.time() - t0,
    }


@torch.no_grad()
def section_affine(S: dict, rng) -> dict:
    model, theta = S["model"], S["theta"]
    keys = sorted(theta)
    rms = branch_rms(model, theta)
    t0 = time.time()

    def branch_dev(eps, a):
        configure(model, theta, eps, 0.0, None)
        base = affine_of(model)
        configure(model, scaled(theta, a), eps, 0.0, None)
        got = affine_of(model)
        dev = {}
        for li in range(len(base)):
            for j, kind in enumerate(("gamma", "beta")):
                dev[f"p{li}_{kind}"] = float((got[li][j] - base[li][j]).abs().max())
        return dev

    def predicted(a, k):
        r, m = rms[k]["rms"], rms[k]["max_abs"]
        return m * abs(a[k] - 1.0) * EPS_IMPL / ((a[k] * r + EPS_IMPL) * (r + EPS_IMPL))

    uniform = []
    for s in AFFINE_SCALES:
        a = {k: s for k in keys}
        m1, m0 = branch_dev(EPS_IMPL, a), branch_dev(0.0, a)
        pred = {k: predicted(a, k) for k in keys}
        uniform.append({"a": s,
                        "eps_impl_max_abs_daffine": max(m1.values()),
                        "eps_impl_predicted_max_abs_daffine": max(pred.values()),
                        "eps_impl_measured_over_predicted": [min(m1[k] / pred[k] for k in keys),
                                                             max(m1[k] / pred[k] for k in keys)],
                        "eps_impl_max_rel_bound": max(eps_bound(s, rms[k]["rms"]) for k in keys),
                        "eps0_max_abs_daffine": max(m0.values())})
    draws, ratio_lo, ratio_hi, eps0_max, rel_max = [], math.inf, 0.0, 0.0, 0.0
    for i in range(AFFINE_DRAWS):
        a = per_branch_draw(keys, 1e-3, 1e3, rng)
        m1, m0 = branch_dev(EPS_IMPL, a), branch_dev(0.0, a)
        for k in keys:
            p = predicted(a, k)
            if p > 1e-12:
                ratio_lo, ratio_hi = min(ratio_lo, m1[k] / p), max(ratio_hi, m1[k] / p)
            rel_max = max(rel_max, eps_bound(a[k], rms[k]["rms"]))
        eps0_max = max(eps0_max, max(m0.values()))
        if i < 3:
            draws.append({"a": a, "eps_impl_max_abs_daffine": max(m1.values()),
                          "eps0_max_abs_daffine": max(m0.values())})
    a = {k: (-1.0 if k == "p0_gamma" else 1.0) for k in keys}
    flip = branch_dev(EPS_IMPL, a)["p0_gamma"]
    configure(model, theta, EPS_IMPL, S["alpha"], S["E"])
    return {"uniform_scales": uniform,
            "per_branch_draws": {"n_draws": AFFINE_DRAWS, "range": [1e-3, 1e3],
                                 "eps_impl_measured_over_predicted_range": [ratio_lo, ratio_hi],
                                 "eps_impl_max_rel_bound": rel_max,
                                 "eps0_max_abs_daffine": eps0_max, "first_draws": draws},
            "sign_flip_p0_gamma_max_abs_dgamma": flip,
            "sign_flip_p0_gamma_expected": 2.0 * rms["p0_gamma"]["max_abs"] / (rms["p0_gamma"]["rms"] + EPS_IMPL),
            "seconds": time.time() - t0}


def forge64(model, theta: dict, scales, seed: int = 11, entropy: bytes = None) -> tuple:
    gen = torch.Generator().manual_seed(seed)
    t_fresh = time.time()
    fresh = make_passport_tensors(model.passport_spec(), entropy)
    t_fresh = time.time() - t_fresh
    variants = [float(s) for s in scales] + [DIST_MATCHED]
    forged = {k: {} for k in variants}
    t_proj = {k: 0.0 for k in variants}
    t_proj[DIST_MATCHED] += t_fresh
    resid = {k: 0.0 for k in variants}
    resid32 = {k: 0.0 for k in variants}
    t_setup = 0.0
    for li, pl in enumerate(model.passport_layers):
        t0 = time.time()
        A = passport_map(pl)
        U, Sv, Vh = torch.linalg.svd(A, full_matrices=False)
        tol = float(Sv.max()) * max(A.shape) * torch.finfo(torch.float64).eps
        r = int((Sv > tol).sum())
        Vr = Vh[:r].contiguous()
        t_setup += time.time() - t0
        for kind in ("gamma", "beta"):
            key = f"p{li}_{kind}"
            shape = theta[key].shape
            v = theta[key].detach().reshape(-1).double()
            t0 = time.time()
            Pv = Vr.T @ (Vr @ v)
            n = torch.randn(v.numel(), dtype=torch.float64, generator=gen)
            u = n - Vr.T @ (Vr @ n)
            u = u / u.norm()
            t_shared = time.time() - t0
            Av = A @ v
            for s in scales:
                t0 = time.time()
                v2 = Pv + float(s) * v.norm() * u
                t_proj[float(s)] += t_shared + (time.time() - t0)
                forged[float(s)][key] = v2.reshape(shape).contiguous()
            t0 = time.time()
            w = fresh[key].reshape(-1).double()
            vd = w + Vr.T @ (Vr @ (v - w))
            t_proj[DIST_MATCHED] += time.time() - t0
            forged[DIST_MATCHED][key] = vd.reshape(shape).contiguous()
            for k in variants:
                x = forged[k][key].reshape(-1)
                resid[k] = max(resid[k], float((A @ x - Av).norm() / Av.norm()))
                resid32[k] = max(resid32[k], float((A @ x.float().double() - Av).norm() / Av.norm()))
        del A, U, Sv, Vh, Vr
    return forged, {"svd_setup_s": t_setup, "project_s": t_proj}, resid, resid32, fresh


class Evaluator:
    def __init__(self, S: dict, bs: int):
        self.S, self.bs, self.refs = S, bs, {}

    def eval(self, theta, eps=EPS_IMPL, alpha=None, E="own", route="conv", mats=None, bs=None, label=""):
        alpha = self.S["alpha"] if alpha is None else alpha
        E = self.S["E"] if E == "own" else E
        configure(self.S["model"], theta, eps, alpha, E, route, mats)
        ev = run(self.S["model"], self.S, bs or self.bs)
        print("  %-48s acc %.4f / %.4f  %.0f s" % (label, ev["acc_classifier"], ev["acc_prototype"],
                                                   ev["seconds"]), flush=True)
        return ev

    def ref(self, name, **kw):
        if name not in self.refs:
            self.refs[name] = self.eval(self.S["theta"], label="reference " + name, **kw)
        return self.refs[name]


def summary_of(ev: dict) -> dict:
    return {"acc_classifier": ev["acc_classifier"], "acc_prototype": ev["acc_prototype"],
            "logit_abs_max": float(ev["logits"].abs().max()), "pass_seconds": ev["seconds"]}


def section_floor(S: dict, X: Evaluator) -> dict:
    t0 = time.time()
    base = X.ref("base")
    out = {"float64": {"reference": summary_of(base), "batch_size_reference": X.bs}}
    for bs in (37, 128):
        ev = X.eval(S["theta"], bs=bs, label="float64 rerun, batch %d" % bs)
        out["float64"]["rerun_batch_%d" % bs] = compare("rerun_batch_%d" % bs, ev, base, "base")
    mats = [passport_map(pl) for pl in S["model"].passport_layers]
    ev = X.eval(S["theta"], route="matrix", mats=mats, label="float64, gamma/beta through explicit A")
    out["float64"]["matrix_route"] = compare("matrix_route", ev, base, "base")
    del mats
    rows = [out["float64"][k] for k in ("rerun_batch_37", "rerun_batch_128", "matrix_route")]
    out["float64"]["floor_max_abs_dlogit"] = max(r["max_abs_dlogit"] for r in rows)
    out["float64"]["floor_max_abs_daffine"] = max(max(r["max_abs_dgamma"], r["max_abs_dbeta"]) for r in rows)

    forged = forged64(S)[0]
    m64 = S["model"]
    S["model"] = copy.deepcopy(m64).float()
    try:
        X32 = Evaluator(S, X.bs)
        b32 = X32.eval(S["theta"], label="float32 reference")
        f32 = {"reference": summary_of(b32)}
        for bs in (37, 128):
            ev = X32.eval(S["theta"], bs=bs, label="float32 rerun, batch %d" % bs)
            f32["rerun_batch_%d" % bs] = compare("rerun_batch_%d" % bs, ev, b32, "float32 base")
        f32["rerun_floor_max_abs_dlogit"] = max(f32["rerun_batch_37"]["max_abs_dlogit"],
                                                f32["rerun_batch_128"]["max_abs_dlogit"])
        ev = X32.eval(as_float32(forged[DIST_MATCHED]), label="float32, nullspace distribution-matched")
        f32["nullspace_distribution_matched"] = compare("nullspace_distribution_matched", ev, b32, "float32 base")
        ev = X32.eval(scaled(S["theta"], {k: 2.0 for k in S["theta"]}), label="float32, uniform a = 2")
        f32["uniform_rescale_2"] = compare("uniform_rescale_2", ev, b32, "float32 base")
        f32["float32_vs_float64_arithmetic"] = compare("float32_vs_float64_arithmetic",
                                                       {**b32, "logits": b32["logits"].double(),
                                                        "affine": [(g.double(), b.double()) for g, b in b32["affine"]]},
                                                       base, "float64 base")
        out["float32_cpu"] = f32
    finally:
        S["model"] = m64
        configure(S["model"], S["theta"], EPS_IMPL, S["alpha"], S["E"])
    out["seconds"] = time.time() - t0
    return out


def section_rescale(S: dict, X: Evaluator, draws: list) -> dict:
    t0 = time.time()
    base, theta = X.ref("base"), S["theta"]
    keys = sorted(theta)
    rms = branch_rms(S["model"], theta)
    rows = []
    for i, a in enumerate(draws):
        ev = X.eval(scaled(theta, a), label="per-branch draw %d" % i)
        rows.append(compare("per_branch_draw_%d" % i, ev, base, "base", scaled(theta, a), theta,
                            scales=a, scale_min=min(a.values()), scale_max=max(a.values()),
                            eps_rel_bound=max(eps_bound(a[k], rms[k]["rms"]) for k in keys),
                            **key_of(scaled(theta, a))))
    t = as_float32(scaled(theta, draws[0]))
    ev = X.eval(t, label="per-branch draw 0, stored in float32")
    rows.append(compare("per_branch_draw_0_float32_stored", ev, base, "base", t, theta, **key_of(t)))
    for s in (2.0, 1000.0):
        t = scaled(theta, {k: s for k in keys})
        ev = X.eval(t, label="uniform a = %g" % s)
        rows.append(compare("uniform_%g" % s, ev, base, "base", t, theta,
                            eps_rel_bound=max(eps_bound(s, rms[k]["rms"]) for k in keys), **key_of(t)))
    for name, a in (("sign_flip_all", {k: -1.0 for k in keys}),
                    ("sign_flip_p0_gamma", {k: (-1.0 if k == "p0_gamma" else 1.0) for k in keys})):
        t = scaled(theta, a)
        ev = X.eval(t, label=name)
        rows.append(compare(name, ev, base, "base", t, theta, **key_of(t)))
    return {"eps": EPS_IMPL, "draw_ranges": [list(r) for r in NET_DRAWS], "reference": summary_of(base),
            "rows": rows, "genuine_key": key_of(theta), "seconds": time.time() - t0}


def forged64(S: dict) -> tuple:
    if "forge64" not in S:
        S["forge64"] = forge64(S["model"], S["theta"], NULL_SCALES, entropy=S["entropy"])
    return S["forge64"]


def section_nullspace(S: dict, X: Evaluator) -> dict:
    t0 = time.time()
    base, theta = X.ref("base"), S["theta"]
    forged, timing, resid, resid32, fresh = forged64(S)
    ref32, timing_ref, _, _ = forge(S["model"], theta, NULL_SCALES, entropy=S["entropy"])
    max_cast_dev = max(float((forged[v][k].float() - ref32[v][k]).abs().max()) for v in ref32 for k in ref32[v])
    rows = []
    for v in list(NULL_SCALES) + [DIST_MATCHED]:
        label = ("scale_%g" % v) if v != DIST_MATCHED else "distribution_matched"
        for store, t in (("float64", forged[v]), ("float32_stored", as_float32(forged[v]))):
            ev = X.eval(t, label="nullspace %s, %s" % (label, store))
            rows.append(compare("nullspace_%s_%s" % (label, store), ev, base, "base", t, theta,
                                nullspace_scale=None if v == DIST_MATCHED else v, storage=store,
                                max_rel_residual_A_theta=(resid if store == "float64" else resid32)[v],
                                forge_seconds_marginal=timing["project_s"][v], **key_of(t)))
    g = torch.Generator().manual_seed(0)
    ulp_all = {}
    for k in sorted(theta):
        t32 = theta[k].float()
        up = torch.rand(t32.shape, generator=g) < 0.5
        ulp_all[k] = torch.where(up, torch.nextafter(t32, torch.full_like(t32, math.inf)),
                                 torch.nextafter(t32, torch.full_like(t32, -math.inf))).double()
    one = {k: v.clone() for k, v in theta.items()}
    one["p0_gamma"].view(-1)[0] = torch.nextafter(theta["p0_gamma"].float().view(-1)[0:1],
                                                  torch.tensor([math.inf])).double()[0]
    for name, t in (("float32_ulp_every_entry", ulp_all), ("float32_ulp_one_entry", one)):
        ev = X.eval(t, label=name)
        rows.append(compare(name, ev, base, "base", t, theta, **key_of(t)))
    dist = forged[DIST_MATCHED]
    return {"reference": summary_of(base), "rows": rows, "genuine_key": key_of(theta),
            "svd_setup_seconds_this_host": timing["svd_setup_s"],
            "per_identity_seconds_this_host": timing["project_s"],
            "float32_cast_equals_negative_results_forge": max_cast_dev == 0.0,
            "float32_cast_max_abs_deviation_from_forge": max_cast_dev,
            "entry_stats": {"genuine": entry_stats(theta), "distribution_matched": entry_stats(dist)},
            "cos_distribution_matched_to_fresh_draw": cosine(fresh, dist)[0],
            "seconds": time.time() - t0}


def foreign_passports(S: dict, a) -> dict:
    out = dict(S["foreign"])
    if a.foreign == "siblings":
        ckdir, base = os.path.dirname(os.path.abspath(a.checkpoint)), os.path.basename(a.checkpoint)
        if "_client" in base:
            tag = base.rsplit("_client", 1)[0]
            for i in range(100):
                p = os.path.join(ckdir, "%s_client%d.pt" % (tag, i))
                if os.path.exists(p) and os.path.abspath(p) != os.path.abspath(a.checkpoint):
                    ck = torch.load(p, map_location="cpu", weights_only=False)
                    out[os.path.basename(p)] = {k: v.detach().cpu().double() for k, v in ck["passport"].items()}
                    S["foreign_sha256"][os.path.basename(p)] = sha256_file(p)
    return out


def section_forward(S: dict, X: Evaluator, a) -> dict:
    t0 = time.time()
    base, theta = X.ref("base"), S["theta"]
    spec = S["model"].passport_spec()
    rows = []
    rand = [("random_theta_0", make_passport_tensors(spec, S["entropy"]))]
    rand += [("random_theta_%d" % i, make_passport_tensors(spec, seed_bytes(S["entropy"], "random-theta-%d" % i, 64)))
             for i in (1, 2)]
    for name, t in rand + [("zero_theta", {k: torch.zeros_like(v) for k, v in theta.items()})]:
        t = {k: v.double() for k, v in t.items()}
        ev = X.eval(t, label=name)
        rows.append(compare(name, ev, base, "base", t, theta))
    for name, t in foreign_passports(S, a).items():
        ev = X.eval(t, label="foreign " + name)
        rows.append(compare("foreign_" + name, ev, base, "base", t, theta))
    fr = [r for r in rows if r["name"].startswith("foreign_")]
    rr = [r for r in rows if r["name"].startswith("random_")]

    def span(rs, key):
        return {"mean": float(np.mean([r[key] for r in rs])), "min": min(r[key] for r in rs),
                "max": max(r[key] for r in rs), "n": len(rs)} if rs else None
    return {"rows": rows, "genuine": summary_of(base), "chance": 1.0 / S["K"],
            "random": {"acc_classifier": span(rr, "acc_classifier"), "acc_prototype": span(rr, "acc_prototype")},
            "foreign": {"acc_classifier": span(fr, "acc_classifier"), "acc_prototype": span(fr, "acc_prototype")},
            "seconds": time.time() - t0}


def section_eps0(S: dict, X: Evaluator, draws: list) -> dict:
    t0 = time.time()
    theta, keys = S["theta"], sorted(S["theta"])
    b0 = X.ref("base_eps0", eps=0.0)
    rows = [compare("eps0_vs_eps_impl_genuine", b0, X.ref("base"), "base")]
    for i, a in enumerate(draws):
        ev = X.eval(scaled(theta, a), eps=0.0, label="eps = 0, per-branch draw %d" % i)
        rows.append(compare("per_branch_draw_%d" % i, ev, b0, "base_eps0", scaled(theta, a), theta,
                            scale_min=min(a.values()), scale_max=max(a.values())))
    for s in (2.0, 1000.0):
        t = scaled(theta, {k: s for k in keys})
        ev = X.eval(t, eps=0.0, label="eps = 0, uniform a = %g" % s)
        rows.append(compare("uniform_%g" % s, ev, b0, "base_eps0", t, theta))
    forged = forged64(S)[0]
    ev = X.eval(forged[DIST_MATCHED], eps=0.0, label="eps = 0, nullspace distribution-matched")
    rows.append(compare("nullspace_distribution_matched_float64", ev, b0, "base_eps0", forged[DIST_MATCHED], theta))
    return {"eps": 0.0, "rows": rows, "seconds": time.time() - t0}


def section_dynamic(S: dict, X: Evaluator, draws: list) -> dict:
    if S["alpha"] != 0.0:
        return {"skipped": "checkpoint trained with alpha_dyn = %g: see identity_swap_eval.py" % S["alpha"]}
    t0 = time.time()
    theta, keys, E, al = S["theta"], sorted(S["theta"]), S["E"], ALPHA_EXTRA
    base = X.ref("base")
    bd = X.ref("base_dyn", alpha=al)
    rows = []
    forged = forged64(S)[0]
    null64, null32 = forged[DIST_MATCHED], as_float32(forged[DIST_MATCHED])
    two = scaled(theta, {k: 2.0 for k in keys})
    for name, t in (("per_branch_draw_0", scaled(theta, draws[0])), ("uniform_2", two),
                    ("nullspace_distribution_matched_float64", null64),
                    ("nullspace_distribution_matched_float32_stored", null32)):
        ev = X.eval(t, alpha=al, label="alpha 0.5, own E, " + name)
        rows.append(compare(name + "_own_E", ev, bd, "base_dyn", t, theta))
    b0 = X.ref("base_dyn_eps0", eps=0.0, alpha=al)
    ev = X.eval(scaled(theta, draws[0]), eps=0.0, alpha=al, label="alpha 0.5, own E, eps 0, draw 0")
    rows.append(compare("per_branch_draw_0_own_E_eps0", ev, b0, "base_dyn_eps0", scaled(theta, draws[0]), theta))

    configure(S["model"], theta, EPS_IMPL, al, None)
    with torch.no_grad():
        same_first_batch = bool(torch.equal(S["model"](S["xs"][:X.bs].double())[0], base["logits"][:X.bs]))
    swap = [compare("genuine_own_E_vs_z0", bd, base, "base (z = 0)")]
    forgery_heads = []
    for name, t in (("rescale_2", two), ("nullspace_distribution_matched_float32_stored", null32)):
        pk = idm.derive_signer(passport_bytes(t), "Ed25519").public_key
        ev = X.eval(t, alpha=al, E=idm.H(b"E0|" + pk), label="alpha 0.5, %s, stand-in head of new key" % name)
        swap.append(compare(name + "_standin_E_new_key_vs_z0", ev, base, "base (z = 0)", t, theta))
        forgery_heads.append((name, t, idm.H(b"E0|" + pk), ev))
    pk0 = idm.derive_signer(passport_bytes(theta), "Ed25519").public_key
    rand_E = seed_bytes(S["entropy"], "passport-equivalence-random-E")
    for name, e in (("genuine_standin_E_genuine_key", idm.H(b"E0|" + pk0)), ("genuine_random_E", rand_E)):
        ev = X.eval(theta, alpha=al, E=e, label="alpha 0.5, " + name)
        swap.append(compare(name + "_vs_z0", ev, base, "base (z = 0)"))
        swap.append(compare(name + "_vs_own_E", ev, bd, "base_dyn"))
    for name, t, e, ev_f in forgery_heads:
        ev = X.eval(theta, alpha=al, E=e, label="alpha 0.5, genuine, stand-in head of %s's key" % name)
        swap.append(compare("genuine_standin_E_of_%s_key_vs_z0" % name, ev, base, "base (z = 0)"))
        swap.append(compare(name + "_standin_E_new_key_vs_genuine_same_head", ev_f, ev,
                            "genuine passport under the same head", t, theta))
    heads = [r for r in swap if r["reference"] == "base (z = 0)" and r["name"].startswith("genuine_")]
    same = [r for r in swap if r["name"].endswith("_vs_genuine_same_head")]
    spread = {"min_d_acc_prototype_pp": min(r["d_acc_prototype_pp"] for r in heads),
              "max_d_acc_prototype_pp": max(r["d_acc_prototype_pp"] for r in heads),
              "min_d_acc_classifier_pp": min(r["d_acc_classifier_pp"] for r in heads),
              "max_d_acc_classifier_pp": max(r["d_acc_classifier_pp"] for r in heads),
              "rows": [r["name"] for r in heads]}
    subst = {"max_predictions_changed": max(max(r["n_classifier_predictions_changed"],
                                                r["n_prototype_predictions_changed"]) for r in same),
             "max_abs_dlogit": max(r["max_abs_dlogit"] for r in same),
             "rows": [r["name"] for r in same]}
    return {"alpha_dyn": al, "E_own": E, "reference_dyn": summary_of(bd),
            "z0_equals_alpha0_first_batch": same_first_batch, "invariance_rows": rows,
            "identity_swap_reanalysis": swap, "head_spread_vs_z0": spread, "substitution_same_head": subst,
            "seconds": time.time() - t0}


def ks_uniform(x: np.ndarray) -> float:
    x = np.sort(x)
    n = len(x)
    F = np.clip((x + 1.0) / 2.0, 0.0, 1.0)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - F), np.max(F - (i - 1) / n)))


def serial_z(theta: dict, lag: int = 1) -> dict:
    xs = [theta[k].double() for k in sorted(theta)]
    v = torch.cat([x.flatten() for x in xs])
    m = float(v.mean())
    s2 = float((v - m).pow(2).mean())
    out = {}
    for ax, dim in (("width", -1), ("height", -2), ("channel", -3)):
        num, n = 0.0, 0
        for x in xs:
            if x.dim() < -dim or x.shape[dim] < lag + 1:
                continue
            x = x - m
            L = x.shape[dim]
            a, b = x.narrow(dim, 0, L - lag), x.narrow(dim, lag, L - lag)
            num += float((a * b).sum())
            n += a.numel()
        out[("serial_z_" if lag == 1 else "serial_z_lag%d_" % lag) + ax] = \
            num / (s2 * math.sqrt(n)) if n and s2 > 0 else None
    return out


def chanvar_z(theta: dict):
    xs = [theta[k].double() for k in sorted(theta)]
    v = torch.cat([x.flatten() for x in xs])
    m = float(v.mean())
    s2 = float((v - m).pow(2).mean())
    q, n = 0.0, 0
    for x in xs:
        if x.dim() < 3:
            continue
        mc = x.mean((-2, -1)).flatten()
        q += float((x.shape[-1] * x.shape[-2] * (mc - m).pow(2)).sum()) / s2 if s2 > 0 else 0.0
        n += mc.numel()
    return (q - n) / math.sqrt(2 * n) if n and s2 > 0 else None


def wellformedness(theta: dict) -> dict:
    v = torch.cat([theta[k].flatten() for k in sorted(theta)]).double()
    g = v * LATTICE
    n = int(v.numel())
    branch = [math.sqrt(t.numel()) * ks_uniform(t.flatten().double().numpy()) for t in (theta[k] for k in sorted(theta))]
    out = {"frac_outside_generator_range": float(((v < -1) | (v >= 1)).double().mean()),
           "frac_on_generator_lattice": float((g == g.round()).double().mean()),
           "ks_vs_uniform": ks_uniform(v.numpy()), "ks_critical_1pct": KS_C01 / math.sqrt(n), "n_entries": n,
           "branch_sqrt_n_ks_max": max(branch), "branches_over_ks_1pct": int(sum(b > KS_C01 for b in branch)),
           "n_branches": len(branch), **serial_z(theta), "z_critical_1pct": Z_C01,
           "entry_stats": entry_stats(theta)}
    zb = [serial_z({k: theta[k]}) for k in sorted(theta)]
    for ax in ("width", "height", "channel"):
        zs = [abs(z["serial_z_" + ax]) for z in zb if z["serial_z_" + ax] is not None]
        out["branch_serial_z_max_abs_" + ax] = max(zs) if zs else None
        out["branches_over_z_1pct_" + ax] = int(sum(z > Z_C01 for z in zs))
    out.update(**serial_z(theta, 2), **serial_z(theta, 3), chanvar_z=chanvar_z(theta))
    return out


def passes_screens(s: dict) -> bool:
    return bool(s["frac_outside_generator_range"] == 0.0 and s["frac_on_generator_lattice"] == 1.0
                and s["ks_vs_uniform"] < s["ks_critical_1pct"] and s["branches_over_ks_1pct"] == 0
                and all(s["serial_z_" + ax] is not None and abs(s["serial_z_" + ax]) < Z_C01
                        for ax in ("width", "height", "channel")))


def shaped_start(draw: dict, beta: float, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k in sorted(draw):
        v = draw[k]
        x = torch.randn(v.shape, generator=g, dtype=torch.float64)
        nb = torch.zeros_like(x)
        nb[..., 1:] += x[..., :-1]
        nb[..., :-1] += x[..., 1:]
        nb[..., 1:, :] += x[..., :-1, :]
        nb[..., :-1, :] += x[..., 1:, :]
        y = (x - beta * nb).flatten()
        vals = torch.sort(v.flatten().double()).values
        o = torch.empty_like(vals)
        o[torch.argsort(y)] = vals
        out[k] = o.reshape(v.shape)
    return out


def wellformed_forgery(model, theta: dict, start: dict, tol: float = 1e-13, max_iter: int = 5000,
                       rank_iters: int = 0, unlink: bool = False) -> tuple:
    lo, hi = -1.0, 1.0 - 1.0 / LATTICE
    out, iters, resid, viol, gap = {}, [], 0.0, 0.0, 0.0
    forced = []
    if unlink:
        mbar = float(torch.cat([theta[k].double().mean((-2, -1)).flatten() for k in sorted(theta)]).mean())
    t0 = time.time()
    for li, pl in enumerate(model.passport_layers):
        A = passport_map(pl)
        _, Sv, Vh = torch.linalg.svd(A, full_matrices=False)
        r = int((Sv > float(Sv.max()) * max(A.shape) * torch.finfo(torch.float64).eps).sum())
        Vr = Vh[:r].contiguous()
        for kind in ("gamma", "beta"):
            key = f"p{li}_{kind}"
            v = theta[key].reshape(-1).double()
            Pv = Vr.T @ (Vr @ v)

            if not unlink:
                def onto_affine(x):
                    return x - Vr.T @ (Vr @ x) + Pv
            else:
                cm = theta[key].double().mean((-2, -1))
                G = torch.stack([v, ((cm - mbar)[..., None, None] * torch.ones_like(theta[key].double())).reshape(-1)], 1)
                H = G - Vr.T @ (Vr @ G)
                c = -(Pv @ G)
                HtH_inv = torch.linalg.inv(H.T @ H)

                def onto_affine(x):
                    y = x - Vr.T @ (Vr @ x)
                    return y - H @ (HtH_inv @ (H.T @ y - c)) + Pv
                s0 = start[key].reshape(-1).double()
                forced.append(float((H @ (HtH_inv @ c)).norm() / (s0 - Vr.T @ (Vr @ s0)).norm()))
            x = start[key].reshape(-1).double()
            if rank_iters:
                values = torch.sort(x).values
                x = onto_affine(x)
                for _ in range(rank_iters):
                    y = torch.empty_like(x)
                    y[torch.argsort(x)] = values
                    x = onto_affine(y)
                y = torch.empty_like(x)
                y[torch.argsort(x)] = values
                gap = max(gap, float((y - x).abs().max()))
            p, q = torch.zeros_like(x), torch.zeros_like(x)
            for it in range(1, max_iter + 1):
                y = onto_affine(x + p)
                p = x + p - y
                x = (y + q).clamp(lo, hi)
                q = y + q - x
                if float((onto_affine(x) - x).abs().max()) < tol:
                    break
            iters.append(it)
            x = onto_affine(x)
            viol = max(viol, float(torch.clamp(lo - x, min=0).max()), float(torch.clamp(x - hi, min=0).max()))
            x = ((x * LATTICE).round() / LATTICE).clamp(lo, hi)
            out[key] = x.reshape(theta[key].shape).contiguous()
            Av = A @ v
            resid = max(resid, float((A @ x - Av).norm() / Av.norm()))
        del A, Sv, Vh, Vr
    diag = {"dykstra_iterations_max": max(iters), "dykstra_iterations_mean": float(np.mean(iters)),
            "box_violation_before_rounding": viol, "max_rel_residual_A_theta": resid,
            "construction_seconds": time.time() - t0}
    if rank_iters:
        diag.update(rank_matching_rounds=rank_iters, rank_matching_final_gap=gap)
    if unlink:
        diag.update(forced_shift_ratio_min=min(forced), forced_shift_ratio_max=max(forced))
    return out, diag


def rowspace_alignment(model, theta: dict, draw: dict) -> dict:
    keys = sorted(theta)
    comp = {name: {"row": {}, "null": {}} for name in ("genuine", "draw")}
    rank, entries = 0, 0
    for li, pl in enumerate(model.passport_layers):
        A = passport_map(pl)
        _, Sv, Vh = torch.linalg.svd(A, full_matrices=False)
        r = int((Sv > float(Sv.max()) * max(A.shape) * torch.finfo(torch.float64).eps).sum())
        Vr = Vh[:r].contiguous()
        for kind in ("gamma", "beta"):
            key = f"p{li}_{kind}"
            rank += r
            entries += theta[key].numel()
            for name, t in (("genuine", theta), ("draw", draw)):
                x = t[key].reshape(-1).double()
                px = Vr.T @ (Vr @ x)
                comp[name]["row"][key] = px.reshape(t[key].shape)
                comp[name]["null"][key] = (x - px).reshape(t[key].shape)
        del A, Sv, Vh, Vr

    def raw_z(t):
        out = {}
        for ax, dim in (("width", -1), ("height", -2)):
            num, n = 0.0, 0
            for k in keys:
                x = t[k].double()
                L = x.shape[dim]
                num += float((x.narrow(dim, 0, L - 1) * x.narrow(dim, 1, L - 1)).sum())
                n += x.narrow(dim, 0, L - 1).numel()
            out[ax] = num / (math.sqrt(n) / 3.0)
        return out

    def energy(t):
        return float(sum(t[k].double().pow(2).sum() for k in keys))
    res = {"expected_rowspace_share_independent": rank / entries}
    for name, t in (("genuine", theta), ("draw", draw)):
        res[name] = {"rowspace_energy_share": energy(comp[name]["row"]) / energy(t),
                     "lag1_z_rowspace_component": raw_z(comp[name]["row"]),
                     "lag1_z_nullspace_component": raw_z(comp[name]["null"]),
                     "lag1_z_total": raw_z(t)}
    return res


def section_wellformed(S: dict, X: Evaluator) -> dict:
    t0 = time.time()
    base, theta = X.ref("base"), S["theta"]
    forged, _, _, _, fresh = forged64(S)
    fresh = {k: v.double() for k, v in fresh.items()}
    wf, diag = wellformed_forgery(S["model"], theta, fresh)
    ev = X.eval(wf, label="well-formed nullspace forgery")
    row = compare("nullspace_wellformed", ev, base, "base", wf, theta, **diag,
                  cos_to_start_draw=cosine(fresh, wf)[0], **key_of(wf))
    rm, diag = wellformed_forgery(S["model"], theta, fresh, rank_iters=RANKMATCH_ITERS)
    ev = X.eval(rm, label="rank-matched nullspace forgery")
    row_rm = compare("nullspace_rankmatched", ev, base, "base", rm, theta, **diag,
                     cos_to_start_draw=cosine(fresh, rm)[0], **key_of(rm))
    screens = {"genuine": wellformedness(theta), "fresh_draw_w": wellformedness(fresh)}
    for v in list(NULL_SCALES) + [DIST_MATCHED]:
        label = ("scale_%g" % v) if v != DIST_MATCHED else "distribution_matched"
        screens["nullspace_%s_float32_stored" % label] = wellformedness(as_float32(forged[v]))
    screens["nullspace_wellformed"] = wellformedness(wf)
    screens["nullspace_rankmatched"] = wellformedness(rm)
    screens["uniform_rescale_2"] = wellformedness(scaled(theta, {k: 2.0 for k in theta}))

    st = shaped_start(fresh, SHAPE_BETA, SHAPE_SEED)
    sh, diag = wellformed_forgery(S["model"], theta, st, rank_iters=RANKMATCH_ITERS)
    ev = X.eval(sh, label="shaped rank-matched nullspace forgery")
    row_sh = compare("nullspace_shaped", ev, base, "base", sh, theta, **diag, shape_beta=SHAPE_BETA,
                     shape_seed=SHAPE_SEED, cos_to_start_draw=cosine(fresh, sh)[0], **key_of(sh))
    screens["shaped_start"] = wellformedness(st)
    screens["nullspace_shaped"] = wellformedness(sh)
    for name in screens:
        screens[name]["passes_screens"] = passes_screens(screens[name])
    shaped = section_shaped_sweep(S, theta, fresh, sh, diag)
    return {"row": row, "row_rankmatched": row_rm, "row_shaped": row_sh, "screens": screens,
            "shaped": shaped,
            "rowspace_alignment": rowspace_alignment(S["model"], theta, fresh),
            "seconds": time.time() - t0}


def max_daffine(S: dict, theta: dict, t: dict) -> float:
    model = S["model"]
    configure(model, theta, EPS_IMPL, S["alpha"], S["E"])
    a0 = affine_of(model)
    configure(model, t, EPS_IMPL, S["alpha"], S["E"])
    a1 = affine_of(model)
    configure(model, theta, EPS_IMPL, S["alpha"], S["E"])
    return max(max(float((g - g0).abs().max()), float((b - b0).abs().max())) for (g, b), (g0, b0) in zip(a1, a0))


def section_shaped_sweep(S: dict, theta: dict, fresh: dict, sh: dict, sh_diag: dict) -> dict:
    t0 = time.time()
    spec = S["model"].passport_spec()
    ks_c = KS_C01 / math.sqrt(sum(v.numel() for v in theta.values()))
    keep = ("ks_vs_uniform", "branch_sqrt_n_ks_max", "branches_over_ks_1pct", "serial_z_width", "serial_z_height",
            "serial_z_channel", "branch_serial_z_max_abs_width", "branch_serial_z_max_abs_height",
            "branches_over_z_1pct_width", "branches_over_z_1pct_height", "frac_outside_generator_range",
            "frac_on_generator_lattice") + EXTRA_SCREENS

    def entry(t, **extra):
        s = wellformedness(t)
        return {**extra, **{k: s[k] for k in keep}, "passes_screens": passes_screens(s)}

    def forgery(beta, seed):
        if beta == SHAPE_BETA and seed == SHAPE_SEED:
            f, d = sh, sh_diag
        else:
            f, d = wellformed_forgery(S["model"], theta, shaped_start(fresh, beta, seed), rank_iters=RANKMATCH_ITERS)
        return entry(f, beta=beta, seed=seed, cos_theta_prime_theta=cosine(theta, f)[0],
                     max_abs_daffine=max_daffine(S, theta, f), **key_of(f),
                     max_rel_residual_A_theta=d["max_rel_residual_A_theta"])

    betas = [forgery(b, SHAPE_SEED) for b in SHAPE_BETAS_REPORTED]
    print("  shaped forgery, betas %s: z width %s" % (SHAPE_BETAS_REPORTED, [round(r["serial_z_width"], 2) for r in betas]),
          flush=True)
    seeds = [forgery(SHAPE_BETA, s) for s in range(SCREEN_SWEEP)]
    print("  shaped forgery, seeds 0-%d: %d pass" % (SCREEN_SWEEP - 1, sum(r["passes_screens"] for r in seeds)), flush=True)
    null = [entry({k: v.double() for k, v in make_passport_tensors(spec, seed_bytes(S["entropy"], "screen-null-%d" % i, 64)).items()},
                  draw=i) for i in range(SCREEN_SWEEP)]
    print("  generator draws 0-%d: %d pass" % (SCREEN_SWEEP - 1, sum(r["passes_screens"] for r in null)), flush=True)

    def summary(rs):
        out = {"n": len(rs), "n_pass": int(sum(r["passes_screens"] for r in rs))}
        out["n_pass_marginal"] = int(sum(r["frac_outside_generator_range"] == 0.0 and r["frac_on_generator_lattice"] == 1.0
                                         and r["ks_vs_uniform"] < ks_c and r["branches_over_ks_1pct"] == 0 for r in rs))
        out["n_pass_lag1"] = int(sum(all(abs(r["serial_z_" + ax]) < Z_C01 for ax in ("width", "height", "channel"))
                                     for r in rs))
        for k in ("serial_z_width", "serial_z_height", "serial_z_channel") + EXTRA_SCREENS:
            z = np.array([r[k] for r in rs])
            out[k] = {"mean": float(z.mean()), "sd": float(z.std(ddof=1)), "max_abs": float(np.abs(z).max())}
        out["ks_max"] = max(r["ks_vs_uniform"] for r in rs)
        out["branches_over_ks_1pct_max"] = max(r["branches_over_ks_1pct"] for r in rs)
        out["branch_serial_z_max_abs"] = max(max(r["branch_serial_z_max_abs_width"], r["branch_serial_z_max_abs_height"])
                                             for r in rs)
        if "max_abs_daffine" in rs[0]:
            out["max_abs_daffine_max"] = max(r["max_abs_daffine"] for r in rs)
            out["cos_theta_prime_theta_range"] = [min(r["cos_theta_prime_theta"] for r in rs),
                                                  max(r["cos_theta_prime_theta"] for r in rs)]
        return out
    return {"beta": SHAPE_BETA, "seed": SHAPE_SEED, "betas_reported": list(SHAPE_BETAS_REPORTED),
            "rank_matching_rounds": RANKMATCH_ITERS, "by_beta": betas, "by_seed": seeds, "generator_draws": null,
            "summary": {"shaped_seeds": summary(seeds), "generator_draws": summary(null)},
            "seconds": time.time() - t0}


def linkage_stats(theta: dict, t: dict) -> dict:
    keys = sorted(theta)
    a = torch.cat([t[k].double().flatten() for k in keys])
    b = torch.cat([theta[k].double().flatten() for k in keys])
    cos = float(a @ b / (a.norm() * b.norm()))
    zb = []
    for k in keys:
        x, y = t[k].double().flatten(), theta[k].double().flatten()
        zb.append(abs(float(x @ y / (x.norm() * y.norm()))) * math.sqrt(x.numel()))
    ma = torch.cat([t[k].double().mean((-2, -1)).flatten() for k in keys])
    mb = torch.cat([theta[k].double().mean((-2, -1)).flatten() for k in keys])
    r = float(np.corrcoef(ma.numpy(), mb.numpy())[0, 1])
    out = {"cos": cos, "z_cos": cos * math.sqrt(a.numel()), "branch_z_cos_max_abs": max(zb),
           "r_chan_means": r, "z_chan_means": r * math.sqrt(ma.numel()), "n_entries": int(a.numel()),
           "n_channels": int(ma.numel())}
    for name, dim in (("row", -1), ("col", -2)):
        ma = torch.cat([t[k].double().mean(dim).flatten() for k in keys])
        mb = torch.cat([theta[k].double().mean(dim).flatten() for k in keys])
        r = float(np.corrcoef(ma.numpy(), mb.numpy())[0, 1])
        out.update({"r_chan_%s_means" % name: r, "z_chan_%s_means" % name: r * math.sqrt(ma.numel())})
    return out


def channel_alignment(model, theta: dict, others: dict = None) -> dict:
    others = others or {}
    layers, tot = [], {"means": 0.0, "means_cos_gt_0.9": 0.0, "row": 0.0, "row_in_C": 0.0}
    tots = {name: dict.fromkeys(tot, 0.0) for name in others}
    for li, pl in enumerate(model.passport_layers):
        A = passport_map(pl)
        _, Sv, Vh = torch.linalg.svd(A, full_matrices=False)
        r = int((Sv > float(Sv.max()) * max(A.shape) * torch.finfo(torch.float64).eps).sum())
        Vr = Vh[:r]
        shape = theta[f"p{li}_gamma"].shape
        cin, hw = shape[1], shape[2] * shape[3]
        B = torch.zeros(cin * hw, cin, dtype=torch.float64)
        for c in range(cin):
            B[c * hw:(c + 1) * hw, c] = 1.0 / math.sqrt(hw)
        U, cosv, _ = torch.linalg.svd(B.T @ Vr.T, full_matrices=False)
        for t, acc in ((theta, tot), *((others[name], tots[name]) for name in others)):
            for kind in ("gamma", "beta"):
                x = t[f"p{li}_{kind}"].reshape(-1).double()
                mu = U.T @ (B.T @ x)
                px = Vr.T @ (Vr @ x)
                acc["means"] += float(mu @ mu)
                acc["means_cos_gt_0.9"] += float((mu[cosv > 0.9] ** 2).sum())
                acc["row"] += float(px @ px)
                acc["row_in_C"] += float((B.T @ px).pow(2).sum())
        layers.append({"layer": li, "cin": cin, "cout": int(A.shape[0]), "cos_min": float(cosv.min()),
                       "cos_median": float(cosv.median()), "cos_max": float(cosv.max()),
                       "n_cos_gt_0.9": int((cosv > 0.9).sum()),
                       **({"n_directions": int(cosv.numel())} if others else {})})
        del A, Sv, Vh, Vr
    out = {"per_layer": layers, "share_of_chan_mean_energy_cos_gt_0.9": tot["means_cos_gt_0.9"] / tot["means"],
           "share_of_rowspace_component_in_C": tot["row_in_C"] / tot["row"],
           "cos_max_all_layers": max(l["cos_max"] for l in layers)}
    if others:
        out["n_cos_gt_0.9_all_layers"] = sum(l["n_cos_gt_0.9"] for l in layers)
        out["n_directions_all_layers"] = sum(l["n_directions"] for l in layers)
        out["others"] = {name: {"share_of_chan_mean_energy_cos_gt_0.9": v["means_cos_gt_0.9"] / v["means"],
                                "share_of_rowspace_component_in_C": v["row_in_C"] / v["row"]}
                         for name, v in tots.items()}
    return out


def pick_shaping_seed(fresh: dict, beta: float, target: float, pool: int = UNLINK_SEED_POOL) -> tuple:
    gaps = [(abs(serial_z(shaped_start(fresh, beta, s))["serial_z_channel"] - target), s) for s in range(pool)]
    gap, seed = min(gaps)
    return seed, gap


def section_linkage(S: dict, X: Evaluator) -> dict:
    t0 = time.time()
    base, theta, model = X.ref("base"), S["theta"], S["model"]
    forged, _, _, _, fresh = forged64(S)
    fresh = {k: v.double() for k, v in fresh.items()}
    wf = wellformed_forgery(model, theta, fresh)[0]
    rm = wellformed_forgery(model, theta, fresh, rank_iters=RANKMATCH_ITERS)[0]
    sh = wellformed_forgery(model, theta, shaped_start(fresh, SHAPE_BETA, SHAPE_SEED), rank_iters=RANKMATCH_ITERS)[0]
    cands = {"fresh_draw": fresh, "nullspace_distribution_matched_float32_stored": as_float32(forged[DIST_MATCHED]),
             "nullspace_minimum_norm_float64": forged[0.0], "nullspace_wellformed": wf,
             "nullspace_rankmatched": rm, "nullspace_shaped": sh}
    stats = {name: {**linkage_stats(theta, t), **serial_z(t, 2), **serial_z(t, 3), "chanvar_z": chanvar_z(t),
                    "theta_sha256": key_of(t)["theta_sha256"]}
             for name, t in cands.items()}
    spec = model.passport_spec()
    null, align_null = [], {}
    for i in range(LINK_NULL):
        d = {k: v.double() for k, v in make_passport_tensors(spec, seed_bytes(S["entropy"], "linkage-null-%d" % i, 64)).items()}
        null.append({**linkage_stats(theta, d), **serial_z(d, 2), **serial_z(d, 3), "chanvar_z": chanvar_z(d)})
        if i < LINK_ALIGN_NULL:
            align_null["linkage-null-%d" % i] = d
    print("  linkage: shaped forgery z_cos %.2f, z_chan_means %.2f; generator draws max |z| %.2f, %.2f" % (
        stats["nullspace_shaped"]["z_cos"], stats["nullspace_shaped"]["z_chan_means"],
        max(abs(s["z_cos"]) for s in null), max(abs(s["z_chan_means"]) for s in null)), flush=True)

    def unlinked_row(name, beta, seed, desc, **extra):
        st = shaped_start(fresh, beta, seed)
        ul, diag = wellformed_forgery(model, theta, st, rank_iters=RANKMATCH_ITERS, unlink=True)
        ev = X.eval(ul, label="unlinked forgery, " + desc)
        row = compare(name, ev, base, "base", ul, theta, **diag, shape_beta=beta, shape_seed=seed, **key_of(ul))
        s = wellformedness(ul)
        s.pop("entry_stats")
        s["passes_screens"] = passes_screens(s)
        s.update(serial_z(ul, 2))
        row.update(screens=s, start_serial_z=serial_z(st), linkage=linkage_stats(theta, ul), **extra)
        return row

    unlinked = [unlinked_row("nullspace_unlinked_beta_%g" % beta, beta, SHAPE_SEED, "beta %g" % beta, label="%g" % beta)
                for beta in UNLINK_BETAS]
    r123 = unlinked_row("nullspace_unlinked_secant_seed_%d" % SHAPE_SEED, UNLINK_SECANT_BETA, SHAPE_SEED,
                        "secant beta %.6f, seed %d" % (UNLINK_SECANT_BETA, SHAPE_SEED), label="secant_seed%d" % SHAPE_SEED,
                        secant_from=[list(p) for p in UNLINK_SECANT_FROM])
    target = r123["start_serial_z"]["serial_z_channel"] - r123["screens"]["serial_z_channel"]
    seed, gap = pick_shaping_seed(fresh, UNLINK_SECANT_BETA, target)
    rpick = unlinked_row("nullspace_unlinked_secant_seed_picked", UNLINK_SECANT_BETA, seed,
                         "secant beta %.6f, picked seed %d" % (UNLINK_SECANT_BETA, seed), label="secant_seed_picked",
                         secant_from=[list(p) for p in UNLINK_SECANT_FROM],
                         seed_pick={"target": target, "abs_gap": gap})
    unlinked += [r123, rpick]
    nz = {k: np.array([s[k] for s in null]) for k in ("z_cos", "z_chan_means", "z_chan_row_means", "z_chan_col_means",
                                                       "serial_z_lag2_width", "serial_z_lag2_height",
                                                       "serial_z_lag2_channel", "serial_z_lag3_width",
                                                       "serial_z_lag3_height", "serial_z_lag3_channel", "chanvar_z")}
    return {"stats": stats,
            "generator_draws": null,
            "generator_draws_summary": {k: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                            "max_abs": float(np.abs(v).max())} for k, v in nz.items()},
            "z_critical_1pct": Z_C01,
            "channel_alignment": channel_alignment(model, theta, align_null),
            "unlinked": unlinked, "unlink_betas": list(UNLINK_BETAS), "unlink_secant_beta": UNLINK_SECANT_BETA,
            "seconds": time.time() - t0}


def table2(out: dict) -> dict:
    t = {}
    fw, ns, fam, fl, rs = (out.get(k) for k in ("forward", "nullspace", "family", "floor", "rescale"))
    if fw:
        t["forward"] = {"correct": fw["genuine"], "random_theta_0": next(
            {"acc_classifier": r["acc_classifier"], "acc_prototype": r["acc_prototype"]}
            for r in fw["rows"] if r["name"] == "random_theta_0"),
            "random": fw["random"], "zero": next({"acc_classifier": r["acc_classifier"], "acc_prototype": r["acc_prototype"]}
                                                 for r in fw["rows"] if r["name"] == "zero_theta"),
            "foreign": fw["foreign"], "chance": fw["chance"]}
    if fam:
        t["inverse_dimensions"] = {k: fam[k] for k in (
            "passport_entries", "numeric_rank_total", "nullspace_dim", "nullspace_fraction",
            "family_dim_eps0", "family_fraction_eps0", "worst_cond_number", "eps_bound_max_over_uniform_scales",
            "eps_bound_sup_a_ge_half")}
    if ns:
        get = {r["name"]: r for r in ns["rows"]}
        d64, d32 = get["nullspace_distribution_matched_float64"], get["nullspace_distribution_matched_float32_stored"]
        null64 = [r for r in ns["rows"] if r["name"].startswith("nullspace_") and r["storage"] == "float64"]
        null32 = [r for r in ns["rows"] if r["name"].startswith("nullspace_") and r["storage"] == "float32_stored"]
        t["inverse_nullspace"] = {
            "acc_classifier_original_forged": [ns["reference"]["acc_classifier"], d32["acc_classifier"]],
            "acc_prototype_original_forged": [ns["reference"]["acc_prototype"], d32["acc_prototype"]],
            "max_abs_dlogit_float64_kept": max(r["max_abs_dlogit"] for r in null64),
            "min_abs_dlogit_float64_kept": min(r["max_abs_dlogit"] for r in null64),
            "max_abs_dlogit_float32_stored": max(r["max_abs_dlogit"] for r in null32),
            "min_abs_dlogit_float32_stored": min(r["max_abs_dlogit"] for r in null32),
            "distribution_matched_float64_max_abs_dlogit": d64["max_abs_dlogit"],
            "distribution_matched_float32_stored_max_abs_dlogit": d32["max_abs_dlogit"],
            "distribution_matched_float32_stored_max_abs_daffine": max(d32["max_abs_dgamma"], d32["max_abs_dbeta"]),
            "cos_minimum_norm": get["nullspace_scale_0_float64"]["cos_theta_prime_theta"],
            "predictions_changed_any_nullspace_row": max(max(r["n_classifier_predictions_changed"],
                                                             r["n_prototype_predictions_changed"]) for r in null64 + null32),
            "float32_ulp_every_entry_max_abs_dlogit": get["float32_ulp_every_entry"]["max_abs_dlogit"],
            "float32_ulp_one_entry_max_abs_dlogit": get["float32_ulp_one_entry"]["max_abs_dlogit"],
            "cos_distribution_matched": d32["cos_theta_prime_theta"],
            "svd_setup_seconds_this_host": ns["svd_setup_seconds_this_host"],
            "per_identity_seconds_this_host": ns["per_identity_seconds_this_host"][DIST_MATCHED]}
    if fl:
        t["floor_float64"] = {"max_abs_dlogit": fl["float64"]["floor_max_abs_dlogit"],
                              "rerun_batch_37": fl["float64"]["rerun_batch_37"]["max_abs_dlogit"],
                              "rerun_batch_128": fl["float64"]["rerun_batch_128"]["max_abs_dlogit"],
                              "matrix_route": fl["float64"]["matrix_route"]["max_abs_dlogit"],
                              "logit_abs_max": fl["float64"]["reference"]["logit_abs_max"]}
        if "float32_cpu" in fl:
            t["floor_float32_cpu"] = {"rerun": fl["float32_cpu"]["rerun_floor_max_abs_dlogit"],
                                      "float32_vs_float64": fl["float32_cpu"]["float32_vs_float64_arithmetic"]["max_abs_dlogit"]}
    if rs:
        t["rescale_eps_impl"] = {r["name"]: {"max_abs_dlogit": r["max_abs_dlogit"],
                                             "predictions_changed": max(r["n_classifier_predictions_changed"],
                                                                        r["n_prototype_predictions_changed"])}
                                 for r in rs["rows"]}
    wf = out.get("wellformed")
    if wf:
        def network(r):
            return {"max_abs_dlogit": r["max_abs_dlogit"],
                    "predictions_changed": max(r["n_classifier_predictions_changed"],
                                               r["n_prototype_predictions_changed"]),
                    "max_abs_daffine": max(r["max_abs_dgamma"], r["max_abs_dbeta"]),
                    "cos_theta_prime_theta": r["cos_theta_prime_theta"]}
        t["wellformed"] = {**network(wf["row"]),
                           "screens": {k: {"outside_range": v["frac_outside_generator_range"],
                                           "on_lattice": v["frac_on_generator_lattice"], "ks": v["ks_vs_uniform"],
                                           **{f: v.get(f) for f in ("branch_sqrt_n_ks_max", "branches_over_ks_1pct",
                                                                    "serial_z_width", "serial_z_height",
                                                                    "serial_z_channel")},
                                           **{f: v[f] for f in ("branch_serial_z_max_abs_width",
                                                                "branch_serial_z_max_abs_height",
                                                                "branches_over_z_1pct_width",
                                                                "branches_over_z_1pct_height", "passes_screens")
                                              + EXTRA_SCREENS if f in v}}
                                       for k, v in wf["screens"].items()},
                           "ks_critical_1pct": wf["screens"]["genuine"]["ks_critical_1pct"]}
        if "row_rankmatched" in wf:
            t["wellformed"]["rankmatched"] = network(wf["row_rankmatched"])
            t["wellformed"]["z_critical_1pct"] = Z_C01
            t["wellformed"]["branch_ks_critical_1pct_sqrt_n"] = KS_C01
            al = wf["rowspace_alignment"]
            t["wellformed"]["rowspace_alignment"] = {
                "expected_share_independent": al["expected_rowspace_share_independent"],
                **{name: {"rowspace_energy_share": al[name]["rowspace_energy_share"],
                          "lag1_z_width_rowspace": al[name]["lag1_z_rowspace_component"]["width"],
                          "lag1_z_width_nullspace": al[name]["lag1_z_nullspace_component"]["width"]}
                   for name in ("genuine", "draw")}}
        if "row_shaped" in wf:
            sw = wf["shaped"]
            t["wellformed"]["shaped"] = {**network(wf["row_shaped"]), "beta": sw["beta"], "seed": sw["seed"],
                                         "by_beta": {"%g" % r["beta"]: {f: r[f] for f in (
                                             "serial_z_width", "serial_z_height", "serial_z_channel", "passes_screens")
                                             + EXTRA_SCREENS if f in r}
                                             for r in sw["by_beta"]},
                                         "sweep": sw["summary"]}
    lk = out.get("linkage")
    if lk:
        t["linkage"] = {"z_cos": {k: v["z_cos"] for k, v in lk["stats"].items()},
                        "z_chan_means": {k: v["z_chan_means"] for k, v in lk["stats"].items()},
                        "r_chan_means": {k: v["r_chan_means"] for k, v in lk["stats"].items()},
                        "generator_draws": lk["generator_draws_summary"], "n_generator_draws": len(lk["generator_draws"]),
                        "share_of_chan_mean_energy_cos_gt_0.9":
                            lk["channel_alignment"]["share_of_chan_mean_energy_cos_gt_0.9"],
                        "share_of_rowspace_component_in_C": lk["channel_alignment"]["share_of_rowspace_component_in_C"],
                        "unlinked": {r.get("label", "%g" % r["shape_beta"]): {
                            "max_abs_dlogit": r["max_abs_dlogit"],
                            "predictions_changed": max(r["n_classifier_predictions_changed"],
                                                       r["n_prototype_predictions_changed"]),
                            "max_abs_daffine": max(r["max_abs_dgamma"], r["max_abs_dbeta"]),
                            "z_cos": r["linkage"]["z_cos"], "z_chan_means": r["linkage"]["z_chan_means"],
                            "ks": r["screens"]["ks_vs_uniform"], "branches_over_ks_1pct": r["screens"]["branches_over_ks_1pct"],
                            "serial_z_width": r["screens"]["serial_z_width"],
                            "serial_z_height": r["screens"]["serial_z_height"],
                            "serial_z_channel": r["screens"]["serial_z_channel"],
                            "serial_z_lag2_width": r["screens"]["serial_z_lag2_width"],
                            "serial_z_lag2_height": r["screens"]["serial_z_lag2_height"],
                            "passes_screens": r["screens"]["passes_screens"],
                            "forced_shift_ratio_max": r["forced_shift_ratio_max"],
                            **{f: r["screens"][f] for f in EXTRA_SCREENS if f in r["screens"]},
                            **{f: r["linkage"][f] for f in ("z_chan_row_means", "z_chan_col_means") if f in r["linkage"]},
                            **({"beta": r["shape_beta"], "seed": r["shape_seed"]} if "label" in r else {}),
                            **({"branches_over_z_1pct_width": r["screens"]["branches_over_z_1pct_width"],
                                "branches_over_z_1pct_height": r["screens"]["branches_over_z_1pct_height"]}
                               if "label" in r else {})}
                            for r in lk["unlinked"]}}
        extra = ("z_chan_row_means", "z_chan_col_means") + EXTRA_SCREENS
        if all(f in v for v in lk["stats"].values() for f in extra):
            t["linkage"]["passport_alone_and_fine_links"] = {k: {f: v[f] for f in extra} for k, v in lk["stats"].items()}
        ca = lk["channel_alignment"]
        if "others" in ca:
            o = list(ca["others"].values())
            t["linkage"]["channel_alignment_generator_draws"] = {
                "n": len(o), **{f: {"min": min(x[f] for x in o), "max": max(x[f] for x in o),
                                    "mean": float(np.mean([x[f] for x in o]))}
                                for f in ("share_of_chan_mean_energy_cos_gt_0.9", "share_of_rowspace_component_in_C")}}
            t["linkage"]["n_principal_directions_cos_gt_0.9"] = [ca["n_cos_gt_0.9_all_layers"], ca["n_directions_all_layers"]]
    return t


MERGE_KEYS = ("checkpoint_sha256", "n_test", "entropy", "rescale_seed", "threads", "batch_size",
              "dtype", "eps_impl", "environment")


def merge(paths: list, out_path: str, provenance: str = None) -> int:
    out, meta, sig0 = {}, None, None
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        m = d["meta"]
        sig = [m.get(k) for k in MERGE_KEYS]
        if meta is None:
            meta, sig0 = copy.deepcopy(m), sig
            meta["seconds_by_section"], meta["merged_from"] = {}, []
            meta.pop("sections_this_invocation", None)
            meta.pop("seconds_this_invocation", None)
        elif sig != sig0:
            raise SystemExit("%s was written with other settings (%s); not merged" % (p, ", ".join(MERGE_KEYS)))
        names = [s for s in SECTIONS if s in d]
        for s in names:
            if s in out:
                raise SystemExit("section %s appears in more than one input" % s)
            out[s] = d[s]
            meta["seconds_by_section"][s] = m.get("seconds_by_section", {}).get(s)
        meta.setdefault("foreign_checkpoint_sha256", {}).update(m.get("foreign_checkpoint_sha256", {}))
        meta["merged_from"].append({"file": rel(p), "sections": names,
                                    "seconds_this_invocation": m.get("seconds_this_invocation")})
    if not meta["foreign_checkpoint_sha256"]:
        del meta["foreign_checkpoint_sha256"]
    meta.pop("provenance", None)
    if provenance:
        meta["provenance"] = provenance
    out["meta"] = meta
    out = {k: out[k] for k in ("meta", *SECTIONS) if k in out}
    out["table2"] = table2(out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=jsonable)
    print("merged %d files into %s: %s" % (len(paths), out_path, ", ".join(k for k in SECTIONS if k in out)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(REPO, "checkpoints", "cifar100_client0.pt"))
    ap.add_argument("--data-root", default=os.path.join(REPO, "data"))
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--foreign", default="siblings", choices=("siblings", "stored"),
                    help="foreign passports: sibling <tag>_client<i>.pt files and any stored in the "
                         "checkpoint, or only those stored in the checkpoint")
    ap.add_argument("--entropy", default=DEFAULT_ENTROPY, help="hex seed of the forgeries and random passports")
    ap.add_argument("--rescale-seed", type=int, default=0, help="seed of the per-branch scale draws")
    ap.add_argument("--batch-size", type=int, default=250)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=0, help="first N test images only (smoke tests); 0 = all")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "passport_equivalence.json"))
    ap.add_argument("--merge", default=None,
                    help="comma-separated output files of --sections runs with equal settings: combine "
                         "their sections into --out and evaluate nothing")
    ap.add_argument("--provenance", default=None, help="with --merge: a provenance note stored in meta")
    a = ap.parse_args()
    if a.merge:
        return merge([p.strip() for p in a.merge.split(",") if p.strip()], a.out, a.provenance)
    names = [s.strip() for s in a.sections.split(",") if s.strip()]
    if any(s not in SECTIONS for s in names):
        raise SystemExit("unknown section in %r; choose from %s" % (a.sections, ",".join(SECTIONS)))
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    t_all = time.time()

    S = load_subject(a.checkpoint, a.data_root)
    if a.n_test:
        S["xs"], S["ys"] = S["xs"][:a.n_test], S["ys"][:a.n_test]
    S["entropy"] = bytes.fromhex(a.entropy)
    S["foreign_sha256"] = {}
    configure(S["model"], S["theta"], EPS_IMPL, S["alpha"], S["E"])
    keys = sorted(S["theta"])
    rng = np.random.default_rng(a.rescale_seed)
    draws = [per_branch_draw(keys, lo, hi, rng) for lo, hi in NET_DRAWS]
    X = Evaluator(S, a.batch_size)

    ck_sha = sha256_file(a.checkpoint)
    out = {}
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as fh:
            out = json.load(fh)
        old = out.get("meta", {})
        if (old.get("checkpoint_sha256"), old.get("n_test"), old.get("entropy"), old.get("rescale_seed")) != \
                (ck_sha, int(len(S["ys"])), a.entropy, a.rescale_seed):
            raise SystemExit("%s holds results of another checkpoint, test set or seed; choose another --out" % a.out)
    meta = out.get("meta", {})
    meta.pop("merged_from", None)
    meta.pop("provenance", None)
    meta.update({
        "environment": environment(), "threads": a.threads, "dtype": "float64",
        "checkpoint": rel(a.checkpoint),
        "checkpoint_sha256": ck_sha,
        "dataset": S["dataset"], "n_test": int(len(S["ys"])), "batch_size": a.batch_size,
        "num_classes": S["K"], "embed_dim": S["D"], "width": S["W"],
        "alpha_dyn_checkpoint": S["alpha"], "E_checkpoint": S["E"],
        "entropy": a.entropy, "rescale_seed": a.rescale_seed,
        "per_branch_draws_network": draws, "eps_impl": EPS_IMPL,
        "sections_this_invocation": names})
    meta.setdefault("seconds_by_section", {})
    run_section = {"family": lambda: section_family(S),
                   "affine": lambda: section_affine(S, np.random.default_rng([a.rescale_seed, 1])),
                   "floor": lambda: section_floor(S, X), "rescale": lambda: section_rescale(S, X, draws),
                   "nullspace": lambda: section_nullspace(S, X), "forward": lambda: section_forward(S, X, a),
                   "eps0": lambda: section_eps0(S, X, draws), "dynamic": lambda: section_dynamic(S, X, draws),
                   "wellformed": lambda: section_wellformed(S, X), "linkage": lambda: section_linkage(S, X)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for name in names:
        print("[%s]" % name, flush=True)
        out[name] = run_section[name]()
        meta["seconds_by_section"][name] = out[name].get("seconds")
        if S["foreign_sha256"]:
            meta["foreign_checkpoint_sha256"] = S["foreign_sha256"]
        meta["seconds_this_invocation"] = time.time() - t_all
        out["meta"] = meta
        out = {k: out[k] for k in ("meta", *SECTIONS) if k in out}
        out["table2"] = table2(out)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=jsonable)
    print("\nwrote %s  (%.1f s)" % (a.out, meta["seconds_this_invocation"]))
    print(json.dumps(out["table2"], indent=1, default=jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
