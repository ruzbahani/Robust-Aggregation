from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

import innovation_replay as IR
import protocol_eval as P
from passportfl.baselines import BASELINES

ARCHIVES = IR.ARCHIVES
RESULTS = IR.RESULTS
EPS = 1e-12
SCHEMA = {"none": "std-baseline-v3", "within": "std-dagmm-within-v1",
          "cross": "std-baseline-v2"}
DEFAULT_OUT = {"none": os.path.join(RESULTS, "std_baseline.json"),
               "within": os.path.join(RESULTS, "std_dagmm_within.json"),
               "cross": None}

CONFIGS = {
    "cifar100_d05": (lambda s: ("arch_cifar100_s%d_c10_r35_s%d.npz" % (s, s)) if s < 3
                     else ("arch_cifar100_s%d_c10_r35_d0.5_s%d.npz" % (s, s)),
                     "cifar100_s%d", (0, 1, 2, 3, 4), (2, 3, 4)),
    "cifar100_d01": (lambda s: "arch_cifar100_d01_s%d_c10_r35_d0.1_s%d.npz" % (s, s),
                     "cifar100_d01_s%d", (0, 1, 2), (2,)),
    "cifar10": (lambda s: "arch_cifar10_s%d_c10_r35_d0.5_s%d.npz" % (s, s),
                "cifar10_s%d", (0, 1, 2), (2,)),
}
CAL_SEEDS = (0, 1)
TRAIN_ROUNDS = (12, 35)
WITHIN_ROUNDS = (2, IR.ATTACK_START - 1)
WITHIN_STEPS = 3000
PCA_Q = 10
REPRO_RULES = ("norm_bound", "krum", "median_dev", "trimmed_mean_dev")
UNTRAINED = ("std_two_sided", "std_low") + REPRO_RULES
DAGMM = {"hidden": (60, 30, 10), "latent": 1, "n_gmm": 4, "est_hidden": 10, "dropout": 0.5,
         "lambda_energy": 0.1, "lambda_cov": 0.005, "lr": 1e-4, "epochs": 200, "batch": 32,
         "cov_eps": 1e-6, "rel_eps": 1e-12, "init_seeds": (0, 1, 2),
         "recon_loss": "per-sample squared L2 error ||x - x'||^2, summed over entries, "
                       "mean over the batch (Zong et al. 2018, Eq. 7)"}
REVISION = {
    "cross_federation_design": "one model per configuration, trained on the honest "
                               "innovations of every client in rounds 12-35 of CAL seeds 0 "
                               "and 1, scored on every seed; headline over the other seeds",
    "why_invalid": "each seed is an independently trained federation (own initialisation, own "
                   "client models), so prototype spaces share no coordinates and a density "
                   "model of one federation's raw innovations does not transfer to another. "
                   "Honest rows only, rounds 12-35, seeds 0-2: the top 10 principal components "
                   "fitted on one seed explain 0.018-0.066 of another seed's variance (CIFAR-100 "
                   "0.5 and CIFAR-10), against 0.72-0.74 for odd rounds of the same seed fitted "
                   "on its even rounds; cosines between seeds' mean innovations are -0.021 to "
                   "0.035; "
                   "on the held-out honest rows, predicting zero (0.995 / 0.939 / 0.972 per "
                   "entry) beats predicting the CAL mean (1.051 / 1.017 / 1.036). The low "
                   "held-out reconstruction quality was first attributed to memorisation; "
                   "it is the absence of shared coordinates.",
    "inspected": "the attacked AUCs of both runs had been computed and read before the "
                 "within-federation design was fixed; that design was fixed on honest data "
                 "only (rows of the seed's own rounds 2-11 and 12-35) before any attacked "
                 "payload was scored with it; afterwards a code-path smoke test with 8 Adam "
                 "steps (CIFAR-10) printed AUCs of an essentially untrained model, which were "
                 "not used for anything",
    "v1": {"schema": "std-baseline-v1",
           "recon_loss": "((x - x')^2).mean() over batch and entries, i.e. the per-sample "
                         "squared L2 error divided by D (12,800 at CIFAR-100, 1,280 at "
                         "CIFAR-10)",
           "train_recon_mse_per_entry_range": [1.14, 1.23],
           "cal_mean_predictor_mse_per_entry": {"cifar100_d05": 0.943, "cifar10": 0.932},
           "heldout_auc_cifar100_d05": {
               "dagmm": {"global_echo": 1.0, "peer_copy": 0.500, "lagged_self": 0.572,
                         "ar2_replay": 0.881, "noise": 0.466, "shrink": 1.0},
               "std_dagmm": {"global_echo": 1.0, "peer_copy": 0.500, "lagged_self": 0.591,
                             "ar2_replay": 0.931, "noise": 0.495, "shrink": 1.0}}},
    "v2": {"schema": "std-baseline-v2", "recon_loss": "as DAGMM.recon_loss; 200 epochs on 480 "
                                                      "rows (3,000 Adam steps)",
           "code_sha256": "0984bda2d0c3f7d053e142c0668691ee1c4db06271860d843d90423a7cc98b30",
           "reproduce": "--dagmm cross --out <file>",
           "train_recon_mse_per_entry_range": [0.78, 0.88],
           "heldout_honest_recon_mse_per_entry_range": [1.08, 1.65],
           "heldout_auc": {
               "cifar100_d05": {
                   "dagmm": {"global_echo": 1.0, "peer_copy": 0.500, "lagged_self": 0.602,
                             "ar2_replay": 0.874, "noise": 0.547, "shrink": 1.0},
                   "std_dagmm": {"global_echo": 1.0, "peer_copy": 0.500, "lagged_self": 0.597,
                                 "ar2_replay": 0.906, "noise": 0.539, "shrink": 1.0}},
               "cifar100_d01": {
                   "dagmm": {"global_echo": 1.0, "peer_copy": 0.500, "lagged_self": 0.576,
                             "ar2_replay": 0.439, "noise": 0.542, "shrink": 0.9998},
                   "std_dagmm": {"global_echo": 1.0, "peer_copy": 0.500, "lagged_self": 0.469,
                                 "ar2_replay": 0.446, "noise": 0.516, "shrink": 1.0}},
               "cifar10": {
                   "dagmm": {"global_echo": 1.0, "peer_copy": 0.500, "lagged_self": 0.503,
                             "ar2_replay": 0.654, "noise": 0.478, "shrink": 1.0},
                   "std_dagmm": {"global_echo": 1.0, "peer_copy": 0.500, "lagged_self": 0.519,
                                 "ar2_replay": 0.736, "noise": 0.481, "shrink": 1.0}}}},
    "within_design_budget": "fixed at 3,000 Adam steps after an honest-only pilot on CAL seed "
                            "0 (rounds 2-11 for training, 12-35 for the check): at 200 epochs "
                            "(800 steps) the reconstruction error per entry on the training "
                            "rows was 0.995 (CIFAR-100 0.5) and 1.023 (CIFAR-10) against 0.871 "
                            "and 0.791 for the training mean; at 3,000 steps 0.847 and 0.800",
}


def _stds(innov):
    return {i: float(np.std(np.asarray(innov[i], dtype=np.float64).ravel())) for i in sorted(innov)}


def std_two_sided(innov):
    sd = _stds(innov)
    med = float(np.median([v for v in sd.values() if v > EPS])) if sd else 0.0
    return {i: abs(np.log((v + EPS) / (med + EPS))) if med > EPS else float("nan")
            for i, v in sd.items()}


def std_low(innov):
    sd = _stds(innov)
    med = float(np.median([v for v in sd.values() if v > EPS])) if sd else 0.0
    return {i: -np.log((v + EPS) / (med + EPS)) if med > EPS else float("nan")
            for i, v in sd.items()}


def untrained_scorers() -> dict:
    scorers = {"std_two_sided": std_two_sided, "std_low": std_low}
    scorers.update({k: BASELINES[k] for k in REPRO_RULES})
    return scorers


def build_dagmm(dim: int, use_std: bool):
    import torch
    import torch.nn as nn

    h = DAGMM["hidden"]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            enc, dec, prev = [], [], dim
            for w in h:
                enc += [nn.Linear(prev, w), nn.Tanh()]
                prev = w
            enc += [nn.Linear(prev, DAGMM["latent"])]
            prev = DAGMM["latent"]
            for w in reversed(h):
                dec += [nn.Linear(prev, w), nn.Tanh()]
                prev = w
            dec += [nn.Linear(prev, dim)]
            self.enc, self.dec = nn.Sequential(*enc), nn.Sequential(*dec)
            zdim = DAGMM["latent"] + 2 + (1 if use_std else 0)
            self.est = nn.Sequential(nn.Linear(zdim, DAGMM["est_hidden"]), nn.Tanh(),
                                     nn.Dropout(DAGMM["dropout"]),
                                     nn.Linear(DAGMM["est_hidden"], DAGMM["n_gmm"]),
                                     nn.Softmax(dim=1))
            self.use_std = use_std

        def forward(self, x):
            zc = self.enc(x)
            xr = self.dec(zc)
            rel = (x - xr).norm(dim=1) / (x.norm(dim=1) + DAGMM["rel_eps"])
            cos = nn.functional.cosine_similarity(x, xr, dim=1)
            feats = [zc, rel[:, None], cos[:, None]]
            if self.use_std:
                feats.append(x.std(dim=1, unbiased=False)[:, None])
            z = torch.cat(feats, dim=1)
            return xr, z, self.est(z)

    return Net()


def gmm_params(z, gamma):
    import torch
    nk = gamma.sum(0)
    phi = nk / gamma.shape[0]
    mu = (gamma.T @ z) / nk[:, None]
    zc = z[:, None, :] - mu[None]
    cov = torch.einsum("nk,nki,nkj->kij", gamma, zc, zc) / nk[:, None, None]
    return phi, mu, cov


def gmm_energy(z, phi, mu, cov):
    import torch
    d = z.shape[1]
    cov = cov + DAGMM["cov_eps"] * torch.eye(d, dtype=z.dtype)
    inv = torch.linalg.inv(cov)
    _, logdet = torch.linalg.slogdet(2.0 * np.pi * cov)
    zc = z[:, None, :] - mu[None]
    maha = torch.einsum("nki,kij,nkj->nk", zc, inv, zc)
    logp = torch.log(phi.clamp_min(1e-30))[None] - 0.5 * maha - 0.5 * logdet[None]
    pen = (1.0 / torch.diagonal(cov, dim1=1, dim2=2)).sum()
    return -torch.logsumexp(logp, dim=1), pen


def train_dagmm(X: np.ndarray, use_std: bool, seed: int, epochs: int | None = None) -> dict:
    import torch
    epochs = DAGMM["epochs"] if epochs is None else int(epochs)
    torch.manual_seed(seed)
    net = build_dagmm(X.shape[1], use_std)
    opt = torch.optim.Adam(net.parameters(), lr=DAGMM["lr"])
    Xt = torch.from_numpy(X.astype(np.float32))
    gen = torch.Generator().manual_seed(seed)
    trace, steps = [], 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), generator=gen)
        tot = []
        for b in range(0, len(Xt), DAGMM["batch"]):
            xb = Xt[perm[b:b + DAGMM["batch"]]]
            if len(xb) < 2:
                continue
            xr, z, gamma = net(xb)
            phi, mu, cov = gmm_params(z, gamma)
            e, pen = gmm_energy(z, phi, mu, cov)
            loss = (((xb - xr) ** 2).sum(dim=1).mean() + DAGMM["lambda_energy"] * e.mean()
                    + DAGMM["lambda_cov"] * pen)
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
            tot.append(loss.item())
        if ep % 20 == 19 or ep == 0:
            trace.append({"epoch": ep + 1, "loss": float(np.mean(tot))})
    net.eval()
    with torch.no_grad():
        xr, z, gamma = net(Xt)
        z64, g64 = z.double(), gamma.double()
        phi, mu, cov = gmm_params(z64, g64)
        e_train, _ = gmm_energy(z64, phi, mu, cov)
        rec = float(((Xt - xr) ** 2).mean())
    return {"net": net, "phi": phi, "mu": mu, "cov": cov, "trace": trace, "steps": steps,
            "train_energy_median": float(e_train.median()), "train_recon_mse": rec}


def dagmm_scorer(state: dict, scale: float):
    import torch

    def score(innov):
        cids = sorted(innov)
        X = np.stack([np.asarray(innov[i], dtype=np.float64).ravel() / scale for i in cids])
        with torch.no_grad():
            _, z, _ = state["net"](torch.from_numpy(X.astype(np.float32)))
            e, _ = gmm_energy(z.double(), state["phi"], state["mu"], state["cov"])
        return {i: float(v) for i, v in zip(cids, e.numpy())}
    return score


def recon_mse(state: dict, X: np.ndarray) -> float:
    import torch
    with torch.no_grad():
        Xt = torch.from_numpy(X.astype(np.float32))
        xr, _, _ = state["net"](Xt)
        return float(((Xt - xr) ** 2).mean())


def innov_rows(arch: dict, rounds) -> np.ndarray:
    C, cbar = arch["C"], arch["cbar"]
    return np.stack([(C[i][r] - cbar[r - 1]).ravel() for r in rounds for i in sorted(C)])


def cal_matrix(arch_name, seeds=CAL_SEEDS) -> np.ndarray:
    rows = []
    for s in seeds:
        arch = P.load_archive(os.path.join(ARCHIVES, arch_name(s)))
        rows.append(innov_rows(arch, range(TRAIN_ROUNDS[0], TRAIN_ROUNDS[1] + 1)))
    return np.concatenate(rows)


def pca_explained(train: np.ndarray, test: np.ndarray, q: int = PCA_Q) -> float:
    m = train.mean(axis=0)
    _, _, vt = np.linalg.svd(train - m, full_matrices=False)
    xt = test - m
    proj = (xt @ vt[:q].T) @ vt[:q]
    return float(1.0 - ((xt - proj) ** 2).sum() / (xt ** 2).sum())


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def summarise(per_seed: dict, seeds, heldout=None) -> dict:
    v = [per_seed[s] for s in seeds]
    out = {"per_seed": {str(s): per_seed[s] for s in seeds}, "mean": float(np.mean(v)),
           "sd_ddof0": float(np.std(v)), "sd_ddof1": float(np.std(v, ddof=1)) if len(v) > 1 else None}
    if heldout is not None:
        out["heldout_mean"] = float(np.mean([per_seed[s] for s in heldout]))
    return out


def replay_seed(arch: dict, seed: int, tag: str, scorers: dict) -> dict:
    ref_path = os.path.join(RESULTS, "innovation_%s.json" % tag)
    ref = (json.load(open(ref_path, encoding="utf-8"))["results"]
           if os.path.exists(ref_path) else None)
    auc, counts, repro, repro_counts = {}, {}, {k: 0.0 for k in REPRO_RULES}, True
    for att in P.ATTACK_ORDER:
        m = IR.replay_matrices(arch, att, seed, scorers)
        auc[att] = {}
        for k, mat in m["scores"].items():
            got = IR.auc_of(mat)
            auc[att][k] = got["auc"]
            counts[(att, k)] = (got["n_pos"], got["n_neg"])
            if k in REPRO_RULES and ref is not None:
                repro[k] = max(repro[k], abs(got["auc"] - ref[att][k]["auc"]))
                repro_counts &= (got["n_pos"] == ref[att][k]["n_pos"]
                                 and got["n_neg"] == ref[att][k]["n_neg"])
    nonfinite = {"%s/%s" % (att, k): c for (att, k), c in counts.items()
                 if c != counts[(att, "norm_bound")]}
    return {"auc": auc, "repro": repro if ref is not None else None,
            "repro_counts": repro_counts if ref is not None else None, "nonfinite": nonfinite}


def within_task(task) -> dict:
    cfg, seed, threads = task
    import torch
    torch.set_num_threads(max(1, int(threads)))
    t0 = time.time()
    arch_name, tag, seeds, _ = CONFIGS[cfg]
    arch = P.load_archive(os.path.join(ARCHIVES, arch_name(seed)))
    _, rounds = P.archive_shape(arch)
    pre_rounds = range(WITHIN_ROUNDS[0], WITHIN_ROUNDS[1] + 1)
    post_rounds = range(IR.EVAL_FROM, rounds + 1)
    assert all((r - 1) in arch["cbar"] for r in pre_rounds)
    X = innov_rows(arch, pre_rounds)
    scale = float(np.sqrt(np.mean(X ** 2)))
    Xs, Ps = X / scale, innov_rows(arch, post_rounds) / scale
    other = seeds[(seeds.index(seed) + 1) % len(seeds)]
    arch_o = P.load_archive(os.path.join(ARCHIVES, arch_name(other)))
    Os = innov_rows(arch_o, range(IR.EVAL_FROM, P.archive_shape(arch_o)[1] + 1)) / scale
    del arch_o
    mean = Xs.mean(axis=0)
    rec = {"rounds": [WITHIN_ROUNDS[0], WITHIN_ROUNDS[1]], "samples": int(Xs.shape[0]),
           "dim": int(Xs.shape[1]), "honest_scored_rows": int(Ps.shape[0]),
           "input_scale_rms": scale,
           "baselines": {"train_mean_predictor_mse_train": float(((Xs - mean) ** 2).mean()),
                         "train_mean_predictor_mse_scored": float(((Ps - mean) ** 2).mean()),
                         "zero_predictor_mse_scored": float((Ps ** 2).mean()),
                         "own_mean_predictor_mse_scored": float(((Ps - Ps.mean(axis=0)) ** 2).mean())},
           "transfer": {"pca_q": PCA_Q,
                        "pca_explained_same_seed": pca_explained(Xs, Ps),
                        "pca_explained_other_seed": pca_explained(Xs, Os),
                        "other_seed": int(other),
                        "cos_mean_same_seed": cosine(mean, Ps.mean(axis=0)),
                        "cos_mean_other_seed": cosine(mean, Os.mean(axis=0))},
           "models": {}}
    del Os
    batches = -(-len(Xs) // DAGMM["batch"])
    epochs = WITHIN_STEPS // batches
    scorers = untrained_scorers()
    for variant, use_std in (("dagmm", False), ("std_dagmm", True)):
        for ms in DAGMM["init_seeds"]:
            t1 = time.time()
            st = train_dagmm(Xs, use_std, ms, epochs=epochs)
            key = "%s_m%d" % (variant, ms)
            scorers[key] = dagmm_scorer(st, scale)
            ps = recon_mse(st, Ps)
            b = rec["baselines"]
            rec["models"][key] = {
                "epochs": epochs, "steps": st["steps"], "loss_trace": st["trace"],
                "train_energy_median": st["train_energy_median"],
                "train_recon_mse": st["train_recon_mse"], "scored_honest_recon_mse": ps,
                "fits_train": st["train_recon_mse"] < b["train_mean_predictor_mse_train"],
                "beats_train_mean_on_scored": ps < b["train_mean_predictor_mse_scored"],
                "beats_zero_on_scored": ps < b["zero_predictor_mse_scored"],
                "train_s": round(time.time() - t1, 1)}
    rep = replay_seed(arch, seed, tag % seed, scorers)
    return {"cfg": cfg, "seed": seed, "training": rec, **rep, "wall_s": round(time.time() - t0, 1)}


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for f in (os.path.abspath(__file__), IR.__file__, P.__file__,
              os.path.join(HERE, "passportfl", "baselines.py")):
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def meta_common(design: str, cfgs, a) -> dict:
    cmd = "python experiments/std_baseline_eval.py"
    if cfgs != list(CONFIGS):
        cmd += " --configs " + ",".join(cfgs)
    if design != "none":
        cmd += " --dagmm " + design
    return {
        "schema": SCHEMA[design],
        "command": cmd,
        "code_sha256": code_fingerprint()
    }


def print_table(results: dict, cols, field: str) -> None:
    for cfg, res in results.items():
        print("\n%s: %s over seeds %s" % (cfg, field, res["seeds"]))
        print("%-12s" % "attack" + "".join("%15s" % c for c in cols))
        for att in P.ATTACK_ORDER:
            row = res["attacks"][att]
            print("%-12s" % att + "".join("%15.3f" % row[c][field] for c in cols))
        print("replay vs innovation_eval: %s, counts %s"
              % ({k: ("%.2g" % v if v is not None else "n.a.")
                  for k, v in res["checks"]["innovation_eval_max_abs_auc_diff"].items()},
                 res["checks"]["innovation_eval_counts_match"]))


def run_within(a, cfgs, t0) -> dict:
    threads = 1 if a.threads is None else a.threads
    tasks = [(cfg, s, threads) for cfg in cfgs for s in CONFIGS[cfg][2]]
    tasks.sort(key=lambda t: t[0] == "cifar10")
    if a.workers <= 1:
        done = []
        for t in tasks:
            done.append(within_task(t))
            print("%s seed %d done (%.0fs)" % (t[0], t[1], time.time() - t0), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            done = []
            for t, d in zip(tasks, ex.map(within_task, tasks)):
                done.append(d)
                print("%s seed %d done (%.0fs in its worker; %.0fs elapsed)"
                      % (t[0], t[1], d["wall_s"], time.time() - t0), flush=True)
    by = {(d["cfg"], d["seed"]): d for d in done}
    results, training = {}, {}
    for cfg in cfgs:
        seeds = CONFIGS[cfg][2]
        res = {"seeds": list(seeds), "attacks": {}}
        for att in P.ATTACK_ORDER:
            row = {k: summarise({s: by[(cfg, s)]["auc"][att][k] for s in seeds}, seeds)
                   for k in UNTRAINED}
            for variant in ("dagmm", "std_dagmm"):
                keys = ["%s_m%d" % (variant, ms) for ms in DAGMM["init_seeds"]]
                per = {s: float(np.mean([by[(cfg, s)]["auc"][att][k] for k in keys]))
                       for s in seeds}
                ent = summarise(per, seeds)
                ent["per_init_mean"] = [float(np.mean([by[(cfg, s)]["auc"][att][k]
                                                       for s in seeds])) for k in keys]
                row[variant] = ent
            res["attacks"][att] = row
        reps = [by[(cfg, s)] for s in seeds]
        res["checks"] = {
            "innovation_eval_max_abs_auc_diff": {
                k: max((r["repro"][k] for r in reps if r["repro"] is not None), default=None)
                for k in REPRO_RULES},
            "innovation_eval_seeds_compared": [r["seed"] for r in reps if r["repro"] is not None],
            "innovation_eval_counts_match": all(r["repro_counts"] for r in reps
                                                if r["repro_counts"] is not None),
            "scores_with_dropped_nonfinite_values": {
                "seed%d/%s" % (r["seed"], k): v for r in reps for k, v in r["nonfinite"].items()}}
        results[cfg] = res
        training[cfg] = {str(s): dict(by[(cfg, s)]["training"], wall_s=by[(cfg, s)]["wall_s"])
                         for s in seeds}
    from passportfl.reproducibility import environment
    meta = meta_common("within", cfgs, a)
    meta["command"] += " --workers %d" % a.workers
    meta["dagmm"] = {
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in DAGMM.items() if k != "epochs"},
        "steps": WITHIN_STEPS,
        "threads_per_worker": threads}
    meta["revision"] = REVISION
    meta["environment"] = environment()
    meta["wall_time_s"] = round(time.time() - t0, 1)
    return {"meta": meta, "configs": results, "dagmm_training": training}


def run_pooled(a, cfgs, t0, design: str) -> dict:
    use_dagmm = design == "cross"
    if use_dagmm:
        import torch
        torch.set_num_threads(max(1, min(4, 4 if a.threads is None else a.threads)))
    results, training = {}, {}
    for cfg in cfgs:
        arch_name, tag, seeds, heldout = CONFIGS[cfg]
        scorers = untrained_scorers()
        if use_dagmm:
            X = cal_matrix(arch_name)
            scale = float(np.sqrt(np.mean(X ** 2)))
            Xs = X / scale
            Xh = cal_matrix(arch_name, heldout) / scale
            cal_mean = Xs.mean(axis=0)
            base = {"cal_mean_predictor_mse_cal": float(((Xs - cal_mean) ** 2).mean()),
                    "cal_mean_predictor_mse_heldout": float(((Xh - cal_mean) ** 2).mean()),
                    "zero_predictor_mse_cal": float((Xs ** 2).mean()),
                    "zero_predictor_mse_heldout": float((Xh ** 2).mean())}
            training[cfg] = {"cal_seeds": list(CAL_SEEDS), "rounds": list(TRAIN_ROUNDS),
                             "samples": int(X.shape[0]), "dim": int(X.shape[1]),
                             "heldout_honest_samples": int(Xh.shape[0]),
                             "input_scale_rms": scale, "baselines": base, "models": {}}
            for variant, use_std in (("dagmm", False), ("std_dagmm", True)):
                for ms in DAGMM["init_seeds"]:
                    t1 = time.time()
                    st = train_dagmm(Xs, use_std, ms)
                    key = "%s_m%d" % (variant, ms)
                    scorers[key] = dagmm_scorer(st, scale)
                    ho = recon_mse(st, Xh)
                    training[cfg]["models"][key] = {
                        "loss_trace": st["trace"], "train_energy_median": st["train_energy_median"],
                        "train_recon_mse": st["train_recon_mse"],
                        "heldout_honest_recon_mse": ho,
                        "fits_cal": st["train_recon_mse"] < base["cal_mean_predictor_mse_cal"],
                        "fits_heldout": ho < base["cal_mean_predictor_mse_heldout"],
                        "train_s": round(time.time() - t1, 1)}
                    print("%s %s trained in %.0fs (final loss %.4f; recon per entry CAL %.4f "
                          "vs mean %.4f, held-out %.4f vs %.4f)"
                          % (cfg, key, time.time() - t1, st["trace"][-1]["loss"],
                             st["train_recon_mse"], base["cal_mean_predictor_mse_cal"], ho,
                             base["cal_mean_predictor_mse_heldout"]), flush=True)
            del Xh

        auc = {att: {k: {} for k in scorers} for att in P.ATTACK_ORDER}
        repro, repro_counts, nonfinite = {k: 0.0 for k in REPRO_RULES}, True, {}
        for s in seeds:
            arch = P.load_archive(os.path.join(ARCHIVES, arch_name(s)))
            rep = replay_seed(arch, s, tag % s, scorers)
            for att in P.ATTACK_ORDER:
                for k in scorers:
                    auc[att][k][s] = rep["auc"][att][k]
            if rep["repro"] is not None:
                repro = {k: max(repro[k], rep["repro"][k]) for k in REPRO_RULES}
                repro_counts &= rep["repro_counts"]
            nonfinite.update({"%s/seed%d" % (k, s): v for k, v in rep["nonfinite"].items()})
            print("%s seed %d replayed (%.0fs)" % (cfg, s, time.time() - t0), flush=True)

        res = {"seeds": list(seeds), "attacks": {}}
        if use_dagmm:
            res["heldout_seeds_for_dagmm"] = list(heldout)
        for att in P.ATTACK_ORDER:
            res["attacks"][att] = {}
            for k in UNTRAINED:
                res["attacks"][att][k] = summarise(auc[att][k], seeds,
                                                   heldout if use_dagmm else None)
            if use_dagmm:
                for variant in ("dagmm", "std_dagmm"):
                    per = {s: float(np.mean([auc[att]["%s_m%d" % (variant, ms)][s]
                                             for ms in DAGMM["init_seeds"]])) for s in seeds}
                    ent = summarise(per, seeds, heldout)
                    ent["per_init_heldout_mean"] = [
                        float(np.mean([auc[att]["%s_m%d" % (variant, ms)][s] for s in heldout]))
                        for ms in DAGMM["init_seeds"]]
                    res["attacks"][att][variant] = ent
        res["checks"] = {"innovation_eval_max_abs_auc_diff": repro,
                         "innovation_eval_counts_match": repro_counts,
                         "scores_with_dropped_nonfinite_values": nonfinite}
        results[cfg] = res

    from passportfl.reproducibility import environment
    meta = meta_common(design, cfgs, a)
    if use_dagmm:
        meta["dagmm"] = {
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in DAGMM.items()},
            "status": "invalid by construction, kept to reproduce the second run; see revision"}
    meta["revision"] = REVISION
    meta["environment"] = environment()
    meta["wall_time_s"] = round(time.time() - t0, 1)
    out = {"meta": meta, "configs": results}
    if use_dagmm:
        out["dagmm_training"] = training
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help="comma-separated subset of: " + ", ".join(CONFIGS))
    ap.add_argument("--dagmm", choices=("none", "within", "cross"), default="none",
                    help="none: standard-deviation and released scores only (std_baseline.json); "
                         "within: per-federation DAGMM (std_dagmm_within.json); cross: the "
                         "invalid CAL-trained design of the second run, to an explicit --out")
    ap.add_argument("--no-dagmm", action="store_true", help="same as --dagmm none")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes over (configuration, seed); --dagmm within only")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads per process (default 1 for within, 4 for cross)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    design = "none" if a.no_dagmm else a.dagmm
    out_path = a.out or DEFAULT_OUT[design]
    if out_path is None:
        raise SystemExit("--dagmm cross reproduces an invalid design; give an explicit --out")
    cfgs = [c.strip() for c in a.configs.split(",") if c.strip()]
    unknown = [c for c in cfgs if c not in CONFIGS]
    if unknown:
        raise SystemExit("unknown configuration(s): %s" % unknown)

    if design == "within":
        payload = run_within(a, cfgs, t0)
    else:
        payload = run_pooled(a, cfgs, t0, design)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    cols = ["std_two_sided", "std_low", "norm_bound"]
    if design == "within":
        print_table(payload["configs"], cols + ["dagmm", "std_dagmm"], "mean")
        for cfg, tr in payload["dagmm_training"].items():
            ms = [m for t in tr.values() for m in t["models"].values()]
            print("%s honest checks: fits_train %d/%d, beats training mean on scored rows %d/%d, "
                  "beats zero %d/%d; PCA-%d explained same seed %s, other seed %s"
                  % (cfg, sum(m["fits_train"] for m in ms), len(ms),
                     sum(m["beats_train_mean_on_scored"] for m in ms), len(ms),
                     sum(m["beats_zero_on_scored"] for m in ms), len(ms), PCA_Q,
                     ["%.3f" % t["transfer"]["pca_explained_same_seed"] for t in tr.values()],
                     ["%.3f" % t["transfer"]["pca_explained_other_seed"] for t in tr.values()]))
    elif design == "cross":
        print_table(payload["configs"], cols + ["dagmm", "std_dagmm"], "heldout_mean")
    else:
        print_table(payload["configs"], cols, "mean")
    print("\nwrote %s (%.0fs)" % (out_path, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
