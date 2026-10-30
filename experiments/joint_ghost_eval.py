from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, hashlib, itertools, json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

import numpy as np
import torch

from passportfl import identity as idm
from passportfl import probe as probe_lib
from passportfl.baselines import bulyan_deviation
from passportfl.combined_attack import BASE_STATS
from passportfl.innovation import InnovationAuditor
from passportfl.limited_access import LimitedAccessAttacker, parse_level
from passportfl.probe import mean_separation, prototype_drift_ratio
from passportfl.reference_baselines import fltrust_scores
from passportfl.reproducibility import environment

import combined_eval as ce
import ghost_extended_eval as gx
import limited_access_eval as lae
from negative_results import DEFAULT_ENTROPY, seed_bytes

REPO = os.path.normpath(os.path.join(_HERE, ".."))
RESULTS = os.path.join(REPO, "results")
ARCHIVES = os.path.join(RESULTS, "archives")
LA_PARTS = os.path.join(REPO, ".reproduction", "limited_access_parts")
PREFIX = "arch_cifar100_s"
K, D, M, T = gx.K, gx.D, gx.M, gx.T
STATS = gx.STATS
CLIENTS = 10
SEEDS, VICTIMS = (0, 1, 2, 3, 4), (0, 1)
CAL_SEEDS, DEV_SEEDS, TEST_SEEDS = (0, 1), (2,), (3, 4)
ATTACK_START, LAST_ROUND = 12, 35
MAD_FLOOR = 1e-2
DEFAULT_LEVELS = ("current,delayed:1,delayed:2,delayed:3,query:16,decision:4,feedback,pseudo,"
                  "shadow:5")
RULES6 = tuple(BASE_STATS)
DETECTORS = ("fused", "auditor") + RULES6 + ("bulyan", "fltrust", "l2")
PAYLOAD_FLAGS = ("fused", "auditor", "norm_bound", "l2")

GEN_GRID = {"tau": (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5), "s_z": (0.0, 0.05, 0.1, 0.2, 0.3),
            "omega": (0.3, 0.5, 0.85), "attach": ("all", "declared")}
CAL_RUNS = (("current", 0, 0), ("current", 0, 1), ("current", 1, 0), ("current", 1, 1))

VERIFIERS = {"deployed_w5": (5, None, 0), "deployed_w0": (0, None, 0),
             "two_sided_w5": (5, "window_12_35", 12), "two_sided_w0": (0, "window_12_35", 12),
             "two_sided_w5_b6": (5, "window_6_35", 6), "two_sided_w5_paper": (5, "paper", 6)}
BEACON_VERIFIERS = ("deployed_w5", "deployed_w0", "two_sided_w5")
JOINT_COMMIT = {"w5": ("deployed_w5", "two_sided_w5"), "w0": ("deployed_w0", "two_sided_w0")}
EXACT_CHECK = {("current", 0, 0), ("current", 3, 0)}


def jsonable(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bytes):
        return o.hex()
    raise TypeError(type(o))


def rate(k: int, n: int) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": None}
    p, lo, hi = gx.clopper_pearson(int(k), int(n))
    return {"k": int(k), "n": int(n), "rate": p, "ci95": [lo, hi]}


def seed_int(entropy: bytes, label: str) -> int:
    return int.from_bytes(seed_bytes(entropy, label)[:8], "little")


def level_file(level: str) -> str:
    return level.replace(":", "-")


_NN_EXACT = probe_lib.nearest_prototype


def nearest_prototype_fast(emb: np.ndarray, C: np.ndarray) -> np.ndarray:
    e, c = np.asarray(emb, np.float64), np.asarray(C, np.float64)
    if e.ndim != 2 or len(e) == 0 or c.ndim != 2 or len(c) < 2:
        return _NN_EXACT(emb, C)
    ee, cc = np.einsum("ij,ij->i", e, e), np.einsum("ij,ij->i", c, c)
    d = ee[:, None] - 2.0 * (e @ c.T) + cc[None, :]
    best = d.argmin(1)
    two = np.partition(d, 1, axis=1)[:, :2]
    amb = (two[:, 1] - two[:, 0]) <= 1e-8 * (ee + cc.max() + 1.0)
    if amb.any():
        best[amb] = _NN_EXACT(np.asarray(emb)[amb], C)
    return best


def install_nn(fast: bool) -> None:
    f = nearest_prototype_fast if fast else _NN_EXACT
    probe_lib.nearest_prototype = f
    gx.nearest_prototype = f


def nn_spot_check(tr: gx.Trajectory, rng: np.random.Generator, pairs: int = 8) -> int:
    R, bad = len(tr.C), 0
    for _ in range(pairs):
        s, q = int(rng.integers(1, R + 1)), int(rng.integers(1, R + 1))
        bad += int(not np.array_equal(nearest_prototype_fast(tr.Z[s - 1], tr.C[q - 1]),
                                      _NN_EXACT(tr.Z[s - 1], tr.C[q - 1])))
    return bad


def frozen_thresholds() -> dict:
    g = json.load(open(os.path.join(RESULTS, "combined_grid.json"), encoding="utf-8"))
    la = json.load(open(os.path.join(RESULTS, "limited_access_grid.json"), encoding="utf-8"))
    pp = json.load(open(os.path.join(RESULTS, "published_predicates.json"), encoding="utf-8"))
    b = pp["combined"]["two-sided l2 (sensitivity)"]["joint_1pct"]["bounds"]
    thr = {"norm": {k: tuple(v) for k, v in g["norm"].items()},
           "normalised": {k: float(v) for k, v in g["thresholds"].items()},
           "fused": float(la["calibration"]["tau"]["F"]),
           "auditor_raw": float(la["calibration"]["tau"]["P"]),
           "l2": (float(b["l2_lower_bound"]), float(b["l2_bound"])),
           "sources": {
               "fused": "limited_access_grid.json calibration.tau.F (= combined_grid.json "
                        "thresholds.combined_max)",
               "auditor_raw": "limited_access_grid.json calibration.tau.P (raw suspicion, CAL 1%)",
               "rules": "combined_grid.json thresholds (CAL-normalised statistics, CAL 1%)",
               "l2": "published_predicates.json combined['two-sided l2 (sensitivity)']"
                     ".joint_1pct.bounds [l2_lower_bound, l2_bound]"}}
    if thr["fused"] != thr["normalised"]["combined_max"]:
        raise RuntimeError("fused threshold differs between the released files")
    return thr


def root_map(seed: int, arch: dict) -> Optional[Dict[int, np.ndarray]]:
    p = os.path.join(ARCHIVES, "fltrust_root_cifar100_d05_r100_s%d.npz" % seed)
    if not os.path.exists(p):
        return None
    g0 = np.load(p)["g0"]
    rounds = [r for r in sorted(arch["cbar"]) if (r - 1) in arch["cbar"]]
    return {r: g0[i] for i, r in enumerate(rounds[:len(g0)])}


def replay_scores(arch: dict, victim: Optional[int], override: Dict[int, np.ndarray],
                  g0map: Optional[dict], norm: Dict[str, tuple], start: int = ATTACK_START
                  ) -> Dict[str, np.ndarray]:
    clients = len(arch["C"])
    rounds = max(max(v) for v in arch["C"].values())
    aud = InnovationAuditor(mad_floor=MAD_FLOOR)
    out: Dict[str, list] = {k: [] for k in DETECTORS}
    rec = []
    for r in range(1, rounds + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        Cb = aud.cbar.get(r - 1)
        present = {c: arch["present"][c][r] for c in range(clients)}
        payloads = {c: (arch["C"][c][r], present[c]) for c in range(clients)}
        if victim is not None and r in override:
            payloads[victim] = (override[r], present[victim])
        rs = aud.observe_round(r, payloads)
        if r < start:
            continue
        raw = ce.raw_round_stats(rs, payloads, Cb, clients, MAD_FLOOR)
        comb = ce.combined_from_raw(raw, norm, clients)
        innov = {c: np.asarray(payloads[c][0], np.float64) - arch["cbar"][r - 1]
                 for c in range(clients)}
        bul = bulyan_deviation({c: innov[c].ravel() for c in range(clients)}, n_byzantine=2)
        flt = (fltrust_scores(innov, g0map[r]) if g0map is not None and r in g0map
               else {c: float("nan") for c in range(clients)})
        rec.append(r)
        out["fused"].append([comb["combined_max"][c] for c in range(clients)])
        out["auditor"].append([raw["suspicion"][c] for c in range(clients)])
        for b in RULES6:
            out[b].append([comb[b][c] for c in range(clients)])
        out["bulyan"].append([bul[c] for c in range(clients)])
        out["fltrust"].append([flt[c] for c in range(clients)])
        out["l2"].append([float(np.linalg.norm(innov[c])) for c in range(clients)])
    res = {k: np.asarray(v, float) for k, v in out.items()}
    res["rounds"] = np.asarray(rec)
    return res


def flags(sc: Dict[str, np.ndarray], thr: dict) -> Dict[str, np.ndarray]:
    f = lambda v, t: np.where(np.isfinite(v), v, -np.inf) > t
    out = {"fused": f(sc["fused"], thr["fused"]), "auditor": f(sc["auditor"], thr["auditor_raw"])}
    for b in RULES6:
        out[b] = f(sc[b], thr["normalised"][b])
    if "bulyan" in thr:
        out["bulyan"] = f(sc["bulyan"], thr["bulyan"])
        out["fltrust"] = f(sc["fltrust"], thr["fltrust"])
    lo, hi = thr["l2"]
    out["l2"] = (sc["l2"] < lo) | (sc["l2"] > hi)
    out["payload_any"] = out["fused"] | out["auditor"] | out["norm_bound"] | out["l2"]
    return out


def fit_extra(honest: Dict[int, dict], thr: dict) -> dict:
    out = {}
    for det in ("bulyan", "fltrust"):
        v = np.concatenate([honest[s][det].ravel() for s in CAL_SEEDS])
        out[det] = ce.fit_threshold(v, 0.01)
    return out


def honest_payload_layer(honest: Dict[int, dict], thr: dict) -> dict:
    fl = {s: flags(honest[s], thr) for s in honest}
    out = {}
    for name, seeds in (("TEST", TEST_SEEDS), ("CAL", CAL_SEEDS), ("DEV", DEV_SEEDS),
                        ("all", SEEDS)):
        seeds = [s for s in seeds if s in fl]
        blk = {}
        for k in fl[seeds[0]]:
            m = np.concatenate([fl[s][k] for s in seeds], axis=0)
            runs = np.concatenate([fl[s][k].any(axis=0) for s in seeds])
            blk[k] = {"client_round": rate(m.sum(), m.size), "client_trajectory": rate(runs.sum(), runs.size)}
        out[name] = blk
    return out


def summarise_detectors(passes: List[Tuple[int, int, dict]], thr: dict) -> dict:
    out = {}
    fl = [(s, v, flags(sc, thr)) for s, v, sc in passes]
    names = list(DETECTORS) + ["l2_upper", "l2_lower"]
    for det in names + ["payload_any"]:
        rec = {}
        if det != "payload_any":
            src = "l2" if det.startswith("l2") else det
            sign = -1.0 if det == "l2_lower" else 1.0
            pos, neg = [], []
            for s, v, sc in passes:
                m = sign * sc[src]
                others = np.ones(m.shape[1], bool); others[v] = False
                pos.extend(m[:, v].tolist()); neg.extend(m[:, others].ravel().tolist())
            rec["auc"] = ce.roc_auc(pos, neg)
            rec["n_pos"], rec["n_neg"] = len(pos), len(neg)
            rec["victim_median"] = float(np.nanmedian(np.asarray(pos) * sign))
            rec["peer_median"] = float(np.nanmedian(np.asarray(neg) * sign))
        if det in ("l2_upper", "l2_lower"):
            out[det] = rec
            continue
        for name, seeds in (("TEST", TEST_SEEDS), ("all", SEEDS)):
            sel = [(s, v, f) for s, v, f in fl if s in seeds]
            k = sum(int(f[det][:, v].sum()) for s, v, f in sel)
            n = sum(int(f[det][:, v].size) for s, v, f in sel)
            rec["flagged_%s" % name] = rate(k, n)
            rec["runs_flagged_%s" % name] = rate(sum(int(f[det][:, v].any()) for s, v, f in sel),
                                                 len(sel))
        out[det] = rec
    return out


def ghost_trajectories() -> Tuple[Dict[str, gx.Trajectory], Dict[str, dict]]:
    ge = json.load(open(os.path.join(RESULTS, "ghost_extended.json"), encoding="utf-8"))
    po, pc = ge["ghosts"]["original"]["params"], ge["ghosts"]["calibrated"]["params"]
    return {"original": gx.original_trajectory(LAST_ROUND, **po),
            "calibrated": gx.calibrated_trajectory(LAST_ROUND, **pc)}, {"original": po, "calibrated": pc}


def part1(paths: Dict[int, str], thr: dict) -> dict:
    trs, params = ghost_trajectories()
    out = {"params": params, "ghosts": {}}
    for kind, tr in trs.items():
        override = {r: tr.C[r - 1] for r in range(1, LAST_ROUND + 1)}
        passes = []
        for s in SEEDS:
            arch = ce.load_archive(paths[s])
            g0 = root_map(s, arch)
            for v in VICTIMS:
                passes.append((s, v, replay_scores(arch, v, override, g0, thr["norm"])))
        l2 = np.concatenate([sc["l2"][:, v] for s, v, sc in passes])
        out["ghosts"][kind] = {"detectors": summarise_detectors(passes, thr),
                               "innovation_l2_median": float(np.median(l2)),
                               "mean_separation_median": float(np.median(
                                   [mean_separation(C) for C in tr.C[ATTACK_START - 1:]]))}
    return out


class CapturingAttacker(LimitedAccessAttacker):
    last = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured: Dict[int, np.ndarray] = {}
        CapturingAttacker.last = self

    def craft(self, aud, score_hist, r, *args, **kwargs):
        C, rec = super().craft(aud, score_hist, r, *args, **kwargs)
        self.captured[int(r)] = np.array(C, dtype=np.float64, copy=True)
        return C, rec


def honest_calibration(paths: Dict[int, str]) -> dict:
    a = SimpleNamespace(attack_start=ATTACK_START, scope_start=ATTACK_START, mad_floor=MAD_FLOOR)
    honest, cached = {}, 0
    for s in SEEDS:
        t = lae.honest_task(paths[s], s, a)
        t.update(defense="H", level="honest", signature="")
        t["signature"] = lae.task_signature(t)
        got = lae.load_part(lae.part_path(LA_PARTS, t), t["signature"])
        if got is None:
            got = lae._honest_pass(t)
        else:
            cached += 1
        honest[s] = got
    calib = lae.calibrate(honest, CAL_SEEDS, TEST_SEEDS, CLIENTS, 0.01)
    calib["release_check"] = lae.check_against_release(calib)
    calib["honest_passes_from_cache"] = cached
    return calib


def attack_task(calib: dict, paths: Dict[int, str], level: str, beta: float, seed: int,
                victim: int) -> dict:
    common = {"phi": 0.0, "attack_start": ATTACK_START, "scope_start": ATTACK_START,
              "mad_floor": MAD_FLOOR, "fb_step": 0.5, "fb_decay": 0.8, "es_sigma0": 0.1,
              "es_tol": 1e-2, "norm": {k: list(v) for k, v in calib["norm"].items()}}
    lv = parse_level(level)
    t = dict(common, path=paths[seed], defense="F", level=lv.name, beta=float(beta), seed=seed,
             victim=victim, solver=[40, 80, 1], tau=calib["tau"]["F"],
             cal_median=calib["cal_median"]["F"])
    if lv.kind == "shadow":
        t["shadow_path"] = paths[lv.shadow_seed]
    t["signature"] = lae.task_signature(t)
    return t


def payload_path(parts: str, task: dict) -> str:
    return os.path.join(parts, "payloads", "F_%s_b%g_s%d_v%d.npz" % (
        level_file(task["level"]), task["beta"], task["seed"], task["victim"]))


def load_payloads(path: str, signature: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        if str(z["signature"]) != signature:
            return None
        return {k: (str(z[k]) if z[k].dtype.kind == "U" and z[k].ndim == 0 else np.array(z[k]))
                for k in z.files}


def regenerate(job: Tuple[dict, str]) -> dict:
    task, path = job
    torch.set_num_threads(1)
    got = load_payloads(path, task["signature"])
    if got is not None:
        return {"path": path, "cached": True, "seconds": float(got["seconds"])}
    orig = lae.LimitedAccessAttacker
    lae.LimitedAccessAttacker = CapturingAttacker
    try:
        res = lae.score_pass(task)
    finally:
        lae.LimitedAccessAttacker = orig
    atk = CapturingAttacker.last
    rounds = [int(e["round"]) for e in res["log"]]
    obj = {"signature": np.asarray(task["signature"]),
           "rounds": np.asarray(rounds),
           "payloads": np.stack([atk.captured[r] for r in rounds]),
           "digests": np.asarray([lae.payload_digest(atk.captured[r]) for r in rounds]),
           "logged_digests": np.asarray([e["payload"] for e in res["log"]]),
           "realised": np.asarray([e["realised"] for e in res["log"]], float),
           "mat": np.asarray(res["mat"], float), "mat_rounds": np.asarray(res["rounds"]),
           "log": np.asarray(json.dumps(res["log"], default=float)),
           "seconds": np.asarray(res["seconds"])}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **obj)
    os.replace(tmp, path)
    return {"path": path, "cached": False, "seconds": float(res["seconds"])}


def regeneration_check(task: dict, got: dict) -> dict:
    ref = lae.load_part(lae.part_path(LA_PARTS, task), task["signature"])
    rec = {"rounds": int(len(got["rounds"])),
           "own_log_digests_match": int(np.sum(got["digests"] == got["logged_digests"])),
           "released_pass_in_cache": ref is not None}
    if ref is not None:
        rd = [e["payload"] for e in ref["log"]]
        rec["digests_match_released"] = int(sum(a == b for a, b in zip(got["digests"], rd)))
        rr = np.asarray([e["realised"] for e in ref["log"]], float)
        rec["realised_max_abs_diff"] = float(np.max(np.abs(rr - got["realised"])))
        rec["mat_max_abs_diff"] = float(np.nanmax(np.abs(np.asarray(ref["mat"]) - got["mat"])))
    return rec


def log_rows(path: str, lo: int, hi: int) -> List[dict]:
    fp = json.load(open(path, encoding="utf-8"))
    return [e for e in fp["verify_log"] if lo <= int(e["round"]) <= hi]


def band(v: Sequence[float]) -> List[float]:
    return [float(x) for x in np.percentile(np.asarray(v, float), gx.BAND_PCT)]


def archive_ratios(paths: Dict[int, str], seeds: Sequence[int], lo: int, hi: int) -> np.ndarray:
    out = []
    for s in seeds:
        arch = ce.load_archive(paths[s])
        for c in range(len(arch["C"])):
            out += [prototype_drift_ratio(arch["C"][c][r], arch["C"][c][r - 1])
                    for r in range(lo, hi + 1)]
    return np.asarray(out)


def honest_ratio_band(paths: Dict[int, str], band: Sequence[float]) -> dict:
    out = {}
    n_r = LAST_ROUND - ATTACK_START + 1
    for name, seeds in (("CAL", CAL_SEEDS), ("DEV", DEV_SEEDS), ("TEST", TEST_SEEDS)):
        v = archive_ratios(paths, seeds, ATTACK_START, LAST_ROUND).reshape(-1, n_r)
        ins = (v >= band[0]) & (v <= band[1])
        out[name] = {"client_rounds_inside": rate(ins.sum(), ins.size),
                     "below": int((v < band[0]).sum()), "above": int((v > band[1]).sum()),
                     "round_12_inside": rate(ins[:, 0].sum(), ins.shape[0]),
                     "trajectories_every_round_inside": rate(ins.all(1).sum(), ins.shape[0]),
                     "median": float(np.median(v)), "round_12_median": float(np.median(v[:, 0]))}
    return out


def default_bands(paths: Dict[int, str]) -> Tuple[dict, dict]:
    bands, meta = {}, {}
    pp = [os.path.join(RESULTS, "fedproto_abl_pp_s%d.json" % s) for s in CAL_SEEDS]
    for lo in (12, 6):
        ratio = archive_ratios(paths, CAL_SEEDS, lo, LAST_ROUND)
        rows = [e for p in pp for e in log_rows(p, lo, LAST_ROUND)]
        key = "window_%d_35" % lo
        bands[key] = {"proto_ratio": band(ratio),
                      "label_stability": band([e["stability"] for e in rows]),
                      "max_drift": band([e["max_drift"] for e in rows])}
        meta[key] = {"proto_ratio": {"source": "CAL archives seeds 0,1 (prototype_drift_ratio "
                                               "of consecutive payloads), rounds %d-35" % lo,
                                     "n": int(len(ratio)), "median": float(np.median(ratio))},
                     "label_stability": {"source": "fedproto_abl_pp_s0/s1.json rounds %d-35" % lo,
                                         "n": len(rows), "median": float(np.median(
                                             [e["stability"] for e in rows]))},
                     "max_drift": {"source": "fedproto_abl_pp_s0/s1.json rounds %d-35" % lo,
                                   "n": len(rows), "median": float(np.median(
                                       [e["max_drift"] for e in rows]))},
                     "ratio_from_abl_pp_logs_for_comparison": band(
                         [e["proto_ratio"] for e in rows])}
    ge = json.load(open(os.path.join(RESULTS, "ghost_extended.json"), encoding="utf-8"))
    bands["paper"] = {k: list(v) for k, v in ge["honest_reference"]["bands_p1_p99"].items()}
    meta["paper"] = {"source": "ghost_extended.json honest_reference.bands_p1_p99 (rounds > 28 "
                               "of fedproto_cifar100.json)"}
    return bands, meta


def bands_from_files(files: Sequence[str], defaults: dict) -> Tuple[dict, dict]:
    rows = {12: [], 6: []}
    bands, meta = {"paper": defaults["paper"]}, {"files": list(files)}
    for p in files:
        fp = json.load(open(p, encoding="utf-8"))
        if "bands" in fp:
            for k, v in fp["bands"].items():
                bands[k] = {s: list(map(float, v[s])) for s in STATS}
        elif "verify_log" in fp:
            for lo in rows:
                rows[lo] += [e for e in fp["verify_log"] if lo <= int(e["round"]) <= LAST_ROUND]
        else:
            raise SystemExit("%s holds neither 'bands' nor 'verify_log'" % p)
    for lo, rr in rows.items():
        if rr:
            bands["window_%d_35" % lo] = {"proto_ratio": band([e["proto_ratio"] for e in rr]),
                                          "label_stability": band([e["stability"] for e in rr]),
                                          "max_drift": band([e["max_drift"] for e in rr])}
            meta["window_%d_35" % lo] = {"n": len(rr)}
    for k in ("window_12_35", "window_6_35"):
        if k not in bands:
            raise SystemExit("--bands-file gives no %s band" % k)
    return bands, meta


def honest_commitment(path: str, bands: dict, lo: int = ATTACK_START) -> dict:
    rows = log_rows(path, lo, LAST_ROUND)
    st = np.asarray([e["stability"] for e in rows]); pr = np.asarray([e["proto_ratio"] for e in rows])
    md = np.asarray([e["max_drift"] for e in rows])
    G = np.asarray([e["G"] for e in rows])
    core = (pr <= gx.MAX_RATIO) & (st >= gx.MIN_STAB)
    sure = md <= gx.DELTA

    def inside(name):
        b = bands[name]
        return ((pr >= b["proto_ratio"][0]) & (pr <= b["proto_ratio"][1])
                & (st >= b["label_stability"][0]) & (st <= b["label_stability"][1])
                & (md >= b["max_drift"][0]) & (md <= b["max_drift"][1]))

    def block(ok_lo, ok_hi):
        per_lo = [bool(ok_lo[G == g].all()) for g in sorted(set(G))]
        per_hi = [bool(ok_hi[G == g].all()) for g in sorted(set(G))]
        return {"client_round_lower": rate(ok_lo.sum(), ok_lo.size),
                "client_round_upper": rate(ok_hi.sum(), ok_hi.size),
                "clients_all_rounds_lower": rate(sum(per_lo), len(per_lo)),
                "clients_all_rounds_upper": rate(sum(per_hi), len(per_hi))}

    out = {"source": os.path.relpath(path, REPO).replace("\\", "/"),
           "rounds": [lo, LAST_ROUND], "n_rows": len(rows),
           "stability_below_0.5": int((st < gx.MIN_STAB).sum()),
           "ratio_above_1": int((pr > gx.MAX_RATIO).sum()),
           "max_drift_above_1_drift_undecided": int((~sure).sum()),
           "medians": {"proto_ratio": float(np.median(pr)), "label_stability": float(np.median(st)),
                       "max_drift": float(np.median(md))},
           "deployed": block(core & sure, core)}
    for name in ("window_12_35", "window_6_35", "paper"):
        ib = inside(name)
        out["two_sided_" + name] = block(core & sure & ib, core & ib)
        out["inside_band_" + name] = rate(ib.sum(), ib.size)
    return out


def fabricate_table(Cseq: Sequence[np.ndarray], declared: np.ndarray, theta: dict,
                    rng: np.random.Generator) -> gx.Trajectory:
    tau, s_z, omega = float(theta["tau"]), float(theta["s_z"]), float(theta["omega"])
    rows = np.arange(K) if theta["attach"] == "all" else np.flatnonzero(np.asarray(declared, bool))
    a = rows[rng.integers(0, len(rows), M)]
    sep = mean_separation(Cseq[0])
    U = rng.normal(0.0, 1.0, (M, D))
    U *= tau * sep / np.linalg.norm(U, axis=1, keepdims=True)
    vU = rng.normal(0.0, 1.0, (M, D))
    tr = gx.Trajectory(Z0=idm.dequantize(Cseq[0][a] + U), params=dict(theta))
    C_prev = None
    for r, C in enumerate(Cseq, 1):
        sep = mean_separation(C)
        vU = omega * vU + (1.0 - omega) * rng.normal(0.0, 1.0, (M, D))
        U = U + s_z * sep * vU / np.linalg.norm(vU, axis=1, keepdims=True)
        U *= tau * sep / np.linalg.norm(U, axis=1, keepdims=True)
        tr.C.append(C); tr.Z.append(idm.dequantize(C[a] + U))
        tr.records.append(gx._record(r, C, C_prev))
        C_prev = C
    return tr


def payload_sequence(arch: dict, victim: int, got: dict) -> List[np.ndarray]:
    crafted = {int(r): got["payloads"][i] for i, r in enumerate(got["rounds"])}
    R = max(arch["C"][victim])
    return [np.ascontiguousarray(crafted[r] if r in crafted else arch["C"][victim][r], np.float64)
            for r in range(1, R + 1)]


class WindowReceiver(gx.ChallengeReceiver):
    def __init__(self, warmup_rounds: int, bands: Optional[dict] = None, band_from: int = 0):
        super().__init__(warmup_rounds, bands)
        self._bands, self.band_from = bands, band_from

    def verify(self, c, challenge_seed=None):
        self.bands = self._bands if c.round >= self.band_from else None
        return super().verify(c, challenge_seed)


def make_verifiers(bands: dict, names: Sequence[str]) -> Dict[str, WindowReceiver]:
    out = {}
    for n in names:
        w, b, f = VERIFIERS[n]
        out[n] = WindowReceiver(w, None if b is None else bands[b], f)
    return out


def verdicts(st: Dict[str, np.ndarray], name: str, bands: dict) -> np.ndarray:
    w, b, f = VERIFIERS[name]
    ok = gx.HistoryReplay.verdicts(st, w, None)
    if b is None:
        return ok
    R = st["proto_ratio"].shape[1]
    rounds = np.arange(1, R + 1)[None, :]
    on = (rounds > w) & (rounds >= f)
    defined = {"proto_ratio": rounds > 1, "label_stability": rounds > 1, "max_drift": st["has_drift"]}
    for s in STATS:
        lo, hi = bands[b][s]
        ok &= ~on | ~defined[s] | ((st[s] >= lo) & (st[s] <= hi))
    return ok


def commit_ok(st: Dict[str, np.ndarray], bands: dict, which: str = "w5") -> np.ndarray:
    a, b = JOINT_COMMIT[which]
    return verdicts(st, a, bands) & verdicts(st, b, bands)


def mc_subsets(rng: np.random.Generator, n: int):
    return lambda r: np.argsort(rng.random((n, M)), axis=1)[:, :T]


def universal_summary(rows: List[dict], bands: dict) -> dict:
    out = {}
    for w in (0, 5):
        sel = [x for x in rows if x["round"] > w]
        ok = [x["worst_drift_rate"] <= gx.DELTA and x["proto_ratio"] <= gx.MAX_RATIO
              and x["stability_lb"] >= gx.MIN_STAB for x in sel]
        out["deployed_w%d" % w] = {
            "rounds_checked": len(sel), "rounds_passing_every_admissible_challenge": int(sum(ok)),
            "failing_rounds": [x["round"] for x, o in zip(sel, ok) if not o],
            "max_worst_drift_rate": max(x["worst_drift_rate"] for x in sel),
            "min_stability_lb": min(x["stability_lb"] for x in sel)}
    b = bands["window_12_35"]
    win = [x for x in rows if ATTACK_START <= x["round"] <= LAST_ROUND]
    out["window_12_35"] = {
        "rounds": len(win),
        "ratio_in_band": int(sum(b["proto_ratio"][0] <= x["proto_ratio"] <= b["proto_ratio"][1]
                                 for x in win)),
        "stability_interval_inside_band": int(sum(
            x["stability_lb"] >= b["label_stability"][0] and x["stability_ub"] <= b["label_stability"][1]
            for x in win)),
        "max_drift_ub_at_most_band_hi": int(sum(x["max_drift_ub"] <= b["max_drift"][1] for x in win))}
    return out


_SEQ_CACHE: Dict[str, tuple] = {}


def cal_sequence(item: Tuple[str, str, int, int]):
    ppath, apath, s, v = item
    if ppath not in _SEQ_CACHE:
        got = np.load(ppath, allow_pickle=False)
        got = {"payloads": np.array(got["payloads"]), "rounds": np.array(got["rounds"])}
        arch = ce.load_archive(apath)
        _SEQ_CACHE[ppath] = (payload_sequence(arch, v, got), np.asarray(arch["present"][v][ATTACK_START], bool))
    return _SEQ_CACHE[ppath]


def evaluate_generator(job) -> dict:
    theta, items, bands, n_seq, entropy_hex, target = job
    install_nn(True)
    entropy = bytes.fromhex(entropy_hex)
    per, pooled = [], {s: [] for s in ("label_stability", "max_drift")}
    for item in items:
        _, _, s, v = item
        Cseq, declared = cal_sequence(item)
        tr = fabricate_table(Cseq, declared, theta,
                             np.random.default_rng(seed_int(entropy, "joint-ghost-s%d-v%d" % (s, v))))
        hr = gx.HistoryReplay(tr)
        st = hr.run(mc_subsets(np.random.default_rng(
            seed_int(entropy, "joint-ghost-calmc-s%d-v%d" % (s, v))), n_seq))
        ok = commit_ok(st, bands, "w5")
        w = slice(ATTACK_START - 1, LAST_ROUND)
        per.append({"seed": s, "victim": v, "full": float(ok.all(1).mean()),
                    "per_round_12_35": float(ok[:, w].mean()),
                    "medians_12_35": {k: float(np.median(st[k][:, w])) for k in STATS}})
        for k in pooled:
            pooled[k].append(st[k][:, w].ravel())
    med = {k: float(np.median(np.concatenate(v))) for k, v in pooled.items()}
    return {"theta": theta, "objective_full": min(p["full"] for p in per),
            "objective_per_round": min(p["per_round_12_35"] for p in per),
            "distance": max(abs(med[k] / target[k] - 1.0) for k in med),
            "pooled_medians_12_35": med, "per_trajectory": per}


def calibrate_generator(items, bands: dict, n_seq: int, entropy_hex: str, target: dict,
                        workers: int) -> dict:
    keys = list(GEN_GRID)
    thetas = [dict(zip(keys, vals)) for vals in itertools.product(*(GEN_GRID[k] for k in keys))]
    jobs = [(t, items, bands, n_seq, entropy_hex, target) for t in thetas]
    res = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(evaluate_generator, j): i for i, j in enumerate(jobs)}
        done = 0
        for f in as_completed(futs):
            res[futs[f]] = f.result()
            done += 1
            if done % 21 == 0 or done == len(jobs):
                print("  generator grid %d/%d" % (done, len(jobs)), flush=True)
    for i, r in enumerate(res):
        r["index"] = i
    prim = min(res, key=lambda r: (-r["objective_full"], -r["objective_per_round"], r["distance"],
                                   r["index"]))
    sec = min(res, key=lambda r: (r["distance"], r["index"]))
    return {"grid": {k: list(v) for k, v in GEN_GRID.items()}, "n_points": len(res),
            "trajectories": [{"level": "current", "beta": 1.0, "seed": s, "victim": v}
                             for _, _, s, v in items],
            "mc_sequences_per_trajectory": n_seq,
            "target_medians": target, "selected_primary": prim, "selected_secondary": sec,
            "points": res}


def run_interactions(tr, signer_seed: bytes, bands: dict, entropy: bytes, n_beacon: int):
    runs = []
    vs = make_verifiers(bands, list(VERIFIERS))
    opened: list = []
    log = gx.run_interaction(gx.GhostProver(tr, "Ed25519", signer_seed), gx.ChainHeadChallenge(),
                             vs, len(tr.C), opened)
    runs.append(("chain_head", None, log, opened))
    for b in range(n_beacon):
        vs = make_verifiers(bands, BEACON_VERIFIERS)
        opened = []
        chal = gx.BeaconChallenge(seed_bytes(entropy, "joint-ghost-beacon-%d" % b))
        log = gx.run_interaction(gx.GhostProver(tr, "Ed25519", signer_seed), chal, vs, len(tr.C), opened)
        runs.append(("beacon", b, log, opened))
    return runs


def evaluate_variant(tr, s: int, v: int, bands: dict, entropy: bytes, n_mc: int, n_beacon: int,
                     payload_ok: np.ndarray, exact_check: bool) -> dict:
    R = len(tr.C)
    signer = seed_bytes(entropy, "joint-ghost-signer-s%d-v%d" % (s, v))
    runs = run_interactions(tr, signer, bands, entropy, n_beacon)
    chain = runs[0][2]
    out = {"theta": tr.params}
    names = list(VERIFIERS)
    out["chain_head"] = {
        "per_round": [{"round": x["round"], **{k: x.get(k) for k in STATS + ("max_drift_rate",)},
                       **{n: bool(x[n]["accepted"]) for n in names}} for x in chain],
        "accepted": {n: int(sum(x[n]["accepted"] for x in chain)) for n in names},
        "rejected_rounds": {n: [x["round"] for x in chain if not x[n]["accepted"]] for n in names},
        "rejection_reasons": {n: sorted({x[n]["reason"] for x in chain if not x[n]["accepted"]})
                              for n in names},
        "stat_mismatch_across_verifiers": int(sum(x["stat_mismatch_across_verifiers"] for x in chain))}
    if exact_check:
        install_nn(False)
        ex = run_interactions(tr, signer, bands, entropy, 0)[0][2]
        install_nn(True)
        out["chain_head"]["exact_nn_rerun_identical"] = bool(json.dumps(ex, default=jsonable)
                                                             == json.dumps(chain, default=jsonable))
    if n_beacon:
        bl = [r[2] for r in runs[1:]]
        out["beacon"] = {"sequences": len(bl),
                         **{n: {"accepted_total": int(sum(x[n]["accepted"] for lg in bl for x in lg)),
                                "rounds_total": R * len(bl),
                                "sequences_all_rounds": int(sum(all(x[n]["accepted"] for x in lg)
                                                                for lg in bl))}
                            for n in BEACON_VERIFIERS}}
    hr = gx.HistoryReplay(tr)
    sdiff, vmis = 0.0, 0
    for _, _, log, opened in runs:
        S = np.asarray(opened)
        st = hr.run(lambda q: S[q - 1][None, :])
        for x in log:
            i = x["round"] - 1
            for k in STATS:
                sdiff = max(sdiff, abs(float(st[k][0, i]) - x[k]))
            sdiff = max(sdiff, abs(float(st["max_drift_rate"][0, i]) - (x["max_drift_rate"] or 0.0)))
        for n in log[0]:
            if n in VERIFIERS:
                ok = verdicts(st, n, bands)[0]
                vmis += int(sum(bool(ok[x["round"] - 1]) != x[n]["accepted"] for x in log))
    out["replay_check"] = {"runs": len(runs), "max_abs_stat_diff": sdiff, "verdict_mismatches": vmis}
    st = hr.run(mc_subsets(np.random.default_rng(
        seed_int(entropy, "joint-ghost-mc-s%d-v%d" % (s, v))), n_mc))
    w = slice(ATTACK_START - 1, LAST_ROUND)
    mc = {"sequences": n_mc}
    for n in names:
        ok = verdicts(st, n, bands)
        mc[n] = {"all_rounds": float(ok.all(1).mean()), "rate_12_35": float(ok[:, w].mean()),
                 "min_round_rate_12_35": float(ok[:, w].mean(0).min()),
                 "per_round_rate": ok.mean(0)}
    mc["medians_12_35"] = {k: float(np.median(st[k][:, w])) for k in STATS}
    mc["stability_below_0.5_12_35"] = float((st["label_stability"][:, w] < gx.MIN_STAB).mean())
    out["monte_carlo"] = mc
    rows = gx.universal_check(tr)
    out["universal"] = universal_summary(rows, bands)
    pr = np.asarray(payload_ok, bool)
    cr = {x["round"]: x for x in chain}
    joint = {}
    for which, (a, b) in JOINT_COMMIT.items():
        ch = np.asarray([cr[r][a]["accepted"] and cr[r][b]["accepted"]
                         for r in range(1, R + 1)])
        okmc = commit_ok(st, bands, which)
        p = okmc[:, w].mean(0)
        joint[which] = {
            "chain_head_commit_rounds_12_35": int(ch[w].sum()),
            "chain_head_joint_client_rounds": int((pr & ch[w]).sum()),
            "chain_head_commit_all_rounds": bool(ch.all()),
            "chain_head_joint_trajectory": bool(pr.all() and ch.all()),
            "mc_joint_client_round_rate": float(np.mean(pr * p)),
            "mc_commit_all_rounds": float(okmc.all(1).mean()),
            "mc_joint_trajectory": float(pr.all() * okmc.all(1).mean())}
    out["joint"] = joint
    out["margins_chain_head"] = margins_commit(chain, bands)
    return out


def margins_commit(chain: List[dict], bands: dict) -> dict:
    b = bands["window_12_35"]
    w = [x for x in chain if ATTACK_START <= x["round"] <= LAST_ROUND]
    rat = [min(x["proto_ratio"] - b["proto_ratio"][0], b["proto_ratio"][1] - x["proto_ratio"]) for x in w]
    stab = [min(x["label_stability"] - gx.MIN_STAB, x["label_stability"] - b["label_stability"][0],
                b["label_stability"][1] - x["label_stability"]) for x in w]
    dr = [min(x["max_drift"] - b["max_drift"][0], b["max_drift"][1] - x["max_drift"]) for x in w
          if x.get("max_drift_rate") is not None]
    return {"proto_ratio": rat, "label_stability": stab, "max_drift": dr}


def evaluate_run(job) -> dict:
    (key, ppath, signature, apath, thetas, bands, thr, n_mc, n_beacon, entropy_hex, exact) = job
    torch.set_num_threads(1)
    install_nn(True)
    t0 = time.perf_counter()
    entropy = bytes.fromhex(entropy_hex)
    level, beta, s, v = key
    got = load_payloads(ppath, signature)
    if got is None:
        raise RuntimeError("payloads of %s missing or stale" % (key,))
    arch = ce.load_archive(apath)
    Cseq = payload_sequence(arch, v, got)
    on_grid = max(float(np.max(np.abs(idm.dequantize(C) - C))) for C in Cseq)
    override = {r: Cseq[r - 1] for r in range(ATTACK_START, len(Cseq) + 1)}
    sc = replay_scores(arch, v, override, root_map(s, arch), thr["norm"])
    fl = flags(sc, thr)
    rounds = sc["rounds"].tolist()
    mat_rounds = got["mat_rounds"].tolist()
    if mat_rounds != rounds:
        raise RuntimeError("round mismatch for %s" % (key,))
    out = {"key": {"level": level, "beta": beta, "seed": s, "victim": v},
           "payload_on_grid_max_abs": on_grid,
           "payload": {"rounds": rounds,
                       **{k: sc[k][:, v] for k in ("fused", "auditor", "norm_bound", "l2")},
                       "flags": {k: fl[k][:, v] for k in PAYLOAD_FLAGS + ("payload_any",)},
                       "fused_vs_logged_max_abs": float(np.max(np.abs(sc["fused"][:, v] - got["realised"]))),
                       "mat_vs_logged_max_abs": float(np.nanmax(np.abs(sc["fused"] - got["mat"])))},
           "_scores": {k: sc[k] for k in ("fused", "auditor", "norm_bound", "l2")},
           "ratio": {"rounds": list(range(2, len(Cseq) + 1)),
                     "published": [prototype_drift_ratio(Cseq[r - 1], Cseq[r - 2])
                                   for r in range(2, len(Cseq) + 1)],
                     "honest_seat": [prototype_drift_ratio(arch["C"][v][r], arch["C"][v][r - 1])
                                     for r in range(2, len(Cseq) + 1)]}}
    payload_ok = ~fl["payload_any"][:, v]
    declared = np.asarray(arch["present"][v][ATTACK_START], bool)
    out["variants"], nn_bad = {}, 0
    for name, theta in thetas.items():
        rng = np.random.default_rng(seed_int(entropy, "joint-ghost-s%d-v%d" % (s, v)))
        tr = fabricate_table(Cseq, declared, theta, rng)
        nn_bad += nn_spot_check(tr, np.random.default_rng(1000 * s + v))
        out["variants"][name] = evaluate_variant(
            tr, s, v, bands, entropy, n_mc, n_beacon if name == "primary" else 0, payload_ok,
            exact and name == "primary")
    out["nn_fast_spot_check_mismatches"] = nn_bad
    out["seconds"] = time.perf_counter() - t0
    return out


def eval_path(parts: str, key, tag: str) -> str:
    level, beta, s, v = key
    return os.path.join(parts, "eval", "F_%s_b%g_s%d_v%d_%s.json" % (level_file(level), beta, s, v, tag))


def summarise_level(runs: List[dict], bands: dict, thr: dict) -> dict:
    out = {"n_runs": len(runs)}
    test = [r for r in runs if r["key"]["seed"] in TEST_SEEDS]
    pl = {}
    for f in PAYLOAD_FLAGS + ("payload_any",):
        rec = {}
        for nm, rr in (("TEST", test), ("all", runs)):
            k = sum(int(np.sum(r["payload"]["flags"][f])) for r in rr)
            n = sum(len(r["payload"]["flags"][f]) for r in rr)
            rec["flagged_" + nm] = rate(k, n)
            rec["runs_flagged_" + nm] = rate(sum(int(np.any(r["payload"]["flags"][f])) for r in rr), len(rr))
        if f != "payload_any":
            pos, neg = [], []
            for r in runs:
                m = r["_scores"][f]
                v = r["key"]["victim"]
                others = np.ones(m.shape[1], bool); others[v] = False
                pos.extend(m[:, v].tolist()); neg.extend(m[:, others].ravel().tolist())
            if f == "l2":
                rec["auc_upper"] = ce.roc_auc(pos, neg)
                rec["auc_lower"] = ce.roc_auc([-x for x in pos], [-x for x in neg])
            else:
                rec["auc"] = ce.roc_auc(pos, neg)
            rec["n_pos"], rec["n_neg"] = len(pos), len(neg)
        pl[f] = rec
    pm = {}
    for f, t in (("fused", thr["fused"]), ("auditor", thr["auditor_raw"]),
                 ("norm_bound", thr["normalised"]["norm_bound"])):
        mg = np.concatenate([t - np.asarray(r["payload"][f]) for r in runs])
        pm[f] = {"min": float(np.nanmin(mg)), "median": float(np.nanmedian(mg))}
    lo, hi = thr["l2"]
    l2 = np.concatenate([np.asarray(r["payload"]["l2"]) for r in runs])
    pm["l2"] = {"min": float(np.min(np.minimum(l2 - lo, hi - l2))),
                "median": float(np.median(np.minimum(l2 - lo, hi - l2))),
                "l2_median": float(np.median(l2)), "l2_range": [float(l2.min()), float(l2.max())]}
    pl["margins_to_threshold"] = pm
    out["payload_layer"] = pl
    b = bands["window_12_35"]["proto_ratio"]
    idx = [i for i, rr in enumerate(runs[0]["ratio"]["rounds"]) if rr >= ATTACK_START]
    g = np.asarray([[r["ratio"]["published"][i] for i in idx] for r in runs])
    h = np.asarray([[r["ratio"]["honest_seat"][i] for i in idx] for r in runs])
    out["published_ratio_12_35"] = {
        "band": b, "inside": rate(((g >= b[0]) & (g <= b[1])).sum(), g.size),
        "below": int((g < b[0]).sum()), "above": int((g > b[1]).sum()),
        "median": float(np.median(g)), "range": [float(g.min()), float(g.max())],
        "round_12_median": float(np.median(g[:, 0])), "round_12_range": [float(g[:, 0].min()), float(g[:, 0].max())],
        "rounds_13_35_inside": rate(((g[:, 1:] >= b[0]) & (g[:, 1:] <= b[1])).sum(), g[:, 1:].size),
        "honest_seat_median": float(np.median(h)),
        "honest_seat_inside": rate(((h >= b[0]) & (h <= b[1])).sum(), h.size),
        "runs_with_every_round_inside": int(sum(((x >= b[0]) & (x <= b[1])).all() for x in g))}
    out["variants"] = {}
    for vname in runs[0]["variants"]:
        V = {}
        for nm, rr in (("TEST", test), ("all", runs)):
            if not rr:
                V[nm] = None
                continue
            vv = [r["variants"][vname] for r in rr]
            pay = [r["payload"]["flags"]["payload_any"] for r in rr]
            blk = {"runs": len(vv)}
            blk["chain_head_rounds_12_35_accepted"] = {
                n: rate(sum(sum(x[n] for x in q["chain_head"]["per_round"] if x["round"] >= ATTACK_START)
                            for q in vv), (LAST_ROUND - ATTACK_START + 1) * len(vv)) for n in VERIFIERS}
            blk["chain_head_trajectories_all_rounds_accepted"] = {
                n: rate(sum(all(x[n] for x in q["chain_head"]["per_round"]) for q in vv), len(vv))
                for n in VERIFIERS}
            blk["mc_rate_12_35_mean"] = {n: float(np.mean([q["monte_carlo"][n]["rate_12_35"] for q in vv]))
                                         for n in VERIFIERS}
            blk["mc_all_rounds_mean"] = {n: float(np.mean([q["monte_carlo"][n]["all_rounds"] for q in vv]))
                                         for n in VERIFIERS}
            blk["mc_all_rounds_min"] = {n: float(np.min([q["monte_carlo"][n]["all_rounds"] for q in vv]))
                                        for n in VERIFIERS}
            nr = (LAST_ROUND - ATTACK_START + 1) * len(vv)
            blk["joint"] = {}
            for which in JOINT_COMMIT:
                j = [q["joint"][which] for q in vv]
                blk["joint"][which] = {
                    "chain_head_client_rounds": rate(sum(x["chain_head_joint_client_rounds"] for x in j), nr),
                    "chain_head_commit_client_rounds": rate(sum(x["chain_head_commit_rounds_12_35"] for x in j), nr),
                    "chain_head_trajectories": rate(sum(x["chain_head_joint_trajectory"] for x in j), len(j)),
                    "mc_client_round_rate_mean": float(np.mean([x["mc_joint_client_round_rate"] for x in j])),
                    "mc_trajectory_rate_mean": float(np.mean([x["mc_joint_trajectory"] for x in j])),
                    "payload_unflagged_client_rounds": rate(sum(int((~np.asarray(p, bool)).sum()) for p in pay), nr),
                    "payload_unflagged_trajectories": rate(sum(int(not np.any(p)) for p in pay), len(pay))}
            blk["universal"] = {
                "deployed_w5_rounds_passing_every_challenge": rate(
                    sum(q["universal"]["deployed_w5"]["rounds_passing_every_admissible_challenge"] for q in vv),
                    sum(q["universal"]["deployed_w5"]["rounds_checked"] for q in vv)),
                "deployed_w0_rounds_passing_every_challenge": rate(
                    sum(q["universal"]["deployed_w0"]["rounds_passing_every_admissible_challenge"] for q in vv),
                    sum(q["universal"]["deployed_w0"]["rounds_checked"] for q in vv)),
                "window_ratio_in_band": rate(sum(q["universal"]["window_12_35"]["ratio_in_band"] for q in vv),
                                             sum(q["universal"]["window_12_35"]["rounds"] for q in vv)),
                "window_stability_interval_inside_band": rate(
                    sum(q["universal"]["window_12_35"]["stability_interval_inside_band"] for q in vv),
                    sum(q["universal"]["window_12_35"]["rounds"] for q in vv))}
            blk["mc_medians_12_35_median_over_runs"] = {
                k: float(np.median([q["monte_carlo"]["medians_12_35"][k] for q in vv])) for k in STATS}
            mg = {k: np.concatenate([np.asarray(q["margins_chain_head"][k], float) for q in vv]) for k in STATS}
            blk["chain_head_margins_12_35"] = {k: {"min": float(m.min()) if m.size else None,
                                                   "median": float(np.median(m)) if m.size else None,
                                                   "negative": int((m < 0).sum()), "n": int(m.size)}
                                               for k, m in mg.items()}
            if nm == "all":
                blk["beacon_sequences_all_rounds"] = {
                    n: rate(sum(q["beacon"][n]["sequences_all_rounds"] for q in vv if "beacon" in q),
                            sum(q["beacon"]["sequences"] for q in vv if "beacon" in q))
                    for n in BEACON_VERIFIERS} if any("beacon" in q for q in vv) else None
                blk["checks"] = {
                    "replay_max_abs_stat_diff": max(q["replay_check"]["max_abs_stat_diff"] for q in vv),
                    "replay_verdict_mismatches": int(sum(q["replay_check"]["verdict_mismatches"] for q in vv)),
                    "stat_mismatch_across_verifiers": int(sum(q["chain_head"]["stat_mismatch_across_verifiers"]
                                                              for q in vv))}
            V[nm] = blk
        out["variants"][vname] = V
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("pilot", "regen", "calibrate", "evaluate", "part1", "all"))
    ap.add_argument("--levels", default=DEFAULT_LEVELS)
    ap.add_argument("--betas", default="1")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--victims", default="0,1")
    ap.add_argument("--workers", type=int, default=4, help="worker processes (capped at 4)")
    ap.add_argument("--mc-seqs", type=int, default=10000,
                    help="uniformly random challenge sequences per evaluated table")
    ap.add_argument("--cal-mc-seqs", type=int, default=2000,
                    help="random challenge sequences per CAL trajectory in the generator search")
    ap.add_argument("--beacon-seqs", type=int, default=5,
                    help="simulated beacon interactions per run (checked against the replay)")
    ap.add_argument("--bands-file", default="",
                    help="comma-separated paired honest summaries replacing the provisional "
                         "bands (files with 'verify_log' rows or with 'bands')")
    ap.add_argument("--honest-dev", default=os.path.join(RESULTS, "fedproto_abl_pp_s2.json"),
                    help="held-out honest log for the commitment rates (comma-separated allowed)")
    ap.add_argument("--entropy", default=DEFAULT_ENTROPY)
    ap.add_argument("--parts", default=os.path.join(REPO, ".reproduction", "joint_ghost"))
    ap.add_argument("--pilot-out", default="")
    ap.add_argument("--out", default=os.path.join(RESULTS, "joint_ghost.json"))
    a = ap.parse_args()
    t_start = time.time()
    torch.set_num_threads(1)
    install_nn(True)
    workers = max(1, min(4, a.workers))
    ints = lambda s: [int(x) for x in s.split(",") if x.strip()]
    levels = [parse_level(x.strip()).name for x in a.levels.split(",") if x.strip()]
    betas = [float(x) for x in a.betas.split(",") if x.strip()]
    seeds, victims = ints(a.seeds), ints(a.victims)
    entropy = bytes.fromhex(a.entropy)
    paths = lae.archive_paths(ARCHIVES, PREFIX)
    thr = frozen_thresholds()

    calib = honest_calibration(paths)
    print("limited_access CAL constants reproduced: %s (tau F %.6f, tau P %.6f)"
          % (calib["release_check"]["identical"], calib["tau"]["F"], calib["tau"]["P"]), flush=True)
    if a.bands_file:
        dflt, _ = default_bands(paths)
        bands, bands_meta = bands_from_files([p for p in a.bands_file.split(",") if p], dflt)
        bands_status = "paired honest (%s)" % a.bands_file
    else:
        bands, bands_meta = default_bands(paths)
        bands_status = ("PROVISIONAL: ratio from CAL archives, stability and max drift from "
                        "fedproto_abl_pp_s0/s1 (not the archive trajectories); to be replaced by "
                        "paired honest runs via --bands-file")
    print("bands 12-35: %s" % {k: [round(x, 4) for x in v] for k, v in bands["window_12_35"].items()},
          flush=True)

    if a.stage == "pilot":
        t = attack_task(calib, paths, "current", 1.0, 0, 0)
        p = payload_path(a.parts, t)
        t0 = time.time()
        info = regenerate((t, p))
        got = load_payloads(p, t["signature"])
        chk = regeneration_check(t, got)
        arch = ce.load_archive(paths[0])
        Cseq = payload_sequence(arch, 0, got)
        override = {r: Cseq[r - 1] for r in range(ATTACK_START, len(Cseq) + 1)}
        sc = replay_scores(arch, 0, override, root_map(0, arch), thr["norm"])
        fl = flags(sc, thr)
        b = bands["window_12_35"]["proto_ratio"]
        rows = []
        for r in range(2, len(Cseq) + 1):
            g = prototype_drift_ratio(Cseq[r - 1], Cseq[r - 2])
            h = prototype_drift_ratio(arch["C"][0][r], arch["C"][0][r - 1])
            rows.append({"round": r, "published_ratio": g, "honest_seat_ratio": h,
                         "inside_band": bool(b[0] <= g <= b[1])})
        print("\nPILOT  F current beta=1 seed 0 victim 0   (%s, %.0fs)"
              % ("cached" if info["cached"] else "regenerated", time.time() - t0))
        print("regeneration: %s" % chk)
        print("band (rounds 12-35) for the drift ratio: [%.4f, %.4f]" % tuple(b))
        print("%5s %10s %10s %7s | %8s %8s %8s %8s  flags" % ("round", "published", "honest", "inband",
                                                              "fused", "auditor", "normb", "l2"))
        for x in rows:
            r = x["round"]
            extra = ""
            if r >= ATTACK_START:
                i = r - ATTACK_START
                extra = "| %8.3f %8.3f %8.3f %8.2f  %s" % (
                    sc["fused"][i, 0], sc["auditor"][i, 0], sc["norm_bound"][i, 0], sc["l2"][i, 0],
                    ",".join(f for f in PAYLOAD_FLAGS if fl[f][i, 0]) or "-")
            print("%5d %10.4f %10.4f %7s %s" % (r, x["published_ratio"], x["honest_seat_ratio"],
                                                x["inside_band"], extra))
        att = [x for x in rows if x["round"] >= ATTACK_START]
        summ = {"inside": sum(x["inside_band"] for x in att), "n": len(att),
                "below": sum(x["published_ratio"] < b[0] for x in att),
                "above": sum(x["published_ratio"] > b[1] for x in att),
                "median": float(np.median([x["published_ratio"] for x in att])),
                "honest_seat_median": float(np.median([x["honest_seat_ratio"] for x in att]))}
        print("rounds 12-35: %d/%d inside the band (%d below, %d above); median %.4f (honest seat %.4f)"
              % (summ["inside"], summ["n"], summ["below"], summ["above"], summ["median"],
                 summ["honest_seat_median"]))
        if a.pilot_out:
            with open(a.pilot_out, "w", encoding="utf-8") as fh:
                json.dump({"regeneration": chk, "band": b, "rows": rows, "summary": summ,
                           "payload": {k: sc[k][:, 0] for k in ("fused", "auditor", "norm_bound", "l2")},
                           "flags": {k: fl[k][:, 0] for k in PAYLOAD_FLAGS}}, fh, indent=1, default=jsonable)
        return 0

    out = {"meta": {"script": "experiments/joint_ghost_eval.py",
                    "command": "python experiments/joint_ghost_eval.py" + "".join(
                        " --%s %s" % (k, getattr(a, k.replace("-", "_"))) for k in
                        ("levels", "betas", "seeds", "victims", "mc-seqs", "cal-mc-seqs",
                         "beacon-seqs", "bands-file", "honest-dev")
                        if getattr(a, k.replace("-", "_")) != ap.get_default(k.replace("-", "_"))),
                    "entropy": a.entropy, "environment": environment()},
           "config": {"levels": levels, "betas": betas, "seeds": seeds, "victims": victims,
                      "cal_seeds": list(CAL_SEEDS), "dev_seeds": list(DEV_SEEDS),
                      "test_seeds": list(TEST_SEEDS), "attack_start": ATTACK_START,
                      "scored_rounds": [ATTACK_START, LAST_ROUND], "phi": 0.0, "defense": "F",
                      "m_probes": M, "t_reveal": T, "mc_sequences": a.mc_seqs,
                      "cal_mc_sequences": a.cal_mc_seqs, "beacon_sequences": a.beacon_seqs,
                      "verifiers": {k: {"warmup": w, "bands": b, "bands_from_round": f}
                                    for k, (w, b, f) in VERIFIERS.items()},
                      "joint_payload_flags": list(PAYLOAD_FLAGS),
                      "joint_commitment": {k: list(v) for k, v in JOINT_COMMIT.items()}}}

    honest = {}
    for s in SEEDS:
        arch = ce.load_archive(paths[s])
        honest[s] = replay_scores(arch, None, {}, root_map(s, arch), thr["norm"])
    thr.update(fit_extra(honest, thr))
    g = json.load(open(os.path.join(RESULTS, "combined_grid.json"), encoding="utf-8"))
    hp = honest_payload_layer(honest, thr)
    rel = {}
    for k, gk in [("fused", "combined_max"), ("auditor", "suspicion")] + [(b_, b_) for b_ in RULES6]:
        rel[k] = {"here": hp["TEST"][k]["client_round"]["rate"], "released": g["honest_fpr_TEST"][gk]["realised"]}
    l2ok, n0, nflt = 0, 0, 0
    for s in (3, 4, 5, 6, 7):
        arch = ce.load_archive(paths[s])
        for r in range(ATTACK_START, LAST_ROUND + 1):
            for c in range(CLIENTS):
                x = float(np.linalg.norm(np.asarray(arch["C"][c][r], np.float64) - arch["cbar"][r - 1]))
                l2ok += int(thr["l2"][0] <= x <= thr["l2"][1]); n0 += 1
    p0 = float(np.mean(np.concatenate([honest[s]["fltrust"].ravel() for s in SEEDS]) >= 0.0))
    pp = json.load(open(os.path.join(RESULTS, "published_predicates.json"), encoding="utf-8"))
    checks = {"honest_TEST_fpr_vs_combined_grid": rel,
              "honest_TEST_fpr_identical": all(v["here"] == v["released"] for v in rel.values()),
              "two_sided_l2_honest_acceptance_seeds_3_7": {
                  "here": rate(l2ok, n0),
                  "released": pp["combined"]["two-sided l2 (sensitivity)"]["joint_1pct"]["honest"]},
              "fltrust_p0_seeds_0_4": {"here": p0, "released": pp["meta"]["checks"]["fltrust_p0"]["root"]["released"]}}
    print("honest checks: TEST FPR identical to combined_grid %s; l2 two-sided %d/%d; FLTrust p0 %.5f"
          % (checks["honest_TEST_fpr_identical"], l2ok, n0, p0), flush=True)
    out["thresholds"] = {"fused": thr["fused"], "auditor_raw": thr["auditor_raw"],
                         "rules_normalised": {b_: thr["normalised"][b_] for b_ in RULES6},
                         "l2_two_sided": list(thr["l2"]), "bulyan_raw_cal1pct": thr["bulyan"],
                         "fltrust_cal1pct": thr["fltrust"], "sources": thr["sources"]}
    out["honest_payload_layer"] = hp
    out["checks"] = checks
    out["calibration_release_check"] = calib["release_check"]

    if a.stage in ("part1", "all"):
        t0 = time.time()
        out["part1_ghost_payloads"] = part1(paths, thr)
        out["part1_ghost_payloads"]["seconds"] = time.time() - t0
        for kind, blk in out["part1_ghost_payloads"]["ghosts"].items():
            d = blk["detectors"]
            print("PART 1 %-10s " % kind + "  ".join(
                "%s %d/%d AUC %.3f" % (k, d[k]["flagged_TEST"]["k"], d[k]["flagged_TEST"]["n"], d[k]["auc"])
                for k in DETECTORS), flush=True)
        if a.stage == "part1":
            return 0

    tasks = [attack_task(calib, paths, lvl, b_, s, v) for lvl in levels for b_ in betas
             for s in seeds for v in victims]
    cost = lambda t: (1.0 + parse_level(t["level"]).queries / 64.0)
    todo = [t for t in tasks if load_payloads(payload_path(a.parts, t), t["signature"]) is None]
    print("%d passes, %d cached, %d to regenerate" % (len(tasks), len(tasks) - len(todo), len(todo)),
          flush=True)
    t0 = time.time()
    if todo:
        todo.sort(key=lambda t: -cost(t))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(regenerate, (t, payload_path(a.parts, t))): t for t in todo}
            for i, f in enumerate(as_completed(futs), 1):
                t, r = futs[f], f.result()
                print("  regen %-11s beta %g seed %d victim %d %6.1fs  [%d/%d, %.0fs]"
                      % (t["level"], t["beta"], t["seed"], t["victim"], r["seconds"], i, len(todo),
                         time.time() - t0), flush=True)
    regen = {}
    for t in tasks:
        got = load_payloads(payload_path(a.parts, t), t["signature"])
        regen["%s|beta=%g|s%d|v%d" % (t["level"], t["beta"], t["seed"], t["victim"])] = dict(
            regeneration_check(t, got), seconds=float(got["seconds"]))
    tot = sum(x["rounds"] for x in regen.values())
    match = sum(x.get("digests_match_released", 0) for x in regen.values())
    out["regeneration"] = {
        "passes": len(regen), "rounds": tot, "digests_match_released": match,
        "passes_in_release_cache": int(sum(x["released_pass_in_cache"] for x in regen.values())),
        "passes_all_digests_match": int(sum(x.get("digests_match_released", -1) == x["rounds"]
                                            for x in regen.values())),
        "max_realised_abs_diff": max(x.get("realised_max_abs_diff", 0.0) for x in regen.values()),
        "compute_s": round(sum(x["seconds"] for x in regen.values()), 1),
        "wall_s_this_call": round(time.time() - t0, 1), "per_pass": regen}
    print("regeneration: %d/%d crafted payloads identical to the released passes (%d/%d passes)"
          % (match, tot, out["regeneration"]["passes_all_digests_match"], len(regen)), flush=True)
    if a.stage == "regen":
        return 0

    out["bands"] = {"status": bands_status, "bands": bands, "meta": bands_meta}
    out["honest_ratio_band_12_35"] = honest_ratio_band(paths, bands["window_12_35"]["proto_ratio"])
    cal_rows =[e for s in CAL_SEEDS for e in log_rows(
        os.path.join(RESULTS, "fedproto_abl_pp_s%d.json" % s), ATTACK_START, LAST_ROUND)]
    target = {"label_stability": float(np.median([e["stability"] for e in cal_rows])),
              "max_drift": float(np.median([e["max_drift"] for e in cal_rows]))}
    out["honest_commitment"] = {
        "DEV": [honest_commitment(p, bands) for p in a.honest_dev.split(",") if p],
        "CAL_in_sample": [honest_commitment(os.path.join(RESULTS, "fedproto_abl_pp_s%d.json" % s), bands)
                          for s in CAL_SEEDS]}

    cal_items = []
    for lvl, s, v in CAL_RUNS:
        t = attack_task(calib, paths, lvl, 1.0, s, v)
        cal_items.append((payload_path(a.parts, t), paths[s], s, v))
        if load_payloads(cal_items[-1][0], t["signature"]) is None:
            raise SystemExit("CAL trajectory %s missing: run the regeneration first" % ((lvl, s, v),))
    ckey = hashlib.sha256(json.dumps([GEN_GRID, bands["window_12_35"], a.cal_mc_seqs, a.entropy,
                                      [os.path.basename(x[0]) for x in cal_items], target],
                                     sort_keys=True, default=str).encode()).hexdigest()[:16]
    cpath = os.path.join(a.parts, "generator_calibration_%s.json" % ckey)
    if os.path.exists(cpath):
        gcal = json.load(open(cpath, encoding="utf-8"))
    else:
        t0 = time.time()
        gcal = calibrate_generator(cal_items, bands, a.cal_mc_seqs, a.entropy, target, workers)
        gcal["seconds"] = time.time() - t0
        os.makedirs(a.parts, exist_ok=True)
        with open(cpath, "w", encoding="utf-8") as fh:
            json.dump(gcal, fh, indent=1, default=jsonable)
    sp, ss = gcal["selected_primary"], gcal["selected_secondary"]
    print("generator: primary %s (full %.4f, per-round %.4f, distance %.3f); secondary %s (distance %.3f)"
          % (sp["theta"], sp["objective_full"], sp["objective_per_round"], sp["distance"],
             ss["theta"], ss["distance"]), flush=True)
    out["generator_calibration"] = gcal
    if a.stage == "calibrate":
        return 0

    thetas = {"primary": sp["theta"], "secondary": ss["theta"]}
    thr_job = {k: thr[k] for k in ("norm", "normalised", "fused", "auditor_raw", "l2", "bulyan", "fltrust")}
    tag = hashlib.sha256(json.dumps([thetas, bands, a.mc_seqs, a.beacon_seqs, a.entropy, thr_job],
                                    sort_keys=True, default=str).encode()).hexdigest()[:12]
    jobs, results = [], {}
    for t in tasks:
        key = (t["level"], t["beta"], t["seed"], t["victim"])
        ep = eval_path(a.parts, key, tag)
        if os.path.exists(ep):
            results[key] = json.load(open(ep, encoding="utf-8"))
            continue
        jobs.append((key, payload_path(a.parts, t), t["signature"], t["path"], thetas, bands, thr_job,
                     a.mc_seqs, a.beacon_seqs, a.entropy,
                     (t["level"], t["seed"], t["victim"]) in EXACT_CHECK))
    print("%d evaluations, %d cached, %d to run" % (len(tasks), len(results), len(jobs)), flush=True)
    t0 = time.time()
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(evaluate_run, j): j[0] for j in jobs}
            for i, f in enumerate(as_completed(futs), 1):
                key, r = futs[f], f.result()
                ep = eval_path(a.parts, key, tag)
                os.makedirs(os.path.dirname(ep), exist_ok=True)
                with open(ep + ".tmp", "w", encoding="utf-8") as fh:
                    json.dump(r, fh, default=jsonable)
                os.replace(ep + ".tmp", ep)
                results[key] = json.load(open(ep, encoding="utf-8"))
                j = r["variants"]["primary"]["joint"]["w5"]
                print("  eval %-11s beta %g seed %d victim %d %5.1fs  payload unflagged %2d/24  "
                      "commit(chain) %2d/24  joint(chain) %2d/24  MC joint c-r %.3f traj %.3f  [%d/%d, %.0fs]"
                      % (key[0], key[1], key[2], key[3], r["seconds"],
                         int(np.sum(~np.asarray(r["payload"]["flags"]["payload_any"], bool))),
                         j["chain_head_commit_rounds_12_35"], j["chain_head_joint_client_rounds"],
                         j["mc_joint_client_round_rate"], j["mc_joint_trajectory"], i, len(jobs),
                         time.time() - t0), flush=True)
    for r in results.values():
        r["_scores"] = {k: np.asarray(v, float) for k, v in r["_scores"].items()}
    cells = {}
    for lvl in levels:
        for b_ in betas:
            rr = [results[(lvl, b_, s, v)] for s in seeds for v in victims]
            cells["%s|beta=%g" % (lvl, b_)] = summarise_level(rr, bands, thr)
    out["cells"] = cells
    la = json.load(open(os.path.join(RESULTS, "limited_access_grid.json"), encoding="utf-8"))
    out["checks"]["fused_detection_vs_limited_access_grid"] = {
        k: {"here_TEST": c["payload_layer"]["fused"]["flagged_TEST"]["k"],
            "released_TEST": la["cells"]["F|%s" % k]["flagged_client_rounds_TEST"],
            "here_auc": c["payload_layer"]["fused"]["auc"], "released_auc": la["cells"]["F|%s" % k]["auc"]}
        for k, c in cells.items() if "F|%s" % k in la["cells"]}
    out["checks"]["fused_rescore_vs_logged_max_abs"] = max(r["payload"]["fused_vs_logged_max_abs"]
                                                           for r in results.values())
    out["checks"]["mat_rescore_vs_logged_max_abs"] = max(r["payload"]["mat_vs_logged_max_abs"]
                                                         for r in results.values())
    out["checks"]["payload_on_grid_max_abs"] = max(r["payload_on_grid_max_abs"] for r in results.values())
    out["checks"]["nn_fast_spot_check_mismatches"] = int(sum(r["nn_fast_spot_check_mismatches"]
                                                             for r in results.values()))
    out["checks"]["exact_nn_chain_head_reruns"] = {
        "%s|s%d|v%d" % (r["key"]["level"], r["key"]["seed"], r["key"]["victim"]):
            r["variants"]["primary"]["chain_head"].get("exact_nn_rerun_identical")
        for r in results.values() if "exact_nn_rerun_identical" in r["variants"]["primary"]["chain_head"]}
    out["per_run"] = [{k: v for k, v in r.items() if k != "_scores"} for r in
                      sorted(results.values(), key=lambda r: (levels.index(r["key"]["level"]),
                                                              r["key"]["beta"], r["key"]["seed"],
                                                              r["key"]["victim"]))]
    out["meta"]["wall_s_this_call"] = round(time.time() - t_start, 1)
    out["meta"]["eval_compute_s"] = round(sum(r["seconds"] for r in results.values()), 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=jsonable)
    report(out)
    print("\nwrote %s (%.0fs)" % (a.out, time.time() - t_start))
    return 0


def report(out: dict) -> None:
    print("\n" + "=" * 150)
    print("JOINT GHOST (primary table). TEST seeds 3,4: 96 attacked client-rounds, 4 trajectories.")
    print("=" * 150)
    h = ("%-12s | %5s %5s %5s %5s %5s | %7s | %7s %7s | %7s %7s | %7s %6s | %6s"
         % ("level", "fused", "aud", "normb", "l2", "any", "ratioIn", "dep5", "two5", "jointCR", "mcCR",
            "jTraj", "mcTraj", "univ5"))
    print(h); print("-" * len(h))
    for k, c in out["cells"].items():
        p = c["payload_layer"]
        v = c["variants"]["primary"]["TEST"]
        j = v["joint"]["w5"]
        print("%-12s | %5d %5d %5d %5d %5d | %7s | %7s %7s | %7s %7.3f | %7s %6.3f | %6s"
              % (k.split("|")[0], p["fused"]["flagged_TEST"]["k"], p["auditor"]["flagged_TEST"]["k"],
                 p["norm_bound"]["flagged_TEST"]["k"], p["l2"]["flagged_TEST"]["k"],
                 p["payload_any"]["flagged_TEST"]["k"],
                 "%d/%d" % (c["published_ratio_12_35"]["inside"]["k"], c["published_ratio_12_35"]["inside"]["n"]),
                 "%d" % v["chain_head_rounds_12_35_accepted"]["deployed_w5"]["k"],
                 "%d" % v["chain_head_rounds_12_35_accepted"]["two_sided_w5"]["k"],
                 "%d" % j["chain_head_client_rounds"]["k"], j["mc_client_round_rate_mean"],
                 "%d/%d" % (j["chain_head_trajectories"]["k"], j["chain_head_trajectories"]["n"]),
                 j["mc_trajectory_rate_mean"],
                 "%d/%d" % (v["universal"]["deployed_w5_rounds_passing_every_challenge"]["k"],
                            v["universal"]["deployed_w5_rounds_passing_every_challenge"]["n"])))
    print("\npayload columns: flagged attacked TEST client-rounds (of 96); ratioIn: published drift "
          "ratio inside the 12-35 band (all seeds, of 240);\ndep5/two5: chain-head commitment "
          "acceptances in rounds 12-35 (of 96); jointCR: chain-head joint evasions (of 96); mcCR: "
          "mean joint\nper-client-round rate under random challenges; jTraj/mcTraj: trajectories "
          "evading jointly; univ5: rounds 6-35 passing every admissible challenge.")
    hp = out["honest_payload_layer"]["TEST"]
    print("\nhonest TEST client-rounds flagged: " + "  ".join(
        "%s %d/%d" % (k, hp[k]["client_round"]["k"], hp[k]["client_round"]["n"]) for k in PAYLOAD_FLAGS + ("payload_any",)))
    hr = out["honest_ratio_band_12_35"]
    print("honest drift ratio inside the 12-35 band: " + "  ".join(
        "%s %d/%d" % (k, hr[k]["client_rounds_inside"]["k"], hr[k]["client_rounds_inside"]["n"]) for k in hr))
    for h in out["honest_commitment"]["DEV"]:
        print("honest commitment (%s, rounds 12-35): deployed-w5 accepted %d-%d/%d, two-sided-w5 (12-35 bands) %d-%d/%d"
              % (h["source"], h["deployed"]["client_round_lower"]["k"], h["deployed"]["client_round_upper"]["k"],
                 h["n_rows"], h["two_sided_window_12_35"]["client_round_lower"]["k"],
                 h["two_sided_window_12_35"]["client_round_upper"]["k"], h["n_rows"]))


if __name__ == "__main__":
    sys.exit(main())
