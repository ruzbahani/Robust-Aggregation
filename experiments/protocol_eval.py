from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import ast
import json
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from passportfl.baselines import BASELINES, score_round
from passportfl.innovation import InnovationAuditor, MAD_SCALE

ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "archives")
TARGETS = (0.10, 0.05, 0.01, 0.005)
MAD_FLOOR_GRID = (1e-12, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)
MIN_COHORT = 5


def attack_global_echo(arch, cid, r, rng):
    return arch["cbar"][r - 1].copy() if (r - 1) in arch["cbar"] else arch["C"][cid][r].copy()


def attack_peer_copy(arch, cid, r, rng):
    others = [j for j in arch["C"] if j != cid]
    return arch["C"][others[cid % len(others)]][r].copy()


def attack_lagged_self(arch, cid, r, rng):
    pub = arch["published"][cid]
    src = max(min(pub), r - 3)
    return pub[src].copy()


def attack_ar2_replay(arch, cid, r, rng):
    rounds = sorted(k for k in arch["C"][cid] if k < r)
    if len(rounds) < 3:
        return arch["C"][cid][r].copy()
    c1, c2, c3 = (arch["C"][cid][rounds[-1]], arch["C"][cid][rounds[-2]], arch["C"][cid][rounds[-3]])
    return c1 + (c1 - c2) + 0.25 * ((c1 - c2) - (c2 - c3))


def attack_noise(arch, cid, r, rng):
    base = arch["cbar"][r - 1] if (r - 1) in arch["cbar"] else arch["C"][cid][r]
    scale = np.linalg.norm(arch["C"][cid][r] - base) / max(1.0, np.sqrt(base.size))
    return base + rng.normal(0, scale, size=base.shape)


def attack_shrink(arch, cid, r, rng):
    base = arch["cbar"][r - 1] if (r - 1) in arch["cbar"] else arch["C"][cid][r]
    return base + 0.1 * (arch["C"][cid][r] - base)


ATTACKS = {
    "global_echo": attack_global_echo,
    "peer_copy": attack_peer_copy,
    "lagged_self": attack_lagged_self,
    "ar2_replay": attack_ar2_replay,
    "noise": attack_noise,
    "shrink": attack_shrink,
}
ATTACK_ORDER = list(ATTACKS)


def _strip_docstring(fn_node: ast.FunctionDef) -> ast.FunctionDef:
    body = list(fn_node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return ast.FunctionDef(name=fn_node.name, args=fn_node.args, body=body or [ast.Pass()],
                           decorator_list=[], returns=None, type_comment=None,
                           type_params=getattr(fn_node, "type_params", []))


def check_attack_definitions() -> Dict[str, object]:
    peer = os.path.join(os.path.dirname(os.path.abspath(__file__)), "innovation_eval.py")
    out = {"checked": False, "match": None, "detail": ""}
    if not os.path.exists(peer):
        out["detail"] = "innovation_eval.py not found"
        return out
    try:
        with open(peer, "r", encoding="utf-8") as f:
            theirs_src = f.read()
        with open(os.path.abspath(__file__), "r", encoding="utf-8") as f:
            ours_src = f.read()
        t_mod, o_mod = ast.parse(theirs_src), ast.parse(ours_src)

        def fns(mod):
            return {n.name: ast.dump(_strip_docstring(n)) for n in mod.body
                    if isinstance(n, ast.FunctionDef) and n.name.startswith("attack_")}

        def amap(mod):
            for n in mod.body:
                if isinstance(n, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "ATTACKS" for t in n.targets):
                    return ast.dump(n.value)
            return None

        tf, of = fns(t_mod), fns(o_mod)
        bad = [k for k in sorted(set(tf) | set(of)) if tf.get(k) != of.get(k)]
        map_ok = amap(t_mod) == amap(o_mod)
        out["checked"] = True
        out["match"] = (not bad) and map_ok
        out["detail"] = ("identical" if out["match"] else
                         f"DIVERGED: bodies={bad} attacks_map_equal={map_ok}")
    except Exception as e:
        out["detail"] = f"check failed: {e!r}"
    return out


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-16:
            break
    return h


def incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log1p(-x) + a * math.log(x)) * _beta_continued_fraction(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if incomplete_beta(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> Tuple[float, float, float]:
    if n <= 0:
        return float("nan"), 0.0, 1.0
    alpha = 1.0 - conf
    lo = 0.0 if k <= 0 else beta_ppf(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k >= n else beta_ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return k / n, lo, hi


def rate_record(k: int, n: int, conf: float = 0.95) -> Dict[str, object]:
    p, lo, hi = clopper_pearson(k, n, conf)
    if n <= 0:
        return {"k": k, "n": n, "point": None, "lo": None, "hi": None,
                "reported": "n/a", "zero_floor_applied": False}
    floor = (k == 0)
    rep = f"<={100*hi:.2f}%" if floor else f"{100*p:.2f}% [{100*lo:.2f}, {100*hi:.2f}]"
    return {"k": int(k), "n": int(n), "point": float(p), "lo": float(lo), "hi": float(hi),
            "reported": rep, "zero_floor_applied": bool(floor)}


_ARCH_CACHE: Dict[str, dict] = {}


def discover_archives(directory: str, prefix: str = "arch_cifar100_s") -> Dict[int, str]:
    out: Dict[int, str] = {}
    if not os.path.isdir(directory):
        return out
    for fn in sorted(os.listdir(directory)):
        if not (fn.startswith(prefix) and fn.endswith(".npz")):
            continue
        m = re.search(r"_s(\d+)\.npz$", fn)
        if not m:
            continue
        seed = int(m.group(1))
        if seed in out:
            raise RuntimeError(
                f"archive seed collision for seed {seed}: {os.path.basename(out[seed])} "
                f"vs {fn}. Tighten --archive-prefix so one configuration is selected."
            )
        out[seed] = os.path.join(directory, fn)
    return out


def load_archive(path: str) -> dict:
    if path in _ARCH_CACHE:
        return _ARCH_CACHE[path]
    z = np.load(path, allow_pickle=True)
    arch = {"C": z["C"].item(), "present": z["present"].item(), "cbar": z["cbar"].item()}
    if len(_ARCH_CACHE) >= 2:
        _ARCH_CACHE.clear()
    _ARCH_CACHE[path] = arch
    return arch


def archive_shape(arch: dict) -> Tuple[int, int]:
    clients = len(arch["C"])
    rounds = max(max(v) for v in arch["C"].values())
    return clients, rounds


def robust_z_score(vals: Dict[int, float], mad_floor: float) -> Dict[int, float]:
    fin = [v for v in vals.values() if np.isfinite(v)]
    if len(fin) < MIN_COHORT:
        return {c: float("nan") for c in vals}
    arr = np.asarray(fin, float)
    med = float(np.median(arr))
    scale = MAD_SCALE * float(np.median(np.abs(arr - med))) + mad_floor
    return {c: ((v - med) / scale if np.isfinite(v) and scale > 0 else float("nan"))
            for c, v in vals.items()}


def stat_names() -> List[str]:
    return ["suspicion"] + list(BASELINES) + [f"{b}_z" for b in BASELINES]


def score_one_archive(task: dict) -> dict:
    path = task["path"]
    attack = task.get("attack")
    victim = task.get("victim")
    mad_floor = task["mad_floor"]
    scope_start = task["scope_start"]
    attack_start = task["attack_start"]
    seed = task["seed"]

    base_arch = load_archive(path)
    clients, rounds = archive_shape(base_arch)
    arch = dict(base_arch)
    arch["published"] = {i: {} for i in range(clients)}

    fn = ATTACKS[attack] if attack else None
    rng = np.random.default_rng([int(seed), int(victim) + 1 if victim is not None else 0,
                                 ATTACK_ORDER.index(attack) + 1 if attack else 0])

    aud = InnovationAuditor(mad_floor=mad_floor,
                            mask_policy=task.get("mask_policy", "declared"))
    names = stat_names()
    rec_rounds: List[int] = []
    mats = {k: [] for k in names}

    for r in range(1, rounds + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        payloads = {}
        for cid in range(clients):
            C = arch["C"][cid][r]
            if fn is not None and cid == victim and r >= attack_start:
                C = fn(arch, cid, r, rng)
            arch["published"][cid][r] = C
            payloads[cid] = (C, arch["present"][cid][r])
        rs = aud.observe_round(r, payloads)

        base: Dict[str, Dict[int, float]] = {}
        Cbar_prev = arch["cbar"].get(r - 1)
        if Cbar_prev is not None:
            innov = {cid: (payloads[cid][0] - Cbar_prev) for cid in payloads}
            if len(innov) >= 3:
                base = score_round(innov)

        if r < scope_start:
            continue
        rec_rounds.append(r)
        row = {k: [float("nan")] * clients for k in names}
        for cid, v in rs.per_client.items():
            row["suspicion"][cid] = float(v.get("suspicion", float("nan")))
        for bname, vals in base.items():
            for cid, v in vals.items():
                row[bname][cid] = float(v) if np.isfinite(v) else float("nan")
            zz = robust_z_score({c: vals.get(c, float("nan")) for c in range(clients)}, mad_floor)
            for cid, v in zz.items():
                row[f"{bname}_z"][cid] = float(v)
        for k in names:
            mats[k].append(row[k])

    return {"key": task["key"], "rounds": rec_rounds,
            "mats": {k: np.asarray(v, dtype=np.float64) for k, v in mats.items()}}


def flag_scores(mat: np.ndarray, thr: float) -> np.ndarray:
    m = np.where(np.isfinite(mat), mat, -np.inf)
    return m > thr


def fit_threshold(vals: np.ndarray, target: float) -> float:
    v = np.where(np.isfinite(vals), vals, -np.inf).ravel()
    n = v.size
    if n == 0:
        return float("inf")
    m = int(math.floor(target * n))
    s = np.sort(v)
    idx = n - 1 - m
    if idx < 0:
        return float("-inf")
    return float(s[idx])


def fit_thresholds(cal_mats: Dict[str, np.ndarray], targets: Sequence[float]) -> Dict[str, Dict[str, float]]:
    return {s: {f"{t:g}": fit_threshold(cal_mats[s], t) for t in targets} for s in cal_mats}


def stack_scores(passes: List[dict], stat: str) -> np.ndarray:
    parts = [p["mats"][stat] for p in passes if p["mats"][stat].size]
    return np.concatenate([p.ravel() for p in parts]) if parts else np.zeros(0)


def select_mad_floor(cal_honest: Dict[float, List[dict]], targets: Sequence[float]) -> dict:
    grid = sorted(cal_honest)
    rows = []
    for floor in grid:
        passes = cal_honest[floor]
        v_all = stack_scores(passes, "suspicion")
        fin = np.isfinite(v_all)
        q = (lambda p: float(np.quantile(v_all[fin], p)) if fin.any() else float("nan"))
        tail = (q(0.995) / q(0.75)) if fin.any() and abs(q(0.75)) > 1e-12 else float("inf")
        transfer = float("nan")
        if len(passes) >= 2:
            errs = []
            for i in range(len(passes)):
                fitv = np.concatenate([passes[j]["mats"]["suspicion"].ravel()
                                       for j in range(len(passes)) if j != i])
                hold = passes[i]["mats"]["suspicion"]
                for t in targets:
                    thr = fit_threshold(fitv, t)
                    real = float(flag_scores(hold, thr).mean())
                    errs.append(abs(real - t) / t)
            transfer = float(np.mean(errs))
        rows.append({"mad_floor": floor, "loso_transfer_err": transfer,
                     "tail_q995_over_q75": tail,
                     "honest_q50": q(0.50), "honest_q99": q(0.99), "honest_q995": q(0.995),
                     "n_client_rounds": int(v_all.size),
                     "frac_finite": float(fin.mean()) if v_all.size else float("nan")})
    usable = [r for r in rows if np.isfinite(r["loso_transfer_err"])]
    if usable:
        best = min(usable, key=lambda r: (round(r["loso_transfer_err"], 6),
                                          r["tail_q995_over_q75"], r["mad_floor"]))
        crit = "leave-one-seed-out transfer error within CAL"
    else:
        best = min(rows, key=lambda r: (r["tail_q995_over_q75"], r["mad_floor"]))
        crit = "tail ratio q99.5/q75 (single CAL seed; LOSO unavailable)"
    return {"grid": rows, "selected": best["mad_floor"], "criterion": crit}


def evaluate_honest(passes: List[dict], stat: str, thr: float, clients: int) -> dict:
    cr_k = cr_n = run_k = run_n = 0
    for p in passes:
        f = flag_scores(p["mats"][stat], thr)
        cr_k += int(f.sum())
        cr_n += int(f.size)
        run_k += int(f.any(axis=0).sum())
        run_n += clients
    return {"client_round": rate_record(cr_k, cr_n), "client_run": rate_record(run_k, run_n)}


def accomplice(attack: str, victim: int, clients: int) -> Optional[int]:
    if attack != "peer_copy":
        return None
    others = [j for j in range(clients) if j != victim]
    return others[victim % len(others)]


def evaluate_attack(passes: List[dict], stat: str, thr: float, clients: int, attack_start: int,
                attack: str = "") -> dict:
    tp_k = tp_n = 0
    det_k = det_n = 0
    by_k = by_n = 0
    un_k = un_n = 0
    ac_k = ac_n = 0
    lat: List[int] = []
    censored = 0
    attr_hit = attr_tot = 0
    pair_hit = 0
    has_acc = False
    top1_hit = top1_tot = 0
    for p in passes:
        victim = p["key"]["victim"]
        acc = accomplice(attack, victim, clients)
        has_acc = has_acc or (acc is not None)
        rounds = np.asarray(p["rounds"])
        sel = rounds >= attack_start
        if not sel.any():
            continue
        mat = p["mats"][stat][sel]
        f = flag_scores(mat, thr)
        rr = rounds[sel]

        tp_k += int(f[:, victim].sum())
        tp_n += int(f.shape[0])
        det_n += 1
        if f[:, victim].any():
            det_k += 1
            lat.append(int(rr[np.argmax(f[:, victim])] - attack_start))
        else:
            censored += 1

        mask = np.ones(clients, bool)
        mask[victim] = False
        by_k += int(f[:, mask].sum())
        by_n += int(f[:, mask].size)
        umask = mask.copy()
        if acc is not None:
            umask[acc] = False
            ac_k += int(f[:, acc].sum())
            ac_n += int(f.shape[0])
        un_k += int(f[:, umask].sum())
        un_n += int(f[:, umask].size)

        attr_tot += int(f.sum())
        attr_hit += int(f[:, victim].sum())
        pair_hit += int(f[:, victim].sum()) + (int(f[:, acc].sum()) if acc is not None else 0)
        for i in range(f.shape[0]):
            if f[i].any():
                top1_tot += 1
                vals = np.where(np.isfinite(mat[i]) & f[i], mat[i], -np.inf)
                if int(np.argmax(vals)) == victim:
                    top1_hit += 1
    return {
        "tpr_client_round": rate_record(tp_k, tp_n),
        "tpr_client_run": rate_record(det_k, det_n),
        "fpr_bystander_client_round": rate_record(by_k, by_n),
        "fpr_uninvolved_client_round": rate_record(un_k, un_n),
        "flag_rate_copy_source": (rate_record(ac_k, ac_n) if has_acc else None),
        "latency_rounds": {
            "median": (float(np.median(lat)) if lat else None),
            "mean": (float(np.mean(lat)) if lat else None),
            "p90": (float(np.quantile(lat, 0.90)) if lat else None),
            "min": (int(min(lat)) if lat else None),
            "max": (int(max(lat)) if lat else None),
            "n_detected": len(lat), "n_censored": int(censored)
        },
        "attribution_precision": rate_record(attr_hit, attr_tot),
        "attribution_top1": rate_record(top1_hit, top1_tot),
        "attribution_pair": (rate_record(pair_hit, attr_tot) if has_acc else None),
    }


def separation_margin(honest: List[dict], attacks: Dict[str, List[dict]], stat: str,
                      attack_start: int) -> dict:
    hv = stack_scores(honest, stat)
    hv = hv[np.isfinite(hv)]
    hmax = float(hv.max()) if hv.size else float("nan")

    by, un = [], []
    for a, ps in attacks.items():
        for p in ps:
            m = p["mats"][stat]
            if not m.size:
                continue
            v = p["key"]["victim"]
            n_cl = m.shape[1]
            acc = accomplice(a, v, n_cl)
            m = m[np.asarray(p["rounds"]) >= attack_start]
            for c in range(n_cl):
                if c == v:
                    continue
                col = m[:, c]
                col = col[np.isfinite(col)]
                by.append(col)
                if acc is None or c != acc:
                    un.append(col)
    by = np.concatenate(by) if by else np.zeros(0)
    un = np.concatenate(un) if un else np.zeros(0)
    pop = {
        "honest_only": hv,
        "uninvolved": np.concatenate([hv, un]) if un.size else hv,
        "bystanders": np.concatenate([hv, by]) if by.size else hv,
    }
    mx = {k: (float(v.max()) if v.size else float("nan")) for k, v in pop.items()}
    nn = {k: int(v.size) for k, v in pop.items()}
    q995 = {k: (float(np.quantile(v, 0.995)) if v.size else float("nan")) for k, v in pop.items()}
    q99 = {k: (float(np.quantile(v, 0.99)) if v.size else float("nan")) for k, v in pop.items()}
    q50 = {k: (float(np.median(v)) if v.size else float("nan")) for k, v in pop.items()}

    per = {}
    worst = float("inf")
    for a, ps in attacks.items():
        vals = []
        for p in ps:
            rr = np.asarray(p["rounds"]) >= attack_start
            col = p["mats"][stat][rr][:, p["key"]["victim"]]
            vals.append(col[np.isfinite(col)])
        v = np.concatenate(vals) if vals else np.zeros(0)
        mn = float(v.min()) if v.size else float("nan")
        per[a] = {"victim_min": mn, "victim_median": float(np.median(v)) if v.size else None,
                  "separated": bool(np.isfinite(mn) and np.isfinite(hmax) and mn > hmax),
                  "separated_vs_uninvolved": bool(np.isfinite(mn) and mn > mx["uninvolved"]),
                  "separated_vs_bystanders": bool(np.isfinite(mn) and mn > mx["bystanders"])}
        if np.isfinite(mn):
            worst = min(worst, mn)
    w = worst if np.isfinite(worst) else None

    def ratio(d, key):
        return (worst / d[key]) if (w is not None and d[key] > 0) else None

    return {"honest_max": hmax, "worst_attack_min": w,
            "margin_ratio": ratio(mx, "honest_only"),
            "honest_max_by_population": mx, "honest_n_by_population": nn,
            "honest_q995_by_population": q995, "honest_q99_by_population": q99,
            "honest_median_by_population": q50,
            "margin_ratio_by_population": {k: ratio(mx, k) for k in pop},
            "robust_margin_by_population": {k: ratio(q995, k) for k in pop},
            "headline_population": "uninvolved",
            "headline_margin_ratio": ratio(mx, "uninvolved"),
            "robust_margin": ratio(q995, "uninvolved"),
            "per_attack": per}


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if not pos.size or not neg.size:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt))
    np.add.at(s, inv, ranks)
    ranks = (s / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_cross_check(attacks: Dict[str, List[dict]], paths: Dict[int, str],
                    attack_start: int) -> dict:
    out: Dict[str, dict] = {}
    per_seed: Dict[int, Dict[str, List[np.ndarray]]] = {}
    for a, ps in attacks.items():
        for p in ps:
            s = p["key"]["seed"]
            m = p["mats"]["suspicion"]
            if not m.size:
                continue
            v = p["key"]["victim"]
            m = m[np.asarray(p["rounds"]) >= attack_start]
            d = per_seed.setdefault(s, {}).setdefault(a, [[], []])
            d[0].append(m[:, v])
            d[1].append(np.delete(m, v, axis=1).ravel())
    for s, byatk in sorted(per_seed.items()):
        base = os.path.basename(paths[s])
        m = re.match(r"arch_(.+?)_c\d+_r\d+_s\d+\.npz$", base)
        ref_path = os.path.join(os.path.dirname(ARCHIVE_DIR), f"innovation_{m.group(1)}.json") \
            if m else ""
        ref = None
        if ref_path and os.path.exists(ref_path):
            try:
                with open(ref_path, "r", encoding="utf-8") as f:
                    ref = json.load(f).get("results", {})
            except Exception:
                ref = None
        rows = {}
        for a in ATTACK_ORDER:
            if a not in byatk:
                continue
            pos = np.concatenate(byatk[a][0])
            neg = np.concatenate(byatk[a][1])
            auc = roc_auc(pos, neg)
            pub = None
            if ref and a in ref and "suspicion" in ref[a]:
                pub = ref[a]["suspicion"].get("auc")
            rows[a] = {"auc_recomputed": auc, "auc_published": pub,
                       "delta": (auc - pub) if isinstance(pub, (int, float)) else None,
                       "n_pos": int(pos.size), "n_neg": int(neg.size)}
        out[str(s)] = {"reference": os.path.basename(ref_path) if ref else None, "attacks": rows}
    return out


def evaluate_pool(honest: List[dict], attacks: Dict[str, List[dict]],
                  thresholds: Dict[str, Dict[str, float]], targets: Sequence[float],
                  clients: int, attack_start: int) -> dict:
    out: Dict[str, dict] = {}
    for stat in thresholds:
        per_t = {}
        for t in targets:
            thr = thresholds[stat][f"{t:g}"]
            h = evaluate_honest(honest, stat, thr, clients) if honest else None
            gap = None
            if h is not None and h["client_round"]["point"] is not None:
                gap = h["client_round"]["point"] - t
            per_t[f"{t:g}"] = {
                "threshold": thr,
                "nominal_fpr": t,
                "honest": h,
                "calibration_transfer_gap": gap,
                "attacks": {a: evaluate_attack(ps, stat, thr, clients, attack_start, a)
                            for a, ps in attacks.items()},
            }
        out[stat] = per_t
    return out


def build_tasks(paths: Dict[int, str], seeds: Sequence[int], mad_floor: float,
                scope_start: int, attack_start: int, clients_by_seed: Dict[int, int],
                with_attacks: bool, pool: str) -> List[dict]:
    tasks = []
    for s in seeds:
        tasks.append({"key": {"pool": pool, "seed": s, "attack": None, "victim": None},
                      "path": paths[s], "attack": None, "victim": None, "seed": s,
                      "mad_floor": mad_floor, "scope_start": scope_start,
                      "attack_start": attack_start})
        if with_attacks:
            for a in ATTACK_ORDER:
                for v in range(clients_by_seed[s]):
                    tasks.append({"key": {"pool": pool, "seed": s, "attack": a, "victim": v},
                                  "path": paths[s], "attack": a, "victim": v, "seed": s,
                                  "mad_floor": mad_floor, "scope_start": scope_start,
                                  "attack_start": attack_start})
    return tasks


def run_tasks(tasks: List[dict], workers: int) -> List[dict]:
    if not tasks:
        return []
    tasks = sorted(tasks, key=lambda t: t["path"])
    if workers <= 1 or len(tasks) <= 2:
        return [score_one_archive(t) for t in tasks]
    chunk = max(1, len(tasks) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(score_one_archive, tasks, chunksize=chunk))


def split_pool(results: List[dict]) -> Tuple[List[dict], Dict[str, List[dict]]]:
    honest = [r for r in results if r["key"]["attack"] is None]
    atk: Dict[str, List[dict]] = {a: [] for a in ATTACK_ORDER}
    for r in results:
        if r["key"]["attack"] is not None:
            atk[r["key"]["attack"]].append(r)
    return honest, {a: v for a, v in atk.items() if v}


def horizontal_rule(c: str = "=", n: int = 108) -> str:
    return c * n


def print_pool_table(title: str, ev: dict, target: float, stats: Sequence[str]) -> None:
    key = f"{target:g}"
    print("\n" + horizontal_rule())
    print(f"{title}  |  nominal per-client-round FPR = {100*target:g}%  (threshold frozen on CAL)")
    print(horizontal_rule())
    hdr = ("%-18s %9s %11s %9s" % ("statistic", "thr", "FPR(c-r)", "gap")) + \
          "".join("%12s" % a[:12] for a in ATTACK_ORDER) + "%10s%9s%7s" % ("worstTPR", "attr", "n/6")
    print(hdr)
    print(horizontal_rule("-"))
    rows = []
    for s in stats:
        if s not in ev:
            continue
        e = ev[s][key]
        h = e["honest"]
        fpr = ("%8.2f%%" % (100 * h["client_round"]["point"])) if h and h["client_round"]["point"] is not None else "     n/a"
        gap = ("%+8.2fpp" % (100 * e["calibration_transfer_gap"])) if e["calibration_transfer_gap"] is not None else "     n/a"
        thr = e["threshold"]
        row = "%-18s %9.3f %11s %9s" % (s, thr, fpr, gap)
        ak = an = 0
        n_with_flags = 0
        worst = None
        for a in ATTACK_ORDER:
            v = e["attacks"].get(a)
            if v is None:
                row += "%12s" % "-"
                continue
            t = v["tpr_client_round"]["point"]
            row += "%11.1f%%" % (100 * t)
            worst = t if worst is None else min(worst, t)
            ap = v["attribution_precision"]
            ak += ap["k"]
            an += ap["n"]
            n_with_flags += int(ap["n"] > 0)
        row += ("%9.1f%%" % (100 * worst)) if worst is not None else "%10s" % "-"
        row += ("%8.1f%%" % (100 * ak / an)) if an else "%9s" % "-"
        row += "%7d" % n_with_flags
        print(row)
        rows.append((s, worst))
    print(horizontal_rule("-"))
    print("  FPR(c-r) = realised honest per-client-round false-alarm rate on this pool.")
    print("  gap      = CALIBRATION-TRANSFER GAP (realised minus nominal), percentage points.")
    print("  per-attack columns = TPR at the client-round level.")
    print("  worstTPR = minimum TPR over the 6-attack suite. This is the headline number: a")
    print("             detector is only as good as the attack it handles worst.")
    print("  attr     = POOLED attribution precision (all flagged cells over all six attacks,")
    print("             fraction naming the true cheat). n/6 = how many attacks produced any")
    print("             flag at all; a statistic with n/6 < 6 is silent on the rest, and its")
    print("             attribution number describes only the attacks it reacted to.")

    ours = [r for r in rows if r[0] == "suspicion"]
    base = [r for r in rows if r[0] != "suspicion" and r[1] is not None]
    if ours and ours[0][1] is not None and base:
        b = max(base, key=lambda r: r[1])
        cal = [r for r in base
               if ev[r[0]][key]["honest"]["client_round"]["point"] is not None
               and ev[r[0]][key]["honest"]["client_round"]["point"] <= target]
        print("  => worst-case-over-suite TPR at nominal %g%%: OURS %.1f%%, best baseline %.1f%% (%s)."
              % (100 * target, 100 * ours[0][1], 100 * b[1], b[0]))
        if cal:
            c = max(cal, key=lambda r: r[1])
            print("     Restricted to baselines whose realised FPR met the target, the best")
            print("     baseline is %s at %.1f%%." % (c[0], 100 * c[1]))
        else:
            print("     No baseline met the nominal FPR on this pool, so even that comparison is")
            print("     generous to them.")


def print_headline(ev: dict, target: float, clients: int) -> None:
    key = f"{target:g}"
    e = ev["suspicion"][key]
    print("\n" + horizontal_rule())
    print(f"HEADLINE (suspicion, nominal {100*target:g}% client-round FPR)  "
          f"-- Clopper-Pearson 95%, client-RUN level")
    print(horizontal_rule())
    h = e["honest"]
    if h:
        print("  honest false-alarm rate   client-run  : %s   (k=%d of n=%d)%s"
              % (h["client_run"]["reported"], h["client_run"]["k"], h["client_run"]["n"],
                 "   [zero-alarm floor applied]" if h["client_run"]["zero_floor_applied"] else ""))
        print("  honest false-alarm rate   client-round: %s   (k=%d of n=%d)  <- dependent unit, CI anti-conservative"
              % (h["client_round"]["reported"], h["client_round"]["k"], h["client_round"]["n"]))
        p_cr, n_run = h["client_round"]["point"], h["client_run"]["n"]
        if p_cr is not None and n_run:
            rounds_per_run = h["client_round"]["n"] / n_run
            print("    for reference, %d independent rounds at %.2f%% each would give a client-run rate of "
                  "%.1f%%;" % (round(rounds_per_run), 100 * p_cr,
                               100 * (1 - (1 - p_cr) ** rounds_per_run)))
            print("    a high client-run rate at a loose nominal target is arithmetic, not a "
                  "detector failure.")
    print("  %-13s %24s %10s %10s %6s   %s"
          % ("attack", "TPR client-run [CP95]", "TPR c-r", "latency", "cens", "attribution precision [CP95]"))
    print(horizontal_rule("-"))
    for a in ATTACK_ORDER:
        v = e["attacks"].get(a)
        if v is None:
            continue
        lat = v["latency_rounds"]
        lt = ("%.1f" % lat["median"]) if lat["median"] is not None else "n/a"
        extra = ""
        if v.get("attribution_pair"):
            extra = "   (cheat-or-copy-source %s)" % v["attribution_pair"]["reported"]
        print("  %-13s %24s %9.1f%% %10s %6d   %s%s"
              % (a, v["tpr_client_run"]["reported"], 100 * v["tpr_client_round"]["point"],
                 lt, lat["n_censored"], v["attribution_precision"]["reported"], extra))
    print(horizontal_rule("-"))
    print("  collateral: flag rate on the OTHER clients while the attack runs --")
    print("  %-14s %28s %28s" % ("attack", "uninvolved bystanders", "copy source (structural)"))
    for a in ATTACK_ORDER:
        v = e["attacks"].get(a)
        if v is None:
            continue
        un = v.get("fpr_uninvolved_client_round")
        cs = v.get("flag_rate_copy_source")
        print("  %-14s %28s %28s"
              % (a, un["reported"] if un else "n/a", cs["reported"] if cs else "-"))
    print("  The raw bystander rate stored in JSON lumps these together, which makes")
    print("  peer_copy look ~8x worse than it is: its copy source is flagged by design")
    print("  (it publishes the copier's innovation), while genuinely uninvolved clients")
    print("  are flagged at the left-hand rate. Both n's are inflated ~%dx by counting each"
          % max(1, clients - 1))
    print("  honest client once per victim replay, so the intervals are not independent.")
    print(horizontal_rule("-"))
    print("  latency = median rounds from attack onset to first flag on the true cheat "
          "(censored runs excluded).")
    print("  attribution precision = of all flagged (client, round) cells after onset, the "
          "fraction naming the")
    print("  actual cheater. Krum and median discard an update without naming anyone, so "
          "this metric has no")
    print("  analogue for them unless they are read as accusers -- which is how they are "
          "scored here.")
    print("  peer_copy is inherently symmetric: the copier and the copied publish the same "
          "innovation, so")
    print("  the payload alone cannot order them. The parenthesised rate is the fraction of "
          "flags landing on")
    print("  one of the two clients that are actually implicated.")


def _synthetic_archive(path: str, clients: int = 8, rounds: int = 14, K: int = 16,
                       d: int = 12, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    tgt = rng.normal(0, 1, (clients, K, d))
    C = {i: {} for i in range(clients)}
    present = {i: {} for i in range(clients)}
    cbar = {}
    cur = {i: rng.normal(0, 0.3, (K, d)) for i in range(clients)}
    for i in range(clients):
        m = np.zeros(K, bool)
        m[rng.choice(K, size=max(3, K - 3), replace=False)] = True
        present[i] = {r: m for r in range(1, rounds + 1)}
    g = np.zeros((K, d))
    for r in range(1, rounds + 1):
        for i in range(clients):
            cur[i] = cur[i] + 0.35 * (tgt[i] - cur[i]) + rng.normal(0, 0.02, (K, d))
            C[i][r] = cur[i].copy()
        g = np.mean([C[i][r] for i in range(clients)], axis=0)
        cbar[r] = g.copy()
        for i in range(clients):
            cur[i] = 0.5 * cur[i] + 0.5 * g
    np.savez_compressed(path, C=C, present=present, cbar=cbar)


def self_test(tmpdir: str) -> int:
    print(horizontal_rule())
    print("SELF-TEST (no archives, no GPU)")
    print(horizontal_rule())
    ok = True

    p, lo, hi = clopper_pearson(0, 20)
    exact = 1.0 - 0.025 ** (1.0 / 20)
    print("  CP k=0  n=20 -> upper %.6f   closed form 1-0.025^(1/20) = %.6f" % (hi, exact))
    ok &= abs(hi - exact) < 1e-9 and lo == 0.0
    p, lo, hi = clopper_pearson(20, 20)
    exact_lo = 0.025 ** (1.0 / 20)
    print("  CP k=n=20    -> lower %.6f   closed form 0.025^(1/20)   = %.6f" % (lo, exact_lo))
    ok &= abs(lo - exact_lo) < 1e-9 and hi == 1.0
    p, lo, hi = clopper_pearson(2, 20)
    print("  CP k=2  n=20 -> [%.4f, %.4f]   (published value [0.0123, 0.3170])" % (lo, hi))
    ok &= abs(lo - 0.012347) < 5e-5 and abs(hi - 0.316990) < 5e-5
    print("  reporting floor 0/20 renders as: %s" % rate_record(0, 20)["reported"])
    ok &= rate_record(0, 20)["reported"].startswith("<=")

    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 20))
    for t in TARGETS:
        thr = fit_threshold(v, t)
        r = float(flag_scores(v, thr).mean())
        print("  fit_threshold target %5.3f -> in-sample exceedance %.4f  (must be <= target)" % (t, r))
        ok &= r <= t + 1e-12
    v2 = v.copy()
    v2[0, 0] = np.nan
    ok &= not flag_scores(v2, -np.inf)[0, 0]
    print("  non-finite statistic never flags: OK")

    os.makedirs(tmpdir, exist_ok=True)
    paths = {}
    for s in (0, 1, 2, 3):
        p_ = os.path.join(tmpdir, f"arch_selftest_c8_r14_s{s}.npz")
        if not os.path.exists(p_):
            _synthetic_archive(p_, seed=s)
        paths[s] = p_
    t0 = time.time()
    res = main_protocol(paths, cal=[0, 1], dev=[2], test=[3], scope_start=6,
                        attack_start=6, targets=TARGETS, workers=4, quiet=False,
                        out_path=os.path.join(tmpdir, "selftest.json"))
    print("  synthetic end-to-end: %.1fs" % (time.time() - t0))
    ok &= res is not None and "TEST" in res["pools_used"] and res["pools_used"]["TEST"] == [3]
    print("\n  SELF-TEST %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main_protocol(paths: Dict[int, str], cal: List[int], dev: List[int], test: List[int],
                  scope_start: int, attack_start: int, targets: Sequence[float],
                  workers: int, quiet: bool, out_path: str,
                  sensitivity: bool = True) -> Optional[dict]:
    avail = sorted(paths)
    req = {"CAL": list(cal), "DEV": list(dev), "TEST": list(test)}
    cal = [s for s in cal if s in paths]
    dev = [s for s in dev if s in paths]
    test = [s for s in test if s in paths]
    missing = {k: [s for s in v if s not in paths] for k, v in req.items()}

    print(horizontal_rule())
    print("CAL / DEV / TEST PROTOCOL -- offline replay over cached archives")
    print(horizontal_rule())
    print("  archives found      : %s" % (", ".join(f"seed {s}" for s in avail) or "NONE"))
    for name, key, pool in (("CAL (fits everything)", "CAL", cal),
                            ("DEV (go/no-go only)", "DEV", dev),
                            ("TEST (frozen, once)", "TEST", test)):
        used = ", ".join(f"seed {s}" for s in pool) if pool else "EMPTY"
        miss = (("   [requested but not on disk: %s]" % ", ".join(f"seed {s}" for s in missing[key]))
                if missing[key] else "")
        print("  %-22s: %s%s" % (name, used, miss))
    if not cal:
        print("\n  no CAL archive present; nothing can be fitted. Aborting.")
        return None

    shapes = {}
    for s in sorted(set(cal + dev + test)):
        shapes[s] = archive_shape(load_archive(paths[s]))
    clients = shapes[cal[0]][0]
    rounds = shapes[cal[0]][1]
    for s, (c, r) in shapes.items():
        if c != clients:
            print(f"  WARNING seed {s} has {c} clients, CAL has {clients}")
    print("  cohort              : %d clients, %d rounds; evaluation scope r >= %d "
          "(attack onset r = %d)" % (clients, rounds, scope_start, attack_start))
    chk = check_attack_definitions()
    print("  attack definitions  : %s vs innovation_eval.py -> %s"
          % ("checked" if chk["checked"] else "NOT CHECKED", chk["detail"]))
    _ARCH_CACHE.clear()

    t0 = time.time()
    grid_tasks = []
    for f in MAD_FLOOR_GRID:
        grid_tasks += build_tasks(paths, cal, f, scope_start, attack_start,
                                  {s: clients for s in cal}, False, "CAL")
        for t in grid_tasks[-len(cal):]:
            t["key"] = dict(t["key"], floor=f)
    gres = run_tasks(grid_tasks, workers)
    cal_honest = {f: [] for f in MAD_FLOOR_GRID}
    for r in gres:
        cal_honest[r["key"]["floor"]].append(r)
    for f in cal_honest:
        cal_honest[f].sort(key=lambda r: r["key"]["seed"])
    sel = select_mad_floor(cal_honest, targets)
    mad_floor = sel["selected"]

    print("\n" + horizontal_rule())
    print("STAGE 1 -- MAD floor selected on CAL HONEST client-rounds only (no attack replayed)")
    print(horizontal_rule())
    print("  criterion: %s" % sel["criterion"])
    print("  %-12s %14s %14s %12s %12s %10s" %
          ("mad_floor", "LOSO transfer", "q99.5/q75", "honest q50", "honest q99.5", "finite"))
    for row in sel["grid"]:
        mark = " <== selected" if row["mad_floor"] == mad_floor else ""
        print("  %-12.0e %14s %14.3f %12.3f %12.3f %9.1f%%%s"
              % (row["mad_floor"],
                 ("%.4f" % row["loso_transfer_err"]) if np.isfinite(row["loso_transfer_err"]) else "n/a",
                 row["tail_q995_over_q75"], row["honest_q50"], row["honest_q995"],
                 100 * row["frac_finite"], mark))

    cal_passes = cal_honest[mad_floor]
    cal_mats = {s: stack_scores(cal_passes, s) for s in stat_names()}
    thresholds = fit_thresholds(cal_mats, targets)
    n_cal = int(cal_mats["suspicion"].size)
    print("\n" + horizontal_rule())
    print("STAGE 2 -- decision thresholds fitted on CAL HONEST client-rounds only (n=%d)" % n_cal)
    print(horizontal_rule())
    print("  The threshold at target t is the (m+1)-th largest honest value, m = floor(t*n).")
    print("  With n=%d that is m = %s allowed exceedances -- so the 0.5%% threshold rests on a"
          % (n_cal, ", ".join("%g%%:%d" % (100 * t, math.floor(t * n_cal)) for t in targets)))
    print("  handful of order statistics and its sampling variance is the dominant uncertainty")
    print("  in the transfer gap below. More CAL seeds, not more rounds, is what tightens it.")
    print("  %-18s" % "statistic" + "".join("%16s" % ("tau @ %g%%" % (100 * t)) for t in targets)
          + "%12s" % "in-samp FPR")
    for s in ["suspicion"] + [f"{b}_z" for b in BASELINES] + list(BASELINES):
        line = "  %-18s" % s + "".join("%16.4g" % thresholds[s][f"{t:g}"] for t in targets)
        thr1 = thresholds[s][f"{targets[-1]:g}"]
        line += "%11.3f%%" % (100 * float(flag_scores(cal_mats[s], thr1).mean()))
        print(line)
    print("  (in-samp FPR is at the tightest target and is <= nominal by construction)")
    struct = int((cal_mats["suspicion"] >= 999.0).sum())
    print("  honest CAL client-rounds hitting the structural score (1e3): %d of %d" % (struct, n_cal))

    out = {
        "protocol": {
            "pools": {"CAL": cal, "DEV": dev, "TEST": test},
            "archives": {str(s): os.path.basename(paths[s]) for s in sorted(paths)},
            "clients": clients, "rounds": rounds,
            "scope_start": scope_start, "attack_start": attack_start,
            "targets": list(targets),
            "attack_def_check": chk,
        },
        "pools_used": {"CAL": cal, "DEV": dev, "TEST": test},
        "pools_requested": req,
        "pools_missing_from_disk": missing,
        "mad_floor": {"selected": mad_floor, "criterion": sel["criterion"], "grid": sel["grid"]},
        "thresholds": thresholds,
        "cal_n_client_rounds": n_cal,
        "results": {},
    }

    pool_atk: Dict[str, Dict[str, List[dict]]] = {}
    for pool_name, seeds in (("DEV", dev), ("TEST", test)):
        if not seeds:
            print("\n" + horizontal_rule())
            print(f"{pool_name}: no archive present -- skipped. "
                  f"(Requested seeds {', '.join(map(str, req[pool_name])) or 'none'}; "
                  f"not on disk: {', '.join(map(str, missing[pool_name])) or 'n/a'}.)")
            out["results"][pool_name] = None
            continue
        tasks = build_tasks(paths, seeds, mad_floor, scope_start, attack_start,
                            {s: clients for s in seeds}, True, pool_name)
        res = run_tasks(tasks, workers)
        honest, atk = split_pool(res)
        pool_atk[pool_name] = atk
        ev = evaluate_pool(honest, atk, thresholds, targets, clients, attack_start)
        out["results"][pool_name] = ev
        sep = separation_margin(honest, atk, "suspicion", attack_start)
        out.setdefault("separation", {})[pool_name] = sep
        if not quiet:
            print("\n" + horizontal_rule())
            print("%s -- THRESHOLD-FREE SEPARATION (suspicion). Largest honest score on this pool"
                  % pool_name)
            print("vs smallest cheating score, so it does not depend on where tau landed.")
            print(horizontal_rule())
            mx, nn = sep["honest_max_by_population"], sep["honest_n_by_population"]
            q995, q50 = sep["honest_q995_by_population"], sep["honest_median_by_population"]
            print("  Three honest populations, because which cells count IS the question:")
            print("    %-28s %7s %10s %10s %10s" % ("population", "n", "median", "q99.5", "max"))
            for k, lab in (("honest_only", "honest-only replays"),
                           ("uninvolved", "+ uninvolved bystanders"),
                           ("bystanders", "+ copy source too")):
                print("    %-28s %7d %10.3f %10.3f %10.3f" % (lab, nn[k], q50[k], q995[k], mx[k]))
            print("    Rows 1-2 agree at the median and at q99.5 and diverge only at the max: the")
            print("    attacker does NOT distort an uninvolved bystander's cohort-relative score,")
            print("    and the honest-only max is just a max over %.0f%% of the honest cells."
                  % (100 * nn["honest_only"] / max(1, nn["uninvolved"])))
            print("    Row 3's tail is shifted by construction -- the copy source is flagged by")
            print("    design -- which is why it is kept separate rather than folded in.")
            print("  %-14s %11s   %11s   %-12s %-12s %-12s"
                  % ("attack", "victim min", "median", "vs honest", "vs uninvol.", "vs bystand."))
            for a in ATTACK_ORDER:
                if a in sep["per_attack"]:
                    pa = sep["per_attack"][a]
                    print("  %-14s %11.3f   %11.3f   %-12s %-12s %-12s"
                          % (a, pa["victim_min"], pa["victim_median"],
                             "SEP" if pa["separated"] else "OVERLAP",
                             "SEP" if pa["separated_vs_uninvolved"] else "OVERLAP",
                             "SEP" if pa["separated_vs_bystanders"] else "OVERLAP"))
            rr = sep["margin_ratio_by_population"]
            rb = sep["robust_margin_by_population"]
            print("  worst cheat score on this pool: %.3f (%s)"
                  % (sep["worst_attack_min"],
                     min(sep["per_attack"], key=lambda a: sep["per_attack"][a]["victim_min"])))
            print("    %-28s %14s %14s" % ("margin vs population", "/ max", "/ q99.5 (robust)"))
            for k, lab in (("honest_only", "honest-only replays"),
                           ("uninvolved", "+ uninvolved bystanders"),
                           ("bystanders", "+ copy source too")):
                print("    %-28s %13s %14s"
                      % (lab, ("%.2fx" % rr[k]) if rr.get(k) else "n/a",
                         ("%.2fx" % rb[k]) if rb.get(k) else "n/a"))
            hm, rmar = sep["headline_margin_ratio"], sep["robust_margin"]
            if hm is not None:
                print("  The max-based column swings by a factor of ~%.0f across populations and is"
                      % (mx["bystanders"] / mx["honest_only"] if mx["honest_only"] > 0 else 0))
                print("  therefore not a stable quantity: quoting the honest-only %.2fx as evidence of"
                      % rr["honest_only"])
                print("  separation is quoting the largest of 240 draws against the smallest of 1440.")
                print("  Counting every honest cell the pool contains, the max-based margin is %.2fx"
                      % hm)
                print("  -- %s." % ("still separable" if hm > 1 else
                                    "NOT separable: honest client-rounds score above the weakest cheat"))
            if rmar is not None:
                print("  ROBUST MARGIN (weakest cheat / honest q99.5) = %.2fx. Unlike the max-based"
                      % rmar)
                print("  figure it barely moves between rows 1 and 2 (%.2fx vs %.2fx), so it is not"
                      % (rb["honest_only"], rb["uninvolved"]))
                print("  hostage to how many honest cells were replayed. This is the number to report.")
                print("  Row 3 is excluded from that comparison for the reason given above.")
        if not quiet:
            print_pool_table(f"{pool_name} POOL (seeds {', '.join(map(str, seeds))})", ev,
                             0.01, ["suspicion"] + [f"{b}_z" for b in BASELINES] + list(BASELINES))

    head_pool = "TEST" if out["results"].get("TEST") else ("DEV" if out["results"].get("DEV") else None)
    if head_pool:
        if head_pool == "DEV":
            want = ", ".join("seed %d" % s for s in req["TEST"]) or "(none requested)"
            print("\n" + horizontal_rule("!"))
            print("!! TEST archives (%s) are absent, so the block below is provisional." % want)
            print("!! It applies the CAL-frozen thresholds to the DEV pool. DEV was not used to fit")
            print("!! anything, neither the threshold nor the MAD floor, so it is a held-out")
            print("!! pool and these numbers are not in-sample. But DEV is the pool reserved for the")
            print("!! configuration check, so rerun with %s present to obtain the TEST headline." % want)
            print(horizontal_rule("!"))
        for t in targets:
            print_headline(out["results"][head_pool], t, clients)
        out["headline_pool"] = head_pool
        out["headline_is_provisional"] = (head_pool == "DEV")

        print("\n" + horizontal_rule())
        print("CALIBRATION-TRANSFER GAP on %s (suspicion): realised honest FPR minus nominal"
              % head_pool)
        print(horizontal_rule())
        print("  %-10s %14s %14s %14s %28s" %
              ("nominal", "tau (CAL)", "realised", "gap (pp)", "client-run FPR [CP95]"))
        for t in targets:
            e = out["results"][head_pool]["suspicion"][f"{t:g}"]
            h = e["honest"]
            print("  %9.2f%% %14.4f %13.3f%% %+13.3f %28s"
                  % (100 * t, e["threshold"], 100 * h["client_round"]["point"],
                     100 * e["calibration_transfer_gap"], h["client_run"]["reported"]))

    atk_at_default_floor = pool_atk.get(head_pool) if abs(mad_floor - 1e-3) <= 1e-12 else None
    if sensitivity and head_pool and abs(mad_floor - 1e-3) > 1e-12:
        alt = 1e-3
        alt_thr = fit_thresholds({s: stack_scores(cal_honest[alt], s) for s in stat_names()}, targets)
        seeds = test if head_pool == "TEST" else dev
        ares = run_tasks(build_tasks(paths, seeds, alt, scope_start, attack_start,
                                     {s: clients for s in seeds}, True, head_pool), workers)
        ah, aa = split_pool(ares)
        atk_at_default_floor = aa
        aev = evaluate_pool(ah, aa, {"suspicion": alt_thr["suspicion"]}, targets,
                            clients, attack_start)
        out["floor_sensitivity"] = {
            "alt_mad_floor": alt, "results": aev}
        print("\n" + horizontal_rule())
        print("STAGE 4: sensitivity to the fitted MAD floor (%.0e selected vs %.0e default)"
              % (mad_floor, alt))
        print(horizontal_rule())
        print("  %-10s %26s %26s" % ("nominal", "realised FPR / worst TPR", "realised FPR / worst TPR"))
        print("  %-10s %26s %26s" % ("", "floor=%.0e (selected)" % mad_floor, "floor=%.0e (default)" % alt))
        for t in targets:
            def cell(ee):
                x = ee["suspicion"][f"{t:g}"]
                w = min(v["tpr_client_round"]["point"] for v in x["attacks"].values())
                return "%7.3f%% / %7.1f%%" % (100 * x["honest"]["client_round"]["point"], 100 * w)
            print("  %9.2f%% %26s %26s"
                  % (100 * t, cell(out["results"][head_pool]), cell(aev)))
        print("  worst TPR = the minimum client-round TPR over the 6-attack suite (the "
              "headline number).")

    if atk_at_default_floor:
        ax = auc_cross_check(atk_at_default_floor, paths, attack_start)
        out["auc_cross_check"] = ax
        print("\n" + horizontal_rule())
        print("STAGE 5 -- semantic agreement with innovation_eval.py (AUC of `suspicion`, "
              "floor 1e-3)")
        print(horizontal_rule())
        worst_d = 0.0
        for s, blk in sorted(ax.items()):
            print("  seed %s   reference: %s" % (s, blk["reference"] or "NOT FOUND -- not compared"))
            print("    %-14s %14s %14s %12s" % ("attack", "recomputed", "published", "delta"))
            for a in ATTACK_ORDER:
                r = blk["attacks"].get(a)
                if not r:
                    continue
                d = r["delta"]
                if d is not None:
                    worst_d = max(worst_d, abs(d))
                print("    %-14s %14.6f %14s %12s"
                      % (a, r["auc_recomputed"],
                         ("%.6f" % r["auc_published"]) if r["auc_published"] is not None else "n/a",
                         ("%+.2e" % d) if d is not None else "n/a"))
        print("  largest |delta| over all compared attacks: %.2e" % worst_d)
        print("  This complements check_attack_definitions: it compares numbers, not")
        print("  syntax. `noise` is expected to differ at ~1e-4 because its RNG is seeded")
        print("  per (seed, victim, attack) here and once per sweep there.")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwrote %s   (total %.1fs)" % (out_path, time.time() - t0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", default=ARCHIVE_DIR)
    ap.add_argument("--cal", default="0,1")
    ap.add_argument("--dev", default="2")
    ap.add_argument("--test", default="3,4")
    ap.add_argument("--attack-start", type=int, default=12)
    ap.add_argument("--eval-from", type=int, default=12)
    ap.add_argument("--workers", type=int, default=min(12, (os.cpu_count() or 4)))
    ap.add_argument("--out", default=os.path.abspath(
        os.path.join(os.path.dirname(ARCHIVE_DIR), "protocol_eval.json")))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-sensitivity", action="store_true",
                    help="skip the MAD-floor sensitivity re-freeze (stage 4)")
    ap.add_argument("--archive-prefix", default="arch_cifar100_s",
                    help="select one configuration; pools must never mix datasets or "
                         "partitions (e.g. arch_cifar10_s, arch_cifar100_d01_s)")
    ap.add_argument("--self-test", action="store_true",
                    help="verify the statistics and the whole pipeline on tiny synthetic "
                         "archives; needs no real archive and no GPU")
    a = ap.parse_args()

    if a.self_test:
        return self_test(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                      "results", "_selftest"))

    def ints(s):
        return [int(x) for x in s.split(",") if x.strip() != ""]

    paths = discover_archives(a.archives, a.archive_prefix)
    if not paths:
        print(f"no archives in {a.archives}")
        return 1
    r = main_protocol(paths, ints(a.cal), ints(a.dev), ints(a.test),
                      max(a.eval_from, a.attack_start), a.attack_start, TARGETS,
                      a.workers, a.quiet, a.out, not a.no_sensitivity)
    return 0 if r else 1


if __name__ == "__main__":
    sys.exit(main())
