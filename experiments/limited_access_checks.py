from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, contextlib, copy, json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

import numpy as np
import torch

from passportfl.adaptive_attack import AttackConfig, WhiteBoxAttacker, build_view
from passportfl.combined_attack import CombinedAttackConfig, CombinedAttacker
from passportfl.innovation import InnovationAuditor
from passportfl import limited_access as la

import adaptive_eval as ae
import combined_eval as ce
import limited_access_eval as le

SEED, VICTIM, BETA = 3, 0, 1.0
LEAK_ROUNDS = (13, 30)
LEAK_LEVELS = ("delayed:1", "delayed:2", "delayed:3", "query:4", "query:16", "query:64",
               "query:256", "decision:4", "decision:16", "decision:64", "feedback", "pseudo",
               "shadow:5", "shadow:6")
POPULATION, GRID_TEST = (0, 1, 2, 3, 4), (3, 4)


class LeakError(AssertionError):
    pass


class Tripwire:
    __slots__ = ("_what",)

    def __init__(self, what: str):
        object.__setattr__(self, "_what", what)

    def _trip(self, *_a, **_k):
        raise LeakError("read of withheld datum: " + object.__getattribute__(self, "_what"))

    def __getattr__(self, _name):
        self._trip()

    __getitem__ = __iter__ = __len__ = __contains__ = __bool__ = __array__ = _trip
    __float__ = __eq__ = __lt__ = __gt__ = __hash__ = _trip


class GuardedDict(dict):
    def __init__(self, data, allowed, what: str):
        super().__init__(data)
        self._allowed, self._what = frozenset(allowed), what

    def _check(self, k):
        if k not in self._allowed:
            raise LeakError("read of withheld datum: %s[%r]" % (self._what, k))

    def __getitem__(self, k):
        self._check(k)
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        self._check(k)
        return dict.get(self, k, default)

    def __contains__(self, k):
        self._check(k)
        return dict.__contains__(self, k)

    def _enumerate(self, *_a, **_k):
        raise LeakError("enumeration of withheld data: " + self._what)

    __iter__ = __len__ = keys = values = items = copy = _enumerate


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, la.Entry):
        return isinstance(b, la.Entry) and same(a.I, b.I) and same(a.mask, b.mask) \
            and same(a.norm, b.norm)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(map(same, a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    if isinstance(a, float) or isinstance(b, float):
        return (a == b) or (np.isnan(a) and np.isnan(b))
    return a == b


VIEW_FIELDS = ("victim", "K", "d", "Cbar_prev", "t_own", "own_prev", "own_prev2", "own_C_prev",
               "own_C_lags", "own_rep_lag_ok", "peer_ids", "peer_I", "peer_u1", "peer_u2",
               "peer_norm", "peer_step", "peer_rep_raw", "peer_x_const", "pool_I",
               "cohort_scale", "peer_I_full")


def view_diff(a, b):
    if a is None or b is None:
        return [] if a is None and b is None else ["<none>"]
    return [f for f in VIEW_FIELDS if not same(getattr(a, f), getattr(b, f))]


def released_constants() -> dict:
    g = json.load(open(le.RELEASED_FUSED, encoding="utf-8"))
    cc = json.load(open(le.RELEASED_TAU_P, encoding="utf-8"))
    return {"tau": {"P": cc["thresholds"]["suspicion"], "F": g["thresholds"]["combined_max"]},
            "norm": {k: list(v) for k, v in g["norm"].items()}}


def archive(seed: int) -> str:
    return le.archive_paths(os.path.join(le.RESULTS, "archives"), "arch_cifar100_s")[seed]


def honest_state(seed: int, defense: str, upto: int, norm):
    arch = ce.load_archive(archive(seed))
    n = len(arch["C"])
    aud = InnovationAuditor(mad_floor=1e-2)
    hist: Dict[int, Dict[int, float]] = {}
    for r in range(1, upto):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        pay = {c: (arch["C"][c][r], arch["present"][c][r]) for c in range(n)}
        rs = aud.observe_round(r, pay)
        hist[r] = le.defended_scores(defense, rs, pay, aud.cbar.get(r - 1), n, 1e-2, norm)
    aud.set_aggregate(upto - 1, arch["cbar"][upto - 1])
    present = {c: arch["present"][c][upto] for c in range(n)}
    pay = {c: (arch["C"][c][upto], present[c]) for c in range(n)}
    return arch, aud, hist, present, pay


def shadow_with_provenance(lvl: la.AccessLevel, defense: str, nt) -> tuple:
    opened = []
    real_load = ce.load_archive

    def recording_load(path):
        opened.append(os.path.basename(path))
        return real_load(path)

    le.ce.load_archive = recording_load
    try:
        sh = le.shadow_federation(archive(lvl.shadow_seed), lvl.shadow_seed, defense, nt, 1e-2)
    finally:
        le.ce.load_archive = real_load
    target = ce.load_archive(archive(SEED))
    shared = sum(int(np.array_equal(c.C, target["C"][j][c.round]))
                 for j, h in sh.hist.items() for c in h)
    rep = {"archives_opened": opened, "source": sh.source,
           "shadow_seed": lvl.shadow_seed, "target_seed": SEED,
           "outside_population": lvl.shadow_seed not in POPULATION,
           "outside_grid_test_seeds": lvl.shadow_seed not in GRID_TEST,
           "payloads_shared_with_target": shared}
    rep["passed"] = bool(opened == [os.path.basename(archive(lvl.shadow_seed))]
                         and sh.source == opened[0] and lvl.shadow_seed != SEED
                         and rep["outside_population"] and rep["outside_grid_test_seeds"]
                         and shared == 0)
    return sh, rep


def tripwire_state(aud: InnovationAuditor, present):
    aud_t = InnovationAuditor(mad_floor=1e-2)
    aud_t.cbar = dict(aud.cbar)
    aud_t.hist = GuardedDict(aud.hist, {VICTIM}, "target auditor records")
    aud_t.scores = Tripwire("target round scores")
    present_t = GuardedDict(present, {VICTIM}, "target round-r masks")
    return aud_t, Tripwire("target defended-score history"), present_t, \
        Tripwire("target round-r peer payloads")


@contextlib.contextmanager
def no_archive_reads():
    saved = (np.load, ce.load_archive)

    def refuse(*_a, **_k):
        raise LeakError("archive read while crafting")

    np.load = ce.load_archive = refuse
    try:
        yield
    finally:
        np.load, ce.load_archive = saved


def make_cfg(defense: str, knowledge: str, norm):
    if defense == "P":
        return AttackConfig(phi=0.0, beta=BETA, knowledge=knowledge, basis_steps=80,
                            full_steps=80, restarts=3, mad_floor=1e-2, aim="mimic")
    return CombinedAttackConfig(phi=0.0, beta=BETA, rule="max", knowledge=knowledge,
                                aim="mimic", basis_steps=40, full_steps=80, restarts=1,
                                warm_families=False, mad_floor=1e-2, schedule="flat",
                                norm={k: tuple(v) for k, v in norm.items()})


def check_views(job: dict) -> dict:
    seed = job["seed"]
    arch = ce.load_archive(archive(seed))
    n = len(arch["C"])
    aud = InnovationAuditor(mad_floor=1e-2)
    hist: Dict[int, Dict[int, float]] = {}
    n_cmp, bad = 0, []
    for r in range(1, max(arch["C"][0]) + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        present = {c: arch["present"][c][r] for c in range(n)}
        pay = {c: (arch["C"][c][r], present[c]) for c in range(n)}
        if r >= 3:
            for v in job["victims"]:
                ref = build_view(aud, r, v, present, None, "peer_lagged")
                info = la.extract_information(aud, hist, r, v, present[v],
                                              la.parse_level("delayed:1"))
                d = view_diff(ref, la.delayed_view(info))
                n_cmp += 1
                if d:
                    bad.append({"round": r, "victim": v, "fields": d})
        rs = aud.observe_round(r, pay)
        hist[r] = {c: float(rs.per_client[c]["suspicion"]) for c in range(n)}
    return {"check": "views", "seed": seed, "compared": n_cmp, "mismatches": bad,
            "passed": n_cmp > 0 and not bad}


def released_pass(job: dict) -> dict:
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    if job["defense"] == "P":
        res = ae.score_one_archive(dict(
            attack_start=12, scope_start=12, mad_floor=1e-2, basis_steps=80, full_steps=80,
            restarts=3, path=archive(SEED), seed=SEED, victim=VICTIM, phi=0.0, beta=BETA,
            knowledge=job["knowledge"], target="suspicion", aim="mimic", honest=False))
        mat = res["mat"]["suspicion"]
    else:
        res = ce.score_one_archive(dict(
            attack_start=12, scope_start=12, mad_floor=1e-2, basis_steps=40, full_steps=80,
            restarts=1, warm=False, path=archive(SEED), seed=SEED, victim=VICTIM, phi=0.0,
            beta=BETA, rule="max", knowledge=job["knowledge"], aim="mimic", schedule="flat",
            honest=False, norm={k: tuple(v) for k, v in job["norm"].items()}))
        mat = res["mat"]["combined_max"]
    return {"mat": mat, "diag": res["diag"], "seconds": time.perf_counter() - t0}


def limited_pass(job: dict) -> dict:
    t = dict(path=archive(SEED), defense=job["defense"], level=job["level"], beta=BETA,
             seed=SEED, victim=VICTIM, phi=0.0, attack_start=12, scope_start=12,
             mad_floor=1e-2, fb_step=0.5, fb_decay=0.8, es_sigma0=0.1, es_tol=1e-2,
             norm=job["norm"], solver=[80, 80, 3] if job["defense"] == "P" else [40, 80, 1],
             tau=job["tau"], cal_median=float("nan"), signature="check")
    return le.score_pass(t)


def check_pass(job: dict) -> dict:
    ref = released_pass(job)
    mine = limited_pass(job)
    err = [abs(x["predicted"] - x["realised"]) for x in mine["log"]
           if np.isfinite(x["predicted"]) and np.isfinite(x["realised"])]
    mean_err = float(np.mean(err)) if err else None
    return {"check": "passes", "defense": job["defense"], "level": job["level"],
            "released_knowledge": job["knowledge"],
            "score_matrix_identical": bool(np.array_equal(ref["mat"], mine["mat"],
                                                          equal_nan=True)),
            "pred_vs_real_mean_abs": [mean_err, ref["diag"]["pred_vs_real_mean_abs"]],
            "seconds": [ref["seconds"], mine["seconds"]],
            "victim_mean_score": float(np.nanmean(mine["mat"][:, VICTIM])),
            "passed": bool(np.array_equal(ref["mat"], mine["mat"], equal_nan=True)
                           and mean_err == ref["diag"]["pred_vs_real_mean_abs"])}


def check_lockstep(job: dict) -> dict:
    torch.set_num_threads(1)
    dfn, norm, tau = job["defense"], job["norm"], job["tau"]
    arch = ce.load_archive(archive(SEED))
    n = len(arch["C"])
    nt = {k: tuple(v) for k, v in norm.items()}
    seed = 1000 * SEED + VICTIM
    ref = (WhiteBoxAttacker(make_cfg(dfn, "peer_lagged", norm), seed=seed) if dfn == "P"
           else CombinedAttacker(make_cfg(dfn, "peer_lagged", norm), seed=seed))
    mine = la.LimitedAccessAttacker(dfn, la.parse_level("delayed:1"),
                                    make_cfg(dfn, "peer_lagged", norm), seed, tau=tau)
    aud = InnovationAuditor(mad_floor=1e-2)
    hist: Dict[int, Dict[int, float]] = {}
    rounds, bad = [], []
    for r in range(1, max(arch["C"][0]) + 1):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        present = {c: arch["present"][c][r] for c in range(n)}
        pay = {c: (arch["C"][c][r], present[c]) for c in range(n)}
        if r >= 12:
            C_hon = arch["C"][VICTIM][r]
            if dfn == "P":
                Cr, ir = ref.craft(aud, r, VICTIM, C_hon, present, None)
            else:
                prev = {c: s for c, s in hist[r - 1].items() if c != VICTIM}
                Cr, ir = ref.craft(aud, r, VICTIM, C_hon, present, None, aim=la.mimic_aim(prev))
            Cm, im = mine.craft(aud, hist, r, VICTIM, C_hon, present, None, None)
            pr = ir.get("predicted_suspicion", ir.get("predicted_combined"))
            ok = (np.array_equal(Cr, Cm) and same(pr, im["predicted"])
                  and same(ir["aim"], im["aim"]) and same(ir["budget"], im["budget"]))
            rounds.append({"round": r, "identical": bool(ok),
                           "digest": le.payload_digest(Cm)})
            if not ok:
                bad.append(r)
            pay[VICTIM] = (Cm, present[VICTIM])
        rs = aud.observe_round(r, pay)
        hist[r] = le.defended_scores(dfn, rs, pay, aud.cbar.get(r - 1), n, 1e-2, nt)
    return {"check": "lockstep", "defense": dfn, "rounds": len(rounds), "mismatch_rounds": bad,
            "per_round": rounds, "passed": len(rounds) == 24 and not bad}


def poison_state(aud: InnovationAuditor, hist, present, victim: int, last: int):
    aud2 = copy.deepcopy(aud)
    for j, h in aud2.hist.items():
        if j == victim:
            continue
        for c in h:
            if c.round > last:
                c.C = np.full_like(c.C, np.nan)
                if c.innovation is not None:
                    c.innovation = np.full_like(c.innovation, np.nan)
                c.u1 = c.u2 = c.norm = float("nan")
                c.present = ~c.present
    for rsc in aud2.scores:
        if rsc.round > last:
            for j, v in rsc.per_client.items():
                if j != victim:
                    for k in list(v):
                        if isinstance(v[k], float):
                            v[k] = float("nan")
    hist2 = {t: {j: (s if (j == victim or t <= last) else float("nan")) for j, s in d.items()}
             for t, d in hist.items()}
    present2 = {j: (m if j == victim else ~np.asarray(m, bool)) for j, m in present.items()}
    return aud2, hist2, present2


def info_equal(a: la.InformationSet, b: la.InformationSet) -> bool:
    if (a.round, a.victim, a.delay) != (b.round, b.victim, b.delay):
        return False
    if a.cbar.keys() != b.cbar.keys() or not all(same(a.cbar[k], b.cbar[k]) for k in a.cbar):
        return False
    rec = lambda c: (c.round, c.C, c.present, c.innovation, c.innov_rows, c.u1, c.u2, c.norm)
    if not same([rec(c) for c in a.own_hist], [rec(c) for c in b.own_hist]):
        return False
    if a.peer_hist.keys() != b.peer_hist.keys():
        return False
    for j in a.peer_hist:
        if not same([rec(c) for c in a.peer_hist[j]], [rec(c) for c in b.peer_hist[j]]):
            return False
    return same(a.present_own, b.present_own) and same(a.peer_scores, b.peer_scores)


def check_leak(job: dict) -> dict:
    torch.set_num_threads(1)
    dfn, lvl_name, norm, tau = job["defense"], job["level"], job["norm"], job["tau"]
    lvl = la.parse_level(lvl_name)
    nt = {k: tuple(v) for k, v in norm.items()}
    out = {"check": "leak", "defense": dfn, "level": lvl_name, "rounds": []}
    passed = True
    shadow = None
    if lvl.kind == "shadow":
        shadow, out["provenance"] = shadow_with_provenance(lvl, dfn, nt)
        passed &= out["provenance"]["passed"]
    for r in LEAK_ROUNDS:
        arch, aud, hist, present, pay = honest_state(SEED, dfn, r, nt)
        n = len(arch["C"])
        last = r - lvl.delay if lvl.sees_peers else -10 ** 9
        aud_p, hist_p, present_p = poison_state(aud, hist, present, VICTIM, last)
        peer_C = {j: arch["C"][j][r] for j in range(n) if j != VICTIM}
        peer_C_p = {j: np.full_like(c, np.nan) for j, c in peer_C.items()}
        info_c = la.extract_information(aud, hist, r, VICTIM, present[VICTIM], lvl)
        info_p = la.extract_information(aud_p, hist_p, r, VICTIM, present_p[VICTIM], lvl)
        inventory = {
            "cbar_rounds": [min(info_c.cbar), max(info_c.cbar)],
            "own_rounds": [info_c.own_hist[0].round, info_c.own_hist[-1].round],
            "peer_latest_round": (max(c.round for h in info_c.peer_hist.values() for c in h)
                                  if info_c.peer_hist else None),
            "n_peer_scores": len(info_c.peer_scores)}
        states = [(aud, hist, present, peer_C), (aud_p, hist_p, present_p, peer_C_p)]
        if not lvl.sees_peers:
            states.append(tripwire_state(aud, present))
        res, trip_error = [], None
        for k_state, (aud_x, hist_x, present_x, peer_x) in enumerate(states):
            atk = la.LimitedAccessAttacker(dfn, lvl, make_cfg(dfn, "peer_lagged", norm),
                                           1000 * SEED + VICTIM, tau=tau,
                                           cal_median=job["cal_median"], shadow=shadow)
            if lvl.kind == "feedback":
                atk.margin = 0.7
            oracle = None
            if lvl.kind in ("query", "decision"):
                Cb = aud.cbar.get(r - 1)
                oracle = la.ExactOracle(
                    aud, r, pay, VICTIM,
                    lambda rs, p: le.defended_scores(dfn, rs, p, Cb, n, 1e-2, nt)[VICTIM],
                    lvl.queries, tau)
            if k_state < 2:
                C, rec = atk.craft(aud_x, hist_x, r, VICTIM, arch["C"][VICTIM][r], present_x,
                                   peer_x, oracle)
            else:
                try:
                    with no_archive_reads():
                        C, rec = atk.craft(aud_x, hist_x, r, VICTIM, arch["C"][VICTIM][r],
                                           present_x, peer_x, oracle)
                except LeakError as e:
                    trip_error, C, rec = str(e), None, {}
            res.append((C, rec))
        keys = ("status", "predicted", "aim", "budget", "cohort_scale", "oracle_calls",
                "rung", "all_flagged", "oracle_score", "margin", "shadow_rho")
        rec_same = all(same(res[0][1].get(k), x[1].get(k)) for x in res[1:] for k in keys)
        pay_same = all(x[0] is not None and np.array_equal(res[0][0], x[0]) for x in res[1:])
        inf_same = info_equal(info_c, info_p)
        calls_ok = res[0][1]["oracle_calls"] <= max(lvl.queries, 0)
        if lvl.sees_peers:
            aud_k, hist_k, _ = poison_state(aud, hist, present, VICTIM, last - 1)
        else:
            aud_k = copy.deepcopy(aud)
            for c in aud_k.hist[VICTIM]:
                if c.round == 11:
                    c.C = np.full_like(c.C, np.nan)
                    c.innovation = np.full_like(c.innovation, np.nan)
                    if lvl.kind == "shadow":
                        c.norm = float("nan")
            hist_k = hist
        info_k = la.extract_information(aud_k, hist_k, r, VICTIM, present[VICTIM], lvl)
        if lvl.kind == "pseudo":
            build = lambda i, sh=None: la.pseudo_view(i, n - 1, 12)
        elif lvl.kind == "shadow":
            build = lambda i, sh=shadow: la.shadow_view(i, sh, la.shadow_scale(i, sh, 12))
        else:
            build = lambda i, sh=None: la.delayed_view(i)
        control = bool(view_diff(build(info_c), build(info_k)))
        if lvl.kind == "shadow":
            sh_k = copy.deepcopy(shadow)
            for j, h in sh_k.hist.items():
                if j != VICTIM:
                    for c in h:
                        if c.round == r:
                            c.C = np.full_like(c.C, np.nan)
                            c.u1 = c.u2 = c.norm = float("nan")
                            if c.innovation is not None:
                                c.innovation = np.full_like(c.innovation, np.nan)
            control = control and bool(view_diff(build(info_c), build(info_c, sh_k)))
        inv_ok = (inventory["cbar_rounds"][1] == r - 1 and inventory["own_rounds"][1] == r - 1
                  and inventory["peer_latest_round"] == (r - lvl.delay if lvl.sees_peers
                                                         else None)
                  and inventory["n_peer_scores"] == (n - 1 if lvl.sees_peers else 0))
        ok = rec_same and pay_same and inf_same and calls_ok and control and inv_ok
        passed &= ok
        row = {"round": r, "payload_identical": pay_same,
               "record_identical": rec_same, "information_identical": inf_same,
               "oracle_calls": res[0][1]["oracle_calls"], "calls_within_Q": calls_ok,
               "control_detected": control, "inventory": inventory,
               "inventory_as_specified": inv_ok,
               "predicted": res[0][1]["predicted"], "aim": res[0][1]["aim"]}
        if not lvl.sees_peers:
            row["tripwire_crafts"] = trip_error is None
            row["tripwire_error"] = trip_error
        if lvl.kind == "shadow":
            row["shadow_rho"] = res[0][1].get("shadow_rho")
            row["budget_unit"] = res[0][1].get("cohort_scale")
        out["rounds"].append(row)
    out["passed"] = bool(passed)
    return out


class _ShadowAt:
    def __init__(self, sh: la.ShadowFederation, last: int):
        self.cbar = {t: c for t, c in sh.cbar.items() if t <= last}
        self.hist = {j: [c for c in h if c.round <= last] for j, h in sh.hist.items()}


def _close(a, b, rtol: float = 1e-9) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    ok = ~np.isnan(a)
    if not ok.any():
        return 0.0
    return float(np.max(np.abs(a[ok] - b[ok]) / np.maximum(1.0, np.abs(b[ok]))))


def check_shadow_views(job: dict) -> dict:
    seed, S = job["seed"], job["shadow_seed"]
    nt = {k: tuple(v) for k, v in job["norm"].items()}
    lvl = la.parse_level("shadow:%d" % S)
    sh = {d: le.shadow_federation(archive(S), S, d, nt, 1e-2) for d in ("P", "F")}
    ns = argparse.Namespace(attack_start=12, scope_start=12, mad_floor=1e-2)
    hon = le._honest_pass(dict(le.honest_task(archive(S), S, ns), signature=""))
    raw, rounds = hon["raw"], hon["rounds"]
    n = raw["suspicion"].shape[1]
    sc_ok = True
    for i, r in enumerate(rounds):
        rd = {k: {c: float(raw[k][i, c]) for c in range(n)} for k in raw}
        cm = ce.combined_from_raw(rd, nt, n)["combined_max"]
        sc_ok &= all(same(sh["P"].scores[r][c], rd["suspicion"][c]) and
                     same(sh["F"].scores[r][c], float(cm[c])) for c in range(n))
    rec = lambda c: (c.round, c.C, c.present, c.innovation, c.innov_rows, c.u1, c.u2, c.norm)
    hist_same = all(same([rec(c) for c in sh["P"].hist[j]], [rec(c) for c in sh["F"].hist[j]])
                    for j in sh["P"].hist)
    arch_s = ce.load_archive(archive(S))
    aud_s = InnovationAuditor(mad_floor=1e-2)
    sh_norm = {}
    for r in range(1, 12):
        if (r - 1) in arch_s["cbar"]:
            aud_s.set_aggregate(r - 1, arch_s["cbar"][r - 1])
        rs = aud_s.observe_round(r, {c: (arch_s["C"][c][r], arch_s["present"][c][r])
                                     for c in range(n)})
        sh_norm[r] = np.asarray([rs.per_client[c]["norm"] for c in range(n)], float)
    shadow_C = {r: {c: arch_s["C"][c][r] for c in range(n)} for r in range(1, 36)}
    shadow_present = {r: {c: arch_s["present"][c][r] for c in range(n)} for r in range(1, 36)}
    arch = ce.load_archive(archive(seed))
    aud = InnovationAuditor(mad_floor=1e-2)
    hist: Dict[int, Dict[int, float]] = {}
    own_fields = ("victim", "K", "d", "Cbar_prev", "t_own", "own_prev", "own_prev2",
                  "own_C_prev", "own_C_lags", "own_rep_lag_ok")
    worst = {k: 0.0 for k in ("peer_I", "u1", "u2", "norm", "step", "rep_raw", "x_const",
                              "pool", "peer_full")}
    n_cmp, n_skip, bad, rho_err, scale_err = 0, 0, [], 0.0, 0.0
    for r in range(1, 36):
        if (r - 1) in arch["cbar"]:
            aud.set_aggregate(r - 1, arch["cbar"][r - 1])
        present = {c: arch["present"][c][r] for c in range(n)}
        pay = {c: (arch["C"][c][r], present[c]) for c in range(n)}
        if r >= 12:
            for v in (0, 1):
                info = la.extract_information(aud, hist, r, v, present[v], lvl)
                rho = la.shadow_scale(info, sh["P"], 12)
                own_n = np.asarray([s.per_client[v]["norm"] for s in aud.scores
                                    if 3 <= s.round <= 11], float)
                meds = []
                for t in range(3, 12):
                    x = np.delete(sh_norm[t], v)
                    meds.append(float(np.median(x[np.isfinite(x) & (x > la.EPS)])))
                rho_ind = float(np.median(own_n[np.isfinite(own_n) & (own_n > la.EPS)])
                                / np.median(meds))
                rho_err = max(rho_err, abs(rho - rho_ind) / rho_ind)
                view = la.shadow_view(info, sh["P"], rho)
                ref_own = build_view(aud, r, v, present, None, "peer_lagged")
                if view is None or ref_own is None:
                    d_own = [] if view is None and ref_own is None else ["<none>"]
                    bad.append({"round": r, "victim": v, "own_fields": d_own or ["<both none>"]})
                    continue
                d_own = [f for f in own_fields if not same(getattr(view, f), getattr(ref_own, f))]
                sp = la.shadow_peers(sh["P"], r, v, view.K)
                ref = build_view(_ShadowAt(sh["P"], r - 1), r, v, shadow_present[r],
                                 {j: shadow_C[r][j] for j in range(n) if j != v}, "peer_current")
                if ref is None:
                    n_skip += 1
                    continue
                n_cmp += 1

                def ent(es):
                    size = view.K * view.d + view.K + 1
                    return np.stack([np.full(size, np.nan) if e is None else
                                     np.concatenate([e.I.ravel(), e.mask, [e.norm]])
                                     for e in es]) if es else np.zeros((0, size))

                diffs = {"peer_I": _close(ent(sp["peer_I"]), ent(ref.peer_I)),
                         "u1": _close(sp["u1"], ref.peer_u1), "u2": _close(sp["u2"], ref.peer_u2),
                         "norm": _close(sp["norm"], ref.peer_norm),
                         "step": _close(sp["step"], ref.peer_step),
                         "rep_raw": _close(sp["rep_raw"], ref.peer_rep_raw),
                         "x_const": _close(sp["x_const"], ref.peer_x_const),
                         "pool": _close(ent(sp["pool"]), ent(ref.pool_I)),
                         "peer_full": _close(np.stack(sp["peer_full"]),
                                             np.stack(ref.peer_I_full))}
                for k, x in diffs.items():
                    worst[k] = max(worst[k], x)
                live = rho * sp["norm"]
                live = live[np.isfinite(live) & (live > la.EPS)]
                scale_err = max(scale_err, abs(view.cohort_scale - float(np.median(live))),
                                float(np.max(np.abs(view.peer_norm - rho * sp["norm"]))))
                if d_own or max(diffs.values()) > 1e-9:
                    bad.append({"round": r, "victim": v, "own_fields": d_own,
                                "peer_rel_diff": diffs})
        rs = aud.observe_round(r, pay)
        hist[r] = {c: float(rs.per_client[c]["suspicion"]) for c in range(n)}
    ok = (sc_ok and hist_same and n_cmp > 0 and not bad and rho_err < 1e-12
          and scale_err == 0.0)
    return {"check": "shadow_views", "seed": seed, "level": lvl.name, "compared": n_cmp,
            "skipped_seat_undefined": n_skip, "mismatches": bad,
            "peer_max_rel_diff_vs_build_view": worst, "rho_max_rel_diff_independent": rho_err,
            "scaling_max_abs_err": scale_err, "scores_equal_honest_replay": bool(sc_ok),
            "records_equal_across_defenses": bool(hist_same), "passed": bool(ok)}


def check_oracle(job: dict) -> dict:
    nt = {k: tuple(v) for k, v in job["norm"].items()}
    arch, aud, hist, present, pay = honest_state(SEED, "F", 20, nt)
    n = len(arch["C"])
    Cb = aud.cbar[19]
    fn = lambda rs, p: le.defended_scores("F", rs, p, Cb, n, 1e-2, nt)[VICTIM]
    before = ({c: len(h) for c, h in aud.hist.items()}, len(aud.scores))
    orc = la.ExactOracle(aud, 20, pay, VICTIM, fn, 2, job["tau"])
    C = la.quantise(Cb)
    s1 = orc.score(C)
    f2 = orc.flagged(C)
    budget = grid = False
    try:
        orc.score(C)
    except RuntimeError:
        budget = True
    try:
        la.ExactOracle(aud, 20, pay, VICTIM, fn, 2, job["tau"]).score(C + 1e-7)
    except ValueError:
        grid = True
    after = ({c: len(h) for c, h in aud.hist.items()}, len(aud.scores))
    pay2 = dict(pay); pay2[VICTIM] = (C, present[VICTIM])
    rs = aud.observe_round(20, pay2)
    real = le.defended_scores("F", rs, pay2, Cb, n, 1e-2, nt)[VICTIM]
    return {"check": "oracle", "budget_enforced": budget, "grid_enforced": grid,
            "state_restored": before == after, "oracle_equals_real_round": s1 == real,
            "flag_consistent": f2 == bool(real > job["tau"]),
            "passed": budget and grid and before == after and s1 == real
            and f2 == bool(real > job["tau"])}


def run_job(job: dict) -> dict:
    t0 = time.perf_counter()
    fn = {"views": check_views, "passes": check_pass, "lockstep": check_lockstep,
          "leak": check_leak, "oracle": check_oracle,
          "shadow_views": check_shadow_views}[job["kind"]]
    out = fn(job)
    out["seconds"] = time.perf_counter() - t0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--only", default="", help="comma-separated subset of: views, passes, "
                                               "lockstep, leak, oracle, shadow_views")
    ap.add_argument("--levels", default="",
                    help="restrict the leak audit to these levels (default: every level)")
    ap.add_argument("--out", default=os.path.join(le.RESULTS, "limited_access_checks.json"))
    a = ap.parse_args()
    t0 = time.time()
    rc = released_constants()
    ns = argparse.Namespace(attack_start=12, scope_start=12, mad_floor=1e-2)
    honest = {s: le._honest_pass(dict(le.honest_task(archive(s), s, ns), signature=""))
              for s in (0, 1, 3, 4)}
    calib = le.calibrate(honest, (0, 1), (3, 4), 10, 0.01)
    release = le.check_against_release(calib)
    base = lambda d: {"norm": rc["norm"], "tau": rc["tau"][d],
                      "cal_median": calib["cal_median"][d]}
    jobs = [{"kind": "views", "seed": s, "victims": [0, 1]} for s in (0, 3)]
    jobs += [{"kind": "shadow_views", "seed": s, "shadow_seed": S, "norm": rc["norm"]}
             for s in (0, 3) for S in (5, 6)]
    for d in ("P", "F"):
        jobs.append(dict(base(d), kind="passes", defense=d, level="current",
                         knowledge="peer_current"))
        jobs.append(dict(base(d), kind="passes", defense=d, level="delayed:1",
                         knowledge="peer_lagged"))
        jobs.append(dict(base(d), kind="lockstep", defense=d))
        for lv in LEAK_LEVELS:
            jobs.append(dict(base(d), kind="leak", defense=d, level=lv))
    jobs.append(dict(base("F"), kind="oracle"))
    if a.only:
        keep = {x.strip() for x in a.only.split(",")}
        jobs = [j for j in jobs if j["kind"] in keep]
    if a.levels:
        keep = {x.strip() for x in a.levels.split(",")}
        jobs = [j for j in jobs if j["kind"] != "leak" or j["level"] in keep]
    order = {"lockstep": 0, "passes": 1, "leak": 2, "shadow_views": 3, "views": 4, "oracle": 5}
    jobs.sort(key=lambda j: order[j["kind"]])
    results = []
    with ProcessPoolExecutor(max_workers=max(1, min(20, a.workers, len(jobs)))) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print("  %-8s %-3s %-12s passed=%s  %.0fs" % (
                r["check"], r.get("defense", ""), r.get("level", r.get("seed", "")),
                r["passed"], r["seconds"]), flush=True)
    results.append({"check": "calibration", "passed": bool(release["identical"]),
                    "release": release, "cal_median": calib["cal_median"], "seconds": 0.0})
    allp = all(r["passed"] for r in results)
    from passportfl.reproducibility import environment
    out = {"meta": {"command": "python experiments/limited_access_checks.py"
                               + (" --only " + a.only if a.only else "")
                               + (" --levels " + a.levels if a.levels else ""),
                    "environment": environment(), "wall_s": round(time.time() - t0, 1),
                    "population": {"seed": SEED, "victim": VICTIM, "beta": BETA,
                                   "leak_rounds": list(LEAK_ROUNDS),
                                   "shadow_view_target_seeds": [0, 3],
                                   "shadow_seeds": [5, 6]}},
           "all_passed": bool(allp),
           "results": sorted(results, key=lambda r: (r["check"], r.get("defense", ""),
                                                     str(r.get("level", r.get("seed", "")))))}
    with open(os.path.abspath(a.out), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("all passed: %s   wrote %s  (%.0fs)" % (allp, os.path.abspath(a.out), time.time() - t0))
    return 0 if allp else 1


if __name__ == "__main__":
    sys.exit(main())
