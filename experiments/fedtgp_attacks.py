from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional

import numpy as np

OFFSET_ROUNDS = tuple(range(2, 12))


def load_archive_ext(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    arch = {"C": z["C"].item(), "present": z["present"].item(), "cbar": z["cbar"].item()}
    if "qbar" in z.files:
        arch["qbar"] = z["qbar"].item()
        arch["qbar_source"] = "archive"
    else:
        arch["qbar"] = recompute_qbar(arch)
        arch["qbar_source"] = "recomputed"
    if "meta" in z.files:
        arch["meta"] = json.loads(str(z["meta"]))
    if "tgp_log" in z.files:
        arch["tgp_log"] = z["tgp_log"].item()
    return arch


def config_key(meta: Optional[dict]) -> Optional[str]:
    if not meta:
        return None
    tgp = {k: v for k, v in (meta.get("tgp") or {}).items() if k not in ("seed", "device")}
    return json.dumps({"arch_group": meta.get("arch_group"), "server": meta.get("server"),
                       "lam": meta.get("lam"), "tgp": tgp}, sort_keys=True)


def attacker_rank(row, victim: int) -> Optional[dict]:
    row = np.asarray(row, float)
    a = row[victim]
    if np.isnan(a):
        return None
    h = np.delete(row, victim)
    nan = np.isnan(h)
    lower = int((h < a).sum())
    ties = int((h == a).sum())
    return {"rank": 1 + lower + ties + int(nan.sum()), "strict": 1 + lower,
            "ties": ties, "nan_honest": int(nan.sum())}


def rank_summary(records: List[Optional[dict]]) -> dict:
    ok = [x for x in records if x is not None]
    rk = np.asarray([x["rank"] for x in ok], float)
    st = np.asarray([x["strict"] for x in ok], float)
    nanf = lambda f, v: float(f(v)) if v.size else float("nan")
    first = lambda v: float((v == 1).sum()) / len(records) if records else float("nan")
    return {"rank_mean": nanf(np.mean, rk), "rank1_frac": first(rk),
            "rank_median": nanf(np.median, rk),
            "rank_mean_strict": nanf(np.mean, st), "rank1_frac_strict": first(st),
            "n_rounds": len(ok), "n_attacker_undefined": len(records) - len(ok),
            "n_rounds_with_ties": sum(1 for x in ok if x["ties"]),
            "n_honest_undefined": sum(x["nan_honest"] for x in ok)}


EXACT_RULES = ("median_dev", "trimmed_mean_dev", "bulyan_dev")


def zero_update_margins(V: np.ndarray) -> dict:
    from passportfl.baselines import bulyan_deviation, median_deviation, trimmed_mean_deviation
    scorers = {"median_dev": median_deviation, "trimmed_mean_dev": trimmed_mean_deviation,
               "bulyan_dev": lambda innov: bulyan_deviation(innov, n_byzantine=2)}
    n = len(V)
    out = {}
    for rule in EXACT_RULES:
        cnorm, mmin, mmed = [], [], []
        for v in range(n):
            innov = {j: (np.zeros_like(V[j]) if j == v else V[j]) for j in range(n)}
            s = scorers[rule](innov)
            h = np.asarray([s[j] for j in range(n) if j != v])
            cnorm.append(s[v])
            mmin.append(float(h.min() - s[v]))
            mmed.append(float(np.median(h) - s[v]))
        out[rule] = {"centre_norm": cnorm, "m_min": mmin, "m_med": mmed,
                     "inverts_frac_seats": float(np.mean(np.asarray(mmin) > 0)),
                     "seat0_centre_norm": cnorm[0], "seat0_m_min": mmin[0], "seat0_m_med": mmed[0]}
    return out


def innovation_geometry(arch: dict, bcast: dict, Q: dict, rounds: int, scope_start: int,
                        exact: bool = True) -> dict:
    clients = len(arch["C"])
    rows, ex_rows = [], []
    for r in range(scope_start, rounds + 1):
        B = bcast[r - 1]
        V = np.stack([(arch["C"][i][r] - B).ravel() for i in range(clients)])
        med = np.median(V, axis=0)
        n = len(V)
        c = int(np.floor(0.2 * n))
        tm = np.sort(V, axis=0)[c:n - c].mean(axis=0)
        dm = np.linalg.norm(V - med, axis=1)
        dt = np.linalg.norm(V - tm, axis=1)
        offset = (Q[r - 1] - B).ravel()
        spread = np.linalg.norm(np.stack([(arch["C"][i][r] - Q[r - 1]).ravel()
                                          for i in range(clients)]), axis=1)
        rows.append({"round": r, "median_centre_norm": float(np.linalg.norm(med)),
                     "median_min_dist": float(dm.min()), "median_med_dist": float(np.median(dm)),
                     "trimmed_centre_norm": float(np.linalg.norm(tm)),
                     "trimmed_min_dist": float(dt.min()), "trimmed_med_dist": float(np.median(dt)),
                     "offset_q_minus_b_fro": float(np.linalg.norm(offset)),
                     "honest_spread_about_q_median": float(np.median(spread)),
                     "honest_innov_norm_median": float(np.median(np.linalg.norm(V, axis=1))),
                     "median_inverts_all": bool(np.linalg.norm(med) < dm.min()),
                     "trimmed_inverts_all": bool(np.linalg.norm(tm) < dt.min())})
        if exact:
            ex = zero_update_margins(V)
            ex_rows.append({"round": r, **{f"exact_{rule}_{k}": ex[rule][k] for rule in EXACT_RULES
                                           for k in ("inverts_frac_seats", "seat0_centre_norm",
                                                     "seat0_m_min", "seat0_m_med")},
                            **{f"exact_{rule}_m_min_seats": ex[rule]["m_min"] for rule in EXACT_RULES}})
    keys = [k for k in rows[0] if k not in ("round", "median_inverts_all", "trimmed_inverts_all")]
    summ = {k: float(np.median([x[k] for x in rows])) for k in keys}
    summ["median_inverts_all_frac"] = float(np.mean([x["median_centre_norm"] < x["median_min_dist"]
                                                     for x in rows]))
    summ["trimmed_inverts_all_frac"] = float(np.mean([x["trimmed_centre_norm"] < x["trimmed_min_dist"]
                                                      for x in rows]))
    out = {"per_round": rows, "median_over_rounds": summ}
    if exact:
        es = {}
        for rule in EXACT_RULES:
            allm = np.concatenate([x[f"exact_{rule}_m_min_seats"] for x in ex_rows])
            es[rule] = {"inverts_frac_round_seats": float(np.mean(allm > 0)),
                        "inverts_all_seats_frac_rounds": float(np.mean(
                            [x[f"exact_{rule}_inverts_frac_seats"] == 1.0 for x in ex_rows])),
                        "m_min_median": float(np.median(allm)),
                        "seat0_centre_norm_median": float(np.median([x[f"exact_{rule}_seat0_centre_norm"]
                                                                     for x in ex_rows])),
                        "seat0_m_min_median": float(np.median([x[f"exact_{rule}_seat0_m_min"]
                                                               for x in ex_rows])),
                        "seat0_m_med_median": float(np.median([x[f"exact_{rule}_seat0_m_med"]
                                                               for x in ex_rows]))}
        out["exact_per_round"] = ex_rows
        out["exact_summary"] = es
    return out


def recompute_qbar(arch: dict) -> Dict[int, np.ndarray]:
    out = {}
    clients = sorted(arch["C"])
    for r in sorted(arch["cbar"]):
        K, d = arch["cbar"][r].shape
        num = np.zeros((K, d))
        cnt = np.zeros(K)
        for i in clients:
            if r not in arch["C"][i]:
                continue
            p = np.asarray(arch["present"][i][r], bool)
            num[p] += arch["C"][i][r][p]
            cnt[p] += 1
        q = arch["cbar"][r].astype(np.float64).copy()
        has = cnt > 0
        q[has] = num[has] / cnt[has, None]
        out[r] = q
    return out


def offset_estimate(arch: dict, cid: int, r: int):
    pub = arch["published"][cid]
    K, d = arch["cbar"][min(arch["cbar"])].shape
    num = np.zeros((K, d))
    cnt = np.zeros(K)
    for t in OFFSET_ROUNDS:
        if t >= r or t not in pub or (t - 1) not in arch["cbar"]:
            continue
        p = np.asarray(arch["present"][cid][t], bool)
        num[p] += (pub[t] - arch["cbar"][t - 1])[p]
        cnt[p] += 1
    has = cnt > 0
    b = np.zeros((K, d))
    b[has] = num[has] / cnt[has, None]
    return b, has


def attack_fedtgp_echo(arch, cid, r, rng):
    return arch["cbar"][r - 1].copy() if (r - 1) in arch["cbar"] else arch["C"][cid][r].copy()


def attack_fedtgp_offset_echo(arch, cid, r, rng):
    if (r - 1) not in arch["cbar"]:
        return arch["C"][cid][r].copy()
    out = arch["cbar"][r - 1].copy()
    b, has = offset_estimate(arch, cid, r)
    rows = np.asarray(arch["present"][cid][r], bool) & has
    out[rows] += b[rows]
    return out


def attack_fedtgp_oracle_echo(arch, cid, r, rng):
    if (r - 1) not in arch["cbar"] or (r - 1) not in arch["qbar"]:
        return arch["C"][cid][r].copy()
    out = arch["cbar"][r - 1].copy()
    rows = np.asarray(arch["present"][cid][r], bool)
    out[rows] = arch["qbar"][r - 1][rows]
    return out


def _usable_rounds(arch: dict, cid: int, r: int) -> List[int]:
    pub = arch["published"][cid]
    return [t for t in OFFSET_ROUNDS if t < r and t in pub and (t - 1) in arch["cbar"]]


def recent_offset(arch: dict, cid: int, r: int):
    K, d = arch["cbar"][min(arch["cbar"])].shape
    off = np.zeros((K, d))
    has = np.zeros(K, bool)
    for t in sorted(_usable_rounds(arch, cid, r), reverse=True):
        p = np.asarray(arch["present"][cid][t], bool) & ~has
        off[p] = (arch["published"][cid][t] - arch["cbar"][t - 1])[p]
        has |= p
    return off, has


def affine_fit(arch: dict, cid: int, r: int) -> Optional[dict]:
    ts = _usable_rounds(arch, cid, r)
    if not ts:
        return None
    t = max(ts)
    p = np.asarray(arch["present"][cid][t], bool)
    if p.sum() < 2:
        return None
    B = np.asarray(arch["cbar"][t - 1], np.float64)
    m = B.mean(axis=0)
    X = B[p] - m
    Y = np.asarray(arch["published"][cid][t], np.float64)[p]
    xm, ym = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - xm, Y - ym
    den = float((Xc * Xc).sum())
    if not np.isfinite(den) or den <= 0:
        return None
    beta = float((Xc * Yc).sum()) / den
    tau = ym - beta * xm
    resid = Y - (beta * X + tau)
    return {"round": t, "beta": beta, "tau": tau, "rows": int(p.sum()),
            "residual_fro": float(np.linalg.norm(resid)),
            "offset_fro": float(np.linalg.norm(Y - B[p]))}


def attack_fedtgp_recent_offset_echo(arch, cid, r, rng):
    if (r - 1) not in arch["cbar"]:
        return arch["C"][cid][r].copy()
    out = arch["cbar"][r - 1].copy()
    off, has = recent_offset(arch, cid, r)
    rows = np.asarray(arch["present"][cid][r], bool) & has
    out[rows] += off[rows]
    return out


def attack_fedtgp_affine_echo(arch, cid, r, rng):
    if (r - 1) not in arch["cbar"]:
        return arch["C"][cid][r].copy()
    out = arch["cbar"][r - 1].copy()
    fit = affine_fit(arch, cid, r)
    if fit is None:
        return out
    B = np.asarray(arch["cbar"][r - 1], np.float64)
    rows = np.asarray(arch["present"][cid][r], bool)
    mapped = fit["beta"] * (B - B.mean(axis=0)) + fit["tau"]
    out[rows] = mapped[rows].astype(out.dtype)
    return out


VARIANTS = {
    "fedtgp_echo": attack_fedtgp_echo,
    "fedtgp_offset_echo": attack_fedtgp_offset_echo,
    "fedtgp_oracle_echo": attack_fedtgp_oracle_echo,
    "fedtgp_recent_offset_echo": attack_fedtgp_recent_offset_echo,
    "fedtgp_affine_echo": attack_fedtgp_affine_echo,
}
VARIANT_ORDER = list(VARIANTS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    a = ap.parse_args()
    arch = load_archive_ext(a.archive)
    clients = len(arch["C"])
    rounds = max(max(v) for v in arch["C"].values())
    qre = recompute_qbar(arch)
    worst = max(float(np.abs(qre[r] - arch["qbar"][r]).max()) for r in arch["qbar"])
    print("qbar source: %s; max |stored - recomputed| = %.3g" % (arch["qbar_source"], worst))
    r = min(12, rounds)
    arch["published"] = {i: {t: arch["C"][i][t] for t in range(1, r)} for i in range(clients)}
    rng = np.random.default_rng(0)
    for cid in range(clients):
        x = {n: f(arch, cid, r, rng) for n, f in VARIANTS.items()}
        Pp = arch["cbar"][r - 1]
        Qp = arch["qbar"][r - 1]
        pres = np.asarray(arch["present"][cid][r], bool)
        absent = ~pres
        fit = affine_fit(arch, cid, r)
        print("client %d: ||v - echo|| offset %.3f recent %.3f affine %.3f oracle %.3f;  "
              "||v - qbar|| on declared rows echo %.3f offset %.3f recent %.3f affine %.3f;  "
              "affine fit t=%s beta %s rows %s;  absent rows carried: %s"
              % (cid, np.linalg.norm(x["fedtgp_offset_echo"] - Pp),
                 np.linalg.norm(x["fedtgp_recent_offset_echo"] - Pp),
                 np.linalg.norm(x["fedtgp_affine_echo"] - Pp),
                 np.linalg.norm(x["fedtgp_oracle_echo"] - Pp),
                 *(np.linalg.norm((x[n] - Qp)[pres]) for n in
                   ("fedtgp_echo", "fedtgp_offset_echo", "fedtgp_recent_offset_echo",
                    "fedtgp_affine_echo")),
                 fit and fit["round"], fit and "%.4f" % fit["beta"], fit and fit["rows"],
                 all(np.array_equal(x[n][absent], Pp[absent]) for n in x)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
