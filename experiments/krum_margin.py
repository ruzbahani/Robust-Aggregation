from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from passportfl.baselines import krum_scores, bulyan_center

ARCHIVES = os.path.join(HERE, "results", "archives")
F_BYZANTINE = 2
EXTRA_CONFIGS = {
    "hetero_fedproto": ("arch_hetero_s*_c10_r35*.npz", 10),
    "hetero_fedtgp_lam1": ("arch_fedtgp_lam1_s*_c10_r35*.npz", 10),
    "hetero_fedtgp_lam10": ("arch_fedtgp_lam10_s*_c10_r35*.npz", 10),
    "hetero_fedtgp_lam01": ("arch_fedtgp_lam01_s*_c10_r35*.npz", 10),
}
EXTRA_ARCHIVES = os.path.join(HERE, "results", "archives_rev2")


def neighbour_means(V: np.ndarray, k: int) -> dict:
    W = np.vstack([V, np.zeros((1, V.shape[1]))])
    dw = ((W[:, None, :] - W[None, :, :]) ** 2).sum(-1)
    dh = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
    with_zero = np.array([np.sort(np.delete(dw[i], i))[:k].mean() for i in range(len(V))])
    honest_only = np.array([np.sort(np.delete(dh[i], i))[:k].mean() for i in range(len(V))])
    return {"k": k,
            "near_mean_with_zero": float(with_zero.mean()),
            "near_mean_honest_only": float(honest_only.mean())}


def pair_stats(V: np.ndarray) -> dict:
    n = len(V)
    sq = (V ** 2).sum(1)
    D = sq[:, None] + sq[None, :] - 2.0 * V @ V.T
    np.fill_diagonal(D, np.nan)
    off = D[~np.isnan(D)]
    mu = V.mean(0)
    sigma2 = float(((V - mu) ** 2).sum(1).mean())

    k = max(1, n - F_BYZANTINE - 2)
    near = np.sort(np.where(np.isnan(D), np.inf, D), axis=1)[:, :k]
    return {
        "pair_mean": float(off.mean()),
        "pair_cv": float(off.std() / max(off.mean(), 1e-12)),
        "near_mean": float(near.mean()),
        "near_over_pair": float(near.mean() / max(off.mean(), 1e-12)),
        "two_sigma2": 2.0 * sigma2,
        "ratio": float(np.sqrt(sigma2) / (np.linalg.norm(mu) + 1e-12)),
    }


def centre_margin(V: np.ndarray, honest_indices=None, rule="median") -> dict:
    if rule == "median":
        c = np.median(V, axis=0)
    elif rule == "trimmed_mean":
        trim = int(np.floor(0.2 * len(V)))
        c = np.sort(V, axis=0)[trim:len(V) - trim].mean(axis=0)
    elif rule == "bulyan":
        if len(V) < 7:
            raise ValueError("Bulyan with at least one Byzantine client requires N >= 7")
        c = bulyan_center(dict(enumerate(V)), F_BYZANTINE)
    else:
        raise ValueError(rule)
    nc = float(np.linalg.norm(c))
    honest = V if honest_indices is None else V[honest_indices]
    dh = np.linalg.norm(honest - c, axis=1)
    return {
        "centre_norm": nc,
        "honest_to_centre_median": float(np.median(dh)),
        "honest_to_centre_min": float(dh.min()),
        "centre_margin": float(np.median(dh) - nc),
        "centre_inverts": bool(nc < np.median(dh)),
        "centre_inverts_all": bool(nc < dh.min()),
        "centre_ratio": float(nc / max(float(np.median(dh)), 1e-12)),
    }


def margin(V: np.ndarray, seat: str = "replace") -> dict:
    if seat == "replace":
        W = V.copy()
        W[0] = 0.0
    elif seat == "insert":
        W = np.vstack([np.zeros((1, V.shape[1])), V])
    else:
        raise ValueError(seat)
    s = krum_scores({i: W[i] for i in range(len(W))}, n_byzantine=F_BYZANTINE)
    zero = s[0]
    hon = np.array([s[i] for i in range(1, len(W))])
    return {
        "krum_zero": float(zero),
        "krum_honest_median": float(np.median(hon)),
        "krum_honest_mean": float(hon.mean()),
        "krum_honest_min": float(hon.min()),
        "margin_median": float(np.median(hon) - zero),
        "margin_min": float(hon.min() - zero),
        "selected": bool(zero < hon.min()),
        "tied_best": bool(np.isclose(zero, hon.min()) and not zero < hon.min()),
        "below_mean": bool(zero < hon.mean()),
        **centre_margin(W, np.arange(1, len(W))),
    }


def exact_collision(d: int = 10) -> dict:
    A = np.zeros((9, d))
    A[:7, 0] = 1.0
    A[7:, 0] = -4.0 / 3.0
    mu = A.mean(0)
    mu_n = float(np.linalg.norm(mu))
    sigma2 = float(((A - mu) ** 2).sum(1).mean())

    P = np.eye(9) - np.ones((9, 9)) / 9.0
    B = np.zeros((9, d))
    B[:, 0] = mu_n
    B[:, 1:10] = np.sqrt(sigma2 * 9.0 / 8.0) * P

    out = {}
    for name, V in (("A_two_groups", A), ("B_regular_simplex", B)):
        m, ps = margin(V, seat="insert"), pair_stats(V)
        muV = V.mean(0)
        out[name] = {
            "mu_norm": float(np.linalg.norm(muV)),
            "sigma2": float(((V - muV) ** 2).sum(1).mean()),
            "ratio": ps["ratio"],
            "near_over_pair": ps["near_over_pair"],
            "krum_zero": m["krum_zero"],
            "krum_honest_min": m["krum_honest_min"],
            "krum_honest_median": m["krum_honest_median"],
            "margin_median": m["margin_median"],
            "margin_min": m["margin_min"],
            "selected": m["selected"],
        }
    a, b = out["A_two_groups"], out["B_regular_simplex"]
    out["moments_match"] = bool(abs(a["mu_norm"] - b["mu_norm"]) < 1e-12
                                and abs(a["sigma2"] - b["sigma2"]) < 1e-12)
    out["outcomes_differ"] = bool(a["selected"] != b["selected"])
    return out


def sweep_clustering(d: int = 64, n: int = 9, trials: int = 300, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for frac in np.linspace(0.0, 1.0, 11):
        sel = bel = 0
        ratios, margins, nops = [], [], []
        for _ in range(trials):
            mu = np.zeros(d)
            mu[0] = 1.0
            iso = rng.normal(0.0, 1.0, size=(n, d))
            iso /= np.linalg.norm(iso, axis=1, keepdims=True).mean()
            g = np.where(rng.random(n) < 7.0 / 9.0, 1.0, -3.5)
            g = g - g.mean()
            split = np.zeros((n, d))
            split[:, 0] = g
            split /= max(np.linalg.norm(split, axis=1).mean(), 1e-12)
            eps = (np.sqrt(1 - frac) * iso + np.sqrt(frac) * split) * 2.0
            V = mu + eps
            ps = pair_stats(V)
            m = margin(V, seat="insert")
            ratios.append(ps["ratio"])
            nops.append(ps["near_over_pair"])
            margins.append(m["margin_median"])
            sel += int(m["selected"])
            bel += int(m["below_mean"])
        out.append({
            "cluster_fraction": float(frac),
            "ratio_median": float(np.median(ratios)),
            "near_over_pair_median": float(np.median(nops)),
            "margin_median": float(np.median(margins)),
            "selected_frac": sel / trials,
            "below_mean_frac": bel / trials,
        })
    return out


SCOPE_START = 12


def measure(pattern: str, clients: int = 10, directory: str | None = None) -> dict | None:
    paths = sorted(glob.glob(os.path.join(directory or ARCHIVES, pattern)))
    if not paths:
        return None
    rows = []
    for path in paths:
        z = np.load(path, allow_pickle=True)
        C, cbar = z["C"].item(), z["cbar"].item()
        for r in sorted(cbar):
            if r < SCOPE_START or (r - 1) not in cbar:
                continue
            V = np.stack([(C[i][r] - cbar[r - 1]).ravel() for i in range(clients)])
            ps = pair_stats(V)
            m = margin(V)
            attacked = V.copy()
            attacked[0] = 0.0
            m["centres_by_rule"] = {
                rule: centre_margin(attacked, np.arange(1, clients), rule)
                for rule in ("median", "trimmed_mean", "bulyan")
                if rule != "bulyan" or clients >= 7
            }
            by_seat = []
            for seat in range(clients):
                W = V.copy()
                W[seat] = 0.0
                s = krum_scores({i: W[i] for i in range(clients)}, n_byzantine=F_BYZANTINE)
                arr = np.array([s[i] for i in range(clients)])
                by_seat.append(int(np.argmin(arr)) == seat)
            m["argmin_by_seat"] = by_seat
            rows.append({**ps, **m})
    if not rows:
        return None

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    return {
        "archives": len(paths), "rounds": len(rows),
        "source_archives": [os.path.basename(p) for p in paths],
        "centres_by_rule": {
            rule: {
                key: float(np.mean([r["centres_by_rule"][rule][key] for r in rows]))
                if key in ("centre_inverts", "centre_inverts_all") else
                float(np.median([r["centres_by_rule"][rule][key] for r in rows]))
                for key in rows[0]["centres_by_rule"][rule]
            } for rule in rows[0]["centres_by_rule"]
        },
        "ratio_median": float(np.median(col("ratio"))),
        "pair_cv_median": float(np.median(col("pair_cv"))),
        "near_over_pair_median": float(np.median(col("near_over_pair"))),
        "near_over_pair_min": float(col("near_over_pair").min()),
        "margin_median": float(np.median(col("margin_median"))),
        "margin_positive_frac": float((col("margin_median") > 0).mean()),
        "margin_min_median": float(np.median(col("margin_min"))),
        "margin_min_positive_frac": float((col("margin_min") > 0).mean()),
        "selected_frac": float(col("selected").mean()),
        "tied_best_frac": float(col("tied_best").mean()),
        "argmin_selected_by_seat": [float(np.mean([r["argmin_by_seat"][k] for r in rows]))
                                    for k in range(len(rows[0]["argmin_by_seat"]))],
        "argmin_selected_all_seats": float(np.mean([np.mean(r["argmin_by_seat"])
                                                    for r in rows])),
        "krum_k": int(len(rows) and (clients - F_BYZANTINE - 2)),
        "below_mean_frac": float(col("below_mean").mean()),
        "centre_norm_median": float(np.median(col("centre_norm"))),
        "honest_to_centre_median": float(np.median(col("honest_to_centre_median"))),
        "honest_to_centre_min_median": float(np.median(col("honest_to_centre_min"))),
        "centre_ratio_median": float(np.median(col("centre_ratio"))),
        "centre_inverts_frac": float(col("centre_inverts").mean()),
        "centre_inverts_all_frac": float(col("centre_inverts_all").mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "krum_margin.json"))
    ap.add_argument("--configs", default="",
                    help="comma-separated configurations to measure in section 3; default: "
                         "the released set. Revision configurations (%s) are read from "
                         "--extra-archives; write them to a separate --out"
                         % ", ".join(EXTRA_CONFIGS))
    ap.add_argument("--extra-archives", default=EXTRA_ARCHIVES)
    args = ap.parse_args()
    if (set(c.strip() for c in args.configs.split(",")) & set(EXTRA_CONFIGS)
            and os.path.abspath(args.out) == os.path.abspath(ap.get_default("out"))):
        raise SystemExit("revision configurations go to a separate --out, not %s"
                         % ap.get_default("out"))

    print("1. The nine-client counterexample.")
    vals = np.array([[1.0]] * 7 + [[-4.0 / 3.0]] * 2)
    ce_stats, ce_margin = pair_stats(vals), margin(vals, seat="insert")
    ps, m = ce_stats, ce_margin
    print("   sigma/|mu| = %.3f, so the closed-form condition ||mu|| < sigma predicts inversion." % ps["ratio"])
    print("   Krum: zero scores %.3f, honest mean %.3f, honest median %.3f."
          % (m["krum_zero"], m["krum_honest_mean"], m["krum_honest_median"]))
    print("   selected: %s.  The closed-form prediction fails." % m["selected"])
    ce_nm = neighbour_means(vals, k=10 - F_BYZANTINE - 2)
    print("   Reason: at the k=%d that Krum uses on this cohort, each honest client's"
          % ce_nm["k"])
    print("   nearest squared distances average %.3f with the zero update available as a"
          % ce_nm["near_mean_with_zero"])
    print("   neighbour and %.3f without it, against the 2*sigma^2 = %.3f the derivation"
          % (ce_nm["near_mean_honest_only"], ps["two_sigma2"]))
    print("   assumed. (rho below is a property of the honest cohort alone and so uses")
    print("   k=%d; its near_mean is %.3f and is not the number to read here.)"
          % (max(1, 9 - F_BYZANTINE - 2), ps["near_mean"]))
    cm0 = centre_margin(vals)
    print("   Two further details of this cohort: the median centre is")
    print("   %.3f and %d of 9 honest clients sit at distance 0 from it, not all nine; and"
          % (cm0["centre_norm"], int(np.sum(np.abs(vals.ravel() - 1.0) < 1e-12))))
    tc = np.sort(np.vstack([vals, [[0.0]]]).ravel())[F_BYZANTINE:-F_BYZANTINE].mean()
    print("   the trimmed mean's centre is %.4f, so the three centre rules do not share a"
          % tc)
    print("   single c here and the identity has to be evaluated per rule.")
    print()

    print("1b. An EXACT collision: same ||mu||, same sigma^2, opposite outcome.")
    ec = exact_collision()
    print("   %-20s %9s %9s %9s %10s %9s"
          % ("cohort", "||mu||", "sigma^2", "s(0)", "min honest", "selected"))
    for nm in ("A_two_groups", "B_regular_simplex"):
        e = ec[nm]
        print("   %-20s %9.6f %9.6f %9.5f %10.5f %9s"
              % (nm, e["mu_norm"], e["sigma2"], e["krum_zero"],
                 e["krum_honest_min"], e["selected"]))
    print("   moments identical: %s ; outcomes differ: %s"
          % (ec["moments_match"], ec["outcomes_differ"]))
    print("   No function of mu and sigma can separate these two, because there is nothing")
    print("   in mu and sigma to separate. A threshold cannot help.")
    print()

    print("2. Clustering sweep: the dispersion ratio also changes.")
    sw = sweep_clustering()
    print("   %-10s %8s %12s %10s %10s"
          % ("clustered", "ratio", "near/pair", "margin", "selected"))
    for row in sw:
        print("   %-10.1f %8.2f %12.3f %10.2f %10.3f"
              % (row["cluster_fraction"], row["ratio_median"],
                 row["near_over_pair_median"], row["margin_median"],
                 row["selected_frac"]))
    print()
    print("   This sweep alone is not a refutation. The margin falls monotonically as the")
    print("   ratio rises, so a decreasing function of the ratio fits the table and the")
    print("   sweep alone exhibits a ratio-only rule rather than refuting one. The")
    print("   refutation needs two cohorts at the same ratio with opposite outcomes,")
    print("   which is what the exact construction above supplies.")
    print()

    print("3. Measurements on the trained federations.")
    configs = {
        "cifar100_d05": ("arch_cifar100_s*_c10_r35*.npz", 10),
        "cifar100_d01": ("arch_cifar100_d01*_s*.npz", 10),
        "cifar10": ("arch_cifar10_s*.npz", 10),
        "late70": ("arch_late70_s*_c10_r70*.npz", 10),
        "cohort05": ("arch_cohort05_s*_c5_r35*.npz", 5),
        "cohort20": ("arch_cohort20_s*_c20_r35*.npz", 20),
        "cohort40": ("arch_cohort40_s*_c40_r35*.npz", 40),
        "nopassport": ("arch_nopp_s*_c10_r35*.npz", 10),
    }
    selected = [c.strip() for c in args.configs.split(",") if c.strip()] or list(configs)
    unknown = [c for c in selected if c not in configs and c not in EXTRA_CONFIGS]
    if unknown:
        raise SystemExit("unknown configuration(s): %s" % unknown)
    measured = {}
    print("   Three separate columns, because a positive median margin is not selection.")
    print("   %-14s %7s %11s %10s %10s %10s"
          % ("config", "ratio", "near/pair", "med marg>0", "min marg>0", "selected"))
    for name in selected:
        if name in configs:
            (pat, nclients), directory = configs[name], ARCHIVES
        else:
            (pat, nclients), directory = EXTRA_CONFIGS[name], args.extra_archives
        mm = measure(pat, clients=nclients, directory=directory)
        if not mm:
            print("   %-14s  (no archive yet)" % name)
            continue
        measured[name] = mm
        print("   %-14s %7.2f %11.3f %10.3f %10.3f %10.3f"
              % (name, mm["ratio_median"], mm["near_over_pair_median"],
                 mm["margin_positive_frac"], mm["margin_min_positive_frac"],
                 mm["selected_frac"]))
    print()
    print("4. Collisions. The exact one is in section 1b; these are the measured cohorts")
    print("   against the sweep points nearest them in ratio, which are close but NOT equal")
    print("   and therefore illustrate rather than prove.")
    print("   %-16s %8s %10s %8s" % ("cohort", "ratio", "selected", "rho"))
    for name in measured:
        m = measured[name]
        near = min(sw, key=lambda r: abs(r["ratio_median"] - m["ratio_median"]))
        print("   %-16s %8.3f %10.3f %8.3f   measured"
              % (name, m["ratio_median"], m["selected_frac"], m["near_over_pair_median"]))
        print("   %-16s %8.3f %10.3f %8.3f   nearest sweep point"
              % ("", near["ratio_median"], near["selected_frac"],
                 near["near_over_pair_median"]))
    print("   rho separates them; the ratio does not.")

    print()
    print("5. The distance-from-centre rules, where the criterion is exact.")
    print("   A zero update scores ||c||, an honest client ||I_j - c||, so the zero update")
    print("   beats the MEDIAN honest client exactly when ||c|| < med ||I_j - c||, and beats")
    print("   EVERY honest client exactly when ||c|| < min_j ||I_j - c||. AUC 0.000 is the")
    print("   second condition, so both are reported.")
    print("   %-16s %10s %10s %10s %11s %11s"
          % ("cohort", "||c||", "med dist", "min dist", "beats med", "beats all"))
    for name in measured:
        m = measured[name]
        print("   %-16s %10.2f %10.2f %10.2f %11.3f %11.3f"
              % (name, m["centre_norm_median"], m["honest_to_centre_median"],
                 m["honest_to_centre_min_median"], m["centre_inverts_frac"],
                 m["centre_inverts_all_frac"]))
    cm = centre_margin(vals)
    print("   %-16s %10.2f %10.2f %10.2f %11s %11s"
          % ("counterexample", cm["centre_norm"], cm["honest_to_centre_median"],
             cm["honest_to_centre_min"], str(cm["centre_inverts"]),
             str(cm["centre_inverts_all"])))

    payload = {"f_byzantine": F_BYZANTINE,
               "counterexample": {**ce_stats, **ce_margin},
               "counterexample_neighbour_means": ce_nm,
               "counterexample_trimmed_centre": float(tc),
               "counterexample_honest_at_centre": int(
                   np.sum(np.abs(vals.ravel() - 1.0) < 1e-12)),
               "exact_collision": ec,
               "cluster_sweep": sw, "measured": measured}
    expect = {"counterexample": dict, "counterexample_neighbour_means": dict,
              "counterexample_trimmed_centre": float, "counterexample_honest_at_centre": int,
              "exact_collision": dict, "cluster_sweep": list, "measured": dict}
    for key, typ in expect.items():
        got = payload.get(key)
        if not isinstance(got, typ):
            raise SystemExit("payload[%r] is %s, expected %s; the payload was assembled "
                             "incorrectly" % (key, type(got).__name__, typ.__name__))

    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
