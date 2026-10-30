from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVES = os.path.join(HERE, "results", "archives")


def frobenius_norm(a):
    return float(np.sqrt((np.asarray(a, dtype=float) ** 2).sum()))


def probe_absent_rows(pattern, lag=3):
    paths = sorted(glob.glob(os.path.join(ARCHIVES, pattern)))
    if not paths:
        return None
    innov_absent, innov_present = [], []
    rep_absent, rep_present, rep_frac = [], [], []
    supports = []
    for path in paths:
        z = np.load(path, allow_pickle=True)
        C, cbar = z["C"].item(), z["cbar"].item()
        P = z["present"].item() if "present" in z else None
        if P is None:
            continue
        n = len(C)
        for i in range(n):
            rounds = sorted(C[i])
            for r in rounds:
                if (r - 1) not in cbar or (r - lag) not in C[i]:
                    continue
                m = np.asarray(P[i][r], dtype=bool).ravel()
                if m.all() or not m.any():
                    supports.append(int(m.sum()))
                    continue
                supports.append(int(m.sum()))

                I = C[i][r] - cbar[r - 1]
                innov_absent.append(frobenius_norm(I[~m]))
                innov_present.append(frobenius_norm(I[m]))

                D = C[i][r] - C[i][r - lag]
                ra, rp = frobenius_norm(D[~m]), frobenius_norm(D[m])
                rep_absent.append(ra)
                rep_present.append(rp)
                tot = np.hypot(ra, rp)
                if tot > 0:
                    rep_frac.append(ra * ra / (tot * tot))
    if not rep_absent:
        return None
    return {
        "client_rounds": len(rep_absent),
        "support_mean": float(np.mean(supports)),
        "innovation_absent_norm_mean": float(np.mean(innov_absent)),
        "innovation_absent_norm_max": float(np.max(innov_absent)),
        "innovation_present_norm_mean": float(np.mean(innov_present)),
        "rep_absent_norm_mean": float(np.mean(rep_absent)),
        "rep_present_norm_mean": float(np.mean(rep_present)),
        "rep_absent_energy_fraction_mean": float(np.mean(rep_frac)),
        "rep_absent_energy_fraction_p90": float(np.percentile(rep_frac, 90)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "absent_row_probe.json"))
    ap.add_argument("--lag", type=int, default=3)
    args = ap.parse_args()

    configs = {
        "cifar100_d05": "arch_cifar100_s*_c10_r35*.npz",
        "cifar100_d01": "arch_cifar100_d01*_s*.npz",
        "cifar10": "arch_cifar10_s*.npz",
        "maskshard10": "arch_maskshard10_*.npz",
        "maskmix": "arch_maskmix_*.npz",
    }
    out = {"lag": args.lag, "configs": {}}
    print("Norm contributed by rows the client does not hold, against rows it does.")
    print("%-14s %8s %12s %12s %12s %12s %10s"
          % ("config", "support", "innov abs", "innov pres", "rep abs", "rep pres", "rep frac"))
    print("-" * 88)
    for name, pat in configs.items():
        p = probe_absent_rows(pat, args.lag)
        if not p:
            print("%-14s  (no archive with masks)" % name)
            continue
        out["configs"][name] = p
        print("%-14s %8.1f %12.2e %12.3f %12.3f %12.3f %10.3f"
              % (name, p["support_mean"], p["innovation_absent_norm_max"],
                 p["innovation_present_norm_mean"], p["rep_absent_norm_mean"],
                 p["rep_present_norm_mean"], p["rep_absent_energy_fraction_mean"]))

    print()
    print("innov abs is the LARGEST absent-row innovation norm seen anywhere, not a mean:")
    print("for the innovation statistics the carried rows cancel exactly and the column is")
    print("at the numerical floor. For rep the absent rows do carry aggregate drift, and")
    print("the last column is the share of the replay distance they account for.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
