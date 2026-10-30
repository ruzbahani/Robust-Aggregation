from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")

TEST_SEEDS = ["3", "4"]
ATTACKS = ["global_echo", "peer_copy", "lagged_self", "ar2_replay", "noise", "shrink"]

ROWS = [
    ("suspicion", "Payload auditor"),
    ("ENS_STATEFUL", "Stateful ensemble, 24 families"),
    ("BEST_STATEFUL", "Best single stateful"),
    ("tb_ar1_burnin_zabs", "AR(1) residual"),
    ("ENS_STATELESS", "Ensemble of the six memoryless rules"),
]


def load_results(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


def auc_test(d, column, attack):
    blk = d["auc"].get(attack, {}).get(column)
    if not blk:
        return None
    per = blk.get("per_seed", {})
    vals = [per[s] for s in TEST_SEEDS if s in per and np.isfinite(per[s])]
    return float(np.mean(vals)) if vals else None


def auc_test_excluding_copy(d, column, attack):
    blk = d.get("auc_excl_copy_source", {}).get(attack, {}).get(column)
    if not blk:
        return None
    per = blk.get("per_seed", {})
    vals = [per[s] for s in TEST_SEEDS if s in per and np.isfinite(per[s])]
    return float(np.mean(vals)) if vals else None


def resolve_archive_path(d, column):
    if column == "BEST_STATEFUL":
        return d["verdict"]["best_stateful_baseline"]["column"]
    if column == "ENS_STATEFUL":
        return "ens_stateful"
    if column == "ENS_STATELESS":
        return "ens_stateless"
    return column


def wilson_interval(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="0.01")
    ap.add_argument("--out", default=os.path.join(RESULTS, "operating_table.json"))
    args = ap.parse_args()

    d = load_results("temporal_eval.json")
    test = d["results"]["TEST"]
    pools = d["protocol"]["pools"]
    assert [str(s) for s in pools["TEST"]] == TEST_SEEDS, pools

    print("Every column below is computed on TEST seeds %s only." % ", ".join(TEST_SEEDS))
    print("Thresholds are fitted on CAL seeds %s and never refit.\n"
          % ", ".join(str(s) for s in pools["CAL"]))

    out = {"test_seeds": TEST_SEEDS, "nominal_fpr": float(args.target), "rows": []}

    hdr = ("%-38s %9s %9s %9s %9s %9s" %
           ("detector", "clean", "uninvol", "worst", "worst", "worst"))
    hdr2 = ("%-38s %9s %9s %9s %9s %9s" %
            ("", "FPR", "FPR", "TPR", "AUC", "AUC*"))
    print(hdr)
    print(hdr2)
    print("-" * 88)

    for column, label in ROWS:
        col = resolve_archive_path(d, column)
        blk = test.get(col, {}).get(args.target)
        if not blk:
            print("%-38s  (absent: %s)" % (label, col))
            continue

        hk = blk["honest"]["client_round"]["k"]
        hn = blk["honest"]["client_round"]["n"]
        clean = hk / hn

        uk = un = 0
        tprs, aucs, aucs_x = [], [], []
        per_attack = {}
        for atk in ATTACKS:
            a = blk["attacks"].get(atk)
            if not a:
                continue
            fp = a.get("fpr_uninvolved_client_round") or a.get("fpr_bystander_client_round")
            if fp:
                uk += fp["k"]
                un += fp["n"]
            t = a["tpr_client_round"]["point"]
            tprs.append(t)
            g = auc_test(d, col, atk)
            gx = auc_test_excluding_copy(d, col, atk)
            if g is not None:
                aucs.append(g)
            if gx is not None:
                aucs_x.append(gx)
            per_attack[atk] = {
                "tpr": t,
                "auc_test": g,
                "auc_test_excl_copy_source": gx,
                "fpr_uninvolved": (fp["k"] / fp["n"]) if fp else None,
                "flag_rate_copy_source": (a["flag_rate_copy_source"]["point"]
                                          if a.get("flag_rate_copy_source") else None),
            }

        worst_tpr = min(tprs) if tprs else None
        worst_auc = min(aucs) if aucs else None
        worst_auc_x = min(aucs_x) if aucs_x else None
        uninv = uk / un if un else None

        print("%-38s %8.2f%% %8.2f%% %8.1f%% %9.3f %9.3f"
              % (label, 100 * clean, 100 * (uninv or 0), 100 * (worst_tpr or 0),
                 worst_auc if worst_auc is not None else float("nan"),
                 worst_auc_x if worst_auc_x is not None else float("nan")))

        lo, hi = wilson_interval(hk, hn)
        out["rows"].append({
            "label": label, "column": col,
            "threshold": blk["threshold"],
            "clean_fpr": clean, "clean_k": hk, "clean_n": hn,
            "clean_ci": [lo, hi],
            "uninvolved_fpr": uninv, "uninvolved_k": uk, "uninvolved_n": un,
            "worst_tpr": worst_tpr, "worst_auc_test": worst_auc,
            "worst_auc_test_excl_copy_source": worst_auc_x,
            "per_attack": per_attack,
        })

    print()
    print("clean FPR   : honest client-rounds, no attacker in the run")
    print("uninvol FPR : honest client-rounds in attacked runs, copy source excluded")
    print("AUC         : worst over the six attacks, TEST seeds only")
    print("AUC*        : the same with the copy source removed from the honest population")

    print()
    print("Consistency of the top row, all quantities on the same TEST population:")
    top = out["rows"][0]
    bound = top["worst_tpr"] * (1 - top["uninvolved_fpr"])
    print("  TPR (1 - uninvolved FPR) = %.4f  is a lower bound on a same-population AUC"
          % bound)
    print("  worst AUC over attacks   = %.4f" % top["worst_auc_test"])
    print("  worst AUC, copy source excluded = %.4f" % top["worst_auc_test_excl_copy_source"])
    ok = top["worst_auc_test_excl_copy_source"] >= bound - 1e-6
    print("  bound satisfied once the copied client is out of the honest population: %s" % ok)
    print()
    pc = top["per_attack"]["peer_copy"]
    print("  peer copy is the whole of the gap. The copied client is flagged at %.1f%%,"
          % (100 * pc["flag_rate_copy_source"]))
    print("  because its innovation and the copier's are the same matrix. Counting it as a")
    print("  ranking error gives AUC %.3f; the rest of the honest cohort gives %.4f."
          % (pc["auc_test"], pc["auc_test_excl_copy_source"]))

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
