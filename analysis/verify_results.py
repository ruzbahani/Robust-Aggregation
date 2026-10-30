from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "..", "results")
MISSING = []


def load_json(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        MISSING.append(name)
        return None
    return json.load(open(p))


def load_innovation_results(pattern, nseeds, attack, key="suspicion"):
    vals = []
    for s in range(nseeds):
        d = load_json(pattern % s)
        if d and attack in d["results"] and key in d["results"][attack]:
            v = d["results"][attack][key]["auc"]
            if np.isfinite(v):
                vals.append(v)
    return float(np.mean(vals)) if vals else None


C100 = ("innovation_cifar100_s%d.json", 5)
C100D01 = ("innovation_cifar100_d01_s%d.json", 3)
C10 = ("innovation_cifar10_s%d.json", 3)

CLAIMS = []


def check_claim(value, source, got, tol=5e-4):
    CLAIMS.append((value, source, got, tol))


for rule, vals in [
    ("krum", {"global echo": 0.011, "peer copy": 0.289, "lagged self": 0.498,
              "AR(2)": 0.977, "noise": 0.587, "shrink": 0.012}),
    ("median_dev", {"global echo": 0.000, "peer copy": 0.244, "lagged self": 0.504,
                    "AR(2)": 0.998, "noise": 0.621, "shrink": 0.000}),
    ("trimmed_mean_dev", {"global echo": 0.000, "peer copy": 0.220, "lagged self": 0.499,
                          "AR(2)": 0.998, "noise": 0.633, "shrink": 0.000}),
    ("norm_bound", {"global echo": 1.000, "peer copy": 0.427, "lagged self": 0.583,
                    "AR(2)": 0.993, "noise": 0.504, "shrink": 1.000}),
]:
    for label, stated in vals.items():
        atk = {"global echo": "global_echo", "peer copy": "peer_copy",
               "lagged self": "lagged_self", "AR(2)": "ar2_replay",
               "noise": "noise", "shrink": "shrink"}[label]
        check_claim(stated, f"ranking table {rule}/{atk}", load_innovation_results(*C100, atk, rule), 1e-3)


for rule, d01, c10 in [("krum", 0.000, 0.001), ("median_dev", 0.000, 0.000),
                       ("trimmed_mean_dev", 0.000, 0.000), ("norm_bound", 1.000, 1.000)]:
    check_claim(d01, f"ranking table {rule} echo d=0.1", load_innovation_results(*C100D01, "global_echo", rule), 1e-3)
    check_claim(c10, f"ranking table {rule} echo cifar10", load_innovation_results(*C10, "global_echo", rule), 1e-3)

for cfg, keys in [("cifar100_d05", {"global_echo": (0.965, 0.500), "peer_copy": (0.500, 0.886),
                                    "lagged_self": (0.134, 0.500), "ar2_replay": (0.649, 0.500),
                                    "noise": (0.914, 0.500), "shrink": (0.500, 0.500)}),
                  ("cifar100_d01", {"global_echo": (0.999, 0.500)}),
                  ("cifar10", {"global_echo": (0.762, 0.544)})]:
    fb = load_json("faithful_baselines_%s.json" % cfg)
    if not fb:
        continue
    for atk, (ft, fg) in keys.items():
        s = fb["summary"].get(atk, {})
        check_claim(ft, "ranking table FLTrust/%s %s" % (atk, cfg), s.get("fltrust_mean"), 1e-3)
        check_claim(fg, "ranking table FoolsGold/%s %s" % (atk, cfg), s.get("foolsgold_mean"), 1e-3)

if km := load_json("krum_margin.json"):
    ce = km["counterexample"]
    check_claim(2.01, "counterexample dispersion ratio", ce["ratio"], 5e-3)
    check_claim(6.00, "counterexample, Krum score of zero", ce["krum_zero"], 5e-3)
    check_claim(5.23, "counterexample, honest mean", ce["krum_honest_mean"], 5e-3)
    check_claim(0.0, "counterexample, zero is selected", float(ce["selected"]), 1e-9)
    ce_nm = km.get("counterexample_neighbour_means", {})
    check_claim(6, "counterexample, neighbour count", ce_nm.get("k"), 1e-9)
    check_claim(0.872, "counterexample, nearest mean with zero available",
          ce_nm.get("near_mean_with_zero"), 5e-4)
    check_claim(1.008, "counterexample, nearest mean among honest only",
          ce_nm.get("near_mean_honest_only"), 5e-4)
    check_claim(1.88, "counterexample, two sigma squared", ce["two_sigma2"], 5e-3)
    check_claim(0.8333, "counterexample, trimmed-mean centre",
          km.get("counterexample_trimmed_centre"), 5e-4)
    check_claim(7, "counterexample, honest clients at the median centre",
          km.get("counterexample_honest_at_centre"), 1e-9)
    check_claim(0.457, "counterexample, rho", ce["near_over_pair"], 5e-4)
    for cfg, rho, pos, sel in [("cifar100_d05", 0.970, 1.000, 1.000),
                               ("cifar100_d01", 0.890, 1.000, 1.000),
                               ("cifar10", 0.909, 1.000, 1.000)]:
        m = km["measured"].get(cfg, {})
        check_claim(rho, "rho median, %s" % cfg, m.get("near_over_pair_median"), 5e-4)
        check_claim(pos, "margin positive, %s" % cfg, m.get("margin_positive_frac"), 5e-4)
        check_claim(sel, "free-rider selected, %s" % cfg, m.get("selected_frac"), 5e-4)
        check_claim(1.000, "min margin positive, %s" % cfg,
              m.get("margin_min_positive_frac"), 5e-4)
        check_claim(1.000, "centre beats every honest client, %s" % cfg,
              m.get("centre_inverts_all_frac"), 5e-4)

    ex = km.get("exact_collision", {})
    if ex:
        check_claim(1.0, "exact collision, moments match", float(ex.get("moments_match")), 1e-9)
        check_claim(1.0, "exact collision, outcomes differ", float(ex.get("outcomes_differ")), 1e-9)
        for side, mu, s2, s0, hmin, sel_ in [
                ("A_two_groups", 13.0 / 27.0, 686.0 / 729.0, 6.0, 0.0, 0.0),
                ("B_regular_simplex", 13.0 / 27.0, 686.0 / 729.0, 190.0 / 27.0,
                 635.0 / 54.0, 1.0)]:
            e = ex[side]
            check_claim(mu, "collision %s, ||mu||" % side, e["mu_norm"], 1e-9)
            check_claim(s2, "collision %s, sigma^2" % side, e["sigma2"], 1e-9)
            check_claim(s0, "collision %s, Krum score of zero" % side, e["krum_zero"], 1e-6)
            check_claim(hmin, "collision %s, best honest score" % side, e["krum_honest_min"], 1e-6)
            check_claim(sel_, "collision %s, zero selected" % side, float(e["selected"]), 1e-9)
    sweep = km["cluster_sweep"]
    check_claim(0.917, "cluster sweep, rho at zero clustering",
          sweep[0]["near_over_pair_median"], 5e-4)
    check_claim(0.457, "cluster sweep, rho at full clustering",
          sweep[-1]["near_over_pair_median"], 5e-4)
    check_claim(13.99, "cluster sweep, margin at zero clustering",
          sweep[0]["margin_median"], 5e-3)
    check_claim(-27.09, "cluster sweep, margin at full clustering",
          sweep[-1]["margin_median"], 5e-3)
    check_claim(1.000, "cluster sweep, selected at zero clustering",
          sweep[0]["selected_frac"], 5e-4)
    check_claim(0.123, "cluster sweep, selected at full clustering",
          sweep[-1]["selected_frac"], 5e-4)
    check_claim(1.57, "cluster sweep, ratio at zero clustering", sweep[0]["ratio_median"], 5e-3)
    check_claim(2.41, "cluster sweep, ratio at full clustering", sweep[-1]["ratio_median"], 5e-3)

    for cfg, cn, dm, inv in [("cifar100_d05", 12.87, 39.28, 1.000),
                             ("cifar100_d01", 5.35, 28.42, 1.000),
                             ("cifar10", 1.77, 6.60, 1.000)]:
        m = km["measured"][cfg]
        check_claim(cn, "centre norm, %s" % cfg, m["centre_norm_median"], 5e-3)
        check_claim(dm, "honest distance to centre, %s" % cfg, m["honest_to_centre_median"], 5e-3)
        check_claim(inv, "centre criterion holds, %s" % cfg, m["centre_inverts_frac"], 5e-4)
    check_claim(36.97, "smallest honest distance to centre, cifar100_d05",
          km["measured"]["cifar100_d05"]["honest_to_centre_min_median"], 5e-3)
    check_claim(0.337, "centre ratio, cifar100_d05",
          km["measured"]["cifar100_d05"]["centre_ratio_median"], 5e-4)
    check_claim(1.00, "counterexample centre norm", ce["centre_norm"], 5e-3)
    check_claim(0.00, "counterexample honest distance to centre",
          ce["honest_to_centre_median"], 5e-3)
    check_claim(0.0, "counterexample centre criterion holds", float(ce["centre_inverts"]), 1e-9)

    def nearest(ratio):
        return min(sweep, key=lambda r: abs(r["ratio_median"] - ratio))
    for cfg, mr, ms, sr, ss in [("cifar100_d01", 2.014, 1.000, 1.974, 0.217),
                                ("cifar10", 2.208, 1.000, 2.232, 0.117),
                                ("cifar100_d05", 2.357, 1.000, 2.405, 0.123)]:
        m = km["measured"][cfg]
        n = nearest(m["ratio_median"])
        check_claim(mr, "collision, measured ratio %s" % cfg, m["ratio_median"], 5e-3)
        check_claim(ms, "collision, measured selected %s" % cfg, m["selected_frac"], 5e-4)
        check_claim(sr, "collision, sweep ratio %s" % cfg, n["ratio_median"], 5e-3)
        check_claim(ss, "collision, sweep selected %s" % cfg, n["selected_frac"], 5e-4)

if te := load_json("temporal_eval.json"):
    pc = te["results"]["TEST"]["suspicion"]["0.01"]["attacks"]["peer_copy"]
    check_claim(0.1264, "peer copy, all-honest FPR", pc["fpr_bystander_client_round"]["point"], 5e-5)
    check_claim(0.0172, "peer copy, uninvolved FPR", pc["fpr_uninvolved_client_round"]["point"], 5e-5)
    check_claim(1.000, "peer copy, copied client accused", pc["flag_rate_copy_source"]["point"], 5e-4)
    check_claim(0.468, "peer copy, share of flags on attacker",
          pc["attribution_precision"]["point"], 5e-4)

if ho := load_json("held_out_summary.json"):
    check_claim(0.0133, "held-out pooled FPR", ho["honest"]["fpr"], 5e-5)
    check_claim(16.0, "held-out false alarms", float(ho["honest"]["flags"]), 1e-9)
    check_claim(1200.0, "held-out honest client-rounds", float(ho["honest"]["rounds"]), 1e-9)
    check_claim(0.0083, "held-out FPR, lowest seed", ho["honest"]["spread"][0], 5e-5)
    check_claim(0.0250, "held-out FPR, highest seed", ho["honest"]["spread"][1], 5e-5)
    check_claim(1.000, "held-out worst-case detection", ho["worst_tpr"], 1e-9)
    check_claim(1200.0, "held-out attacked client-rounds per attack",
          float(ho["tpr"]["global_echo"]["n"]), 1e-9)

if fs := load_json("fresh_seed_transfer.json"):
    check_claim(0.0250, "fresh seed 5 FPR", fs["honest"]["per_seed"]["5"]["fpr"], 5e-5)
    check_claim(0.0125, "fresh seed 6 FPR", fs["honest"]["per_seed"]["6"]["fpr"], 5e-5)
    check_claim(0.0083, "fresh seed 7 FPR", fs["honest"]["per_seed"]["7"]["fpr"], 5e-5)
    check_claim(1.000, "fresh seeds, worst-case detection", fs["worst_tpr"], 1e-9)
    assert float(fs["tpr"]["global_echo"]["n"]) == 720.0, "fresh-seed round count moved"

ar = load_json("absent_row_probe.json")
if ar:
    for cfg, frac in [("cifar100_d05", 0.021), ("cifar100_d01", 0.288),
                      ("cifar10", 0.021), ("maskshard10", 0.910)]:
        check_claim(frac, "absent-row share of rep / %s" % cfg,
              ar["configs"].get(cfg, {}).get("rep_absent_energy_fraction_mean"), 5e-4)
    mx = max(v["innovation_absent_norm_max"] for v in ar["configs"].values())
    check_claim(5.3e-4, "largest absent-row innovation norm", mx, 5e-5)

b = load_json("binding_eval.json")
if b:
    check_claim(0.5419, "passport, correct", b["static"]["correct"])
    check_claim(0.0100, "passport, random", b["static"]["random_theta"])
    check_claim(0.0102, "passport, zero", b["static"]["zero_theta"])
    check_claim(0.0100, "passport, foreign", b["static"]["foreign_theta_B"])
    check_claim(0.4451, "version, random E", b["version"]["random_E"])
    check_claim(0.4315, "version, stale E", b["version"]["E_first"])
    check_claim(0.5120, "version, term removed", b["version"]["no_E"])
    check_claim(0.5122, "version, head trained on", b["version"]["E_trained_on"])
    check_claim(0.4338, "version, head minus two", b["version"]["E_minus_2"])
    check_claim(0.4566, "version, head minus five", b["version"]["E_minus_5"])
    check_claim(0.3113, "ambiguity, 2000 steps", b["ambiguity_attack"]["curve"][-1][1])

n1 = load_json("n1_with_rescale.json")
if n1:
    rd, N = n1["rank_deficiency"], n1["N1"]
    check_claim(401408, "passport entries", rd["passport_entries_total"], 1e-9)
    check_claim(7168, "nominal constraints", rd["nominal_constraints_total"], 1e-9)
    check_claim(7168, "numeric rank", rd["numeric_rank_total"], 1e-9)
    check_claim(394240, "nullspace dimension", rd["nullspace_dim_total"], 1e-9)
    check_claim(0.9821, "nullspace share of the passport",
          rd["nullspace_dim_total"] / rd["passport_entries_total"], 5e-5)
    r2 = [r for r in N["rescale_forgeries"] if r["rescale"] == 2.0][0]
    check_claim(1.0, "rescale, cosine to the original", r2["cos_theta_prime_theta"], 1e-9)
    check_claim(2.384e-7, "rescale, max absolute gamma change", r2["max_abs_dgamma_float32"], 1e-9)
    check_claim(2.065e-7, "rescale, max relative gamma change", r2["max_rel_dgamma_float32"], 1e-9)
    check_claim(5.722e-6, "rescale, max logit deviation", r2["max_abs_dlogit"], 1e-9)
    check_claim(0.2604, "rescale, classifier accuracy", r2["acc_classifier"], 5e-5)
    check_claim(0.0, "rescale, predictions changed", r2["frac_predictions_changed"], 1e-9)
    check_claim(1.0, "rescale, signing key changes",
          float(r2["pk_Ed25519"] != N["legit"]["pk_Ed25519"]), 1e-9)
    check_claim(1.0, "rescale, passport digest changes",
          float(r2["theta_sha256"] != N["legit"]["theta_sha256"]), 1e-9)
    dm = [r for r in N["forgeries"] if r["name"] == "forged_distribution_matched"][0]
    check_claim(0.0256, "nullspace, cosine to the original", dm["cos_theta_prime_theta"], 5e-4)
    check_claim(7.629e-6, "nullspace, max logit deviation", dm["max_abs_dlogit"], 1e-9)
    check_claim(21.80, "logit range on this checkpoint",
          N["float32_noise_floor"]["logit_abs_max"], 5e-3)
    check_claim(1.907e-6, "batch-size noise floor, this CPU host",
          N["float32_noise_floor_hi"], 1e-9)
    sw = {(s["forgery"], s["alpha_dyn"]): s for s in N["identity_swap"]}
    check_claim(0.0, "identity swap at alpha 0, rescale, accuracy change",
          sw[("rescale_2", 0.0)]["d_acc_prototype_pp"], 1e-9)
    check_claim(-7.24, "identity swap at alpha 0.5, rescale, accuracy change",
          sw[("rescale_2", 0.5)]["d_acc_prototype_pp"], 5e-3)
    check_claim(-3.66, "identity swap at alpha 0.5, nullspace, accuracy change",
          sw[("nullspace", 0.5)]["d_acc_prototype_pp"], 5e-3)
    check_claim(0.530, "identity swap at alpha 0.5, rescale, predictions moved",
          sw[("rescale_2", 0.5)]["frac_predictions_changed"], 5e-4)
    check_claim(0.4991, "identity swap at alpha 0.5, nullspace, predictions moved",
          sw[("nullspace", 0.5)]["frac_predictions_changed"], 5e-4)

for name, stated in [("global_echo", 1.000), ("peer_copy", 0.944), ("lagged_self", 0.999),
                     ("ar2_replay", 1.000), ("noise", 1.000), ("shrink", 1.000)]:
    check_claim(stated, f"payload auditor AUC / {name}", load_innovation_results(*C100, name), 1e-3)

fb2 = load_json("faithful_baselines_cifar100_d05.json")
if fb2:
    s = fb2["summary"]
    for atk, uni, imp in [("global_echo", 0.500, 0.500), ("peer_copy", 0.886, 0.902),
                          ("lagged_self", 0.500, 0.500), ("ar2_replay", 0.500, 0.500),
                          ("noise", 0.500, 0.500), ("shrink", 0.500, 0.500)]:
        check_claim(uni, "FoolsGold uniform weights / %s" % atk,
              s[atk].get("foolsgold_mean"), 5e-4)
        check_claim(imp, "FoolsGold indicative features / %s" % atk,
              s[atk].get("foolsgold_imp10_mean"), 5e-4)

sw = load_json("sweep_summary.json")
if sw:
    def row(label):
        for k, v in sw.items():
            if k.startswith(label):
                return v
        return None
    for label, krum, worst in [("late", 0.039, 0.944), ("cohort     5", 0.227, 0.823),
                               ("cohort    20", 0.038, 0.974),
                               ("cohort    40", 0.044, 0.987)]:
        r = row(label)
        check_claim(krum, "sweep Krum AUC / %s" % label.strip(),
              r["inverters"]["krum"]["mean"], 5e-4)
        check_claim(0.000, "sweep median AUC / %s" % label.strip(),
              r["inverters"]["median_dev"]["mean"], 5e-4)
        check_claim(0.000, "sweep trimmed AUC / %s" % label.strip(),
              r["inverters"]["trimmed_mean_dev"]["mean"], 5e-4)
        check_claim(1.000, "sweep norm-bound control / %s" % label.strip(),
              r["inverters"]["norm_bound"]["mean"], 5e-4)
        check_claim(worst, "sweep auditor worst / %s" % label.strip(), r["auditor_worst"], 5e-4)
    for label in ("baseline", "cohort    20", "cohort    40"):
        r = row(label)
        n = r["clients"]
        check_claim((n - 1.5) / (n - 1), "tie ceiling predicts peer copy at N=%d" % n,
              r["auditor"]["peer_copy"]["mean"], 5e-4)

NOPP = ("innovation_nopp_s%d.json", 3)
for rule, stated in (("krum", 0.000), ("median_dev", 0.000), ("trimmed_mean_dev", 0.000),
                     ("norm_bound", 1.000)):
    check_claim(stated, "no-passport federation, %s on global echo" % rule,
          load_innovation_results(*NOPP, "global_echo", rule), 5e-4)
check_claim(0.944, "no-passport federation, auditor worst case",
      min(load_innovation_results(*NOPP, a) for a in ("global_echo", "peer_copy", "lagged_self",
                                     "ar2_replay", "noise", "shrink")), 5e-4)
if km2 := load_json("krum_margin.json"):
    npp = km2["measured"]["nopassport"]
    if npp:
        check_claim(1.000, "no-passport, min margin positive", npp["margin_min_positive_frac"], 5e-4)
        check_claim(1.000, "no-passport, free-rider selected", npp["selected_frac"], 5e-4)
        check_claim(1.000, "no-passport, centre beats every honest", npp["centre_inverts_all_frac"], 5e-4)
        check_claim(0.970, "no-passport, rho", npp["near_over_pair_median"], 5e-4)
        check_claim(2.29, "no-passport, dispersion ratio", npp["ratio_median"], 5e-3)

km2 = load_json("krum_margin.json")
if km2:
    for cfg in ("late70", "cohort20", "cohort40"):
        m = km2["measured"][cfg]
        check_claim(1.000, "min margin positive / %s" % cfg, m["margin_min_positive_frac"], 5e-4)
        check_claim(1.000, "free-rider selected / %s" % cfg, m["selected_frac"], 5e-4)
        check_claim(1.000, "centre beats every honest / %s" % cfg,
              m["centre_inverts_all_frac"], 5e-4)
    c5 = km2["measured"].get("cohort05")
    if c5:
        check_claim(1, "cohort05, Krum neighbourhood k", c5["krum_k"], 1e-9)
        check_claim(0.000, "cohort05, min margin positive", c5["margin_min_positive_frac"], 5e-4)
        check_claim(0.000, "cohort05, free-rider strictly selected", c5["selected_frac"], 5e-4)
        check_claim(1.000, "cohort05, free-rider tied for best", c5["tied_best_frac"], 5e-4)
        for seat, stated in enumerate((1.000, 0.361, 0.194, 0.014, 0.000)):
            check_claim(stated, "cohort05, argmin picks free-rider in seat %d" % seat,
                  c5["argmin_selected_by_seat"][seat], 5e-4)
        check_claim(0.314, "cohort05, argmin picks free-rider over all seats",
              c5["argmin_selected_all_seats"], 5e-4)
        check_claim(1.000, "cohort05, median margin positive", c5["margin_positive_frac"], 5e-4)
        check_claim(1.000, "cohort05, centre beats every honest", c5["centre_inverts_all_frac"], 5e-4)
    ratios = [km2["measured"][c]["ratio_median"] for c in km2["measured"]]
    check_claim(1.76, "smallest dispersion ratio across configurations", min(ratios), 5e-3)
    check_claim(4.34, "largest dispersion ratio across configurations", max(ratios), 5e-3)

for k, ar2, noise in [(5, 0.553, 0.889), (10, 0.631, 0.984),
                      (20, 0.992, 0.997), (40, 1.000, 1.000)]:
    ms = load_json("mask_eval_shard%02d.json" % k)
    if ms:
        atk = ms["configs"]["paper"]["attacks"]
        check_claim(ar2, "support sweep AR(2) at %d classes" % k, atk["ar2_replay"]["auc_strict"], 5e-4)
        check_claim(noise, "support sweep noise at %d classes" % k, atk["noise"]["auc_strict"], 5e-4)

res = load_json("reduced_effort_summary.json")
if res:
    for frac, steps, move, sus, tpr, nb, kr, md, tm, cc, fg in (
            ("0.1", 14, 0.933, 0.896, 0.194, 0.675, 0.354, 0.249, 0.238, 0.109, 0.836),
            ("0.5", 71, 0.936, 0.799, 0.069, 0.656, 0.378, 0.282, 0.274, 0.197, 0.781)):
        r = res[frac]
        check_claim(steps, "reduced effort %s, victim steps" % frac, r["victim_steps"], 1e-9)
        check_claim(142, "reduced effort %s, full steps" % frac, r["full_steps"], 1e-9)
        check_claim(3, "reduced effort %s, seeds" % frac, r["n"], 1e-9)
        check_claim(move, "reduced effort %s, innovation over cohort" % frac,
              r["innovation_over_cohort"][0], 5e-4)
        check_claim(sus, "reduced effort %s, auditor AUC" % frac, r["auc"]["suspicion"][0], 5e-4)
        check_claim(tpr, "reduced effort %s, auditor TPR at frozen tau" % frac, r["tpr"][0], 5e-4)
        check_claim(nb, "reduced effort %s, norm bound AUC" % frac, r["auc"]["norm_bound"][0], 5e-4)
        check_claim(kr, "reduced effort %s, Krum AUC" % frac, r["auc"]["krum"][0], 5e-4)
        check_claim(md, "reduced effort %s, median AUC" % frac, r["auc"]["median_dev"][0], 5e-4)
        check_claim(tm, "reduced effort %s, trimmed mean AUC" % frac, r["auc"]["trimmed_mean_dev"][0], 5e-4)
        check_claim(cc, "reduced effort %s, cohort cosine AUC" % frac, r["auc"]["cosine_cohort"][0], 5e-4)
        check_claim(fg, "reduced effort %s, max-cosine AUC" % frac, r["auc"]["foolsgold_maxcos"][0], 5e-4)

zc = load_json("zero_innovation_cost.json")
if zc:
    a = zc["all"]
    check_claim(17200, "zero-innovation cost, honest client-rounds", a["client_rounds"], 1e-9)
    check_claim(0.129, "zero-innovation cost, smallest honest ratio", a["min_ratio"], 5e-4)
    check_claim(0, "zero-innovation cost, honest at or below 1e-6", a["count_at_or_below"]["1e-06"], 1e-9)
    dirichlet = [zc["groups"][g]["min_ratio"] for g in ("cifar100", "cifar10", "cohort05",
                                                        "cohort20", "cohort40", "late70")]
    check_claim(0.593, "zero-innovation cost, smallest ratio in a Dirichlet partition", min(dirichlet), 5e-4)

fr = load_json("fltrust_reference_eval.json")
if fr:
    S = fr["summary"]
    check_claim(0.071, "FLTrust root reference, honest share at trust zero", S["global_echo"]["root"]["p0_honest"], 5e-4)
    check_claim(0.965, "FLTrust root reference, global echo AUC", S["global_echo"]["root"]["auc_mean"], 5e-4)
    check_claim(0.965, "FLTrust root reference, 1 - p0/2", S["global_echo"]["root"]["auc_if_attacker_at_ceiling"], 5e-4)
    check_claim(0.000, "FLTrust root reference, TPR at 1% FPR", S["global_echo"]["root"]["tpr_at_fpr_pooled"]["fpr0.01"], 1e-9)
    check_claim(0.000, "FLTrust root reference, TPR at 5% FPR", S["global_echo"]["root"]["tpr_at_fpr_pooled"]["fpr0.05"], 1e-9)
    check_claim(1.000, "FLTrust root reference, TPR at 10% FPR", S["global_echo"]["root"]["tpr_at_fpr_pooled"]["fpr0.1"], 1e-9)
    check_claim(0.503, "FLTrust random reference, honest share at trust zero", S["global_echo"]["random"]["p0_honest"], 5e-4)
    check_claim(0.749, "FLTrust random reference, global echo AUC", S["global_echo"]["random"]["auc_mean"], 5e-4)
    check_claim(1.000, "FLTrust cohort-mean reference, global echo AUC", S["global_echo"]["cohort_mean"]["auc_mean"], 5e-4)
    check_claim(1.000, "FLTrust cohort-mean reference, TPR at 1% FPR", S["global_echo"]["cohort_mean"]["tpr_at_fpr_pooled"]["fpr0.01"], 1e-9)
    check_claim(0.056, "FLTrust cohort-mean reference, peer copy AUC", S["peer_copy"]["cohort_mean"]["auc_mean"], 5e-4)
    check_claim(0.073, "FLTrust cohort-mean reference, lagged self AUC", S["lagged_self"]["cohort_mean"]["auc_mean"], 5e-4)
    check_claim(0.092, "FLTrust cohort-median reference, lagged self AUC", S["lagged_self"]["cohort_median"]["auc_mean"], 5e-4)
    check_claim(0.219, "FLTrust cohort-mean reference, AR(2) AUC", S["ar2_replay"]["cohort_mean"]["auc_mean"], 5e-4)
    check_claim(0.649, "FLTrust root reference, AR(2) AUC", S["ar2_replay"]["root"]["auc_mean"], 5e-4)
    check_claim(0.134, "FLTrust root reference, lagged self AUC", S["lagged_self"]["root"]["auc_mean"], 5e-4)
    check_claim(0.500, "FLTrust root reference, peer copy AUC", S["peer_copy"]["root"]["auc_mean"], 5e-4)

ed = load_json("effort_decomposition.json")
if ed:
    assert ed.get("schema") == "fixed-readout-v2", "stale augmented-readout effort result"
    check_claim(3, "effort decomposition, seeds", len(ed["seeds"]), 1e-9)
    check_claim(12, "effort decomposition, reported from round", ed["report_from_round"], 1e-9)
    for seed in ed["seeds"]:
        raw = load_json("effort_decomposition_s%d.json" % seed)
        if raw is None:
            continue
        assert raw["schema"] == "fixed-readout-v2"
        check_claim(24, "fixed-readout rounds, seed %d" % seed, len(raw["rounds"]), 1e-9)
        zero_ratios = [r["by_steps"]["0"]["move"] / r["by_steps"][str(r["full_steps"])]["move"]
                       for r in raw["rounds"].values()]
        check_claim(0, "zero-step movement/full, max, seed %d" % seed, max(zero_ratios), 1e-6)
        for row in raw["rounds"].values():
            for steps, value in row["by_steps"].items():
                if int(steps) == 0:
                    continue
                b, d = row["offset"], value["move"]
                reconstructed = b*b + d*d + 2*b*d*value["cos_move_offset"]
                assert np.isclose(value["innov"] ** 2, reconstructed, rtol=1e-5), "effort norm decomposition fails"

fc = load_json("feature_combination.json")
if fc:
    R = fc["results"]
    check_claim(0.0, "combiners, max_z reproduces the auditor exactly", fc["max_z_vs_auditor_max_abs_dev"], 1e-9)
    check_claim(480, "combiners, calibration honest client-rounds", fc["fit_sizes"]["honest_cells"], 1e-9)
    check_claim(2400, "combiners, logistic regression positives", fc["fit_sizes"]["logreg_positives"], 1e-9)
    check_claim(26400, "combiners, logistic regression negatives",
          fc["fit_sizes"]["logreg_cells"] - fc["fit_sizes"]["logreg_positives"], 1e-9)
    for name, clean, worst, auc in [("max_z", 1.04, 100.0, 0.944), ("sum_z", 0.83, 19.4, 0.423),
                                    ("max_zsd", 1.88, 4.6, 0.923), ("logreg_z", 1.25, 74.8, 0.941),
                                    ("mahal_raw", 1.04, 100.0, 0.946), ("iforest_raw", 0.63, 18.8, 0.909)]:
        check_claim(clean, "combiners, %s clean FPR (%%)" % name, 100 * R[name]["clean_fpr"], 5.1e-3)
        check_claim(worst, "combiners, %s worst TPR (%%)" % name, 100 * R[name]["worst_tpr"], 5.1e-2)
        check_claim(auc, "combiners, %s worst AUC" % name, R[name]["worst_auc"], 5e-4)
    check_claim(R["sum_z"]["worst_tpr"] * 100, "combiners, mean_z equals sum_z", 100 * R["mean_z"]["worst_tpr"], 1e-6)
    check_claim(50.2, "combiners, sum_z noise TPR (%)", 100 * R["sum_z"]["per_attack"]["noise"]["tpr"], 5e-2)
    check_claim(19.4, "combiners, sum_z AR(2) TPR (%)", 100 * R["sum_z"]["per_attack"]["ar2_replay"]["tpr"], 5e-2)
    check_claim(4.6, "combiners, max_zsd peer copy TPR (%)", 100 * R["max_zsd"]["per_attack"]["peer_copy"]["tpr"], 5e-2)
    check_claim(74.8, "combiners, logreg lagged self TPR (%)", 100 * R["logreg_z"]["per_attack"]["lagged_self"]["tpr"], 5e-2)
    check_claim(18.8, "combiners, iforest peer copy TPR (%)", 100 * R["iforest_raw"]["per_attack"]["peer_copy"]["tpr"], 5.1e-2)
    check_claim(35.2, "combiners, iforest shrink TPR (%)", 100 * R["iforest_raw"]["per_attack"]["shrink"]["tpr"], 5e-2)

ue = load_json("unpredictable_eval.json")
if ue:
    for beta, stated in ((0.5, 0.988), (1, 0.114), (2, 0.063)):
        key = "sub3|worst|peer_current|phi=0|beta=%g" % beta
        check_claim(stated, "beacon-randomised AUC, beta=%g" % beta, ue["adaptive"][key]["auc"], 5e-4)

acv = load_json("adaptive_converge.json")
if acv:
    s = acv["summary"]
    secs = [v["secs"] for v in s.values()]
    check_claim(11.2, "effort sweep, cheapest pass seconds", min(secs), 5e-2)
    check_claim(1261.0, "effort sweep, costliest pass seconds", max(secs), 5e-1)
    used = [v["bused_mean"] for v in s.values()]
    check_claim(0.987, "effort sweep, lowest budget utilization", min(used), 5e-4)
    check_claim(0.999, "effort sweep, highest budget utilization", max(used), 5e-4)
    sm = [v["susp_mean"] for v in s.values()]
    check_claim(1.045, "effort sweep, lowest achieved score", min(sm), 5e-4)
    check_claim(1.117, "effort sweep, highest achieved score", max(sm), 5e-4)

tfm = load_json("temporal_eval_fullmask.json")
tdm = load_json("temporal_eval.json")
if tfm and tdm:
    fa = tfm["results"]["TEST"]["suspicion"]["0.01"]
    da = tdm["results"]["TEST"]["suspicion"]["0.01"]
    check_claim(da["threshold"], "full-mask policy, same frozen threshold", fa["threshold"], 5e-5)
    check_claim(da["honest"]["client_round"]["k"], "full-mask policy, same honest false alarms",
          fa["honest"]["client_round"]["k"], 1e-9)
    for atk in ("global_echo", "peer_copy", "lagged_self", "ar2_replay", "noise", "shrink"):
        check_claim(1.0, "full-mask policy, detection / %s" % atk,
              fa["attacks"][atk]["tpr_client_round"]["point"], 1e-9)
    for atk, before, after in (("peer_copy", 0.0172, 0.0193), ("lagged_self", 0.0102, 0.0116),
                               ("noise", 0.0049, 0.0051)):
        check_claim(before, "uninvolved FPR, declared policy / %s" % atk,
              da["attacks"][atk]["fpr_uninvolved_client_round"]["point"], 5e-5)
        check_claim(after, "uninvolved FPR, full policy / %s" % atk,
              fa["attacks"][atk]["fpr_uninvolved_client_round"]["point"], 5e-5)

mpd = load_json("mask_policy_delta.json")
if mpd:
    check_claim(29280, "mask policy delta, scored client-rounds", mpd["scored_client_rounds"], 1e-9)
    check_claim(0.296, "mask policy delta, share of scores that move", mpd["share_changed"], 5e-4)
    check_claim(13.6, "mask policy delta, largest score change", mpd["abs_change_max"], 5e-2)
    check_claim(0.10, "mask policy delta, mean score change", mpd["abs_change_mean"], 5e-3)
    check_claim(17, "mask policy delta, decisions that flip", mpd["decision_flips_total"], 1e-9)
    check_claim(0, "mask policy delta, flips in honest replays", mpd["decision_flips"]["honest_replay"], 1e-9)
    check_claim(0, "mask policy delta, flips on victim rounds", mpd["decision_flips"]["victim"], 1e-9)
    check_claim(82.0, "largest statistical suspicion in any test replay",
          mpd["largest_statistical_suspicion"], 5e-2)
    check_claim(6.61, "largest honest suspicion, unattacked replays",
          mpd["largest_honest_suspicion_declared"], 5e-3)

us = load_json("utility_seed_spread.json")
if us:
    check_claim(0.0039, "unattacked range across seeds", us["honest_range"], 5e-5)
    check_claim(5, "utility seeds", len(us["seeds"]), 1e-9)
    for name, m, sd in [("fused_b1", 0.0088, 0.0075), ("fused_b2", 0.0157, 0.0163),
                        ("fused_b1_x3", 0.0161, 0.0031)]:
        r = us["scenarios"][name]
        check_claim(m, "utility drop mean / %s" % name, r["mean_drop"], 5e-5)
        check_claim(sd, "utility drop sd / %s" % name, r["sd_drop"], 5e-5)
    check_claim(1.0, "collusion drop has one sign in every seed",
          float(us["scenarios"]["fused_b1_x3"]["consistent_sign"]), 1e-9)
    check_claim(0.0, "beta=1 drop has one sign in every seed",
          float(us["scenarios"]["fused_b1"]["consistent_sign"]), 1e-9)

sws = load_json("sweeps.json")
if sws:
    fs = sws["fraction_sweep"]
    for n, stated in (("1", 0.944), ("5", 0.563)):
        check_claim(stated, "collusion sweep, %s attacker(s) of ten" % n,
              fs[n]["suspicion"], 5e-4)

pe = load_json("protocol_eval_d05.json")
if pe:
    sep = pe["separation"]["TEST"]
    check_claim(4.29, "weakest cheating score on TEST", sep["worst_attack_min"], 5e-3)
    check_claim(8.83, "largest uninvolved honest score on TEST",
          sep["honest_max_by_population"]["uninvolved"], 5e-3)

pa = load_json("passport_ablation.json")
if pa:
    check_claim(0.0287, "passport cost, prototype accuracy", pa["mean_d_prototype"], 5e-5)
    check_claim(0.0185, "passport cost, classifier accuracy", pa["mean_d_classifier"], 5e-5)
    check_claim(0.0013, "passport cost, prototype sd", pa["sd_d_prototype"], 5e-5)
    check_claim(0.0021, "passport cost, classifier sd", pa["sd_d_classifier"], 5e-5)
    check_claim(3, "passport ablation, paired seeds", len(pa["per_seed"]), 1e-9)

ac = load_json("attacker_cost.json")
if ac:
    check_claim(1.34, "attacker craft seconds per round",
          ac["attacker_craft_seconds"]["median"], 5e-3)
    check_claim(37.6, "honest local training seconds per round",
          ac["honest_local_train_seconds"]["median"], 5e-2)
    check_claim(28.1, "honest cost over attacker cost", ac["honest_over_attacker"], 5e-2)
for dev, craft, honest, ratio in (("cpu", 0.90, 27.9, 30.9), ("cuda", 0.90, 2.75, 3.0)):
    if act := load_json(f"attacker_cost_{dev}.json"):
        check_claim(craft, f"attacker craft seconds per round ({dev} host)", act["attacker_craft_seconds"]["median"], 5e-3)
        check_claim(honest, f"honest local training seconds per round on {dev}", act["honest_local_train_seconds"]["median"],
                    5e-2 if honest >= 10 else 5e-3)
        check_claim(ratio, f"honest cost over attacker cost ({dev})", act["honest_over_attacker"], 5e-2)
        check_claim(1, f"timing host recorded ({dev})",
                    int(act["hardware"]["cpu"].startswith("AMD Ryzen 9 9950X3D") and act["hardware"]["gpu"] == "NVIDIA GeForce RTX 5090"), 0)

fgo = load_json("adaptive_fg_observer.json")
if fgo:
    for beta, cc in ((0, 1.0000), (1, 0.9248)):
        key = "suspicion/mimic|peer_current|phi=0|beta=%d" % beta
        st = fgo["cells"][key]["stats"]
        check_claim(0.500, "FoolsGold observer AUC, beta=%d" % beta,
              st["foolsgold_history"]["auc"], 5e-4)
        check_claim(0.500, "FoolsGold observer with importance, beta=%d" % beta,
              st["foolsgold_history_imp10"]["auc"], 5e-4)
        check_claim(cc, "cohort cosine as observer, beta=%d" % beta,
              st["cosine_cohort"]["auc"], 5e-4)

g = load_json("combined_grid.json")
if g:
    def cell(phi, beta, rule):
        for k, v in g["cells"].items():
            p = k.split("|")
            if float(p[4].split("=")[1]) == phi and float(p[5].split("=")[1]) == beta:
                return v["stats"][rule]["auc"]
        return None
    for beta, mx, sm, tw in [(0.0, 1.000, 1.000, 1.000), (0.5, 0.999, 0.812, 1.000),
                             (1.0, 0.498, 0.074, 0.418), (2.0, 0.506, 0.135, 0.452)]:
        check_claim(mx, f"adaptive table max beta={beta}", cell(0.0, beta, "combined_max"), 1e-3)
        check_claim(sm, f"adaptive table sum beta={beta}", cell(0.0, beta, "combined_sum"), 1e-3)
        check_claim(tw, f"adaptive table two beta={beta}", cell(0.0, beta, "combined_two"), 1e-3)

u = load_json("utility_impact_gpu_s0.json")
if u:
    S = u["summary"]
    base = S["honest"]["utility"]["tail_acc_prototype_honest"]
    check_claim(0.1629, "honest tail accuracy", base, 5e-5)
    for key, stated in [("fused_b1", 0.0041), ("fused_b2", 0.0426), ("fused_b1_x3", 0.0172)]:
        if key in S:
            check_claim(stated, f"accuracy lost on seed 0, {key}",
                  base - S[key]["utility"]["tail_acc_prototype_honest"], 1e-4)

if ai := load_json("aggregate_impact.json"):
    d05 = ai["configs"]["cifar100_d05"]["rules"]
    for rule, ratio, kept, kept_loo in [("median", 0.99, 0.75, 1.00), ("trimmed_mean", 0.93, 0.84, 1.00),
                                        ("bulyan", 0.94, 0.68, 0.94)]:
        check_claim(ratio, f"echo/removal displacement, {rule}", d05[rule]["global_echo"]["shift_over_loo"]["median"], 5e-3)
        check_claim(kept, f"retained step under echo, {rule}", d05[rule]["global_echo"]["progress_own"]["median"], 5e-3)
        check_claim(kept_loo, f"retained step after removal, {rule}", d05[rule]["leave_one_out"]["progress_own"]["median"], 5e-3)
    check_claim(0.90, "retained step under echo, fedproto", d05["fedproto"]["global_echo"]["progress_own"]["median"], 5e-3)
    robust = [ai["configs"][c]["rules"][r]["global_echo"]["progress_own"]["median"]
              for c in ai["configs"] for r in ("median", "trimmed_mean", "bulyan") if r in ai["configs"][c]["rules"]]
    check_claim(0.55, "smallest retained step, robust rules", min(robust), 5e-3)
    check_claim(0.95, "largest retained step, robust rules", max(robust), 5e-3)
    guard = [g["krum_guarantee"] for g in ai["configs"].values() if g["krum_guarantee"]["n_gt_2f_plus_2"]]
    check_claim(11.1, "smallest eta*ratio in any round", min(g["eta_times_ratio_min"] for g in guard), 5e-2)
    check_claim(12.1, "smallest configuration median eta*ratio", min(g["eta_times_ratio"]["median"] for g in guard), 5e-2)
    check_claim(40.9, "largest configuration median eta*ratio", max(g["eta_times_ratio"]["median"] for g in guard), 5e-2)
    cl = ai["closed_loop_reference"]
    check_claim(0.0013, "closed-loop echo accuracy drop", cl["echo"]["mean_drop"], 5e-5)
    check_claim(0.0019, "closed-loop echo drop s.d.", cl["echo"]["sd_drop"], 5e-5)
    check_claim(0.0039, "unattacked accuracy range", cl["honest_range"], 5e-5)

if pp := load_json("published_predicates.json"):
    rows = {**pp["results"], **{"joint_" + k: v["joint_1pct"] for k, v in pp["combined"].items()}}
    for key, vals in [("l2_bound", (99.3, 100, 100, 100, 91.4, 99.3)), ("linf_bound", (99.2, 100, 100, 100, 100, 99.2)),
                      ("median_l2_bound", (98.5, 100, 100, 100, 93.6, 98.9)), ("range_submission", (99.0, 100, 87.7, 99.9, 100, 99.0)),
                      ("norm_ball", (100, 100, 99.8, 100, 100, 100)), ("zeno", (100, 100, 0.0, 100, 100, 100)),
                      ("cosine_last_update", (99.9, 100, 100, 0.0, 69.8, 99.9)), ("joint_EIFFeL", (100, 100, 0.0, 0.0, 77.9, 100))]:
        e = rows[key]
        got = [e["honest"]["rate"]] + [e["attacks"][a]["rate"] for a in
                                       ("global_echo", "drift_extrapolation", "drift_reversal", "lagged_self", "peer_copy")]
        for stated, g, col in zip(vals, got, ("honest", "echo", "drift", "reversed", "lagged", "copy")):
            check_claim(stated, f"predicate table {key}/{col}", 100 * g, 0.05)

if ge := load_json("ghost_extended.json"):
    med = ge["calibration_search"]["selected"]["medians"]
    check_claim(0.156, "calibrated ghost prototype ratio", med["proto_ratio"], 5e-4)
    check_claim(0.776, "calibrated ghost stability", med["label_stability"], 5e-4)
    acc = {(r["ghost"], r["verifier"], r["regime"], r["scheme"], r["warmup"]): r.get("accepted", r.get("accepted_total"))
           for r in ge["acceptance_table"]}
    check_claim(40, "calibrated ghost, deployed", acc[("calibrated", "deployed", "chain_head", "Ed25519", 5)], 0)
    check_claim(800, "calibrated ghost, post-commitment", acc[("calibrated", "deployed", "beacon_post_commit", "Ed25519", 5)], 0)
    check_claim(40, "calibrated ghost, two-sided", acc[("calibrated", "two_sided", "chain_head", "Ed25519", 5)], 0)
    check_claim(5, "original ghost, two-sided (warm-up only)", acc[("original", "two_sided", "chain_head", "Ed25519", 5)], 0)
    for w in (5, 0):
        check_claim(800, f"original ghost, post-commitment, warm-up {w}",
                    acc[("original", "deployed", "beacon_post_commit", "Ed25519", w)], 0)
    mc = ge["ghosts"]["calibrated"]["random_history_monte_carlo"]
    check_claim(10000, "calibrated ghost, random sequences fully accepted (deployed)", mc["deployed_w5"]["sequences_all_rounds_accepted"], 0)
    check_claim(99.2, "calibrated ghost, random sequences accepted (two-sided, %)",
                100 * mc["two_sided_w5"]["sequences_all_rounds_accepted"] / mc["sequences"], 0.05)
    h = ge["honest_reference"]["acceptance"]
    check_claim(8.2, "honest rounds rejected by the two-sided verifier (%)", 100 * (1 - h["two_sided"] / ge["honest_reference"]["n_rows"]), 0.05)
    check_claim(3, "honest clients never rejected by the two-sided verifier", h["clients_all_post_warmup_rounds_accepted"]["two_sided"], 0)

if fe := load_json("frozen_encoder_prover.json"):
    for name, w5 in (("stale_trained", 40), ("untrained", 8)):
        check_claim(w5, f"frozen encoder {name}, accepted rounds at warm-up 5",
                    fe["variants"][name]["summary"]["per_warmup"]["5"]["accepted_rounds"], 0)

if la := load_json("limited_access_grid.json"):
    for level, p_auc, p_det, f_auc, f_det in [
            ("current", 0.514, 0, 0.498, 0), ("delayed:1", 0.605, 1, 0.690, 9), ("delayed:2", 0.662, 2, 0.713, 8),
            ("delayed:3", 0.700, 2, 0.767, 11), ("query:4", 0.588, 1, 0.655, 2), ("query:16", 0.545, 0, 0.560, 0),
            ("query:64", 0.522, 0, 0.526, 0), ("query:256", 0.522, 0, 0.509, 0), ("decision:4", 0.606, 0, 0.709, 0),
            ("decision:16", 0.608, 0, 0.712, 0), ("decision:64", 0.606, 0, 0.711, 0), ("feedback", 0.604, 1, 0.685, 8),
            ("pseudo", 0.968, 77, 0.970, 81), ("shadow:5", 0.573, 1, 0.885, 10), ("shadow:6", 0.610, 4, 0.885, 21)]:
        for defense, auc, det in (("P", p_auc, p_det), ("F", f_auc, f_det)):
            c = la["cells"][f"{defense}|{level}|beta=1"]
            check_claim(auc, f"access table {defense}/{level} AUC", c["auc"], 5e-4)
            check_claim(det, f"access table {defense}/{level} detected", c["flagged_client_rounds_TEST"], 0)
    for level in ("shadow:5", "shadow:6"):
        check_claim(1.0, f"every attacked test trajectory flagged, F/{level}", la["cells"][f"F|{level}|beta=1"]["tpr_run_TEST"], 0)

if ai:
    for rule, kept in (("median", 0.74), ("trimmed_mean", 0.73), ("bulyan", 0.83)):
        check_claim(kept, f"retained step under a -3x Byzantine update, {rule}",
                    ai["configs"]["cifar100_d05"]["rules"][rule]["byzantine_ref"]["progress_own"]["median"], 5e-3)
    check_claim(11.0, "eta*ratio exceeds 11.0 in every round (lower bound)",
                min(min(g["eta_times_ratio_min"] for g in guard), 11.0), 0)

if pp:
    rows = {**pp["results"], **{"joint_" + k: v["joint_1pct"] for k, v in pp["combined"].items()}}
    for key, vals in [("linf_bound", (99.3, 100, 100, 100, 100, 99.3)),
                      ("joint_two-sided l2 (sensitivity)", (99.5, 0.0, 92.0, 92.0, 95.3, 99.5))]:
        e = rows[key]
        got = [e["honest"]["rate"]] + [e["attacks"][a]["rate"] for a in
                                       ("global_echo", "drift_extrapolation", "drift_reversal", "lagged_self", "peer_copy")]
        for stated, g, col in zip(vals, got, ("honest", "echo", "drift", "reversed", "lagged", "copy")):
            check_claim(stated, f"predicate table {key}/{col} (as printed)", 100 * g, 0.05 + 1e-9)

if ul := load_json("utility_lagged_summary.json"):
    sc = ul["scenarios"]
    for key, mean, sd in (("fused_b1_lag", 0.0133, 0.0061), ("fused_b1", 0.0075, 0.0068), ("fused_b1_x3_lag", 0.0175, 0.0056)):
        check_claim(mean, f"closed loop {key} accuracy drop", sc[key]["drop"]["mean"], 5e-5)
        check_claim(sd, f"closed loop {key} drop s.d.", sc[key]["drop"]["sd"], 5e-5)
    check_claim(1, "delayed attacker loses accuracy in every seed", int(min(sc["fused_b1_lag"]["per_seed_drop"]) > 0), 0)
    check_claim(9.4, "delayed attacker flagged client-rounds (%)", 100 * sc["fused_b1_lag"]["flag_rate"]["mean"], 0.05)
    check_claim(4, "delayed attacker runs flagged at least once", sum(sc["fused_b1_lag"]["run_flagged"]["per_seed"]), 0)
    check_claim(0.0, "current-information attacker flagged client-rounds", sc["fused_b1"]["flag_rate"]["mean"], 0)
    check_claim(8.6, "delayed colluders flagged client-rounds (%)", 100 * sc["fused_b1_x3_lag"]["flag_rate"]["mean"], 0.05)

if ur := load_json("utility_robust_summary.json"):
    med = ur["rules"]["median"]["across_seeds"]
    check_claim(-0.0030, "median closed loop: echo change in honest prototype accuracy (sign: gain)",
                med["drop"]["echo"]["proto"]["mean"], 5e-5)
    check_claim(0.0017, "median closed loop: echo change s.d.", med["drop"]["echo"]["proto"]["sd"], 5e-5)
    check_claim(1, "median closed loop: echo improves prototype accuracy in every seed",
                int(max(med["drop"]["echo"]["proto"]["per_seed"]) < 0), 0)
    check_claim(-0.0002, "median closed loop: echo change in classifier accuracy", med["drop"]["echo"]["cls"]["mean"], 5e-5)
    check_claim(0.0017, "median closed loop: classifier change s.d.", med["drop"]["echo"]["cls"]["sd"], 5e-5)
    check_claim(0.0004, "median closed loop: replicate honest bound", med["noise_floor_max_abs"]["proto"], 5e-5)
    x = ur["cross_rule"]["median_minus_mean"]["proto"]["honest_tail"]
    check_claim(-0.0097, "median minus mean honest prototype accuracy", x["mean"], 5e-5)
    check_claim(0.0014, "median minus mean s.d.", x["sd"], 5e-5)
    fr = med["free_rider"]["free_rider"]
    for rule in ("krum", "median_dev", "trimmed_mean_dev"):
        check_claim(1.0, f"median closed loop: echo ranked first by {rule} in every attacked round",
                    fr[rule]["rank1_fraction"]["min"], 0)
    check_claim(85, "median closed loop: attacked rounds", sum(
        len(ur["rules"]["median"]["per_seed"][s]["free_rider"]["krum"]["rank"]) for s in ur["rules"]["median"]["per_seed"]), 0)

if ic := load_json("inversion_condition.json"):
    cf = ic["configs"]
    check_claim(0.0, "FedProto identity residual", max(c["identity"]["identity_ls_max_abs"] for c in cf.values()), 1e-13)
    for c in cf:
        check_claim(1.0, f"exact inversion scored rounds {c}", cf[c]["scored"]["coverage_seat_rounds"]["all_rules_exact"], 1e-9)
    check_claim(11, "early rounds with exact failure", sum(c["early"]["rounds_with_exact_failure"] for c in cf.values()), 0)
    d = cf["cifar100_d05"]["scored"]
    for k, v in [("median", .627), ("trimmed_mean", .593), ("bulyan", .398)]:
        check_claim(v, f"SC1 {k} d05", d["coverage_seat_rounds"][k]["SC1"], 5e-4)
    check_claim(.663, "Krum-tight d05", d["coverage_seat_rounds"]["krum_f2"]["krum_tight"], 5e-4)
    check_claim(.967, "uniform-vs-step gap d05 (median)", d["ranges"]["gap_over_mu"]["median"], 5e-4)
    check_claim(.869, "uniform-vs-step gap, scored min over configs", min(c["scored"]["ranges"]["gap_over_mu"]["min"] for c in cf.values()), 5e-4)
    check_claim(1.210, "uniform-vs-step gap, scored max over configs", max(c["scored"]["ranges"]["gap_over_mu"]["max"] for c in cf.values()), 5e-4)
    dec = cf["cifar100_d05"]["decomposition_at"]
    for lab, own, prev, off, st, mu, mus, ss in [
            ("round_2", 71.19, 99.77, 42.29, 28.66, 37.59, .955, .728),
            ("round_12", 26.09, 35.09, 34.50, 11.02, 14.72, .463, .349),
            ("last_round", 17.36, 46.55, 46.46, 5.58, 17.06, .388, .129)]:
        x = dec[lab]
        check_claim(own, f"own move {lab}", x["own_move_median"], 5e-3)
        check_claim(prev, f"prev offset {lab}", x["prev_offset_median"], 5e-3)
        check_claim(off, f"offset {lab}", x["offset_median"], 5e-3)
        check_claim(st, f"step {lab}", x["step"], 5e-3)
        check_claim(mu, f"||mu|| {lab}", x["mu_norm"], 5e-3)
        check_claim(mus, f"||mu||/sigma {lab}", x["mu_over_sigma"], 5e-4)
        check_claim(ss, f"||s||/sigma {lab}", x["step_over_sigma"], 5e-4)
    for c, ok, fail in [("cifar100_d05", .799, .755), ("cifar10", .443, .556), ("late70", .744, .806),
                        ("cohort05", .751, .713), ("nopassport", .731, .738)]:
        t = cf[c]["step_over_offset_min_threshold"]
        check_claim(ok, f"max ||s||/min a where all invert {c}", t["max_where_every_seat_inverts"], 5e-4)
        check_claim(fail, f"min ||s||/min a where a seat fails {c}", t["min_where_some_seat_fails"], 5e-4)
if c5 := load_json("cohort05_f1.json"):
    s, g = c5["summary"]["global_echo"], c5["krum_guarantee"]["f1"]
    check_claim(0.001, "N=5 Krum f=1 echo AUC", s["krum_f1"]["auc_mean"], 5e-4)
    check_claim(0.227, "N=5 Krum f=2 echo AUC", s["krum_f2"]["auc_mean"], 5e-4)
    check_claim(1.0, "N=5 f=1 strict selection", s["krum_f1_selection"]["strict"], 1e-9)
    check_claim(0.314, "N=5 f=2 tie-broken selection", s["krum_f2_selection"]["selected"], 5e-4)
    check_claim(4.243, "eta(5,1)", g["eta"], 5e-4)
    check_claim(7.45, "eta*ratio median", g["eta_times_ratio"]["median"], 5e-3)
    check_claim(6.66, "eta*ratio min", g["eta_times_ratio"]["min"], 5e-3)
    check_claim(0.0, "precondition holds", g["frac_rounds_precondition_holds"], 1e-9)
if bp := load_json("bulyan_per_seed.json"):
    for cfg, vals in [("cifar100_d05", {"global_echo": 0.000, "peer_copy": 0.241,
                                        "lagged_self": 0.497, "ar2_replay": 0.998,
                                        "noise": 0.674, "shrink": 0.000}),
                      ("cifar100_d01", {"global_echo": 0.000}),
                      ("cifar10", {"global_echo": 0.000}),
                      ("late70", {"global_echo": 0.000, "shrink": 0.000}),
                      ("cohort20", {"global_echo": 0.000, "shrink": 0.000}),
                      ("cohort40", {"global_echo": 0.000, "shrink": 0.000}),
                      ("nopassport", {"global_echo": 0.000, "shrink": 0.000})]:
        for atk, stated in vals.items():
            check_claim(stated, "ranking table Bulyan (per-seed mean)/%s %s" % (atk, cfg),
                        bp["configs"].get(cfg, {}).get(atk, {}).get("auc_mean"), 1e-3)
    check_claim(0.0, "Bulyan echo AUC in every seed of every configuration",
                max(v for c in bp["configs"].values() for v in c["global_echo"]["per_seed"].values()), 1e-9)
    check_claim(0.022, "Bulyan largest s.d. over seeds, ranking-table configurations",
                max(bp["configs"][c][a]["auc_sd_ddof0"] for c in ("cifar100_d05", "cifar100_d01", "cifar10")
                    for a in bp["configs"][c]), 5e-4)
if bl := load_json("bulyan_eval.json"):
    for atk, stated in [("global_echo", 0.000), ("peer_copy", 0.241), ("lagged_self", 0.496),
                        ("ar2_replay", 0.998), ("noise", 0.673), ("shrink", 0.000)]:
        check_claim(stated, "Bulyan pooled over seeds/%s cifar100_d05" % atk, bl["cifar100_d05"][atk], 1e-3)
if sb := load_json("std_baseline.json"):
    for cfg, rows in [("cifar100_d05", [("global_echo", 1, 1), ("peer_copy", .428, .499), ("lagged_self", .582, .317),
                                        ("ar2_replay", .993, .003), ("noise", .505, .500), ("shrink", 1, 1)]),
                      ("cifar100_d01", [("global_echo", 1, 1), ("peer_copy", .399, .492), ("lagged_self", .756, .136),
                                        ("ar2_replay", .869, .098), ("noise", .501, .501), ("shrink", 1, 1)]),
                      ("cifar10", [("global_echo", 1, 1), ("peer_copy", .418, .502), ("lagged_self", .418, .368),
                                   ("ar2_replay", .789, .106), ("noise", .499, .499), ("shrink", 1, 1)])]:
        for atk, two, low in rows:
            r = sb["configs"][cfg]["attacks"][atk]
            check_claim(two, f"STD two-sided {atk} {cfg}", r["std_two_sided"]["mean"], 1e-3)
            check_claim(low, f"STD low {atk} {cfg}", r["std_low"]["mean"], 1e-3)
    check_claim(0.0, "STD two-sided vs norm bound, all-seed means", max(
        abs(r["std_two_sided"]["mean"] - r["norm_bound"]["mean"])
        for res in sb["configs"].values() for r in res["attacks"].values()), 1.2e-3)
    check_claim(0.0, "STD two-sided vs norm bound, per seed", max(
        abs(r["std_two_sided"]["per_seed"][s] - r["norm_bound"]["per_seed"][s])
        for res in sb["configs"].values() for r in res["attacks"].values()
        for s in r["norm_bound"]["per_seed"]), 3.5e-3)
if sw := load_json("std_dagmm_within.json"):
    for cfg, rows in [("cifar100_d05", [("global_echo", 1, 1), ("lagged_self", .182, .411), ("ar2_replay", .844, .987),
                                        ("noise", .979, .978), ("shrink", 1, 1)]),
                      ("cifar100_d01", [("global_echo", 1, 1), ("lagged_self", .472, .583), ("ar2_replay", .830, .926),
                                        ("noise", .981, .993), ("shrink", 1, 1)]),
                      ("cifar10", [("global_echo", 1, 1), ("lagged_self", .313, .301), ("ar2_replay", .677, .831),
                                   ("noise", .760, .782), ("shrink", 1, 1)])]:
        for atk, dg, sdg in rows:
            r = sw["configs"][cfg]["attacks"][atk]
            check_claim(dg, f"DAGMM within-federation {atk} {cfg}", r["dagmm"]["mean"], 1e-3)
            check_claim(sdg, f"STD-DAGMM within-federation {atk} {cfg}", r["std_dagmm"]["mean"], 1e-3)
    check_claim(0.0, "DAGMM / STD-DAGMM peer copy |AUC - 0.5| per seed", max(
        abs(v - 0.5) for res in sw["configs"].values() for k in ("dagmm", "std_dagmm")
        for v in res["attacks"]["peer_copy"][k]["per_seed"].values()), 2e-4)
    check_claim(0.0, "within-file untrained scores equal std_baseline.json", max(
        abs(sw["configs"][c]["attacks"][a][k]["mean"] - sb["configs"][c]["attacks"][a][k]["mean"])
        for c in sw["configs"] for a in sw["configs"][c]["attacks"]
        for k in ("std_two_sided", "std_low", "norm_bound")) if sb else None, 0)
    check_claim(0.0, "within-file replay vs innovation_eval", max(
        v for res in sw["configs"].values() for v in res["checks"]["innovation_eval_max_abs_auc_diff"].values()), 0)
    tr = sw["dagmm_training"]
    for cfg, n, fit, mean_sc, zero_sc in [("cifar100_d05", 30, 23, 12, 30), ("cifar100_d01", 18, 18, 18, 18),
                                          ("cifar10", 18, 2, 12, 16)]:
        ms = [m for s in tr[cfg].values() for m in s["models"].values()]
        check_claim(n, f"DAGMM within models {cfg}", len(ms), 0)
        check_claim(fit, f"DAGMM within fits training rows {cfg}", sum(m["fits_train"] for m in ms), 0)
        check_claim(mean_sc, f"DAGMM within beats training mean on scored honest rows {cfg}",
                    sum(m["beats_train_mean_on_scored"] for m in ms), 0)
        check_claim(zero_sc, f"DAGMM within beats zero on scored honest rows {cfg}",
                    sum(m["beats_zero_on_scored"] for m in ms), 0)
    tf = [s["transfer"] for t in tr.values() for s in t.values()]
    check_claim(.243, "PCA-10 of rounds 2-11 explains rounds 12-35, same seed (min)", min(t["pca_explained_same_seed"] for t in tf), 5e-4)
    check_claim(.383, "PCA-10 of rounds 2-11 explains rounds 12-35, same seed (max)", max(t["pca_explained_same_seed"] for t in tf), 5e-4)
    check_claim(.145, "PCA-10 of rounds 2-11 explains rounds 12-35, other seed (max)", max(t["pca_explained_other_seed"] for t in tf), 5e-4)

if ph0 := load_json("paired_honest_summary.json"):
    check_claim(0, "paired honest: logged verdicts recomputed", ph0["verdict_recomputation"]["total_mismatches"], 0)
    for s, c in ph0["verdict_recomputation"]["run_consistency"].items():
        check_claim(0, f"paired honest s{s}: server rejections", c["server_rejections"], 0)
        check_claim(0, f"paired honest s{s}: statistic mismatches",
                    c["receiver_stat_mismatches"] + c["server_stat_mismatches"] + c["recompute_mismatches"], 0)
    rc = ph0["random_challenges"]
    check_claim(0, "paired honest: tables vs committed roots", rc["root_mismatches"], 0)
    check_claim(0, "paired honest: realised replay verdicts", rc["realised_verdict_mismatches"], 0)
    check_claim(0, "paired honest: realised replay statistics", rc["realised_max_abs_stat_diff"], 1e-12)
    check_claim(1, "paired honest: released payload rates re-derived",
                int(ph0["payload_layer"]["regression_released_archives"]["all_identical"]), 0)
if jp := load_json("joint_ghost_paired.json"):
    check_claim(2160, "joint ghost (paired bands): regenerated payloads identical", jp["regeneration"]["digests_match_released"], 0)
    b = jp["bands"]["bands"]["window_12_35"]
    for st, (lo, hi) in {"proto_ratio": (0.16730, 0.56975), "label_stability": (0.19963, 0.82087),
                         "max_drift": (0.75170, 3.03208)}.items():
        check_claim(lo, f"paired CAL band {st} low", b[st][0], 5e-5); check_claim(hi, f"paired CAL band {st} high", b[st][1], 5e-5)
    for lv, pay, rin, dep, two, jch, jmc in [("current", 0, 217, 73, 65, 65, .6563), ("delayed:1", 12, 200, 53, 50, 46, .5087),
            ("delayed:2", 12, 214, 57, 57, 53, .5435), ("delayed:3", 16, 209, 63, 63, 58, .6003), ("query:16", 1, 203, 63, 59, 59, .6338),
            ("decision:4", 3, 202, 62, 60, 60, .6486), ("feedback", 9, 201, 64, 62, 60, .6278), ("pseudo", 88, 216, 64, 59, 0, 0.0),
            ("shadow:5", 14, 228, 77, 64, 58, .5905)]:
        c = jp["cells"][f"{lv}|beta=1"]; v = c["variants"]["primary"]["TEST"]
        check_claim(pay, f"joint ghost paired {lv}: payload flags", c["payload_layer"]["payload_any"]["flagged_TEST"]["k"], 0)
        check_claim(rin, f"joint ghost paired {lv}: ratio in band", c["published_ratio_12_35"]["inside"]["k"], 0)
        check_claim(dep, f"joint ghost paired {lv}: deployed w5", v["chain_head_rounds_12_35_accepted"]["deployed_w5"]["k"], 0)
        check_claim(two, f"joint ghost paired {lv}: two-sided w5", v["chain_head_rounds_12_35_accepted"]["two_sided_w5"]["k"], 0)
        check_claim(jch, f"joint ghost paired {lv}: joint chain-head", v["joint"]["w5"]["chain_head_client_rounds"]["k"], 0)
        check_claim(jmc, f"joint ghost paired {lv}: joint MC", v["joint"]["w5"]["mc_client_round_rate_mean"], 5e-4)
        check_claim(0, f"joint ghost paired {lv}: joint trajectories", v["joint"]["w5"]["chain_head_trajectories"]["k"], 0)
if ph := load_json("paired_honest_summary.json"):
    fa = ph["false_alarms_rounds_12_35"]["TEST"]
    for vname, k in (("deployed_w5", 223), ("two_sided_window_w5", 234), ("two_sided_paper_w5", 351)):
        check_claim(k, f"paired honest TEST rejections 12-35 {vname}", fa[vname]["k"], 0)
    pl = ph["payload_layer"]["pooled"]["TEST"]
    check_claim(13, "paired honest TEST any payload flag", pl["any_payload_flag"]["k"], 0)
    for vname, k in (("deployed_w5", 229), ("two_sided_window_w5", 240), ("two_sided_paper_w5", 354)):
        check_claim(k, f"paired honest TEST joint rejected {vname}", pl["joint_rejected_" + vname]["k"], 0)
        check_claim(0, f"paired honest TEST trajectories fully accepted {vname}", pl["trajectories_fully_accepted_" + vname]["k"], 0)
    check_claim(836, "paired honest deployed w5 accepted rounds 6-R, all seeds",
                sum(ph["per_seed"][s][w]["acceptance"]["deployed_w5"]["accepted"]["k"]
                    for s in ph["per_seed"] for w in ph["per_seed"][s] if w.startswith("rounds_6_")), 0)
if f3 := load_json("paired_fig3a.json"):
    check_claim(1, "paired Fig. 3a: inputs consistent", int(f3["checks"]["all_consistent"]), 0)
    bars = {x["key"]: x for x in f3["panel_a"]["bars"]}
    for key, g, h in (("rounds_accepted", 1.0, 0.566), ("label_stability", 1.000, 0.580), ("drift_ratio", 0.001, 0.338)):
        check_claim(g, f"paired Fig. 3a ghost {key}", bars[key]["ghost"], 5e-4)
        check_claim(h, f"paired Fig. 3a honest {key}", bars[key]["honest"], 5e-4)
    check_claim(0, "paired Fig. 3a: honest clients accepted every round", f3["panel_a"]["honest_clients_accepted_every_round"][0], 0)
    cb = {x["key"]: x for x in f3["panel_a"]["calibrated_ghost_bars"]}
    check_claim(0.776, "calibrated ghost stability", cb["label_stability"]["ghost"], 5e-4)
    check_claim(0.156, "calibrated ghost ratio", cb["drift_ratio"]["ghost"], 5e-4)

if sr := load_json("score_range.json"):
    so = sr["summary_offline"]["0.01"]
    check_claim(8.9, "largest unattacked statistical score, seeds 0-7, eta 0.01", so["max_unattacked"], 5e-2)
    check_claim(102, "largest statistical score in offline replays, eta 0.01", so["max_victim"], 0.5)
    check_claim(82.0, "largest statistical score on test seeds 3 and 4, eta 0.01", so["max_test_seeds_3_4"], 5e-2)
    check_claim(0, "offline client-rounds at or above the clip", so["at_or_above_clip"] + sr["summary_offline"]["0.001"]["at_or_above_clip"], 0)
    check_claim(0, "unattacked or non-victim structural client-rounds",
                so["degenerate_unattacked"] + so["degenerate_nonvictim"], 0)
    cl = [(f, s, v) for f, per in sr["closed_loop"].items() for s, v in per.items() if v["at_or_above_clip"]]
    check_claim(1, "closed loop: only echo attackers reach the clip",
                int(all(s == "echo" and v["roles_at_clip"] == ["attacker"] for _, s, v in cl)), 0)
    check_claim(2, "closed loop: at most two clipped rounds per run", max(v["at_or_above_clip"] for _, _, v in cl), 0)


eb = load_json("effort_decomposition_bn.json")
if eb:
    assert eb.get("schema") == "fixed-readout-bn-split-v1", "not a parameter/statistics split result"
    check_claim(3, "BN split, seeds", len(eb["seeds"]), 1e-9)
    check_claim(12, "BN split, reported from round", eb["report_from_round"], 1e-9)
    for key, n in (("modules", 20), ("non_affine", 12), ("statistic_buffers", 60)):
        check_claim(n, "BN split, BatchNorm %s" % key, eb["batchnorm"][key], 1e-9)
    check_claim(0.1, "BN split, BatchNorm momentum", max(eb["batchnorm"]["momenta"]), 1e-12)
    check_claim(0, "BN split, zero-step total movement/full, max", eb["zero_step_part_over_move_full"]["move"], 1e-6)
    check_claim(0, "BN split, zero-step largest part/full, max", eb["zero_step_max_part_over_move_full"], 1e-6)
    F = eb["by_fraction"]
    check_claim(0.983, "BN split, offset/innovation at full", eb["offset_over_innov_full"][0], 5e-4)
    check_claim(0.576, "BN split, full movement/offset", F["full"]["move_over_offset"][0], 5e-4)
    check_claim(-0.261, "BN split, cos(full movement, offset)", F["full"]["cos_move_offset"][0], 5e-4)
    for lab, mv, cs, inn in (("0", 0.000, None, 0.983), ("0.01", 0.235, 0.084, 0.979), ("0.02", 0.433, 0.172, 0.985),
                             ("0.05", 0.815, 0.403, 0.999), ("0.1", 1.036, 0.512, 1.010), ("0.2", 0.990, 0.492, 0.991),
                             ("0.5", 1.005, 0.501, 1.001)):
        check_claim(mv, "BN split %s, movement/full" % lab, F[lab]["move_over_move_full"][0], 5e-4)
        check_claim(inn, "BN split %s, innovation/full" % lab, F[lab]["innov_over_innov_full"][0], 5e-4)
        if cs is not None:
            check_claim(cs, "BN split %s, cos with full movement" % lab, F[lab]["cos_move_full"][0], 5e-4)
    caps = [lab for lab in F if lab != "0"]
    check_claim(1.001, "BN split, smallest parameter share", min(F[l]["share_par"][0] for l in caps), 5e-4)
    check_claim(1.025, "BN split, largest parameter share", max(F[l]["share_par"][0] for l in caps), 5e-4)
    check_claim(-0.054, "BN split, lowest cos(statistics part, movement)", min(F[l]["cos_bn_move"][0] for l in caps), 5e-4)
    check_claim(0.002, "BN split, highest cos(statistics part, movement)", max(F[l]["cos_bn_move"][0] for l in caps), 5e-4)
    for lab, bn, rc, fw, fi in (("0.01", 0.080, 0.069, 0.077, 0.977), ("0.1", 0.452, 0.458, 0.338, 0.971),
                                ("0.2", 0.554, 0.463, 0.394, 0.971), ("0.5", 0.674, 0.513, 0.398, 0.968),
                                ("full", 0.759, 1.000, 0.404, 0.971)):
        check_claim(bn, "BN split %s, statistics-only displacement/full" % lab, F[lab]["bn_over_move_full"][0], 5e-4)
        check_claim(rc, "BN split %s, recalibrated cos with full" % lab, F[lab]["cos_recal_full"][0], 5e-4)
        check_claim(fw, "BN split %s, forward-only movement/full" % lab, F[lab]["fwd_over_move_full"][0], 5e-4)
        check_claim(fi, "BN split %s, forward-only innovation/full" % lab, F[lab]["innov_fwd_over_innov_full"][0], 5e-4)
    check_claim(0.918, "BN split 0.1, recalibrated movement/its full", F["0.1"]["recal_over_recal_full"][0], 5e-4)
    payload = {}
    steps_at = {"0.01": [], "0.02": []}
    for seed in eb["seeds"]:
        raw = load_json("effort_decomposition_bn_s%d.json" % seed)
        if raw is None:
            continue
        full = raw["victim_full_steps"]
        for cap in raw["caps"]:
            lab = "%g" % round(cap / full, 2)
            if lab in steps_at:
                steps_at[lab].append((cap, full))
            dist, cosine = [], []
            for row in raw["rounds"].values():
                k, f = row["by_steps"][str(cap)], row["by_steps"][str(full)]
                ckf = 0.0 if cap == 0 else k["cos_full"]["move"]
                cko = 0.0 if cap == 0 else k["cos_move_offset"]
                dist.append(np.sqrt(max(k["move"] ** 2 + f["move"] ** 2 - 2 * k["move"] * f["move"] * ckf, 0.0))
                            / row["innov_full"])
                b = row["offset"]
                cosine.append((b * b + b * f["move"] * f["cos_move_offset"] + b * k["move"] * cko
                               + k["move"] * f["move"] * ckf) / (k["innov"] * row["innov_full"]))
            payload.setdefault(lab, []).append((np.median(dist), np.median(cosine)))
    if len(payload.get("0", [])) == 3:
        P = {lab: np.mean(v, axis=0) for lab, v in payload.items()}
        caps_lab = [lab for lab in P if lab != "0"]
        check_claim(0.559, "BN split 0, distance to full-training innovation/its norm", P["0"][0], 5e-4)
        check_claim(0.842, "BN split 0, cos with full-training innovation", P["0"][1], 5e-4)
        check_claim(0.556, "BN split 0.01, distance to full-training innovation/its norm", P["0.01"][0], 5e-4)
        check_claim(0.841, "BN split 0.01, cos with full-training innovation", P["0.01"][1], 5e-4)
        check_claim(0.544, "BN split, smallest reduced-budget distance/norm", min(P[l][0] for l in caps_lab), 5e-4)
        check_claim(0.564, "BN split, largest reduced-budget distance/norm", max(P[l][0] for l in caps_lab), 5e-4)
        check_claim(0.839, "BN split, smallest reduced-budget cos", min(P[l][1] for l in caps_lab), 5e-4)
        check_claim(0.854, "BN split, largest reduced-budget cos", max(P[l][1] for l in caps_lab), 5e-4)
        check_claim(1, "BN split 0.01, fewest steps", min(c for c, _ in steps_at["0.01"]), 1e-9)
        check_claim(2, "BN split 0.01, most steps", max(c for c, _ in steps_at["0.01"]), 1e-9)
        check_claim(3, "BN split 0.02, fewest steps", min(c for c, _ in steps_at["0.02"]), 1e-9)
        check_claim(4, "BN split 0.02, most steps", max(c for c, _ in steps_at["0.02"]), 1e-9)
    higher = {}
    for seed in eb["seeds"]:
        old, new = load_json("effort_decomposition_s%d.json" % seed), load_json("effort_decomposition_bn_s%d.json" % seed)
        if old is None or new is None:
            continue
        full = str(new["victim_full_steps"])
        for cap in new["caps"][1:]:
            lab = "%g" % round(cap / new["victim_full_steps"], 2)
            m = [np.median([v["by_steps"][str(cap)]["move"] / v["by_steps"][full]["move"] for v in d["rounds"].values()])
                 for d in (old, new)]
            higher.setdefault(lab, []).append(m[0] > m[1])
    if all(len(v) == 3 for v in higher.values()):
        for lab, n in (("0.02", 3), ("0.05", 3), ("0.1", 0), ("0.2", 0), ("0.5", 0)):
            check_claim(n, "BN split %s, seeds whose released movement/full exceeds the corrected" % lab, sum(higher[lab]), 1e-9)
    for seed in eb["seeds"]:
        raw = load_json("effort_decomposition_bn_s%d.json" % seed)
        if raw is None:
            continue
        assert raw["schema"] == "fixed-readout-bn-split-v1"
        check_claim(24, "BN split rounds, seed %d" % seed, len(raw["rounds"]), 1e-9)
        for row in raw["rounds"].values():
            for steps, v in row["by_steps"].items():
                if int(steps) == 0:
                    continue
                b, d = row["offset"], v["move"]
                assert np.isclose(v["innov"] ** 2, b * b + d * d + 2 * b * d * v["cos_move_offset"], rtol=1e-5), \
                    "BN split norm decomposition fails"
                assert abs(v["share"]["par"] + v["share"]["bn"] - 1) < 1e-9, "Shapley shares do not sum to one"
                n, c = v["norm"], v["cos_move"]
                for p, q in (("shap_par", "shap_bn"), ("recal", "noise")):
                    assert abs(n[p] * c[p] + n[q] * c[q] - d) <= 1e-6 * d, "BN split parts do not add up"

rm = load_json("reduced_effort_summary_modes.json")
if rm:
    for tag, passes in (("fwd", 142), ("zero", 0)):
        r = rm[tag]
        check_claim(3, "reduced effort %s, seeds" % tag, r["n"], 1e-9)
        check_claim(0, "reduced effort %s, victim optimizer steps" % tag, r["victim_steps"], 1e-9)
        check_claim(142, "reduced effort %s, full steps" % tag, r["full_steps"], 1e-9)
        check_claim(passes, "reduced effort %s, forward passes" % tag, r["victim_forward_passes"], 1e-9)
    for tag, inn, sus, tpr, fpr, nb, kr, md in (("fwd", 0.906, 0.995, 0.958, 0.0093, 0.764, 0.339, 0.217),
                                                ("zero", 0.911, 0.999, 0.958, 0.0046, 0.734, 0.339, 0.216)):
        r = rm[tag]
        check_claim(inn, "reduced effort %s, innovation over cohort" % tag, r["innovation_over_cohort"][0], 5e-4)
        check_claim(sus, "reduced effort %s, auditor AUC" % tag, r["auc"]["suspicion"][0], 5e-4)
        check_claim(23 / 24, "reduced effort %s, auditor TPR at frozen tau" % tag, r["tpr"][0], 1e-9)
        check_claim(0, "reduced effort %s, TPR spread over seeds" % tag, r["tpr"][1], 1e-9)
        check_claim(fpr, "reduced effort %s, honest FPR" % tag, r["honest_fpr"][0], 5e-5)
        check_claim(nb, "reduced effort %s, norm bound AUC" % tag, r["auc"]["norm_bound"][0], 5e-4)
        check_claim(kr, "reduced effort %s, Krum AUC" % tag, r["auc"]["krum"][0], 5e-4)
        check_claim(md, "reduced effort %s, median AUC" % tag, r["auc"]["median_dev"][0], 5e-4)
    for frac in ("0.1", "0.5"):
        released = load_json("reduced_effort_summary.json")
        if released:
            entry = dict(rm["steps_f" + frac])
            assert entry.pop("mode") == "steps" and entry == released[frac], "modes summary step rows differ"

dr = load_json("reduced_effort_drivers.json")
if dr:
    assert dr.get("schema") == "reduced-effort-drivers-v3", "not a reduced-effort drivers result with the control"
    S, R = dr["summary"], dr["runs"]
    for tag in ("f0.1", "f0.5", "fwd", "zero"):
        check_claim(72, "drivers %s, attacked client-rounds" % tag, S[tag]["n_attacked"], 1e-9)
        check_claim(648, "drivers %s, honest client-rounds" % tag, S[tag]["n_honest"], 1e-9)
        for s in range(3):
            run = R["%s_s%d" % (tag, s)]
            assert abs(run["replayed"]["tpr"] * run["n_attacked"] - run["n_flagged"]) < 1e-9, "drivers replay counts"
    for tag, n in (("f0.1", 14), ("f0.5", 5), ("fwd", 69), ("zero", 69)):
        check_claim(n, "drivers %s, flagged attacked rounds" % tag, S[tag]["n_flagged"], 1e-9)
    check_claim(66, "drivers fwd, flagged rounds attained by self-replay", S["fwd"]["argmax_flagged"]["rep"], 1e-9)
    check_claim(64, "drivers zero, flagged rounds attained by self-replay", S["zero"]["argmax_flagged"]["rep"], 1e-9)
    check_claim(8, "drivers fwd+zero, flagged rounds attained by u1",
                S["fwd"]["argmax_flagged"]["u1"] + S["zero"]["argmax_flagged"]["u1"], 1e-9)
    no_rep = S["fwd"]["per_seed_flagged_without_rep"] + S["zero"]["per_seed_flagged_without_rep"]
    check_claim(5, "drivers fwd/zero, fewest rounds flagged without self-replay", min(no_rep), 1e-9)
    check_claim(11, "drivers fwd/zero, most rounds flagged without self-replay", max(no_rep), 1e-9)
    check_claim(5, "drivers fwd/zero, runs whose unflagged round is round 12",
                sum(R["%s_s%d" % (t, s)]["unflagged_rounds"] == [12] for t in ("fwd", "zero") for s in range(3)), 1e-9)
    r12 = [next(p for p in R["%s_s%d" % (t, s)]["per_round"] if p["round"] == 12) for t in ("fwd", "zero") for s in range(3)]
    check_claim(0.785, "drivers fwd/zero, smallest round-12 self-replay distance/cohort step",
                min(p["replay_dist"] for p in r12), 5e-4)
    check_claim(2.47, "drivers fwd/zero, largest round-12 z_rep", max(p["oriented"]["rep"] for p in r12), 5e-3)
    zr = S["fwd"]["per_seed_median_oriented_z"]["rep"] + S["zero"]["per_seed_median_oriented_z"]["rep"]
    check_claim(5.75, "drivers fwd/zero, lowest per-seed median z_rep", min(zr), 5e-3)
    check_claim(8.46, "drivers fwd/zero, highest per-seed median z_rep", max(zr), 5e-3)
    st = S["fwd"]["per_seed_raw"]["step_over_cohort_step"]["victim"] + S["zero"]["per_seed_raw"]["step_over_cohort_step"]["victim"]
    check_claim(0.27, "drivers fwd/zero, smallest payload step/cohort step", min(st), 5e-3)
    check_claim(0.35, "drivers fwd/zero, largest payload step/cohort step", max(st), 5e-3)
    hs = [x for t in S for x in S[t]["per_seed_raw"]["step_over_cohort_step"]["honest"]]
    check_claim(0.99, "drivers, smallest honest payload step/cohort step", min(hs), 5e-3)
    check_claim(1.01, "drivers, largest honest payload step/cohort step", max(hs), 5e-3)
    u1v = S["fwd"]["per_seed_raw"]["u1"]["victim"] + S["zero"]["per_seed_raw"]["u1"]["victim"]
    u1h = S["fwd"]["per_seed_raw"]["u1"]["honest"] + S["zero"]["per_seed_raw"]["u1"]["honest"]
    check_claim(0.96, "drivers fwd/zero, victim median u1", float(np.mean(u1v)), 5e-3)
    check_claim(0.81, "drivers fwd/zero, honest median u1", float(np.mean(u1h)), 5e-3)
    check_claim(0, "drivers f0.1, flagged rounds attained by self-replay", S["f0.1"]["argmax_flagged"]["rep"], 1e-9)
    check_claim(11, "drivers f0.1, flagged rounds attained by u1", S["f0.1"]["argmax_flagged"]["u1"], 1e-9)
    check_claim(3, "drivers f0.1, flagged rounds attained by g", S["f0.1"]["argmax_flagged"]["g"], 1e-9)
    z1 = S["f0.1"]["per_seed_median_oriented_z"]["rep"]
    check_claim(-0.40, "drivers f0.1, lowest per-seed median z_rep", min(z1), 5e-3)
    check_claim(0.66, "drivers f0.1, highest per-seed median z_rep", max(z1), 5e-3)
    s1 = S["f0.1"]["per_seed_raw"]["step_over_cohort_step"]["victim"]
    check_claim(1.04, "drivers f0.1, smallest payload step/cohort step", min(s1), 5e-3)
    check_claim(1.22, "drivers f0.1, largest payload step/cohort step", max(s1), 5e-3)
    r1 = S["f0.1"]["per_seed_raw"]["replay_dist_over_cohort_step"]["victim"]
    check_claim(0.93, "drivers f0.1, smallest self-replay distance/cohort step", min(r1), 5e-3)
    check_claim(1.04, "drivers f0.1, largest self-replay distance/cohort step", max(r1), 5e-3)
    for tag, n in (("f0.1", 3), ("f0.5", 2), ("fwd", 2), ("zero", 0)):
        check_claim(n, "drivers %s, rounds in which g alone exceeds tau" % tag, S[tag]["exceeds_tau_alone"]["g"], 1e-9)
    tol_auc = {"u1": 1e-3, "u2": 5e-3, "x": 1e-3}
    for tag, vals in (("f0.1", (0.867, 0.54, 0.782, 0.908, 0.483)), ("f0.5", (0.696, 0.55, 0.794, 0.874, 0.505)),
                      ("fwd", (0.985, 0.56, 0.901, 0.887, 0.986)), ("zero", (0.994, 0.57, 0.873, 0.912, 0.977))):
        for k, v in zip(("u1", "u2", "g", "x", "rep"), vals):
            check_claim(v, "drivers %s, single-statistic AUC of %s" % (tag, k), S[tag]["auc_single"][k][0],
                        tol_auc.get(k, 5e-4))
    gs = [R["%s_s%d" % (t, s)]["auc_single"]["g"] for t in S for s in range(3)]
    check_claim(0.564, "drivers, lowest per-seed single-statistic AUC of g", min(gs), 5e-4)
    check_claim(0.987, "drivers, highest per-seed single-statistic AUC of g", max(gs), 5e-4)
    M = {t: S[t]["magnitude"] for t in S}
    check_claim(12, "drivers, runs in which the victim's magnitude falls", sum(M[t]["n_victim_falls"] for t in M), 1e-9)
    check_claim(9, "drivers, runs in which the victim's fall is the largest of 10",
                sum(M[t]["n_victim_largest_fall"] for t in M), 1e-9)
    check_claim(7, "drivers, runs in which the victim's attacked magnitude is the smallest of 10",
                sum(M[t]["n_victim_smallest_attack"] for t in M), 1e-9)
    vch = [x for t in M for x in M[t]["per_seed_victim_change"]]
    check_claim(0.169, "drivers, largest relative fall of the victim's magnitude", 1 - np.exp(min(vch)), 5e-4)
    check_claim(0.016, "drivers, smallest relative fall of the victim's magnitude", 1 - np.exp(max(vch)), 5e-4)
    check_claim(-0.077, "drivers, most negative honest relative magnitude change",
                np.exp(min(M[t]["honest_change_range"][0] for t in M)) - 1, 5e-4)
    check_claim(0.092, "drivers, most positive honest relative magnitude change",
                np.exp(max(M[t]["honest_change_range"][1] for t in M)) - 1, 5e-4)
    for tag, v in (("fwd", 0.092), ("zero", 0.077), ("f0.1", 0.074), ("f0.5", 0.064)):
        check_claim(v, "drivers %s, victim's fall, mean log change over seeds" % tag, 1 - np.exp(M[tag]["victim_change"][0]), 5e-4)
    b2 = [M[t]["per_seed_victim_ratio"]["baseline"][2] for t in M]
    check_claim(0.932, "drivers, seed 2, lowest pre-attack magnitude ratio of client 0", min(b2), 5e-4)
    check_claim(0.978, "drivers, seed 2, highest pre-attack magnitude ratio of client 0", max(b2), 5e-4)
    fz = [x for t in ("fwd", "zero") for x in S[t]["per_seed_raw"]["absent_row_share_of_step"]["victim"]]
    check_claim(0.114, "drivers fwd/zero, smallest absent-row share of the squared step", min(fz), 5e-4)
    check_claim(0.206, "drivers fwd/zero, largest absent-row share of the squared step", max(fz), 5e-4)
    ft = [x for t in ("f0.1", "f0.5") for x in S[t]["per_seed_raw"]["absent_row_share_of_step"]["victim"]]
    check_claim(0.017, "drivers f0.1/f0.5, largest absent-row share of the squared step", max(ft), 5e-4)
    ho = [x for t in S for x in S[t]["per_seed_raw"]["absent_row_share_of_step"]["honest"]]
    check_claim(0.010, "drivers, smallest honest absent-row share of the squared step", min(ho), 5e-4)
    check_claim(0.014, "drivers, largest honest absent-row share of the squared step", max(ho), 5e-4)
    ar = [x for t in S for x in S[t]["per_seed_raw"]["absent_rows"]["victim"]]
    check_claim(8, "drivers, fewest median absent rows of the victim", min(ar), 1e-9)
    check_claim(9, "drivers, most median absent rows of the victim", max(ar), 1e-9)
    hr = [x for t in S for x in S[t]["per_seed_raw"]["replay_dist_over_cohort_step"]["honest"]]
    check_claim(0.99, "drivers, smallest honest self-replay distance/cohort step", min(hr), 5e-3)
    check_claim(1.04, "drivers, largest honest self-replay distance/cohort step", max(hr), 5e-3)
    Cn = dr["control"]
    CS = Cn["summary"]
    assert Cn["declared_classes_equal_to_every_run"] and len(Cn["compared_config_keys"]) == 8, "control not comparable"
    check_claim(72, "drivers control, client-0 rounds of the attack window", CS["n_attacked"], 1e-9)
    check_claim(0, "drivers control, flagged client-0 rounds", CS["n_flagged"], 1e-9)
    crel = [float(np.exp(x)) - 1 for x in CS["magnitude"]["per_seed_victim_change"]]
    check_claim(-0.007, "drivers control, seed 0 client-0 relative magnitude change", crel[0], 5e-4)
    check_claim(-0.050, "drivers control, seed 1 client-0 relative magnitude change", crel[1], 5e-4)
    check_claim(0.015, "drivers control, seed 2 client-0 relative magnitude change", crel[2], 5e-4)
    check_claim(2, "drivers control, smallest client-0 rank by fall", min(CS["magnitude"]["per_seed_victim_change_rank"]), 1e-9)
    check_claim(6, "drivers control, largest client-0 rank by fall", max(CS["magnitude"]["per_seed_victim_change_rank"]), 1e-9)
    for s, v in enumerate((0.255, 0.608, 0.696)):
        check_claim(v, "drivers control, seed %d single-statistic AUC of g" % s, Cn["runs"]["s%d" % s]["auc_single"]["g"], 5e-4)
    check_claim(0.519, "drivers control, mean single-statistic AUC of g", CS["auc_single"]["g"][0], 5e-4)
    check_claim(0.612, "drivers control, mean single-statistic AUC of x", CS["auc_single"]["x"][0], 1e-3)
    check_claim(0.409, "drivers control, mean single-statistic AUC of u1", CS["auc_single"]["u1"][0], 1e-3)
    check_claim(0.529, "drivers control, mean single-statistic AUC of rep", CS["auc_single"]["rep"][0], 5e-4)
    cs = CS["per_seed_raw"]["step_over_cohort_step"]["victim"]
    check_claim(0.95, "drivers control, smallest client-0 payload step/cohort step", min(cs), 5e-3)
    check_claim(1.03, "drivers control, largest client-0 payload step/cohort step", max(cs), 5e-3)
    cr = CS["per_seed_raw"]["replay_dist_over_cohort_step"]["victim"]
    check_claim(0.97, "drivers control, smallest client-0 self-replay distance/cohort step", min(cr), 5e-3)
    check_claim(1.08, "drivers control, largest client-0 self-replay distance/cohort step", max(cr), 5e-3)
    check_claim(0.020, "drivers control, largest client-0 absent-row share of the squared step",
                max(CS["per_seed_raw"]["absent_row_share_of_step"]["victim"]), 5e-4)
    CC = Cn["comparison"]
    check_claim(12, "drivers, runs in which the victim falls more than client 0 without attack",
                sum(CC[t]["n_victim_below_control"] for t in CC), 1e-9)
    check_claim(12, "drivers, runs in which the victim's AUC of g exceeds the control's",
                sum(a > b for t in CC for a, b in zip(CC[t]["per_seed_auc_g"], CC[t]["per_seed_control_auc_g"])), 1e-9)
    check_claim(12, "drivers, runs in which the victim's AUC of x exceeds the control's",
                sum(R["%s_s%d" % (t, s)]["auc_single"]["x"] > Cn["runs"]["s%d" % s]["auc_single"]["x"]
                    for t in CC for s in range(3)), 1e-9)
    dif = [x for t in CC for x in CC[t]["per_seed_difference"]]
    check_claim(0.009, "drivers, smallest excess fall of the victim over the control", 1 - np.exp(max(dif)), 5e-4)
    check_claim(0.125, "drivers, largest excess fall of the victim over the control", 1 - np.exp(min(dif)), 5e-4)
_ratio = {(t, s): load_json("reduced_effort_%s_s%d.json" % (t, s))["victim_innovation_over_cohort_median"]["median"]
          for t in ("zero", "fwd", "f0.1", "f0.5") for s in range(3)}
_gaps = [min(_ratio[("f0.1", s)], _ratio[("f0.5", s)]) - max(_ratio[("zero", s)], _ratio[("fwd", s)]) for s in range(3)]
check_claim(12, "no-step clients below both partially training clients, within-seed pairs",
            sum(_ratio[(a, s)] < _ratio[(b, s)] for s in range(3) for a in ("zero", "fwd") for b in ("f0.1", "f0.5")), 0)
check_claim(0.005, "smallest within-seed gap, no steps vs partial training", min(_gaps), 5e-4)
check_claim(0.040, "largest within-seed gap, no steps vs partial training", max(_gaps), 5e-4)
check_claim(0.911, "no-computation client innovation ratio, mean", float(np.mean([_ratio[("zero", s)] for s in range(3)])), 5e-4)
check_claim(0.906, "forward-only client innovation ratio, mean", float(np.mean([_ratio[("fwd", s)] for s in range(3)])), 5e-4)

if env := load_json("envelope_eval.json"):
    rows, test = env["summary"]["rows"], env["results"]["TEST"]
    cmp_ = env["summary"]["comparators"]
    new = [a for a in rows if rows[a]["family"] != "reference"]
    fam_ad = [a for a in new if rows[a]["family"] in ("a", "d")]
    raw_rules = ("krum", "median_dev", "trimmed_mean_dev", "bulyan")
    z_rules = tuple(c + "_z" for c in raw_rules)
    robust = z_rules + ("norm_bound_z",)
    env_k = lambda a, c, t="0.01": test["attacks"][a][c][t]["tpr"][0]
    env_auc = lambda a, c, kind="all": env["auc"][a][c][kind]["TEST"]["mean"]
    env_best = lambda a: max(env_k(a, c) for c in robust)
    check_claim(1, "envelope: main file written by the replay command, from the repository root",
                int(env["config"]["command"].startswith("python experiments/envelope_eval.py")
                    and "--load-scores" not in env["config"]["command"]), 0)
    check_claim(3.786, "envelope: frozen auditor threshold", env["thresholds"]["suspicion"]["0.01"], 5e-4)
    check_claim(28, "envelope: new attack variants", len(new), 0)
    check_claim(13, "envelope: aggregate-centred and copy variants", len(fam_ad), 0)
    stateful = ("ens_stateful", "ens_stateful_causal", "ens_stateful_burnin", "ens_all")
    for col, k in [("suspicion", 16), ("abl_full_nostruct", 16), ("ens_stateful", 12), ("ens_stateful_causal", 14),
                   ("ens_stateful_burnin", 7), ("ens_all", 13), ("ens_stateless", 29), ("krum_z", 32),
                   ("median_dev_z", 36), ("trimmed_mean_dev_z", 36), ("bulyan_z", 31), ("norm_bound_z", 17)]:
        check_claim(k, f"envelope: TEST honest flags /1200, {col}", test["honest"][col]["0.01"]["client_round"][0],
                    1 if col in stateful else 0)
    for col, k in [("suspicion", 13), ("ens_stateful", 10), ("ens_all", 7), ("ens_stateless", 16)]:
        check_claim(k, f"envelope: TEST honest flags /1200, {col} at its frozen 0.5% threshold",
                    test["honest"][col]["0.005"]["client_round"][0], 1 if col in stateful else 0)
    for at, k, auc in [("echo_eps_1e-05", 1200, 1.000), ("echo_eps_1", 1200, 1.000), ("agg_frozen_innov", 1150, 0.997),
                       ("drift_pos", 1200, 1.000), ("drift_neg", 1200, 1.000), ("lagged_6", 91, 0.783),
                       ("lagged_8", 117, 0.775), ("lagged_10", 225, 0.788), ("lag3_noise_0.25", 1124, 0.998),
                       ("lag3_noise_0.5", 462, 0.963), ("lag3_noise_1", 959, 0.990), ("lag3_noise_2", 1173, 0.999),
                       ("lag6_noise_0.25", 102, 0.784), ("lag6_noise_0.5", 235, 0.868), ("lag6_noise_1", 941, 0.991),
                       ("lag6_noise_2", 1175, 0.999), ("lag3_rot_0.5", 434, 0.967), ("peer_avg_2", 1200, 0.897),
                       ("rows_half_echo", 1200, 0.945), ("rows_half_lag3", 1200, 0.945), ("mosaic_3", 880, 0.864),
                       ("mosaic_9", 59, 0.490)]:
        check_claim(k, f"envelope: TEST auditor flags /1200, {at}", env_k(at, "suspicion"), 0)
        check_claim(auc, f"envelope: TEST auditor AUC, {at}", rows[at]["auc_auditor"], 5e-4)
    for at, k in [("lagged_6", 2), ("lagged_8", 8), ("lagged_10", 31), ("mosaic_9", 50)]:
        check_claim(k, f"envelope: TEST runs flagged in the onset round /50, {at}",
                    test["attacks"][at]["suspicion"]["0.01"]["onset"][0], 0)
    m9 = test["attacks"]["mosaic_9"]["suspicion"]["0.01"]
    check_claim(9, "envelope: mosaic_9 auditor flags after the onset round", m9["tpr"][0] - m9["onset"][0], 0)
    check_claim(1150, "envelope: mosaic_9 attacked client-rounds after the onset round", m9["tpr"][1] - m9["onset"][1], 0)
    check_claim(9, "envelope: mosaic_9 runs flagged at least twice /50", m9["run_ge2"][0], 0)
    check_claim(50, "envelope: mosaic_9 auditor flags triggered by u1", test["triggers"]["mosaic_9"].get("u1", 0), 0)
    for at, auc in [("peer_avg_2", 1.000), ("rows_half_echo", 1.000), ("rows_half_lag3", 1.000), ("mosaic_3", 0.984)]:
        check_claim(auc, f"envelope: TEST auditor AUC without copy sources, {at}", rows[at]["auc_auditor_excl_sources"], 5e-4)
        check_claim(1.0 if at != "mosaic_3" else 0.566, f"envelope: copy sources' flag rate, {at}",
                    rows[at]["flag_rate_sources"], 5e-4)
    check_claim(1, "envelope: rep-triggered auditor flags, lag-6 replay + noise (all four scales)",
                sum(test["triggers"][f"lag6_noise_{x}"].get("rep", 0) for x in ("0.25", "0.5", "1", "2")), 0)
    check_claim(771, "envelope: rep-triggered auditor flags, frozen innovation", test["triggers"]["agg_frozen_innov"]["rep"], 0)
    check_claim(1023, "envelope: global echo flags /1200 without the structural tier", env_k("global_echo", "abl_full_nostruct"), 0)
    check_claim(0.912, "envelope: global echo AUC without the structural tier", env_auc("global_echo", "abl_full_nostruct"), 5e-4)
    check_claim(28, "envelope: new attacks with identical flag counts with and without the structural tier",
                sum(env_k(a, "abl_full_nostruct") == env_k(a, "suspicion") for a in new), 0)
    groups = {"release": ["global_echo", "peer_copy", "lagged_self", "ar2_replay", "noise", "shrink"],
              "a": [a for a in new if rows[a]["family"] == "a"], "b": ["lagged_6", "lagged_8", "lagged_10"],
              "c lag-3 noise": [f"lag3_noise_{x}" for x in ("0.25", "0.5", "1", "2")],
              "c lag-6 noise": [f"lag6_noise_{x}" for x in ("0.25", "0.5", "1", "2")],
              "c rotation": [f"lag3_rot_{x}" for x in ("0.05", "0.2", "0.5", "1")],
              "d copies": ["peer_avg_2", "rows_half_echo", "rows_half_lag3"], "d mosaic 3": ["mosaic_3"],
              "d mosaic 9": ["mosaic_9"]}
    table = {
        "release": ((1200, 1200), (1023, 1200), (191, 1198), (105, 1199), (41, 1200), (3, 1200)),
        "a": ((1150, 1200), (1150, 1200), (265, 1199), (167, 1196), (396, 1200), (471, 1200)),
        "b": ((91, 225), (91, 225), (389, 482), (102, 281), (64, 228), (63, 219)),
        "c lag-3 noise": ((462, 1173), (462, 1173), (167, 1200), (116, 1200), (46, 1200), (46, 1200)),
        "c lag-6 noise": ((102, 1175), (102, 1175), (253, 1200), (117, 1200), (74, 1200), (75, 1200)),
        "c rotation": ((434, 1200), (434, 1200), (152, 709), (106, 763), (39, 220), (38, 219)),
        "d copies": ((1200, 1200), (1200, 1200), (68, 193), (170, 390), (1198, 1200), (7, 1163)),
        "d mosaic 3": ((880, 880), (880, 880), (221, 221), (737, 737), (721, 721), (0, 0)),
        "d mosaic 9": ((59, 59), (59, 59), (222, 222), (737, 737), (500, 500), (0, 0))}
    cols = [("suspicion", 0), ("abl_full_nostruct", 0), ("ens_stateful_causal", 3), ("ens_stateful_burnin", 3),
            ("ens_stateless", 0), ("best robust rule", 0)]
    for g, cells in table.items():
        for (col, tol), (lo, hi) in zip(cols, cells):
            ks = [env_best(a) if col == "best robust rule" else env_k(a, col) for a in groups[g]]
            check_claim(lo, f"envelope table, {g}: {col}, fewest flags /1200", min(ks), tol)
            check_claim(hi, f"envelope table, {g}: {col}, most flags /1200", max(ks), tol)
    for g, lo, hi in [("release", 0.944, 1.000), ("a", 0.997, 1.000), ("b", 0.775, 0.788), ("c lag-3 noise", 0.963, 0.999),
                      ("c lag-6 noise", 0.784, 0.999), ("c rotation", 0.967, 1.000), ("d copies", 0.897, 0.945)]:
        check_claim(lo, f"envelope table, {g}: lowest auditor AUC", min(env_auc(a, "suspicion") for a in groups[g]), 5e-4)
        check_claim(hi, f"envelope table, {g}: highest auditor AUC", max(env_auc(a, "suspicion") for a in groups[g]), 5e-4)
    for col, gt, lt in [("ens_stateful", 11, 17), ("ens_stateful_causal", 11, 17), ("ens_stateful_burnin", 11, 17),
                        ("ens_all", 11, 13), ("ens_stateless", 4, 18)]:
        check_claim(gt, f"envelope: new attacks with {col} TPR > auditor",
                    sum(env_k(a, col) > env_k(a, "suspicion") for a in new), 0)
        check_claim(lt, f"envelope: new attacks with auditor TPR > {col}",
                    sum(env_k(a, col) < env_k(a, "suspicion") for a in new), 0)
        check_claim(lt, f"envelope: of these, still higher with the auditor at its frozen 0.5% threshold, vs {col}",
                    sum(env_k(a, "suspicion", "0.005") > env_k(a, col) for a in new
                        if env_k(a, col) < env_k(a, "suspicion")),
                    1 if col == "ens_all" else 0)
        check_claim(gt, f"envelope: summary block, {col} TPR wins", cmp_["n_tpr_gt_reference"][col], 0)
    for col, k in [("ens_stateful", 11), ("ens_all", 7), ("ens_stateless", 3)]:
        check_claim(k, f"envelope: new attacks with {col} at its 0.5% threshold above the auditor at 1%",
                    sum(env_k(a, col, "0.005") > env_k(a, "suspicion") for a in new), 0)
        check_claim(k, f"envelope: summary block, {col} at 0.5% still above the auditor",
                    cmp_["comparators_at_lower_target"]["columns"][col]["n_still_gt_reference"], 0)
    for col, k in [("ens_stateful", 15), ("ens_stateful_causal", 13), ("ens_stateful_burnin", 16), ("ens_all", 15)]:
        check_claim(k, f"envelope: new attacks with {col} AUC > auditor",
                    sum(env_auc(a, col) > env_auc(a, "suspicion") for a in new), 1)
    check_claim(8, "envelope: new attacks with ens_stateless AUC > auditor",
                sum(env_auc(a, "ens_stateless") > env_auc(a, "suspicion") for a in new), 0)
    d = 0.002
    for col, kind, gt, lt in [("ens_stateful", "all", 13, 10), ("ens_stateful", "excl_sources", 9, 13),
                              ("ens_stateful_causal", "all", 10, 14), ("ens_stateful_causal", "excl_sources", 8, 15),
                              ("ens_stateful_burnin", "all", 13, 10)]:
        xy = [(env_auc(a, col, kind), env_auc(a, "suspicion", kind)) for a in new]
        xy = [(x, y) for x, y in xy if x is not None and y is not None]
        check_claim(gt, f"envelope: new attacks with {col} AUC ({kind}) above the auditor's by > {d}",
                    sum(x - y > d for x, y in xy), 1)
        check_claim(lt, f"envelope: new attacks with auditor AUC ({kind}) above {col}'s by > {d}",
                    sum(y - x > d for x, y in xy), 1)
    for at, col, k in [("mosaic_9", "ens_stateful", 738), ("mosaic_9", "ens_stateful_causal", 222),
                       ("mosaic_9", "ens_stateful_burnin", 737), ("lagged_6", "ens_stateful", 318),
                       ("lagged_6", "ens_stateful_causal", 392), ("lagged_6", "ens_stateful_burnin", 102),
                       ("lagged_8", "ens_stateful_causal", 389), ("lagged_10", "ens_stateful", 430),
                       ("lagged_10", "ens_stateful_causal", 482), ("agg_frozen_innov", "ens_stateful", 1192),
                       ("lag6_noise_0.25", "ens_stateful", 237), ("lag3_noise_1", "ens_stateful", 1067),
                       ("lag3_noise_0.25", "ens_stateful", 190), ("lag3_rot_0.05", "ens_stateful", 276),
                       ("rows_half_lag3", "ens_stateful", 178)]:
        check_claim(k, f"envelope: TEST {col} flags /1200, {at}", env_k(at, col), 3)
    for at, col, auc in [("peer_avg_2", "ens_stateful", 0.977), ("rows_half_echo", "ens_stateful", 0.976),
                         ("mosaic_3", "ens_stateful", 0.982), ("mosaic_9", "ens_stateful", 0.981)]:
        check_claim(auc, f"envelope: TEST {col} AUC, {at}", env_auc(at, col), 2e-3)
    mm9 = test["members_ens_stateless"]["mosaic_9"]
    check_claim(500, "envelope: TEST memoryless-ensemble flags /1200, mosaic_9", env_k("mosaic_9", "ens_stateless"), 0)
    check_claim(258, "envelope: TEST memoryless-ensemble flags /1200 at its 0.5% threshold, mosaic_9",
                env_k("mosaic_9", "ens_stateless", "0.005"), 0)
    check_claim(0.977, "envelope: TEST memoryless-ensemble AUC, mosaic_9", env_auc("mosaic_9", "ens_stateless"), 5e-4)
    check_claim(500, "envelope: mosaic_9 memoryless flags carried by a negative cohort z",
                sum(v for key, v in mm9.items() if key.endswith("-")), 0)
    check_claim(440, "envelope: of these, by the negated cosine to the cohort mean", mm9.get("cosine_cohort_z-", 0), 3)
    check_claim(0, "envelope: memoryless flags whose largest member |z| differs from the ensemble",
                sum(m.get("mismatch", 0) for m in test["members_ens_stateless"].values()), 0)
    check_claim(0.423, "envelope: largest raw ranking-rule AUC over the aggregate-centred and copy variants (Krum, drift_pos)",
                max(env_auc(a, c) for a in fam_ad for c in raw_rules), 5e-4)
    check_claim(13, "envelope: aggregate-centred and copy variants with every raw ranking-rule AUC < 0.5",
                sum(max(env_auc(a, c) for c in raw_rules) < 0.5 for a in fam_ad), 0)
    near0 = lambda c: [a for a in fam_ad if env_auc(a, c) < 0.05]
    check_claim(5, "envelope: aggregate-centred and copy variants with raw Krum AUC < 0.05", len(near0("krum")), 0)
    check_claim(0.041, "envelope: largest raw Krum AUC among them (echo_eps_0.5)",
                max(env_auc(a, "krum") for a in near0("krum")), 5e-4)
    for c in ("median_dev", "trimmed_mean_dev", "bulyan"):
        check_claim(7, f"envelope: aggregate-centred and copy variants with raw {c} AUC < 0.05", len(near0(c)), 0)
        check_claim(1, f"envelope: {c} AUC < 0.05 on negative drift and on half rows + echo",
                    int("drift_neg" in near0(c) and "rows_half_echo" in near0(c)), 0)
    check_claim(0.022, "envelope: largest raw median / trimmed-mean / Bulyan AUC on those 7 variants (trimmed mean, drift_neg)",
                max(env_auc(a, c) for c in ("median_dev", "trimmed_mean_dev", "bulyan") for a in near0(c)), 5e-4)
    for at, lo, hi in [("drift_neg", 0.019, 0.022), ("rows_half_echo", 0.002, 0.011)]:
        cs = [env_auc(at, c) for c in ("median_dev", "trimmed_mean_dev", "bulyan")]
        check_claim(lo, f"envelope: smallest raw median / trimmed-mean / Bulyan AUC, {at}", min(cs), 5e-4)
        check_claim(hi, f"envelope: largest raw median / trimmed-mean / Bulyan AUC, {at}", max(cs), 5e-4)
    rest = [a for a in fam_ad if min(env_auc(a, c) for c in raw_rules) >= 0.05]
    check_claim(6, "envelope: aggregate-centred and copy variants with every raw ranking-rule AUC >= 0.05", len(rest), 0)
    check_claim(0.130, "envelope: smallest raw ranking-rule AUC on those 6 variants (Bulyan, echo_eps_1)",
                min(env_auc(a, c) for a in rest for c in raw_rules), 5e-4)
    check_claim(0.423, "envelope: largest raw ranking-rule AUC on those 6 variants (Krum, drift_pos)",
                max(env_auc(a, c) for a in rest for c in raw_rules), 5e-4)
    check_claim(0.459, "envelope: largest single-seed raw ranking-rule AUC over these variants",
                env["summary"]["ranking_inversion"]["raw"]["families_a_d"]["max_seed_auc"], 5e-4)
    check_claim(0.497, "envelope: smallest raw ranking-rule AUC over the replay families (b) and (c)",
                min(env_auc(a, c) for a in new if rows[a]["family"] in ("b", "c") for c in raw_rules), 5e-4)
    check_claim(0.396, "envelope: largest cohort-z ranking-rule AUC over the aggregate-centred and copy variants",
                max(env_auc(a, c) for a in fam_ad for c in z_rules), 5e-4)
    check_claim(0.423, "envelope: summary block, raw ranking-rule maximum",
                env["summary"]["ranking_inversion"]["raw"]["families_a_d"]["max_auc"], 5e-4)
    check_claim(0.879, "envelope: mosaic_9, raw x AUC (post-hoc diagnostic)", env_auc("mosaic_9", "raw_x"), 5e-4)
if ll := load_json("envelope_eval_longlags_w5.json"):
    check_claim(367, "W=5 lagged_12 (single clamped round): TEST auditor flags /1200",
                ll["results"]["TEST"]["attacks"]["lagged_12"]["suspicion"]["0.01"]["tpr"][0], 0)
    check_claim(0, "W=5 lagged_12: rep-triggered flags", ll["summary"]["rows"]["lagged_12"]["triggers"].get("rep", 0), 0)
if w34 := load_json("envelope_eval_w34.json"):
    t = w34["results"]["TEST"]
    check_claim(16, "W=34 sensitivity: TEST honest flags /1200", t["honest"]["suspicion"]["0.01"]["client_round"][0], 0)
    for at, k in [("lagged_6", 1200), ("lagged_8", 1200), ("lagged_10", 1200), ("lag3_noise_0.5", 936),
                  ("lag6_noise_0.5", 904), ("lag3_rot_0.5", 944), ("mosaic_9", 59)]:
        check_claim(k, f"W=34 sensitivity: TEST auditor flags /1200, {at}", t["attacks"][at]["suspicion"]["0.01"]["tpr"][0], 0)

if ast := load_json("auditor_statistics.json"):
    check_claim(1.0, "auditor statistics, all consistency checks pass", float(ast["all_checks_ok"]), 0)
    sel = ast["selection_best_single_stateful"]["cal_only"]
    check_claim(1.0, "CAL-only selection is tb_maha_pca_burnin_zabs",
                float(sel["selected"] == "tb_maha_pca_burnin_zabs"), 0)
    check_claim(0.792, "CAL-only selection, worst AUC on CAL seeds 0-1", sel["worst_auc"], 2e-3)
    F = ast["frozen_threshold"]
    TL = ("0.1", "0.05", "0.01", "0.005")
    for pool, det, tprs, tol in [
            ("TEST", "suspicion", (1.0, 1.0, 1.0, 1.0), 5e-4),
            ("TEST", "abl_full_nostruct", (0.917, 0.917, 0.817, 0.777), 5e-4),
            ("TEST", "ens_stateful", (0.742, 0.421, 0.144, 0.140), 2e-3),
            ("TEST", "tb_maha_pca_burnin_zabs", (0.596, 0.465, 0.158, 0.123), 2e-3),
            ("TEST", "ens_stateless", (0.269, 0.0875, 0.010, 0.006), 5e-4),
            ("TEST", "ens_all", (0.725, 0.385, 0.142, 0.046), 2e-3),
            ("TEST", "ens_stateful_x_rep", (0.969, 0.960, 0.771, 0.756), 2e-3),
            ("HELDOUT", "suspicion", (1.0, 1.0, 1.0, 1.0), 5e-4),
            ("HELDOUT", "abl_full_nostruct", (0.917, 0.917, 0.8525, 0.810), 5e-4),
            ("HELDOUT", "ens_stateful", (0.852, 0.584, 0.232, 0.225), 2e-3),
            ("HELDOUT", "ens_stateful_causal", (0.578, 0.371, 0.159, 0.124), 3e-3),
            ("HELDOUT", "ens_stateful_burnin", (0.602, 0.343, 0.0875, 0.0825), 2e-3),
            ("HELDOUT", "tb_maha_pca_burnin_zabs", (0.613, 0.5075, 0.187, 0.1375), 2e-3),
            ("HELDOUT", "ens_stateless", (0.370, 0.175, 0.034, 0.0175), 5e-4),
            ("HELDOUT", "ens_all", (0.842, 0.550, 0.228, 0.096), 2e-3),
            ("HELDOUT", "ens_stateful_x", (0.852, 0.584, 0.232, 0.225), 2e-3),
            ("HELDOUT", "ens_stateful_fg", (0.852, 0.584, 0.232, 0.225), 2e-3),
            ("HELDOUT", "ens_stateful_x_rep", (0.982, 0.974, 0.799, 0.782), 2e-3),
            ("HELDOUT", "ens_all_fg", (0.842, 0.550, 0.228, 0.096), 2e-3)]:
        ops = F[pool]["detectors"][det]["operating_points"]
        for t, v in zip(TL, tprs):
            check_claim(v, f"worst-case detection {pool}/{det} at nominal {t}", ops[t]["worst_tpr"], tol)
    for pool, det, fprs, tol in [("TEST", "suspicion", (0.1458, 0.0604, 0.0104, 0.0083), 5e-5),
                                 ("HELDOUT", "suspicion", (0.1342, 0.0592, 0.0133, 0.0108), 5e-5),
                                 ("HELDOUT", "ens_stateful", (0.1233, 0.0400, 0.0100, 0.0083), 2e-3)]:
        ops = F[pool]["detectors"][det]["operating_points"]
        for t, v in zip(TL, fprs):
            check_claim(v, f"clean false alarms {pool}/{det} at nominal {t}", ops[t]["clean_fpr"], tol)
    for det, auc, tol in [("suspicion", 0.944, 5e-4), ("abl_full_nostruct", 0.912, 5e-4),
                          ("ens_stateful", 0.948, 2e-3), ("ens_stateful_x_rep", 0.945, 2e-3),
                          ("tb_maha_pca_burnin_zabs", 0.839, 2e-3), ("ens_all", 0.940, 2e-3),
                          ("ens_stateless", 0.752, 5e-4), ("ens_stateful_causal", 0.835, 3e-3),
                          ("ens_stateful_burnin", 0.864, 2e-3)]:
        check_claim(auc, f"worst AUC on seeds 3-7, {det}", F["HELDOUT"]["detectors"][det]["worst_auc"], tol)
    check_claim(0.921, "worst AUC on seeds 3-4, ens_stateful", F["TEST"]["detectors"]["ens_stateful"]["worst_auc"], 2e-3)
    B = ast["bootstrap"]["pools"]["HELDOUT"]
    d = B["seed"]["ens_stateful"]["auditor_minus_this/worst_auc"]
    check_claim(-0.003, "seed bootstrap, auditor minus stateful ensemble, worst AUC", d["point"], 2e-3)
    check_claim(-0.021, "seed bootstrap, AUC difference, 2.5% bound", d["lo"], 5e-3)
    check_claim(0.017, "seed bootstrap, AUC difference, 97.5% bound", d["hi"], 5e-3)
    d = B["seed"]["ens_stateful"]["auditor_minus_this/worst_tpr@0.01"]
    check_claim(0.768, "seed bootstrap, detection difference at 1%", d["point"], 2e-3)
    check_claim(0.704, "seed bootstrap, detection difference at 1%, 2.5% bound", d["lo"], 5e-3)
    check_claim(0.833, "seed bootstrap, detection difference at 1%, 97.5% bound", d["hi"], 5e-3)
    d = B["seat"]["ens_stateful"]["auditor_minus_this/worst_auc"]
    check_claim(-0.014, "trajectory bootstrap, AUC difference, 2.5% bound", d["lo"], 5e-3)
    check_claim(0.009, "trajectory bootstrap, AUC difference, 97.5% bound", d["hi"], 5e-3)
    c = B["seed"]["suspicion"]["clean_fpr@0.01"]
    check_claim(0.0075, "seed bootstrap, auditor clean false alarms at 1%, 2.5% bound", c["lo"], 1e-3)
    check_claim(0.020, "seed bootstrap, auditor clean false alarms at 1%, 97.5% bound", c["hi"], 1e-3)
    TR = ast["trajectories"]
    for pool, k1, k2, n_tr in [("TEST", 5, 0, 20), ("HELDOUT", 14, 2, 50), ("FRESH", 9, 2, 30), ("DEV", 2, 0, 10)]:
        r = TR[pool]["suspicion"]["0.01"]
        check_claim(k1, f"clean trajectories flagged at least once, {pool}", r["k_of_24_rules_post_hoc"]["1"]["clean_trajectories_flagged"], 0)
        check_claim(k2, f"clean trajectories flagged at least twice, {pool}", r["k_of_24_rules_post_hoc"]["2"]["clean_trajectories_flagged"], 0)
        check_claim(n_tr, f"clean trajectories, {pool}", r["k_of_24_rules_post_hoc"]["1"]["clean_trajectories"], 0)
        check_claim(0, f"clean trajectories flagged three times or more, {pool}", r["k_of_24_rules_post_hoc"]["3"]["clean_trajectories_flagged"], 0)
        for a, v in r["attacked"].items():
            check_claim(n_tr, f"attacked trajectories flagged in all 24 rounds, {pool}/{a}", v["all_rounds_flagged"], 0)
        k3 = r["k_of_24_rules_post_hoc"]["3"]
        check_claim(n_tr, f"copied honest clients flagged in all 24 rounds, {pool}",
                    r["copy_source"]["peer_copy"]["all_rounds_flagged"], 0)
        check_claim(54 * n_tr, f"honest trajectories in attacked runs, {pool}",
                    k3["honest_trajectories_in_attacked_runs"], 0)
    for pool, k1, k2, k3, u3 in [("TEST", 264, 36, 22, 2), ("HELDOUT", 658, 143, 54, 4),
                                 ("FRESH", 394, 107, 32, 2), ("DEV", 84, 17, 11, 1)]:
        r = TR[pool]["suspicion"]["0.01"]["k_of_24_rules_post_hoc"]
        for k, v in (("1", k1), ("2", k2), ("3", k3)):
            check_claim(v, f"honest trajectories in attacked runs flagged at least {k} times, {pool}",
                        r[k]["honest_trajectories_in_attacked_runs_flagged"], 0)
        check_claim(u3, f"uninvolved trajectories flagged three times or more, {pool}",
                    r["3"]["uninvolved_trajectories_flagged"], 0)
    for pool, one_sided, two_sided in [("TEST", 0.861, 0.832), ("HELDOUT", 0.942, 0.929)]:
        att = TR[pool]["suspicion"]["0.01"]["k_of_24_rules_post_hoc"]["1"]["attacked"]
        check_claim(one_sided, f"attacked trajectories flagged, exact one-sided 95% lower bound, {pool}",
                    min(v["one_sided_95_lower"] for v in att.values()), 5e-4)
        check_claim(two_sided, f"attacked trajectories flagged, exact two-sided 95% interval, lower end, {pool}",
                    min(v["ci"][0] for v in att.values()), 5e-4)
    c = TR["HELDOUT"]["suspicion"]["0.01"]["k_of_24_rules_post_hoc"]["1"]["clean_ci"]
    check_claim(0.162, "clean trajectories flagged at least once, seeds 3-7, 95% interval, lower end", c[0], 5e-4)
    check_claim(0.425, "clean trajectories flagged at least once, seeds 3-7, 95% interval, upper end", c[1], 5e-4)
    ps = TR["HELDOUT"]["suspicion"]["0.01"]["clean"]["flagged_rounds_per_seed_and_seat"]
    for s, v in (("3", 1), ("4", 4), ("5", 6), ("6", 3), ("7", 2)):
        check_claim(v, f"auditor clean false alarms at 1%, seed {s} (of 240)", sum(ps[s]), 0)
    B2 = B["two_stage"]
    c = B2["suspicion"]["clean_fpr@0.01"]
    check_claim(0.0058, "two-stage bootstrap, auditor clean false alarms at 1%, 2.5% bound", c["lo"], 1e-3)
    check_claim(0.0225, "two-stage bootstrap, auditor clean false alarms at 1%, 97.5% bound", c["hi"], 1e-3)
    c = ast["bootstrap"]["pools"]["TEST"]["seat"]["suspicion"]["clean_fpr@0.01"]
    check_claim(0.0042, "trajectory bootstrap, auditor clean false alarms at 1%, seeds 3-4, 2.5% bound", c["lo"], 1e-3)
    check_claim(0.0188, "trajectory bootstrap, auditor clean false alarms at 1%, seeds 3-4, 97.5% bound", c["hi"], 1e-3)
    d = B2["ens_stateful"]["auditor_minus_this/worst_auc"]
    check_claim(-0.024, "two-stage bootstrap, auditor minus stateful ensemble, worst AUC, 2.5% bound", d["lo"], 5e-3)
    check_claim(0.020, "two-stage bootstrap, auditor minus stateful ensemble, worst AUC, 97.5% bound", d["hi"], 5e-3)
    d = B2["ens_stateful"]["auditor_minus_this/worst_tpr@0.01"]
    check_claim(0.70, "two-stage bootstrap, detection difference at 1%, 2.5% bound", d["lo"], 5e-3)
    check_claim(0.84, "two-stage bootstrap, detection difference at 1%, 97.5% bound", d["hi"], 5e-3)
    d = B2["ens_stateful_x_rep"]["auditor_minus_this/worst_tpr@0.01"]
    check_claim(0.201, "two-stage bootstrap, auditor minus stateful + x + rep, detection at 1%", d["point"], 2e-3)
    for pool, det, k, att, cln, hon, tol in [
            ("HELDOUT", "suspicion", "3", 300, 0, 54, 0), ("HELDOUT", "suspicion", "2", 300, 2, 143, 0),
            ("HELDOUT", "ens_stateful", "3", 292, 0, 1, 1), ("HELDOUT", "ens_stateful", "2", 299, 1, 40, 1),
            ("HELDOUT", "ens_stateful_x_rep", "3", 300, 0, 51, 1), ("HELDOUT", "ens_all", "3", 293, 0, 54, 1),
            ("TEST", "suspicion", "3", 120, 0, 22, 0), ("TEST", "ens_stateful", "3", 112, 0, 0, 1),
            ("FRESH", "ens_stateful", "3", 180, 0, 1, 1)]:
        r = TR[pool][det]["0.01"]["k_of_24_rules_post_hoc"][k]
        check_claim(att, f"{det}, at least {k} of 24 flags, attacked trajectories reached, {pool}",
                    r["attacked_all_attacks"]["detected"], tol)
        check_claim(cln, f"{det}, at least {k} of 24 flags, clean trajectories reached, {pool}",
                    r["clean_trajectories_flagged"], tol)
        check_claim(hon, f"{det}, at least {k} of 24 flags, honest trajectories of attacked runs reached, {pool}",
                    r["honest_trajectories_in_attacked_runs_flagged"], tol)
    r = TR["HELDOUT"]["ens_stateful"]["0.01"]["k_of_24_rules_post_hoc"]["3"]["attacked"]
    check_claim(43, "stateful ensemble, at least 3 flags, lagged self-replay trajectories reached (of 50)",
                r["lagged_self"]["detected"], 1)
    check_claim(11, "stateful ensemble, lagged self-replay, third flag: median attacked-round index "
                    "(0 = first attacked round) over the 43 of 50 trajectories that receive three flags",
                r["lagged_self"]["round_of_kth_flag"]["median"], 1)
    check_claim(20, "stateful ensemble, lagged self-replay, third flag: latest attacked-round index "
                    "(0 = first attacked round) over the 43 of 50 trajectories that receive three flags",
                r["lagged_self"]["round_of_kth_flag"]["max"], 1)
    check_claim(1, "stateful ensemble, peer copy, trajectories that never receive three flags (of 50)",
                r["peer_copy"]["round_of_kth_flag"]["not_reached"], 1)
    for a, v in (("ar2_replay", 49), ("noise", 46), ("peer_copy", 31), ("global_echo", 5), ("shrink", 3),
                 ("lagged_self", 0)):
        check_claim(v, f"stateful ensemble, {a}: third flag in the third attacked round (earliest possible), of 50",
                    r[a]["round_of_kth_flag"]["at_earliest_possible_round"], 2)
    for a in ("global_echo", "shrink"):
        check_claim(3, f"stateful ensemble, {a}: third flag no later than attacked-round index 3 (fourth round)",
                    r[a]["round_of_kth_flag"]["max"], 1)
    for det, v, tol in (("suspicion", 300, 0), ("ens_stateful", 134, 2), ("ens_all", 246, 2),
                        ("ens_stateful_x_rep", 167, 2)):
        check_claim(v, f"{det}: attacked trajectories with the third flag in the third attacked round, seeds 3-7 (of 300)",
                    TR["HELDOUT"][det]["0.01"]["k_of_24_rules_post_hoc"]["3"]["attacked_all_attacks"]
                    ["kth_flag_at_earliest_possible_round"], tol)
    r = TR["HELDOUT"]["ens_stateful"]["0.01"]["k_of_24_rules_post_hoc"]
    for k, att, cln, hon, early in (("1", 300, 11, 433, 153), ("2", 299, 1, 40, 138)):
        check_claim(att, f"ens_stateful, at least {k} of 24 flags, attacked trajectories reached, HELDOUT",
                    r[k]["attacked_all_attacks"]["detected"], 1)
        check_claim(cln, f"ens_stateful, at least {k} of 24 flags, clean trajectories reached, HELDOUT",
                    r[k]["clean_trajectories_flagged"], 1)
        check_claim(hon, f"ens_stateful, at least {k} of 24 flags, honest trajectories of attacked runs reached, HELDOUT",
                    r[k]["honest_trajectories_in_attacked_runs_flagged"], 1)
        check_claim(early, f"ens_stateful, k-th flag ({k}) in attacked round {k} (earliest possible), HELDOUT, of 300",
                    r[k]["attacked_all_attacks"]["kth_flag_at_earliest_possible_round"], 2)
    so = ast["single_stateful_outputs"]["HELDOUT"]
    check_claim(144, "single stateful outputs evaluated at frozen thresholds", so["outputs"], 0)
    for t, v, clean in (("0.1", 0.613, 0.0908), ("0.05", 0.508, 0.0375), ("0.01", 0.267, 0.0258),
                        ("0.005", 0.237, 0.0175)):
        check_claim(v, f"best of the 144 single stateful outputs, worst-case detection at nominal {t}, seeds 3-7",
                    so["nominal"][t]["max_worst_tpr"], 2e-3)
        check_claim(clean, f"that output's realised clean false alarms at nominal {t}, seeds 3-7",
                    so["nominal"][t]["argmax_clean_fpr"], 2e-3)
    r = TR["HELDOUT"]["suspicion"]["0.01"]["k_of_24_rules_post_hoc"]["3"]["attacked"]
    check_claim(2, "auditor: third flag in the third attacked round (index 2) for every attacked trajectory",
                max(v["round_of_kth_flag"]["max"] for v in r.values()), 0)
    ns = ast["auditor_without_structural_tier"]
    check_claim(1023, "no structural tier: echo rounds detected at 1%, seeds 3-7 (of 1200)",
                sum(ns[s]["global_echo_detected_rounds"]["0.01"] for s in "34567"), 0)
    check_claim(631, "no structural tier: echo rounds detected at 1%, seeds 5-7 (of 720)",
                sum(ns[s]["global_echo_detected_rounds"]["0.01"] for s in "567"), 0)
    check_claim(50, "no structural tier: echo trajectories flagged at least once at 1%, seeds 3-7",
                sum(ns[s]["global_echo_trajectories_flagged"]["0.01"] for s in "34567"), 0)
    for s, v in (("3", 0.913), ("4", 0.911), ("5", 0.910), ("6", 0.909), ("7", 0.916)):
        check_claim(v, f"no structural tier: echo AUC, seed {s}", ns[s]["auc_global_echo"], 5e-4)
if es := load_json("eta_sensitivity.json"):
    check_claim(1.0, "eta sensitivity, all consistency checks pass", float(es["all_checks_ok"]), 0)
    c100, c10 = es["configs"]["cifar100"], es["configs"]["cifar10"]
    check_claim(0.01, "CIFAR-100 floor selected on CAL", c100["cal_selection"]["selected"], 1e-12)
    check_claim(0.3, "CIFAR-10 floor selected on CAL (largest grid value)", c10["cal_selection"]["selected"], 1e-12)
    check_claim(9, "CIFAR-100 floors refitted (the full CAL selection grid)", len(c100["grid"]), 0)
    check_claim(9, "CIFAR-10 floors refitted (the full CAL selection grid)", len(c10["grid"]), 0)
    for eta, tpr, fpr in [("1e-12", 0.9883, 0.0158), ("0.0001", 0.9883, 0.0158), ("0.0003", 0.9892, 0.0150),
                          ("0.001", 0.9892, 0.0142), ("0.003", 0.9942, 0.0125), ("0.01", 1.0, 0.0133),
                          ("0.03", 1.0, 0.0167), ("0.1", 1.0, 0.0158), ("0.3", 1.0, 0.0192)]:
        o = c100["grid"][eta]["pools"]["HELDOUT"]["auditor"]["0.01"]
        check_claim(tpr, f"CIFAR-100 eta={eta}: worst-case detection at 1%, seeds 3-7", o["worst_tpr"], 5e-5)
        check_claim(fpr, f"CIFAR-100 eta={eta}: clean false alarms at 1%, seeds 3-7", o["clean_fpr"], 5e-5)
        check_claim(0.9444, f"CIFAR-100 eta={eta}: worst AUC, seeds 0-4", c100["grid"][eta]["auc"]["seeds_0_4"]["worst"], 1e-4)
    check_claim(0.0192, "CIFAR-100 at the CIFAR-10 floor 0.3: clean false alarms at 1%, seeds 3-7 (23/1200)",
                c100["grid"]["0.3"]["pools"]["HELDOUT"]["auditor"]["0.01"]["clean_fpr"], 5e-5)
    for eta, tpr in [("1e-12", 0.304), ("0.001", 0.354), ("0.01", 0.533), ("0.1", 0.7875), ("0.3", 0.8417)]:
        o = c10["grid"][eta]["pools"]["DEV"]["auditor"]["0.01"]
        check_claim(tpr, f"CIFAR-10 eta={eta}: worst-case detection at 1%, DEV seed 2", o["worst_tpr"], 5e-4)
    check_claim(0.0042, "CIFAR-10 eta=0.3: clean false alarms at 1%, DEV", c10["grid"]["0.3"]["pools"]["DEV"]["auditor"]["0.01"]["clean_fpr"], 5e-5)
    check_claim(0.9438, "CIFAR-10 eta=0.3: worst AUC, seeds 0-2", c10["grid"]["0.3"]["auc"]["seeds_0_2"]["worst"], 1e-4)
if te10 := load_json("temporal_eval_cifar10.json"):
    check_claim(0.3, "CIFAR-10 temporal evaluation floor", te10["protocol"]["mad_floor"], 1e-12)
    check_claim(0.944, "CIFAR-10 auditor worst AUC (peer copy)", te10["verdict"]["ours_worst_auc"], 5e-4)
if ted := load_json("temporal_eval_cifar100_d01.json"):
    check_claim(0.01, "CIFAR-100 delta=0.1 temporal evaluation floor", ted["protocol"]["mad_floor"], 1e-12)
    check_claim(0.946, "CIFAR-100 delta=0.1 auditor worst AUC (peer copy)", ted["verdict"]["ours_worst_auc"], 5e-4)
if ec := load_json("eta_convention.json"):
    check_claim(1.0, "eta convention, all consistency checks pass", float(ec["all_checks_ok"]), 0)
    cc = ec["configuration_controls"]
    for g, old, new in [("cohort05", 0.8227, 0.8666), ("cohort20", 0.9737, 0.9737), ("cohort40", 0.9872, 0.9872),
                        ("late70", 0.9443, 0.9444), ("nopp", 0.9444, 0.9445), ("cifar10", 0.9302, 0.9346),
                        ("cifar100_d01", 0.9449, 0.9457), ("cifar100_d05", 0.9444, 0.9444)]:
        check_claim(old, f"auditor worst AUC at eta=1e-3, {g}", cc[g]["released_eta_1e-3"]["worst"], 1e-4)
        check_claim(new, f"auditor worst AUC at eta=0.01, {g}", cc[g]["eta_0.01"]["worst"], 1e-4)
    cd = ec["collusion_detail"]
    check_claim(0.563, "collusion five of ten, eta=1e-3", cd["0.001"]["5"]["auc_mean"], 5e-4)
    check_claim(0.564, "collusion five of ten, eta=0.01", cd["0.01"]["5"]["auc_mean"], 5e-4)
    check_claim(0.650, "collusion five of ten, eta=1e-3, duplicated honest client excluded",
                cd["0.001"]["5"]["auc_excl_duplicated_honest_mean"], 5e-4)
    check_claim(0.653, "collusion five of ten, eta=0.01, duplicated honest client excluded",
                cd["0.01"]["5"]["auc_excl_duplicated_honest_mean"], 5e-4)
    per = cd["0.01"]["5"]["per_archive"]
    check_claim(57, "collusion: the duplicating colluder, rounds flagged of 72",
                sum(p["frozen_threshold_1pct"]["per_attacker_rounds_flagged"]["9"] for p in per), 0)
    check_claim(8, "collusion: the four relabelling colluders, rounds flagged of 288",
                sum(p["frozen_threshold_1pct"]["per_attacker_rounds_flagged"][c] for p in per for c in "5678"), 0)
    check_claim(57, "collusion: the duplicated honest client, rounds flagged of 72",
                sum(p["frozen_threshold_1pct"]["duplicated_honest_rounds_flagged"] for p in per), 0)

if pl := load_json("fedtgp_pilot.json"):
    fp = pl["geometry"]["fedproto_cbar"]["exact_summary"]
    tg = pl["geometry"]["fedtgp_refit"]["exact_summary"]
    check_claim(14.20, "pilot exact: seat-0 ||c||, median rule, FedProto broadcast",
                fp["median_dev"]["seat0_centre_norm_median"], 0.005)
    check_claim(29.42, "pilot exact: seat-0 margin M_med, median rule, FedProto broadcast",
                fp["median_dev"]["seat0_m_med_median"], 0.005)
    check_claim(69.1, "pilot exact: seat-0 ||c||, median rule, re-fitted FedTGP broadcast",
                tg["median_dev"]["seat0_centre_norm_median"], 0.1)
    check_claim(-27.5, "pilot exact: seat-0 margin M_min, median rule, re-fitted FedTGP broadcast",
                tg["median_dev"]["seat0_m_min_median"], 0.1)
    for rule in ("median_dev", "trimmed_mean_dev", "bulyan_dev"):
        check_claim(1.0, f"pilot exact: share of (round, seat) pairs with M_min > 0, {rule}, FedProto broadcast",
                    fp[rule]["inverts_frac_round_seats"], 0)
        check_claim(0.0, f"pilot exact: share of (round, seat) pairs with M_min > 0, {rule}, FedTGP re-fit",
                    tg[rule]["inverts_frac_round_seats"], 0)
    v = pl["variants"]["fedtgp_refit"]
    for name, stat, auc in [("fedtgp_echo", "median_dev", 0.978), ("fedtgp_echo", "krum", 0.994),
                            ("fedtgp_offset_echo", "median_dev", 0.997),
                            ("fedtgp_oracle_echo", "median_dev", 0.000),
                            ("fedtgp_recent_offset_echo", "median_dev", 1.000),
                            ("fedtgp_affine_echo", "median_dev", 0.100),
                            ("fedtgp_affine_echo", "krum", 0.147),
                            ("fedtgp_affine_echo", "trimmed_mean_dev", 0.089),
                            ("fedtgp_affine_echo", "bulyan_dev", 0.100),
                            ("fedtgp_affine_echo", "suspicion", 0.996)]:
        check_claim(auc, f"pilot: {stat} AUC of {name}, re-fitted FedTGP broadcast",
                    v[name][stat]["auc"], 1e-3)
    check_claim(9.646, "pilot: mean rank of the plain echo, median rule (of 10)",
                v["fedtgp_echo"]["median_dev"]["rank_mean"], 1e-3)
    check_claim(1.625, "pilot: mean rank of the affine echo, median rule (of 10)",
                v["fedtgp_affine_echo"]["median_dev"]["rank_mean"], 1e-3)
    check_claim(0.775, "pilot: affine echo ranked first, median rule, share of 240 (victim, round) pairs",
                v["fedtgp_affine_echo"]["median_dev"]["rank1_frac"], 1e-3)
    check_claim(0.796, "pilot: affine echo ranked first, Krum, share of 240 (victim, round) pairs",
                v["fedtgp_affine_echo"]["krum"]["rank1_frac"], 1e-3)
    check_claim(0, "pilot: rounds with tied or undefined honest scores (any variant, any rule)",
                sum(e.get("n_rounds_with_ties", 0) + e.get("n_honest_undefined", 0)
                    for b in pl["variants"].values() for ent in b.values() for e in ent.values()), 0)

rev2_auc = {
    'hetero': {
        'krum': (0.000, 0.185, 0.581, 0.990, 0.677, 0.000),
        'median_dev': (0.000, 0.225, 0.638, 0.995, 0.651, 0.000),
        'trimmed_mean_dev': (0.000, 0.203, 0.634, 0.995, 0.666, 0.000),
        'norm_bound': (1.000, 0.440, 0.614, 0.989, 0.501, 1.000),
        'suspicion': (1.000, 0.945, 0.990, 0.998, 1.000, 1.000),
        'bulyan': (0.000, 0.270, 0.634, 0.994, 0.647, 0.000),
    },
    'fedtgp_lam1': {
        'krum': (0.300, 0.192, 0.710, 0.998, 1.000, 0.158),
        'median_dev': (0.304, 0.179, 0.793, 1.000, 1.000, 0.100),
        'trimmed_mean_dev': (0.293, 0.161, 0.795, 1.000, 1.000, 0.090),
        'norm_bound': (1.000, 0.434, 0.889, 0.949, 0.499, 1.000),
        'suspicion': (1.000, 0.842, 0.997, 0.998, 1.000, 1.000),
        'bulyan': (0.308, 0.223, 0.813, 1.000, 1.000, 0.126),
    },
    'fedtgp_lam10': {
        'krum': (0.349, 0.317, 0.828, 0.985, 0.992, 0.257),
        'median_dev': (0.199, 0.268, 0.888, 0.999, 0.999, 0.072),
        'trimmed_mean_dev': (0.205, 0.260, 0.891, 0.999, 1.000, 0.074),
        'norm_bound': (1.000, 0.432, 0.946, 0.978, 0.500, 1.000),
        'suspicion': (1.000, 0.861, 0.998, 0.997, 0.999, 1.000),
        'bulyan': (0.168, 0.266, 0.901, 0.999, 1.000, 0.061),
    },
    'fedtgp_lam01': {
        'krum': (0.904, 0.197, 0.479, 0.994, 1.000, 0.672),
        'median_dev': (0.952, 0.180, 0.508, 1.000, 1.000, 0.802),
        'trimmed_mean_dev': (0.951, 0.164, 0.512, 1.000, 1.000, 0.783),
        'norm_bound': (1.000, 0.408, 0.858, 0.959, 0.499, 1.000),
        'suspicion': (1.000, 0.913, 0.997, 0.998, 1.000, 1.000),
        'bulyan': (0.955, 0.237, 0.507, 1.000, 1.000, 0.769),
    },
}
for cfg, rows in rev2_auc.items():
    tag = {'hetero': 'hetero_fedproto'}.get(cfg, 'hetero_' + cfg)
    inn = [load_json(f'innovation_{cfg}_s{s}.json') for s in range(5)]
    bps = load_json('bulyan_per_seed_rev2.json')
    for rule, vals in rows.items():
        for atk, stated in zip(('global_echo', 'peer_copy', 'lagged_self', 'ar2_replay', 'noise', 'shrink'), vals):
            if rule == 'bulyan':
                got = bps['configs'][tag][atk]['auc_mean'] if bps else None
            else:
                got = float(np.mean([d['results'][atk][rule]['auc'] for d in inn])) if all(inn) else None
            check_claim(stated, f'{cfg}: five-seed mean AUC, {rule}, {atk}', got, 1e-3)
for cfg, v in {
    'hetero': {
        'fedtgp_echo|all|krum|auc': 0.000,
        'fedtgp_echo|all|krum|rank1': 1.000,
        'fedtgp_echo|all|median_dev|auc': 0.000,
        'fedtgp_echo|all|median_dev|rank1': 1.000,
        'fedtgp_echo|TEST|krum|auc': 0.000,
        'fedtgp_echo|TEST|krum|rank1': 1.000,
        'fedtgp_echo|TEST|median_dev|auc': 0.000,
        'fedtgp_echo|TEST|median_dev|rank1': 1.000,
        'fedtgp_offset_echo|all|krum|auc': 0.066,
        'fedtgp_offset_echo|all|krum|rank1': 0.869,
        'fedtgp_offset_echo|all|median_dev|auc': 0.012,
        'fedtgp_offset_echo|all|median_dev|rank1': 0.916,
        'fedtgp_offset_echo|TEST|krum|auc': 0.052,
        'fedtgp_offset_echo|TEST|krum|rank1': 0.896,
        'fedtgp_offset_echo|TEST|median_dev|auc': 0.008,
        'fedtgp_offset_echo|TEST|median_dev|rank1': 0.944,
        'fedtgp_recent_offset_echo|all|krum|auc': 0.383,
        'fedtgp_recent_offset_echo|all|krum|rank1': 0.237,
        'fedtgp_recent_offset_echo|all|median_dev|auc': 0.353,
        'fedtgp_recent_offset_echo|all|median_dev|rank1': 0.278,
        'fedtgp_recent_offset_echo|TEST|krum|auc': 0.365,
        'fedtgp_recent_offset_echo|TEST|krum|rank1': 0.250,
        'fedtgp_recent_offset_echo|TEST|median_dev|auc': 0.334,
        'fedtgp_recent_offset_echo|TEST|median_dev|rank1': 0.269,
        'fedtgp_affine_echo|all|krum|auc': 0.052,
        'fedtgp_affine_echo|all|krum|rank1': 0.988,
        'fedtgp_affine_echo|all|median_dev|auc': 0.006,
        'fedtgp_affine_echo|all|median_dev|rank1': 0.988,
        'fedtgp_affine_echo|TEST|krum|auc': 0.044,
        'fedtgp_affine_echo|TEST|krum|rank1': 1.000,
        'fedtgp_affine_echo|TEST|median_dev|auc': 0.004,
        'fedtgp_affine_echo|TEST|median_dev|rank1': 1.000,
        'fedtgp_echo|rounds_12_20|krum|auc': 0.000,
        'fedtgp_echo|rounds_12_20|krum|rank1': 1.000,
        'fedtgp_echo|rounds_12_20|median_dev|auc': 0.000,
        'fedtgp_echo|rounds_12_20|median_dev|rank1': 1.000,
        'fedtgp_echo|rounds_28_35|krum|auc': 0.000,
        'fedtgp_echo|rounds_28_35|krum|rank1': 1.000,
        'fedtgp_echo|rounds_28_35|median_dev|auc': 0.000,
        'fedtgp_echo|rounds_28_35|median_dev|rank1': 1.000,
        'fedtgp_affine_echo|rounds_12_20|krum|auc': 0.005,
        'fedtgp_affine_echo|rounds_12_20|krum|rank1': 0.998,
        'fedtgp_affine_echo|rounds_12_20|median_dev|auc': 0.001,
        'fedtgp_affine_echo|rounds_12_20|median_dev|rank1': 0.998,
        'fedtgp_affine_echo|rounds_28_35|krum|auc': 0.005,
        'fedtgp_affine_echo|rounds_28_35|krum|rank1': 0.980,
        'fedtgp_affine_echo|rounds_28_35|median_dev|auc': 0.003,
        'fedtgp_affine_echo|rounds_28_35|median_dev|rank1': 0.980,
        'exact|median_dev|all|share': 1.000,
        'exact|median_dev|12_20|share': 1.000,
        'exact|median_dev|28_35|share': 1.000,
    },
    'fedtgp_lam1': {
        'fedtgp_echo|all|krum|auc': 0.300,
        'fedtgp_echo|all|krum|rank1': 0.453,
        'fedtgp_echo|all|median_dev|auc': 0.304,
        'fedtgp_echo|all|median_dev|rank1': 0.412,
        'fedtgp_echo|TEST|krum|auc': 0.302,
        'fedtgp_echo|TEST|krum|rank1': 0.473,
        'fedtgp_echo|TEST|median_dev|auc': 0.334,
        'fedtgp_echo|TEST|median_dev|rank1': 0.404,
        'fedtgp_offset_echo|all|krum|auc': 0.307,
        'fedtgp_offset_echo|all|krum|rank1': 0.341,
        'fedtgp_offset_echo|all|median_dev|auc': 0.195,
        'fedtgp_offset_echo|all|median_dev|rank1': 0.398,
        'fedtgp_offset_echo|TEST|krum|auc': 0.301,
        'fedtgp_offset_echo|TEST|krum|rank1': 0.290,
        'fedtgp_offset_echo|TEST|median_dev|auc': 0.197,
        'fedtgp_offset_echo|TEST|median_dev|rank1': 0.325,
        'fedtgp_recent_offset_echo|all|krum|auc': 0.602,
        'fedtgp_recent_offset_echo|all|krum|rank1': 0.017,
        'fedtgp_recent_offset_echo|all|median_dev|auc': 0.639,
        'fedtgp_recent_offset_echo|all|median_dev|rank1': 0.020,
        'fedtgp_recent_offset_echo|TEST|krum|auc': 0.614,
        'fedtgp_recent_offset_echo|TEST|krum|rank1': 0.019,
        'fedtgp_recent_offset_echo|TEST|median_dev|auc': 0.655,
        'fedtgp_recent_offset_echo|TEST|median_dev|rank1': 0.021,
        'fedtgp_affine_echo|all|krum|auc': 0.335,
        'fedtgp_affine_echo|all|krum|rank1': 0.681,
        'fedtgp_affine_echo|all|median_dev|auc': 0.298,
        'fedtgp_affine_echo|all|median_dev|rank1': 0.598,
        'fedtgp_affine_echo|TEST|krum|auc': 0.338,
        'fedtgp_affine_echo|TEST|krum|rank1': 0.667,
        'fedtgp_affine_echo|TEST|median_dev|auc': 0.301,
        'fedtgp_affine_echo|TEST|median_dev|rank1': 0.569,
        'fedtgp_echo|rounds_12_20|krum|auc': 0.713,
        'fedtgp_echo|rounds_12_20|krum|rank1': 0.004,
        'fedtgp_echo|rounds_12_20|median_dev|auc': 0.799,
        'fedtgp_echo|rounds_12_20|median_dev|rank1': 0.000,
        'fedtgp_echo|rounds_28_35|krum|auc': 0.012,
        'fedtgp_echo|rounds_28_35|krum|rank1': 0.988,
        'fedtgp_echo|rounds_28_35|median_dev|auc': 0.011,
        'fedtgp_echo|rounds_28_35|median_dev|rank1': 0.955,
        'fedtgp_affine_echo|rounds_12_20|krum|auc': 0.087,
        'fedtgp_affine_echo|rounds_12_20|krum|rank1': 0.953,
        'fedtgp_affine_echo|rounds_12_20|median_dev|auc': 0.033,
        'fedtgp_affine_echo|rounds_12_20|median_dev|rank1': 0.942,
        'fedtgp_affine_echo|rounds_28_35|krum|auc': 0.453,
        'fedtgp_affine_echo|rounds_28_35|krum|rank1': 0.305,
        'fedtgp_affine_echo|rounds_28_35|median_dev|auc': 0.542,
        'fedtgp_affine_echo|rounds_28_35|median_dev|rank1': 0.170,
        'exact|median_dev|all|share': 0.412,
        'exact|median_dev|12_20|share': 0.000,
        'exact|median_dev|28_35|share': 0.955,
    },
    'fedtgp_lam10': {
        'fedtgp_echo|all|krum|auc': 0.349,
        'fedtgp_echo|all|krum|rank1': 0.595,
        'fedtgp_echo|all|median_dev|auc': 0.199,
        'fedtgp_echo|all|median_dev|rank1': 0.565,
        'fedtgp_echo|TEST|krum|auc': 0.341,
        'fedtgp_echo|TEST|krum|rank1': 0.571,
        'fedtgp_echo|TEST|median_dev|auc': 0.195,
        'fedtgp_echo|TEST|median_dev|rank1': 0.537,
        'fedtgp_offset_echo|all|krum|auc': 0.674,
        'fedtgp_offset_echo|all|krum|rank1': 0.002,
        'fedtgp_offset_echo|all|median_dev|auc': 0.806,
        'fedtgp_offset_echo|all|median_dev|rank1': 0.001,
        'fedtgp_offset_echo|TEST|krum|auc': 0.684,
        'fedtgp_offset_echo|TEST|krum|rank1': 0.000,
        'fedtgp_offset_echo|TEST|median_dev|auc': 0.819,
        'fedtgp_offset_echo|TEST|median_dev|rank1': 0.000,
        'fedtgp_recent_offset_echo|all|krum|auc': 0.453,
        'fedtgp_recent_offset_echo|all|krum|rank1': 0.292,
        'fedtgp_recent_offset_echo|all|median_dev|auc': 0.399,
        'fedtgp_recent_offset_echo|all|median_dev|rank1': 0.289,
        'fedtgp_recent_offset_echo|TEST|krum|auc': 0.464,
        'fedtgp_recent_offset_echo|TEST|krum|rank1': 0.263,
        'fedtgp_recent_offset_echo|TEST|median_dev|auc': 0.419,
        'fedtgp_recent_offset_echo|TEST|median_dev|rank1': 0.250,
        'fedtgp_affine_echo|all|krum|auc': 0.673,
        'fedtgp_affine_echo|all|krum|rank1': 0.169,
        'fedtgp_affine_echo|all|median_dev|auc': 0.719,
        'fedtgp_affine_echo|all|median_dev|rank1': 0.160,
        'fedtgp_affine_echo|TEST|krum|auc': 0.660,
        'fedtgp_affine_echo|TEST|krum|rank1': 0.179,
        'fedtgp_affine_echo|TEST|median_dev|auc': 0.714,
        'fedtgp_affine_echo|TEST|median_dev|rank1': 0.169,
        'fedtgp_echo|rounds_12_20|krum|auc': 0.486,
        'fedtgp_echo|rounds_12_20|krum|rank1': 0.073,
        'fedtgp_echo|rounds_12_20|median_dev|auc': 0.468,
        'fedtgp_echo|rounds_12_20|median_dev|rank1': 0.053,
        'fedtgp_echo|rounds_28_35|krum|auc': 0.028,
        'fedtgp_echo|rounds_28_35|krum|rank1': 0.988,
        'fedtgp_echo|rounds_28_35|median_dev|auc': 0.005,
        'fedtgp_echo|rounds_28_35|median_dev|rank1': 0.975,
        'fedtgp_affine_echo|rounds_12_20|krum|auc': 0.419,
        'fedtgp_affine_echo|rounds_12_20|krum|rank1': 0.449,
        'fedtgp_affine_echo|rounds_12_20|median_dev|auc': 0.428,
        'fedtgp_affine_echo|rounds_12_20|median_dev|rank1': 0.424,
        'fedtgp_affine_echo|rounds_28_35|krum|auc': 0.998,
        'fedtgp_affine_echo|rounds_28_35|krum|rank1': 0.000,
        'fedtgp_affine_echo|rounds_28_35|median_dev|auc': 1.000,
        'fedtgp_affine_echo|rounds_28_35|median_dev|rank1': 0.000,
        'exact|median_dev|all|share': 0.565,
        'exact|median_dev|12_20|share': 0.053,
        'exact|median_dev|28_35|share': 0.975,
    },
    'fedtgp_lam01': {
        'fedtgp_echo|all|krum|auc': 0.904,
        'fedtgp_echo|all|krum|rank1': 0.001,
        'fedtgp_echo|all|median_dev|auc': 0.952,
        'fedtgp_echo|all|median_dev|rank1': 0.000,
        'fedtgp_echo|TEST|krum|auc': 0.911,
        'fedtgp_echo|TEST|krum|rank1': 0.000,
        'fedtgp_echo|TEST|median_dev|auc': 0.965,
        'fedtgp_echo|TEST|median_dev|rank1': 0.000,
        'fedtgp_offset_echo|all|krum|auc': 0.647,
        'fedtgp_offset_echo|all|krum|rank1': 0.000,
        'fedtgp_offset_echo|all|median_dev|auc': 0.726,
        'fedtgp_offset_echo|all|median_dev|rank1': 0.000,
        'fedtgp_offset_echo|TEST|krum|auc': 0.658,
        'fedtgp_offset_echo|TEST|krum|rank1': 0.000,
        'fedtgp_offset_echo|TEST|median_dev|auc': 0.755,
        'fedtgp_offset_echo|TEST|median_dev|rank1': 0.000,
        'fedtgp_recent_offset_echo|all|krum|auc': 0.805,
        'fedtgp_recent_offset_echo|all|krum|rank1': 0.000,
        'fedtgp_recent_offset_echo|all|median_dev|auc': 0.895,
        'fedtgp_recent_offset_echo|all|median_dev|rank1': 0.000,
        'fedtgp_recent_offset_echo|TEST|krum|auc': 0.818,
        'fedtgp_recent_offset_echo|TEST|krum|rank1': 0.000,
        'fedtgp_recent_offset_echo|TEST|median_dev|auc': 0.912,
        'fedtgp_recent_offset_echo|TEST|median_dev|rank1': 0.000,
        'fedtgp_affine_echo|all|krum|auc': 0.110,
        'fedtgp_affine_echo|all|krum|rank1': 0.998,
        'fedtgp_affine_echo|all|median_dev|auc': 0.024,
        'fedtgp_affine_echo|all|median_dev|rank1': 0.990,
        'fedtgp_affine_echo|TEST|krum|auc': 0.103,
        'fedtgp_affine_echo|TEST|krum|rank1': 1.000,
        'fedtgp_affine_echo|TEST|median_dev|auc': 0.015,
        'fedtgp_affine_echo|TEST|median_dev|rank1': 0.998,
        'fedtgp_echo|rounds_12_20|krum|auc': 0.987,
        'fedtgp_echo|rounds_12_20|krum|rank1': 0.000,
        'fedtgp_echo|rounds_12_20|median_dev|auc': 1.000,
        'fedtgp_echo|rounds_12_20|median_dev|rank1': 0.000,
        'fedtgp_echo|rounds_28_35|krum|auc': 0.704,
        'fedtgp_echo|rounds_28_35|krum|rank1': 0.003,
        'fedtgp_echo|rounds_28_35|median_dev|auc': 0.809,
        'fedtgp_echo|rounds_28_35|median_dev|rank1': 0.000,
        'fedtgp_affine_echo|rounds_12_20|krum|auc': 0.028,
        'fedtgp_affine_echo|rounds_12_20|krum|rank1': 1.000,
        'fedtgp_affine_echo|rounds_12_20|median_dev|auc': 0.001,
        'fedtgp_affine_echo|rounds_12_20|median_dev|rank1': 1.000,
        'fedtgp_affine_echo|rounds_28_35|krum|auc': 0.003,
        'fedtgp_affine_echo|rounds_28_35|krum|rank1': 0.995,
        'fedtgp_affine_echo|rounds_28_35|median_dev|auc': 0.005,
        'fedtgp_affine_echo|rounds_28_35|median_dev|rank1': 0.970,
        'exact|median_dev|all|share': 0.000,
        'exact|median_dev|12_20|share': 0.000,
        'exact|median_dev|28_35|share': 0.000,
    },
}.items():
    if fe := load_json(f'fedtgp_echo_{cfg}.json'):
        tf = fe['threshold_free']
        for key, stated in v.items():
            a, b, c, d = key.split('|')
            if a == 'exact':
                geo = [fe['geometry'][str(s)] for s in range(5)]
                if c == 'all':
                    got = float(np.mean([g['exact_summary'][b]['inverts_frac_round_seats'] for g in geo]))
                else:
                    lo, hi = map(int, c.split('_'))
                    got = float(np.mean([x[f'exact_{b}_inverts_frac_seats'] for g in geo
                                         for x in g['exact_per_round'] if lo <= x['round'] <= hi]))
                check_claim(stated, f'{cfg}: exact condition (zero update ranked first), {b}, rounds {c}, share of (round, seat) pairs', got, 1e-3)
                continue
            blk = (tf[a]['all_seeds']['pooled'] if b == 'all' else tf[a]['TEST'] if b == 'TEST'
                   else tf[a]['windows'][b]['pooled'])[c]
            got = blk['auc_mean_over_seeds'] if d == 'auc' else blk['rank1_frac']
            check_claim(stated, f'{cfg}: {a} {d}, {c}, seeds {b}', got, 1e-3)
for cfg, (honest, tprs, var_tprs) in {
    'hetero': (0.0167, {'global_echo': 1.0000, 'peer_copy': 1.0000, 'lagged_self': 1.0000, 'ar2_replay': 0.9646, 'noise': 1.0000, 'shrink': 1.0000}, {'fedtgp_echo': 1.0000, 'fedtgp_offset_echo': 0.9229, 'fedtgp_recent_offset_echo': 0.8417, 'fedtgp_affine_echo': 0.9792}),
    'fedtgp_lam1': (0.0104, {'global_echo': 1.0000, 'peer_copy': 0.0688, 'lagged_self': 0.8604, 'ar2_replay': 0.9208, 'noise': 0.9958, 'shrink': 1.0000}, {'fedtgp_echo': 1.0000, 'fedtgp_offset_echo': 0.2812, 'fedtgp_recent_offset_echo': 0.2458, 'fedtgp_affine_echo': 0.5958}),
    'fedtgp_lam10': (0.0167, {'global_echo': 1.0000, 'peer_copy': 0.2146, 'lagged_self': 0.9729, 'ar2_replay': 0.9313, 'noise': 1.0000, 'shrink': 1.0000}, {'fedtgp_echo': 1.0000, 'fedtgp_offset_echo': 0.3083, 'fedtgp_recent_offset_echo': 0.3104, 'fedtgp_affine_echo': 0.7771}),
    'fedtgp_lam01': (0.0104, {'global_echo': 1.0000, 'peer_copy': 0.7042, 'lagged_self': 0.9875, 'ar2_replay': 0.9812, 'noise': 1.0000, 'shrink': 1.0000}, {'fedtgp_echo': 1.0000, 'fedtgp_offset_echo': 0.6562, 'fedtgp_recent_offset_echo': 0.4833, 'fedtgp_affine_echo': 0.7229}),
}.items():
    if pe := load_json(f'protocol_eval_{cfg}.json'):
        e = pe['results']['TEST']['suspicion']['0.01']
        check_claim(honest, f'{cfg}: auditor honest client-round FPR at the frozen 1% threshold, TEST (480)',
                    e['honest']['client_round']['point'], 0.0025)
        for atk, tpr in tprs.items():
            check_claim(tpr, f'{cfg}: auditor TPR (client-round, TEST, 480) of {atk}', e['attacks'][atk]['tpr_client_round']['point'], 0.0025)
    if fe := load_json(f'fedtgp_echo_{cfg}.json'):
        e = fe['thresholded']['TEST']['suspicion']['0.01']
        for var, tpr in var_tprs.items():
            check_claim(tpr, f'{cfg}: auditor TPR (client-round, TEST, 480) of {var}', e['attacks'][var]['tpr_client_round']['point'], 0.0025)
if (kmr := load_json('krum_margin_rev2.json')) and (air := load_json('aggregate_impact_rev2.json')):
    for cfg, (sel, echo_zero, kept) in {
        'hetero_fedproto': (1.000, 1.000, (0.75, 0.84, 0.68)),
        'hetero_fedtgp_lam1': (0.350, 0.453, (0.89, 0.88, 0.85)),
        'hetero_fedtgp_lam10': (0.500, 0.595, (0.87, 0.88, 0.82)),
        'hetero_fedtgp_lam01': (0.000, 0.001, (0.91, 0.90, 0.95)),
    }.items():
        check_claim(sel, f'{cfg}: Krum selects the zero update at seat 0, share of 120 rounds', kmr['measured'][cfg]['selected_frac'], 1e-3)
        r = air['configs'][cfg]['rules']
        check_claim(echo_zero, f'{cfg}: Krum outputs the echo (zero update), share of 1200 seat-rounds', r['krum']['global_echo']['zero_output_frac'], 1e-3)
        for rule, x in zip(('median', 'trimmed_mean', 'bulyan'), kept):
            check_claim(x, f'{cfg}: step retained under global echo, {rule}', r[rule]['global_echo']['progress_own']['median'], 0.01)
_temporal = {
    'hetero': (0.944, 0.81, 0.93, 0.93, 0.524, 0.998, 0.524, 0.90),
    'fedtgp_lam1': (0.839, 0.79, 0.85, 0.82, 0.881, 0.879, 0.884, 0.85),
    'fedtgp_lam10': (0.879, 0.77, 0.84, 0.81, 0.894, 0.926, 0.943, 0.84),
    'fedtgp_lam01': (0.919, 0.74, 0.90, 0.84, 0.771, 0.969, 0.771, 0.86),
}
for cfg, (ours, stateful, ens_all, ens_sf, ens_sl, ours_x, ens_sl_x, ens_bi) in _temporal.items():
    if te := load_json(f'temporal_eval_{cfg}.json'):
        v = te['verdict']
        check_claim(ours, f'{cfg}: auditor worst-case AUC over the six attacks', v['ours_worst_auc'], 1e-3)
        check_claim(stateful, f'{cfg}: best stateful temporal baseline, worst-case AUC', v['best_stateful_baseline']['worst_auc'], 0.01)
        check_claim(ens_all, f'{cfg}: ensemble of all baselines, worst-case AUC', v['ensembles_worst_auc']['ens_all'], 0.01)
        check_claim(ens_bi, f'{cfg}: ensemble of the stateful baselines with burn-in fit, worst-case AUC', v['ensembles_worst_auc']['ens_stateful_burnin'], 0.01)
        check_claim(ens_sf, f'{cfg}: ensemble of the stateful baselines, worst-case AUC', v['ensembles_worst_auc']['ens_stateful'], 0.01)
        check_claim(ens_sl, f'{cfg}: ensemble of the stateless rules, worst-case AUC', v['ensembles_worst_auc']['ens_stateless'], 1e-3)
        check_claim(ours_x, f'{cfg}: auditor worst-case AUC, copy source excluded from the negatives', v['ours_worst_auc_excl_copy_source'], 1e-3)
        check_claim(ens_sl_x, f'{cfg}: stateless ensemble worst-case AUC, copy source excluded', v['ensembles_worst_auc_excl_copy_source']['ens_stateless'], 1e-3)
for cfg, cells in {
    'hetero': {'P|current|beta=1': (0.50, 0), 'P|delayed:1|beta=1': (0.54, 0), 'P|query:16|beta=1': (0.51, 0), 'P|decision:16|beta=1': (0.54, 0), 'P|feedback|beta=1': (0.54, 0), 'P|pseudo|beta=1': (0.88, 47), 'F|current|beta=1': (0.50, 0), 'F|delayed:1|beta=1': (0.64, 2), 'F|query:16|beta=1': (0.53, 0), 'F|decision:16|beta=1': (0.63, 0), 'F|feedback|beta=1': (0.61, 2), 'F|pseudo|beta=1': (0.96, 51)},
    'fedtgp_lam1': {'P|current|beta=1': (0.52, 0), 'P|delayed:1|beta=1': (0.55, 1), 'P|query:16|beta=1': (0.52, 0), 'P|decision:16|beta=1': (0.55, 0), 'P|feedback|beta=1': (0.54, 1), 'P|pseudo|beta=1': (0.92, 23), 'F|current|beta=1': (0.52, 0), 'F|delayed:1|beta=1': (0.56, 3), 'F|query:16|beta=1': (0.53, 0), 'F|decision:16|beta=1': (0.55, 0), 'F|feedback|beta=1': (0.55, 3), 'F|pseudo|beta=1': (0.96, 23)},
    'fedtgp_lam10': {'P|current|beta=1': (0.51, 0), 'P|delayed:1|beta=1': (0.56, 0), 'P|query:16|beta=1': (0.52, 0), 'P|decision:16|beta=1': (0.56, 0), 'P|feedback|beta=1': (0.55, 0), 'P|pseudo|beta=1': (0.82, 8), 'F|current|beta=1': (0.51, 0), 'F|delayed:1|beta=1': (0.55, 3), 'F|query:16|beta=1': (0.51, 0), 'F|decision:16|beta=1': (0.56, 0), 'F|feedback|beta=1': (0.54, 3), 'F|pseudo|beta=1': (0.84, 7)},
    'fedtgp_lam01': {'P|current|beta=1': (0.51, 0), 'P|delayed:1|beta=1': (0.59, 1), 'P|query:16|beta=1': (0.53, 0), 'P|decision:16|beta=1': (0.59, 0), 'P|feedback|beta=1': (0.59, 1), 'P|pseudo|beta=1': (0.97, 67), 'F|current|beta=1': (0.49, 0), 'F|delayed:1|beta=1': (0.62, 5), 'F|query:16|beta=1': (0.53, 0), 'F|decision:16|beta=1': (0.62, 2), 'F|feedback|beta=1': (0.60, 4), 'F|pseudo|beta=1': (0.99, 88)},
}.items():
    if la := load_json(f'limited_access_{cfg}.json'):
        for cell, (auc, det) in cells.items():
            check_claim(auc, f'{cfg}: limited access {cell} AUC (240 attacked, 2160 honest client-rounds)', la['cells'][cell]['auc'], 0.01)
            check_claim(det, f'{cfg}: limited access {cell} detected of 96 TEST attacked client-rounds', la['cells'][cell]['flagged_client_rounds_TEST'], 2)
for cfg, (floor, cells) in {
    'hetero': (0.1, {'P|current|beta=1': (0.48, 0), 'P|delayed:1|beta=1': (0.48, 0), 'P|query:16|beta=1': (0.49, 0), 'P|decision:16|beta=1': (0.48, 0), 'P|feedback|beta=1': (0.48, 0), 'P|pseudo|beta=1': (0.86, 45), 'F|current|beta=1': (0.50, 0), 'F|delayed:1|beta=1': (0.56, 2), 'F|query:16|beta=1': (0.51, 0), 'F|decision:16|beta=1': (0.55, 0), 'F|feedback|beta=1': (0.56, 2), 'F|pseudo|beta=1': (0.96, 59)}),
    'fedtgp_lam1': (0.0003, {'P|current|beta=1': (0.52, 0), 'P|delayed:1|beta=1': (0.56, 0), 'P|query:16|beta=1': (0.52, 0), 'P|decision:16|beta=1': (0.55, 0), 'P|feedback|beta=1': (0.54, 0), 'P|pseudo|beta=1': (0.91, 6), 'F|current|beta=1': (0.51, 0), 'F|delayed:1|beta=1': (0.55, 3), 'F|query:16|beta=1': (0.53, 0), 'F|decision:16|beta=1': (0.55, 0), 'F|feedback|beta=1': (0.54, 4), 'F|pseudo|beta=1': (0.96, 23)}),
    'fedtgp_lam01': (0.003, {'P|current|beta=1': (0.51, 0), 'P|delayed:1|beta=1': (0.58, 2), 'P|query:16|beta=1': (0.53, 0), 'P|decision:16|beta=1': (0.58, 0), 'P|feedback|beta=1': (0.58, 2), 'P|pseudo|beta=1': (0.97, 56), 'F|current|beta=1': (0.50, 0), 'F|delayed:1|beta=1': (0.62, 4), 'F|query:16|beta=1': (0.53, 0), 'F|decision:16|beta=1': (0.63, 0), 'F|feedback|beta=1': (0.61, 3), 'F|pseudo|beta=1': (0.99, 66)}),
}.items():
    if la := load_json(f'limited_access_{cfg}_calfloor.json'):
        check_claim(floor, f'{cfg}: limited access sensitivity run uses the CAL-selected MAD floor', la['config']['mad_floor'], 0)
        for cell, (auc, det) in cells.items():
            check_claim(auc, f'{cfg}: limited access (CAL-selected MAD floor) {cell} AUC', la['cells'][cell]['auc'], 0.01)
            check_claim(det, f'{cfg}: limited access (CAL-selected MAD floor) {cell} detected of 96 TEST attacked client-rounds', la['cells'][cell]['flagged_client_rounds_TEST'], 2)
for arm, cohorts in {
    'lam1': {'fedtgp_echo': (0.52, 0.71, 0.88), 'fedtgp_offset_echo': (0.65, 0.80, 1.09), 'fedtgp_recent_offset_echo': (0.62, 0.85, 1.05), 'fedtgp_affine_echo': (0.56, 0.88, 0.96), 'leave_one_out': (0.59, 0.94, None)},
    'lam10': {'fedtgp_echo': (0.54, 0.62, 0.92), 'fedtgp_offset_echo': (0.59, 0.91, 0.99), 'fedtgp_recent_offset_echo': (0.50, 0.88, 0.85), 'fedtgp_affine_echo': (0.53, 0.90, 0.91), 'leave_one_out': (0.59, 0.90, None)},
    'lam01': {'fedtgp_echo': (0.55, 0.93, 0.93), 'fedtgp_offset_echo': (0.68, 0.74, 1.18), 'fedtgp_recent_offset_echo': (0.75, 0.76, 1.28), 'fedtgp_affine_echo': (0.58, 0.86, 0.97), 'leave_one_out': (0.59, 0.96, None)},
}.items():
    if li := load_json(f'fedtgp_live_impact_{arm}.json'):
        for c, (shift, prog, rel) in cohorts.items():
            check_claim(shift, f'fedtgp_{arm}: broadcast step displacement / clean step, {c}, median of 240 seat-rounds', li['summary'][c]['shift_own']['median'], 0.01)
            check_claim(prog, f'fedtgp_{arm}: broadcast step retained, {c}, median of 240 seat-rounds', li['summary'][c]['progress_own']['median'], 0.01)
            if rel is not None:
                check_claim(rel, f'fedtgp_{arm}: displacement relative to removing the client, {c}, median of 240 seat-rounds', li['summary'][c]['shift_over_loo']['median'], 0.01)
        check_claim(0.0, f'fedtgp_{arm}: clean server re-fit vs archived broadcast, max |diff| over 120 seed-rounds',
                    li['meta']['clean_vs_archived_max_abs'], 0.3 if arm == 'lam01' else 0.1)
_geo = {
    'hetero': {'offset|12': 16.3, 'spread|12': 36.3, 'ratio|12': 0.44, 'offset|35': 17.4, 'spread|35': 40.6, 'ratio|35': 0.43},
    'fedtgp_lam1': {'offset|12': 53.6, 'spread|12': 40.5, 'ratio|12': 1.32, 'offset|35': 48.7, 'spread|35': 48.8, 'ratio|35': 1.01},
    'fedtgp_lam10': {'offset|12': 19.3, 'spread|12': 15.1, 'ratio|12': 1.27, 'offset|35': 21.8, 'spread|35': 21.5, 'ratio|35': 1.02},
    'fedtgp_lam01': {'offset|12': 121.9, 'spread|12': 79.0, 'ratio|12': 1.56, 'offset|35': 110.0, 'spread|35': 91.1, 'ratio|35': 1.20},
}
_geo_rows = {}
for cfg, ent in _geo.items():
    if fe := load_json(f'fedtgp_echo_{cfg}.json'):
        G = fe['geometry']
        rows = [(x['offset_q_minus_b_fro'], x['honest_spread_about_q_median'], e['exact_median_dev_inverts_frac_seats'], x['round'])
                for s in sorted(G, key=int) for x, e in zip(G[s]['per_round'], G[s]['exact_per_round'])]
        _geo_rows[cfg] = rows
        for key, stated in ent.items():
            what, r = key.split('|')
            sel = [x for x in rows if x[3] == int(r)]
            vals = {'offset': [x[0] for x in sel], 'spread': [x[1] for x in sel], 'ratio': [x[0] / x[1] for x in sel]}[what]
            check_claim(stated, f'{cfg}: {what} at round {r}, median over seeds 0-4', float(np.median(vals)),
                        0.01 if what == 'ratio' else 0.1)
if len(_geo_rows) == 4:
    _rs = np.asarray([o / s for c in ('fedtgp_lam1', 'fedtgp_lam10', 'fedtgp_lam01') for o, s, _, _ in _geo_rows[c]])
    _sh = np.asarray([f for c in ('fedtgp_lam1', 'fedtgp_lam10', 'fedtgp_lam01') for _, _, f, _ in _geo_rows[c]])
    check_claim(50, 'FedTGP arms: seed-rounds with offset/spread < 1.05', int((_rs < 1.05).sum()), 0)
    check_claim(0.994, 'FedTGP arms: mean share of seats with M_min > 0 where offset/spread < 1.05', float(_sh[_rs < 1.05].mean()), 1e-3)
    check_claim(176, 'FedTGP arms: seed-rounds with offset/spread >= 1.20', int((_rs >= 1.20).sum()), 0)
    check_claim(0.003, 'FedTGP arms: mean share of seats with M_min > 0 where offset/spread >= 1.20', float(_sh[_rs >= 1.20].mean()), 1e-3)
    check_claim(1.18, 'fedtgp_lam01: smallest offset/spread over seeds 0-4 and rounds 12-35', float(min(o / s for o, s, _, _ in _geo_rows['fedtgp_lam01'])), 0.01)
    check_claim(0.48, 'heterogeneous FedProto: largest offset/spread over seeds 0-4 and rounds 12-35', float(max(o / s for o, s, _, _ in _geo_rows['hetero'])), 0.01)
    check_claim(0.41, 'heterogeneous FedProto: smallest offset/spread over seeds 0-4 and rounds 12-35', float(min(o / s for o, s, _, _ in _geo_rows['hetero'])), 0.01)
    check_claim(0.96, 'FedTGP arms: smallest offset/spread over seeds 0-4 and rounds 12-35', float(_rs.min()), 0.01)
    check_claim(1.59, 'FedTGP arms: largest offset/spread over seeds 0-4 and rounds 12-35', float(_rs.max()), 0.01)
def _c14_rank(a):
    a = np.asarray(a, float)
    r = np.empty(len(a))
    r[np.argsort(a, kind='mergesort')] = np.arange(len(a))
    for v in np.unique(a):
        r[a == v] = r[a == v].mean()
    return r
if len(_geo_rows) == 4:
    _mid = (_rs >= 1.05) & (_rs < 1.20)
    check_claim(134, 'FedTGP arms: seed-rounds with 1.05 <= offset/spread < 1.20', int(_mid.sum()), 0)
    check_claim(0.500, 'FedTGP arms: mean share of seats with M_min > 0 where 1.05 <= offset/spread < 1.20', float(_sh[_mid].mean()), 1e-3)
    check_claim(0.0, 'FedTGP arms: smallest share in the band 1.05-1.20', float(_sh[_mid].min()), 0)
    check_claim(1.0, 'FedTGP arms: largest share in the band 1.05-1.20', float(_sh[_mid].max()), 0)
    _lo, _hi = float(_rs[_sh < 1].min()), float(_rs[_sh > 0].max())
    check_claim(1.043, 'FedTGP arms: smallest offset/spread of a seed-round with a seat not meeting M_min > 0', _lo, 1e-3)
    check_claim(1.219, 'FedTGP arms: largest offset/spread of a seed-round with a seat meeting M_min > 0', _hi, 1e-3)
    check_claim(42, 'FedTGP arms: seed-rounds below that overlap region (all seats meet the condition)', int((_rs < _lo).sum()), 0)
    check_claim(159, 'FedTGP arms: seed-rounds above that overlap region (no seat meets the condition)', int((_rs > _hi).sum()), 0)
    _btw = (_rs >= _lo) & (_rs <= _hi)
    check_claim(159, 'FedTGP arms: seed-rounds inside the overlap region', int(_btw.sum()), 0)
    check_claim(0.474, 'FedTGP arms: mean share of seats with M_min > 0 inside the overlap region', float(_sh[_btw].mean()), 1e-3)
    _r01 = [(o / s, f) for o, s, f, _ in _geo_rows['fedtgp_lam01'] if _lo <= o / s <= _hi]
    check_claim(9, 'fedtgp_lam01: seed-rounds inside the overlap region', len(_r01), 0)
    check_claim(0.0, 'fedtgp_lam01: largest share of seats with M_min > 0 inside the overlap region', max(f for _, f in _r01), 0)
    check_claim(1.177, 'fedtgp_lam01: smallest offset/spread inside the overlap region', min(r for r, _ in _r01), 1e-3)
    check_claim(1.213, 'fedtgp_lam01: largest offset/spread inside the overlap region', max(r for r, _ in _r01), 1e-3)
    for cfg, stated in {'fedtgp_lam1': -0.92, 'fedtgp_lam10': -0.85}.items():
        x = [o / s for o, s, _, _ in _geo_rows[cfg]]
        y = [f for _, _, f, _ in _geo_rows[cfg]]
        check_claim(stated, f'{cfg}: Spearman correlation of offset/spread and share of seats with M_min > 0, 120 seed-rounds',
                    float(np.corrcoef(_c14_rank(x), _c14_rank(y))[0, 1]), 0.01)
for cfg, (c_off, d_spr, rho_cd) in {'fedtgp_lam1': (0.85, 0.9, 0.95), 'fedtgp_lam10': (0.81, 0.89, 0.85), 'fedtgp_lam01': (0.9, 0.92, 0.96)}.items():
    if fe := load_json(f'fedtgp_echo_{cfg}.json'):
        G = fe['geometry']
        rr = [(x['offset_q_minus_b_fro'], x['honest_spread_about_q_median'], e['exact_median_dev_seat0_centre_norm'],
               e['exact_median_dev_seat0_m_min']) for s in sorted(G, key=int) for x, e in zip(G[s]['per_round'], G[s]['exact_per_round'])]
        check_claim(c_off, f'{cfg}: median over 120 seed-rounds of ||c|| / offset (seat 0, median rule)', float(np.median([c / o for o, s, c, m in rr])), 0.01)
        check_claim(d_spr, f'{cfg}: median over 120 seed-rounds of min_j ||I_j - c|| / spread (seat 0, median rule)', float(np.median([(c + m) / s for o, s, c, m in rr])), 0.01)
        check_claim(rho_cd, f'{cfg}: Spearman correlation of offset/spread and ||c|| / min_j ||I_j - c||, 120 seed-rounds',
                    float(np.corrcoef(_c14_rank([o / s for o, s, c, m in rr]), _c14_rank([c / (c + m) for o, s, c, m in rr]))[0, 1]), 0.01)
if la := load_json('limited_access_fedtgp_lam01.json'):
    check_claim(1.00, 'fedtgp_lam01: limited access F|delayed:1 attacked TEST runs flagged at least once (of 4)', la['cells']['F|delayed:1|beta=1']['tpr_run_TEST'], 0.25)
    check_claim(1.00, 'fedtgp_lam01: limited access F|feedback attacked TEST runs flagged at least once (of 4)', la['cells']['F|feedback|beta=1']['tpr_run_TEST'], 0.25)
    check_claim(0.15, 'fedtgp_lam01: limited access honest TEST runs flagged at least once by F (of 20)', la['calibration']['honest']['F']['fpr_run_TEST'], 0.05)
if fe := load_json('fedtgp_echo_fedtgp_lam01.json'):
    pr = fe['threshold_free']['fedtgp_echo']['per_round']
    for key, stated in {'krum|rank|27': 10.0, 'krum|rank|29': 9.5, 'krum|rank|34': 5.32, 'krum|rank|35': 4.22, 'median_dev|rank|29': 9.92, 'median_dev|rank|34': 6.44, 'median_dev|rank|35': 5.68, 'trimmed_mean_dev|rank|34': 6.48, 'trimmed_mean_dev|rank|35': 5.76, 'bulyan_dev|rank|34': 7.4, 'bulyan_dev|rank|35': 7.52, 'krum|rank1|35': 0.02, 'bulyan_dev|rank1|34': 0.08, 'bulyan_dev|rank1|35': 0.04}.items():
        rule, what, r = key.split('|')
        j = pr[rule]['rounds'].index(int(r))
        got = pr[rule]['rank_mean'][j] if what == 'rank' else pr[rule]['rank1_frac'][j]
        check_claim(stated, f'fedtgp_lam01: plain echo {what} under {rule} at round {r} (50 victim-seed pairs)', got, 0.01 if what == 'rank' else 1e-3)
    check_claim(0.080, 'fedtgp_lam01: largest per-round rank-1 share of the plain echo, any of the four rules, rounds 12-35', max(max(pr[r]['rank1_frac']) for r in ('krum', 'median_dev', 'trimmed_mean_dev', 'bulyan_dev')), 1e-3)
    _ps = pr['krum']['per_seed_rank_mean']
    for s, stated in enumerate([4.4, 4.5, 2.9, 4.8, 4.5]):
        got = (_ps[str(s)] if isinstance(_ps, dict) else _ps[s])[pr['krum']['rounds'].index(35)]
        check_claim(stated, f'fedtgp_lam01: plain echo mean Krum rank at round 35, seed {s}', got, 0.1)
for cfg, v in {
    'hetero': {'krum': (0.000, 1.000, 1.00), 'median_dev': (0.000, 1.000, 1.00), 'trimmed_mean_dev': (0.000, 1.000, 1.00), 'bulyan_dev': (0.000, 1.000, 1.00)},
    'fedtgp_lam1': {'krum': (0.300, 0.453, 4.15), 'median_dev': (0.304, 0.412, 4.24), 'trimmed_mean_dev': (0.293, 0.412, 4.14), 'bulyan_dev': (0.308, 0.419, 4.13)},
    'fedtgp_lam10': {'krum': (0.349, 0.595, 2.76), 'median_dev': (0.199, 0.565, 2.69), 'trimmed_mean_dev': (0.205, 0.572, 2.56), 'bulyan_dev': (0.168, 0.573, 2.60)},
    'fedtgp_lam01': {'krum': (0.904, 0.001, 9.24), 'median_dev': (0.952, 0.000, 9.48), 'trimmed_mean_dev': (0.951, 0.000, 9.47), 'bulyan_dev': (0.955, 0.005, 9.49)},
}.items():
    if fe := load_json(f'fedtgp_echo_{cfg}.json'):
        b = fe['threshold_free']['fedtgp_echo']['all_seeds']['pooled']
        for rule, (auc, r1, rk) in v.items():
            check_claim(auc, f'{cfg}: plain echo AUC, {rule}, seeds 0-4', b[rule]['auc_mean_over_seeds'], 1e-3)
            check_claim(r1, f'{cfg}: plain echo ranked first, {rule}, share of 1200 (victim, round) pairs', b[rule]['rank1_frac'], 1e-3)
            check_claim(rk, f'{cfg}: plain echo mean rank of 10, {rule}, seeds 0-4', b[rule]['rank_mean'], 0.01)
for cfg in ('hetero', 'fedtgp_lam1', 'fedtgp_lam10', 'fedtgp_lam01'):
    if fe := load_json(f'fedtgp_echo_{cfg}.json'):
        c = fe['meta']['checks']
        check_claim(0.0, f'{cfg}: echo replay honest pass = protocol_eval', c['honest_vs_protocol_eval_max_abs'], 0)
        check_claim(0.0, f'{cfg}: plain echo = protocol_eval global_echo', c['echo_vs_protocol_eval_global_echo_max_abs'], 0)
        check_claim(0.0, f'{cfg}: identity check, exact condition = echo rank-1 fraction, every seed and round',
                    c['exact_condition_vs_echo_rank1_max_abs'], 0)
if xc := load_json('rev2_auc_cross_check.json'):
    check_claim(1.0, 'revision archives: protocol_eval suspicion AUC = innovation_<tag>.json within 1e-5 (noise exempt)',
                float(xc['all_ok']), 0)
for cfg, (dl, fb) in {'hetero': (0.50, 0.50), 'fedtgp_lam1': (0.50, 0.50), 'fedtgp_lam10': (0.75, 0.75)}.items():
    if la := load_json(f'limited_access_{cfg}.json'):
        check_claim(dl, f'{cfg}: limited access F|delayed:1 attacked TEST runs flagged at least once (of 4)',
                    la['cells']['F|delayed:1|beta=1']['tpr_run_TEST'], 0.25)
        check_claim(fb, f'{cfg}: limited access F|feedback attacked TEST runs flagged at least once (of 4)',
                    la['cells']['F|feedback|beta=1']['tpr_run_TEST'], 0.25)
if la := load_json('limited_access_grid.json'):
    for cell, det in {'F|delayed:1|beta=1': 9, 'F|feedback|beta=1': 8, 'P|pseudo|beta=1': 77, 'F|pseudo|beta=1': 81}.items():
        check_claim(det, f'release: limited access {cell} detected of 96 TEST attacked client-rounds',
                    la['cells'][cell]['flagged_client_rounds_TEST'], 2)
for cfg, rules in {
    'hetero': {'krum': (0.0, 0.001), 'median_dev': (0.0, 0.0), 'trimmed_mean_dev': (0.0, 0.0)},
    'fedtgp_lam01': {'krum': (0.870, None), 'median_dev': (0.930, None), 'trimmed_mean_dev': (0.927, None)},
    'fedtgp_lam1': {'krum': (0.280, 0.330), 'median_dev': (0.269, 0.351), 'trimmed_mean_dev': (0.256, 0.342)},
    'fedtgp_lam10': {'krum': (0.328, 0.388), 'median_dev': (0.178, 0.219), 'trimmed_mean_dev': (0.179, 0.220)},
}.items():
    inn = [load_json(f'innovation_{cfg}_s{s}.json') for s in range(5)]
    if all(inn):
        for rule, (lo, hi) in rules.items():
            vals = [d['results']['global_echo'][rule]['auc'] for d in inn]
            check_claim(lo, f'{cfg}: global echo seed-level AUC, {rule}, minimum over seeds 0-4', min(vals), 1e-3)
            if hi is not None:
                check_claim(hi, f'{cfg}: global echo seed-level AUC, {rule}, maximum over seeds 0-4', max(vals), 1e-3)
if fe := load_json('fedtgp_echo_hetero.json'):
    pe = fe['honest_by_architecture']['TEST']['suspicion']['0.01']['per_encoder']
    check_claim(8, 'hetero: honest TEST false alarms on the CNN4 clients (of 96)', pe['cnn4']['k'], 0)
    check_claim(0, 'hetero: honest TEST false alarms on the other encoders (of 384)',
                sum(pe[e]['k'] for e in pe if e != 'cnn4'), 0)

if ic := load_json("inversion_condition.json"):
    check_claim(4, "Krum condition: early CIFAR-10 seat-rounds certified (of 300)",
                round(300 * ic["configs"]["cifar10"]["early"]["coverage_seat_rounds"]["krum_f2"]["krum_tight"]), 0)
if fp := load_json("frozen_encoder_prover.json"):
    v = fp["variants"]["untrained"]
    check_claim(0.014, "frozen random encoder: per-round movement of present-class prototypes",
                v["summary"]["medians_rounds_6_on"]["proto_velocity"] * 100 / v["shard"]["classes_present"], 5e-4)
if te := load_json("temporal_eval.json"):
    T = te["results"]["TEST"]
    for det, atk, stated in [("abl_no_x", "peer_copy", 5.6), ("abl_no_rep", "lagged_self", 6.0)]:
        check_claim(stated, f"feature ablation {det}: worst-case detection (%), seeds 3-4",
                    100 * T[det]["0.01"]["attacks"][atk]["tpr_client_round"]["point"], 5e-2)
    for det, atk, cost in [("abl_no_u1", "ar2_replay", 39.6), ("abl_no_u2", "ar2_replay", 4.2), ("abl_no_g", "shrink", 16.3)]:
        check_claim(cost, f"feature ablation {det}: lost detection of {atk} (percentage points), seeds 3-4",
                    100 * (1 - T[det]["0.01"]["attacks"][atk]["tpr_client_round"]["point"]), 5.01e-2)
if rd := load_json("reduced_effort_drivers.json"):
    for s, stated in zip((0, 1, 2), (0.25, 0.61, 0.70)):
        check_claim(stated, f"control client 0, AUC of g alone, seed {s} (two decimals)",
                    rd["control"]["runs"][f"s{s}"]["auc_single"]["g"], 5e-3)
if cc := load_json("adaptive_costcurve.json"):
    check_claim(0.992, "auditor-targeted attacker at beta 2: budget used",
                cc["cells"]["suspicion/mimic|peer_current|phi=0|beta=2"]["budget_used_mean"], 5e-4)
if cg := load_json("combined_grid.json"):
    check_claim(0.515, "fused-maximum attacker at beta 2: budget used",
                cg["cells"]["max|mimic|flat|peer_current|phi=0|beta=2"]["budget_used_mean"], 5e-4)
if fo := load_json("adaptive_fg_observer.json"):
    check_claim(1.000, "cohort-mean cosine on the crafted payloads at beta 0",
                fo["cells"]["suspicion/mimic|peer_current|phi=0|beta=0"]["stats"]["cosine_cohort"]["auc"], 5e-4)
if la := load_json("limited_access_grid.json"):
    for d, n in zip((1, 2, 3), (2, 3, 6)):
        check_claim(n, f"fused maximum, delayed by {d} at beta 2: flagged of 96 TEST attacked client-rounds",
                    la["cells"][f"F|delayed:{d}|beta=2"]["flagged_client_rounds_TEST"], 0)
if fb := load_json("faithful_baselines_cifar10.json"):
    check_claim(0.544, "FoolsGold on CIFAR-10 global echo", fb["summary"]["global_echo"]["foolsgold_mean"], 5e-4)
if sh := load_json("shard_summary.json"):
    check_claim(52, "archived client 0: classes with at least 20 images",
                sh["archived_fedproto_cifar100"]["client0"]["classes_with_min_images"], 0)

for arm, last, vals in [
        ('r60', 60, {'auc': {'krum': 0.528, 'median_dev': 0.528, 'trimmed_mean_dev': 0.534, 'bulyan_dev': 0.533},
                     'first_median': 690, 'n': 2450, 'win': ((1.000, 0.817)), 'ratio': (1.55, 1.00),
                     'affine': (0.208, 633, 980), 'auditor': (13, 980, 980, 495)}),
        ('euc', 35, {'auc': {'krum': 0.279, 'median_dev': 0.260, 'trimmed_mean_dev': 0.256, 'bulyan_dev': 0.260},
                     'first_median': 545, 'n': 1200, 'win': ((0.724, 0.010)), 'ratio': (1.30, 0.96),
                     'affine': (0.227, 325, 480), 'auditor': (5, 480, 480, 323)})]:
    cfg = f'fedtgp_lam01_{arm}'
    if fe := load_json(f'fedtgp_echo_{cfg}.json'):
        tf = fe['threshold_free']
        pooled = tf['fedtgp_echo']['all_seeds']['pooled']
        for rule, auc in vals['auc'].items():
            check_claim(auc, f'{cfg}: plain echo AUC, {rule}, seeds 0-4, rounds 12-{last}', pooled[rule]['auc_mean_over_seeds'], 5.01e-4)
        check_claim(vals['n'], f'{cfg}: victim-rounds pooled', pooled['median_dev']['n_rounds'], 0)
        check_claim(vals['first_median'], f'{cfg}: plain echo ranked first (median rule), victim-rounds',
                    round(pooled['median_dev']['rank1_frac'] * pooled['median_dev']['n_rounds']), 0)
        for w, v in zip(('rounds_12_20', 'rounds_28_35'), vals['win']):
            check_claim(v, f'{cfg}: plain echo median-rule AUC, {w}', tf['fedtgp_echo']['windows'][w]['pooled']['median_dev']['auc_mean_over_seeds'], 5.01e-4)
        rat = {}
        for g in fe['geometry'].values():
            for x in g['per_round']:
                rat.setdefault(x['round'], []).append(x['offset_q_minus_b_fro'] / x['honest_spread_about_q_median'])
        check_claim(vals['ratio'][0], f'{cfg}: offset/spread at round 12 (median over seeds)', float(np.median(rat[12])), 5.01e-3)
        check_claim(vals['ratio'][1], f'{cfg}: offset/spread at round {last} (median over seeds)', float(np.median(rat[last])), 5.01e-3)
        a = tf['fedtgp_affine_echo']['TEST']['median_dev']
        auc, first, n = vals['affine']
        check_claim(auc, f'{cfg}: affine echo median-rule AUC, TEST', a['auc_mean_over_seeds'], 5.01e-4)
        check_claim(n, f'{cfg}: affine echo TEST victim-rounds', a['n_rounds'], 0)
        check_claim(first, f'{cfg}: affine echo ranked first (median rule), TEST', round(a['rank1_frac'] * a['n_rounds']), 0)
        pr = tf['fedtgp_echo']['per_round']

        def share(rule, lo, hi):
            return float(np.mean([v for r, v in zip(pr[rule]['rounds'], pr[rule]['rank1_frac']) if lo <= r <= hi]))
        if arm == 'r60':
            check_claim(0.943, 'r60: plain echo rank-1 share, rounds 53-60, median rule (decision)', share('median_dev', 53, 60), 5.01e-4)
            check_claim(0.970, 'r60: plain echo rank-1 share, rounds 53-60, Krum (decision)', share('krum', 53, 60), 5.01e-4)
            check_claim(1.0, 'r60: pre-specified decision is INVERTED (> 0.5 under both)',
                        float(share('median_dev', 53, 60) > 0.5 and share('krum', 53, 60) > 0.5), 0)
            first_round = [next(r for r, v in zip(pr[rule]['rounds'], pr[rule]['rank1_frac']) if v >= 0.5)
                           for rule in ('krum', 'median_dev', 'trimmed_mean_dev', 'bulyan_dev')]
            check_claim(42, 'r60: earliest round with rank-1 share >= 0.5 over the four rules', min(first_round), 0)
            check_claim(49, 'r60: latest such round over the four rules', max(first_round), 0)
            check_claim(0.003, 'r60: rank-1 share, rounds 12-35, Krum (35-round arm: 0.001)', share('krum', 12, 35), 5.01e-4)
            check_claim(0.0, 'r60: rank-1 share, rounds 12-35, median rule', share('median_dev', 12, 35), 5.01e-4)
            check_claim(0.0, 'r60: rank-1 share, rounds 12-35, trimmed mean', share('trimmed_mean_dev', 12, 35), 5.01e-4)
            check_claim(10, 'r60: plain echo ranked first under Bulyan, rounds 12-35, of 1200 victim-rounds',
                        round(share('bulyan_dev', 12, 35) * 1200), 0)
        else:
            check_claim(0.0, 'euc: plain echo rank-1 share, rounds 12-20, median rule',
                        tf['fedtgp_echo']['windows']['rounds_12_20']['pooled']['median_dev']['rank1_frac'], 5.01e-4)
            for rule, sh, auc in (('median_dev', 0.983, 0.010), ('krum', 0.990, 0.014)):
                w = tf['fedtgp_echo']['windows']['rounds_28_35']['pooled'][rule]
                check_claim(sh, f'euc: plain echo rank-1 share, rounds 28-35, {rule}', w['rank1_frac'], 5.01e-4)
                check_claim(auc, f'euc: plain echo AUC, rounds 28-35, {rule}', w['auc_mean_over_seeds'], 5.01e-4)
    if pe := load_json(f'protocol_eval_{cfg}.json'):
        e = pe['results']['TEST']['suspicion']['0.01']
        hk, n, ek, ck = vals['auditor']
        check_claim(hk, f'{cfg}: auditor honest flags, TEST', e['honest']['client_round']['k'], 0)
        check_claim(n, f'{cfg}: TEST client-rounds per attack', e['attacks']['global_echo']['tpr_client_round']['n'], 0)
        check_claim(ek, f'{cfg}: auditor flags on global echo, TEST', e['attacks']['global_echo']['tpr_client_round']['k'], 0)
        check_claim(ck, f'{cfg}: auditor flags on peer copy, TEST', e['attacks']['peer_copy']['tpr_client_round']['k'], 0)
_rs_all = []
for cfg in ('fedtgp_lam1', 'fedtgp_lam10', 'fedtgp_lam01', 'fedtgp_lam01_r60', 'fedtgp_lam01_euc'):
    if fe := load_json(f'fedtgp_echo_{cfg}.json'):
        _rs_all += [x['offset_q_minus_b_fro'] / x['honest_spread_about_q_median'] for g in fe['geometry'].values() for x in g['per_round']]
if _rs_all:
    check_claim(0.94, 'FedTGP, all runs: smallest offset/spread over seed-rounds', min(_rs_all), 5.01e-3)
    check_claim(1.59, 'FedTGP, all runs: largest offset/spread over seed-rounds', max(_rs_all), 5.01e-3)

bad = unver = 0
print("%-42s %10s %12s  %s" % ("claim", "reference", "in results", "status"))
print("-" * 86)
for stated, src, got, tol in CLAIMS:
    if got is None:
        print("%-42s %10.4f %12s  UNVERIFIED" % (src, stated, "n/a"))
        unver += 1
        continue
    ok = abs(stated - got) <= tol
    if not ok:
        bad += 1
    print("%-42s %10.4f %12.4f  %s" % (src, stated, got, "ok" if ok else "MISMATCH"))

print()
if MISSING:
    print("SOURCE FILES ABSENT, so every claim keyed to them was skipped:")
    for name in sorted(set(MISSING)):
        print("   %s" % name)
    print()

print("%d data assertions checked, %d mismatched, %d unverified, %d source files absent"
      % (len(CLAIMS), bad, unver, len(set(MISSING))))
sys.exit(1 if (bad or unver or MISSING) else 0)
