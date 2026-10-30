from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)

import numpy as np

from passportfl.reproducibility import environment

RESULTS = os.path.join(REPO, "results")
STATS = ("proto_ratio", "label_stability", "max_drift")
BARS = (("rounds_accepted", "rounds accepted", None),
        ("label_stability", "label stability", "label_stability"),
        ("drift_ratio", "drift ratio", "proto_ratio"))
EXTRA = (("max_drift", "maximum drift", "max_drift"),)
VERIFIER = "deployed_w5"
SUPPLEMENTARY_VERIFIER = "two_sided_paper_w5"


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def quartiles(v: Sequence[float]) -> dict:
    a = np.asarray(v, float)
    return {"median": float(np.median(a)), "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)), "n": int(a.size)}


def honest_rows(runs: Dict[int, dict], lo: int, hi: int) -> List[dict]:
    return [dict(x, seed=s) for s, d in runs.items() for x in d["client_rounds"] if lo <= x["round"] <= hi]


def honest_block(runs: Dict[int, dict], lo: int, hi: int) -> dict:
    rows = honest_rows(runs, lo, hi)
    out = {"seeds": sorted(runs), "rounds": [lo, hi], "client_rounds": len(rows)}
    for v in (VERIFIER, SUPPLEMENTARY_VERIFIER):
        ok = [bool(x["verdicts"][v]["accepted"]) for x in rows]
        per = {}
        for x, o in zip(rows, ok):
            per.setdefault((x["seed"], x["client"]), []).append(o)
        frac = [float(np.mean(p)) for p in per.values()]
        out[v] = {"accepted": int(sum(ok)), "n": len(ok), "fraction": float(np.mean(ok)),
                  "per_client_fraction": {"min": min(frac), "median": float(np.median(frac)),
                                          "max": max(frac)},
                  "clients": len(per), "clients_accepted_every_round": int(sum(all(p) for p in per.values()))}
    for s in STATS:
        out[s] = quartiles([x[s] for x in rows if x["defined"][s]])
    return out


def honest_per_round(run: dict, R: int) -> List[dict]:
    out = []
    for r in range(1, R + 1):
        rows = [x for x in run["client_rounds"] if x["round"] == r]
        rec = {"round": r, "clients": len(rows),
               "accepted": int(sum(x["verdicts"][VERIFIER]["accepted"] for x in rows))}
        for s in STATS:
            v = [x[s] for x in rows if x["defined"][s]]
            rec[s] = quartiles(v) if v else None
        out.append(rec)
    return out


def ghost_series(per_round: List[dict], lo: int, hi: int, accepted_key) -> dict:
    rows = [p for p in per_round if lo <= p["round"] <= hi]
    ok = [bool(accepted_key(p)) for p in rows]
    out = {"rounds": [lo, hi], "accepted": int(sum(ok)), "n": len(ok), "fraction": float(np.mean(ok))}
    for s in STATS:
        out[s] = quartiles([p[s] for p in rows if p["round"] > 1])
    return out


def bars(ghost: dict, honest: dict, ghost_acc: Optional[dict] = None,
         honest_acc: Optional[dict] = None, verifier: str = VERIFIER) -> List[dict]:
    ga, ha = ghost_acc or ghost, honest_acc or honest
    out = []
    for key, label, stat in BARS + EXTRA:
        g = ga["fraction"] if stat is None else ghost[stat]["median"]
        h = ha[verifier]["fraction"] if stat is None else honest[stat]["median"]
        out.append({"key": key, "label": label, "ghost": g, "honest": h,
                    "in_released_panel": key != "max_drift",
                    "rounds": ga["rounds"] if stat is None else ghost["rounds"],
                    "ghost_n": ga["n"] if stat is None else ghost[stat]["n"],
                    "honest_n": ha[verifier]["n"] if stat is None else honest[stat]["n"]})
    return out


def unpaired_reference() -> dict:
    fp = load(os.path.join(RESULTS, "fedproto_cifar100.json"))
    warm = int(fp["config"]["warmup"])
    rows = [e for e in fp["verify_log"] if int(e["round"]) > warm]
    return {"source": "results/fedproto_cifar100.json verify_log, rounds > %d (warm-up %d)" % (warm, warm),
            "client_rounds": len(rows),
            "label_stability": float(np.median([e["stability"] for e in rows])),
            "proto_ratio": float(np.median([e["proto_ratio"] for e in rows])),
            "max_drift": float(np.median([e["max_drift"] for e in rows])),
            "accepted_logged": int(sum(bool(e["accepted"]) for e in rows))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="paired honest run of the primary series")
    ap.add_argument("--sensitivity-seeds", default="0,1,2",
                    help="40-round paired runs pooled in the sensitivity variant")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--post-from", type=int, default=6, help="first post-warm-up round")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--out", default=os.path.join(RESULTS, "paired_fig3a.json"))
    a = ap.parse_args()
    R, lo = a.rounds, a.post_from
    sens = [int(x) for x in a.sensitivity_seeds.split(",") if x.strip()]
    paths = {s: os.path.join(a.results, "paired_honest_s%d.json" % s) for s in sorted({a.seed, *sens})}
    runs = {s: load(p) for s, p in paths.items()}
    for s, d in runs.items():
        if int(d["config"]["rounds"]) < R:
            raise SystemExit("paired_honest_s%d.json has %s rounds, fewer than %d" % (s, d["config"]["rounds"], R))
    neg_path = os.path.join(a.results, "negative_results.json")
    gx_path = os.path.join(a.results, "ghost_extended.json")
    n3 = load(neg_path)["N3_ghost_prover"]
    ge = load(gx_path)
    if int(n3["rounds"]) != R or int(n3["warmup_rounds"]) != lo - 1:
        raise SystemExit("N3_ghost_prover has %s rounds and warm-up %s" % (n3["rounds"], n3["warmup_rounds"]))
    orig, cal = ge["ghosts"]["original"]["chain_head"], ge["ghosts"]["calibrated"]["chain_head"]

    ghost = {"original": {w: ghost_series(n3["per_round"], *rng, lambda p: p["accepted"])
                          for w, rng in (("post_warmup", (lo, R)), ("all_rounds", (1, R)))},
             "calibrated": {w: ghost_series(cal["per_round"], *rng, lambda p: p[VERIFIER]["accepted"])
                            for w, rng in (("post_warmup", (lo, R)), ("all_rounds", (1, R)))}}
    for kind, ch in (("original", orig), ("calibrated", cal)):
        for w, (l_, h_) in (("post_warmup", (lo, R)), ("all_rounds", (1, R))):
            rows = [p for p in ch["per_round"] if l_ <= p["round"] <= h_]
            ghost[kind][w]["two_sided_paper_w5_accepted"] = int(sum(p["two_sided_w5"]["accepted"] for p in rows))

    prim = {s: runs[s] for s in (a.seed,)}
    pooled = {s: runs[s] for s in sens}
    honest = {"post_warmup": honest_block(prim, lo, R), "all_rounds": honest_block(prim, 1, R)}
    honest_sens = {"post_warmup": honest_block(pooled, lo, R), "all_rounds": honest_block(pooled, 1, R)}

    out = {"meta": {"script": "experiments/paired_fig3a_data.py",
                    "command": "python experiments/paired_fig3a_data.py" + "".join(
                        " --%s %s" % (k, getattr(a, k.replace("-", "_"))) for k in
                        ("seed", "sensitivity-seeds", "rounds", "post-from")
                        if getattr(a, k.replace("-", "_")) != ap.get_default(k.replace("-", "_"))),
                    "environment": environment(),
                    "inputs": {os.path.relpath(p, REPO).replace("\\", "/"): sha256(p)
                               for p in [*paths.values(), neg_path, gx_path]}},
           "config": {"rounds": R, "warmup": lo - 1, "post_warmup_rounds": [lo, R], "primary_seed": a.seed,
                      "sensitivity_seeds": sens, "verifier": VERIFIER,
                      "verifier_spec": runs[a.seed]["receivers"][VERIFIER],
                      "supplementary_verifier": SUPPLEMENTARY_VERIFIER,
                      "supplementary_verifier_bands": runs[a.seed]["receivers"][SUPPLEMENTARY_VERIFIER]["bands"],
                      "ghost_scheme": ge["reproduction"]["stored_signature_scheme"],
                      "honest_scheme": runs[a.seed]["config"]["scheme"]}}

    out["panel_a"] = {
        "ghost_series": "N3_ghost_prover (= ghost_extended original)",
        "honest_series": "paired_honest_s%d.json, every client-round, logged %s verdicts" % (a.seed, VERIFIER),
        "bars": bars(ghost["original"]["post_warmup"], honest["post_warmup"]),
        "calibrated_ghost_bars": bars(ghost["calibrated"]["post_warmup"], honest["post_warmup"]),
        "honest_per_client_acceptance": honest["post_warmup"][VERIFIER]["per_client_fraction"],
        "honest_clients_accepted_every_round": [honest["post_warmup"][VERIFIER]["clients_accepted_every_round"],
                                                honest["post_warmup"][VERIFIER]["clients"]]}
    out["alternatives"] = {
        "released_windows": {
            "bars": bars(ghost["original"]["post_warmup"], honest["post_warmup"],
                         ghost["original"]["all_rounds"], honest["all_rounds"])},
        "sensitivity_seeds_%s" % "_".join(map(str, sens)): {
            "bars": bars(ghost["original"]["post_warmup"], honest_sens["post_warmup"]),
            "calibrated_ghost_bars": bars(ghost["calibrated"]["post_warmup"], honest_sens["post_warmup"]),
            "honest_per_client_acceptance": honest_sens["post_warmup"][VERIFIER]["per_client_fraction"],
            "honest_clients_accepted_every_round": [
                honest_sens["post_warmup"][VERIFIER]["clients_accepted_every_round"],
                honest_sens["post_warmup"][VERIFIER]["clients"]]},
        "two_sided_paper_bands_w5": {
            "ghost_original_post_warmup_accepted": [ghost["original"]["post_warmup"]["two_sided_paper_w5_accepted"],
                                                    ghost["original"]["post_warmup"]["n"]],
            "ghost_calibrated_post_warmup_accepted": [ghost["calibrated"]["post_warmup"]["two_sided_paper_w5_accepted"],
                                                      ghost["calibrated"]["post_warmup"]["n"]],
            "honest_post_warmup": {k: honest["post_warmup"][SUPPLEMENTARY_VERIFIER][k]
                                   for k in ("accepted", "n", "fraction", "clients_accepted_every_round", "clients")}}}
    out["ghost"], out["honest"], out["honest_sensitivity"] = ghost, honest, honest_sens
    out["per_round"] = {"ghost_original": [{"round": p["round"], "accepted": bool(p["accepted"]),
                                            **{s: p[s] for s in STATS}} for p in n3["per_round"]],
                        "ghost_calibrated": [{"round": p["round"], "accepted": bool(p[VERIFIER]["accepted"]),
                                              **{s: p[s] for s in STATS}} for p in cal["per_round"]],
                        "honest_seed_%d" % a.seed: honest_per_round(runs[a.seed], R)}
    out["released_unpaired_reference"] = unpaired_reference()

    chk = {}
    chk["n3_equals_ghost_extended_original_max_abs"] = float(max(
        abs(p[s] - q[s]) for p, q in zip(n3["per_round"], orig["per_round"]) for s in STATS))
    chk["n3_accepted_equals_ghost_extended_deployed_w5"] = all(
        bool(p["accepted"]) == bool(q[VERIFIER]["accepted"]) for p, q in zip(n3["per_round"], orig["per_round"]))
    chk["n3_accepted_rounds_logged"] = [int(n3["accepted_rounds"]), ghost["original"]["all_rounds"]["accepted"]]
    logged = runs[a.seed]["summary"]
    chk["honest_counts_vs_run_summary"] = {
        w: [logged[w][VERIFIER], honest[k][VERIFIER]["accepted"]]
        for w, k in (("rounds_1_%d" % R, "all_rounds"), ("rounds_%d_%d" % (lo, R), "post_warmup")) if w in logged}
    sp = os.path.join(a.results, "paired_honest_summary.json")
    if os.path.exists(sp):
        f3 = load(sp).get("fig3a", {})
        if f3.get("seed") == a.seed:
            hb = f3["honest"]["rounds_%d_%d" % (lo, R)]
            chk["paired_honest_summary_fig3a"] = {
                "fraction_accepted": [hb[VERIFIER]["fraction_accepted"], honest["post_warmup"][VERIFIER]["fraction"]],
                "medians": {s: [hb["medians"][s], honest["post_warmup"][s]["median"]] for s in STATS}}
            out["meta"]["inputs"]["results/paired_honest_summary.json"] = sha256(sp)
    chk["all_consistent"] = bool(
        chk["n3_equals_ghost_extended_original_max_abs"] == 0.0 and chk["n3_accepted_equals_ghost_extended_deployed_w5"]
        and chk["n3_accepted_rounds_logged"][0] == chk["n3_accepted_rounds_logged"][1]
        and all(x == y for x, y in chk["honest_counts_vs_run_summary"].values())
        and ("paired_honest_summary_fig3a" not in chk or (
            chk["paired_honest_summary_fig3a"]["fraction_accepted"][0] == chk["paired_honest_summary_fig3a"]["fraction_accepted"][1]
            and all(x == y for x, y in chk["paired_honest_summary_fig3a"]["medians"].values()))))
    out["checks"] = chk

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("paired Fig. 3a, rounds %d-%d, honest seed %d (%d client-rounds), verifier %s:"
          % (lo, R, a.seed, honest["post_warmup"]["client_rounds"], VERIFIER))
    for b in out["panel_a"]["bars"]:
        print("  %-16s ghost %.4f   honest %.4f%s" % (b["label"], b["ghost"], b["honest"],
                                                     "" if b["in_released_panel"] else "   (not in the released panel)"))
    for b in out["panel_a"]["calibrated_ghost_bars"]:
        print("  calibrated %-16s ghost %.4f" % (b["label"], b["ghost"]))
    u = out["released_unpaired_reference"]
    print("released unpaired reference (%d client-rounds): stability %.4f ratio %.4f"
          % (u["client_rounds"], u["label_stability"], u["proto_ratio"]))
    print("checks consistent: %s" % chk["all_consistent"])
    print("wrote %s" % a.out)
    return 0 if chk["all_consistent"] else 1


if __name__ == "__main__":
    sys.exit(main())
