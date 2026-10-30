from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

ARCHIVE_NAME = re.compile(r"arch_(.+?)_c\d+_r\d+(?:_d[0-9.]+)?_s\d+\.npz$")
SEEDED_DIFFERENTLY = ("noise",)


def check_file(path: str, results_dir: str, tol: float) -> dict:
    with open(path, encoding="utf-8") as fh:
        pe = json.load(fh)
    archives = pe.get("protocol", {}).get("archives", {})
    block = pe.get("auc_cross_check")
    out = {"protocol_file": os.path.basename(path), "seeds": {}, "ok": True, "problems": []}
    if not block:
        out["ok"] = False
        out["problems"].append("no auc_cross_check block (stage 5 did not run)")
        return out
    worst, worst_noise, worst_pairs = 0.0, 0.0, 0.0
    for seed, ent in sorted(block.items(), key=lambda kv: int(kv[0])):
        name = archives.get(seed, "")
        m = ARCHIVE_NAME.match(name)
        ref_path = os.path.join(results_dir, "innovation_%s.json" % m.group(1)) if m else ""
        rec = {"archive": name, "reference": os.path.basename(ref_path) if ref_path else None,
               "attacks": {}}
        if not ref_path or not os.path.exists(ref_path):
            out["ok"] = False
            out["problems"].append("seed %s: reference %s not found" % (seed, ref_path or name))
            rec["reference_found"] = False
            out["seeds"][seed] = rec
            continue
        rec["reference_found"] = True
        with open(ref_path, encoding="utf-8") as fh:
            ref = json.load(fh).get("results", {})
        for atk, r in ent.get("attacks", {}).items():
            sus = ref.get(atk, {}).get("suspicion", {})
            pub = sus.get("auc")
            mine = r.get("auc_recomputed")
            d = (mine - pub) if isinstance(pub, (int, float)) and isinstance(mine, (int, float)) else None
            n_pairs = (sus["n_pos"] * sus["n_neg"]) if isinstance(sus.get("n_pos"), int) \
                and isinstance(sus.get("n_neg"), int) else None
            pairs = abs(d) * n_pairs if d is not None and n_pairs else None
            rec["attacks"][atk] = {"auc_recomputed": mine, "auc_published": pub, "delta": d,
                                   "delta_pairs": pairs}
            if d is None:
                out["ok"] = False
                out["problems"].append("seed %s %s: no comparable value" % (seed, atk))
            elif atk in SEEDED_DIFFERENTLY:
                worst_noise = max(worst_noise, abs(d))
            else:
                worst = max(worst, abs(d))
                if pairs is not None:
                    worst_pairs = max(worst_pairs, pairs)
        out["seeds"][seed] = rec
    out["max_abs_delta"] = worst
    out["max_abs_delta_pairs"] = worst_pairs
    out["max_abs_delta_noise"] = worst_noise
    if worst > tol:
        out["ok"] = False
        out["problems"].append("max |delta| %.3g exceeds %.3g" % (worst, tol))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", nargs="+", required=True,
                    help="protocol_eval.py outputs on revision archives")
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"),
                    help="directory holding innovation_<tag>.json")
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="largest |recomputed - published| accepted for the attacks other than "
                         "noise (default: about 5 tie flips of 518,400 pairs; see the docstring)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "rev2_auc_cross_check.json"))
    a = ap.parse_args()
    res = {os.path.basename(p): check_file(p, a.results_dir, a.tol) for p in a.protocol}
    for name, r in res.items():
        print("%-36s ok=%s  max|delta| %s (%s pairs)  noise %s%s"
              % (name, r["ok"], "%.2e" % r["max_abs_delta"] if "max_abs_delta" in r else "-",
                 "%.1f" % r["max_abs_delta_pairs"] if "max_abs_delta_pairs" in r else "-",
                 "%.2e" % r["max_abs_delta_noise"] if "max_abs_delta_noise" in r else "-",
                 ("  " + "; ".join(r["problems"])) if r["problems"] else ""))
    from passportfl.reproducibility import environment

    def rel(path: str) -> str:
        try:
            return os.path.relpath(os.path.abspath(path), HERE).replace(os.sep, "/")
        except ValueError:
            return os.path.abspath(path)

    payload = {"config": {"protocol": [rel(p) for p in a.protocol],
                          "results_dir": rel(a.results_dir), "tol": a.tol,
                          "exempt": list(SEEDED_DIFFERENTLY),
                          "archive_name_pattern": ARCHIVE_NAME.pattern},
               "meta": {"command": "python experiments/rev2_auc_cross_check.py "
                                   + " ".join(sys.argv[1:]),
                        "tolerance": a.tol, "exempt": list(SEEDED_DIFFERENTLY),
                        "environment": environment()},
               "files": res, "all_ok": all(r["ok"] for r in res.values())}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print("wrote %s" % a.out)
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
