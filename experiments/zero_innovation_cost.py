from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = os.path.join(HERE, "results", "archives")
OUT = os.path.join(HERE, "results", "zero_innovation_cost.json")
STRUCTURAL_RATIO = 1e-6
LEVELS = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)


def group_of(name: str) -> str:
    m = re.match(r"arch_([a-z0-9]+)_", name)
    return m.group(1) if m else name


def archive_rows(path: str):
    z = np.load(path, allow_pickle=True)
    C, present, cbar = z["C"].item(), z["present"].item(), z["cbar"].item()
    cids = sorted(C)
    rounds = sorted(r for r in cbar if (r - 1) in cbar)
    rows = []
    for r in rounds:
        norms = {}
        for cid in cids:
            if r not in C[cid]:
                continue
            pr = np.asarray(present[cid][r], bool)
            if pr.sum() == 0:
                continue
            I = (np.asarray(C[cid][r], np.float64) - np.asarray(cbar[r - 1], np.float64))[pr]
            norms[cid] = float(np.linalg.norm(I))
        if len(norms) < 2:
            continue
        med = float(np.median(list(norms.values())))
        for cid, n in norms.items():
            rows.append((int(r), int(cid), n, n / med if med > 1e-12 else float("nan")))
    return rows


def summarize(rows):
    ratio = np.asarray([x[3] for x in rows], float)
    ratio = ratio[np.isfinite(ratio)]
    rr = np.asarray([x[0] for x in rows])
    last = rr.max()
    late = np.asarray([x[3] for x in rows if x[0] > last - 5], float)
    imin = int(np.argmin(ratio))
    return {
        "client_rounds": int(len(ratio)),
        "min_ratio": float(ratio.min()),
        "min_at": {"round": int(rows[imin][0]), "client": int(rows[imin][1])},
        "p0.1": float(np.percentile(ratio, 0.1)),
        "p1": float(np.percentile(ratio, 1)),
        "p5": float(np.percentile(ratio, 5)),
        "median": float(np.median(ratio)),
        "last5_rounds_min_ratio": float(late.min()) if len(late) else None,
        "count_at_or_below": {("%g" % lv): int((ratio <= lv).sum()) for lv in LEVELS},
    }


def main() -> int:
    paths = sorted(glob.glob(os.path.join(ARCH, "arch_*.npz")))
    paths = [p for p in paths if "reduced" not in os.path.basename(p)]
    by_group, all_rows = {}, []
    for p in paths:
        rows = archive_rows(p)
        g = group_of(os.path.basename(p))
        by_group.setdefault(g, []).extend(rows)
        all_rows.extend(rows)
        print("%-45s %5d client-rounds  min ratio %.3e" % (os.path.basename(p), len(rows),
              np.nanmin([x[3] for x in rows])))
    out = {"structural_ratio": STRUCTURAL_RATIO,
           "archives": [os.path.basename(p) for p in paths],
           "groups": {g: summarize(rows) for g, rows in sorted(by_group.items())},
           "all": summarize(all_rows)}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\n%-12s %8s %12s %10s %10s %14s" % ("group", "n", "min ratio", "p0.1", "p1", "last5 min"))
    for g, s in sorted(out["groups"].items()):
        print("%-12s %8d %12.3e %10.3e %10.3e %14.3e" % (g, s["client_rounds"], s["min_ratio"],
              s["p0.1"], s["p1"], s["last5_rounds_min_ratio"] or float("nan")))
    a = out["all"]
    print("\nall honest client-rounds: %d; min ratio %.3e at round %d client %d; at or below 1e-6: %d, 1e-2: %d, 1e-1: %d"
          % (a["client_rounds"], a["min_ratio"], a["min_at"]["round"], a["min_at"]["client"],
             a["count_at_or_below"]["1e-06"], a["count_at_or_below"]["0.01"], a["count_at_or_below"]["0.1"]))
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
