from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)


def _no_release_check(calib: dict) -> dict:
    return {"identical": False, "applies": False}


def _no_fused_check(cells: dict) -> dict:
    return {"applies": False}


def _check_population(archives: str, prefix: str) -> None:
    import re
    import numpy as np
    from fedtgp_attacks import config_key
    if not os.path.isdir(archives):
        return
    seen = {}
    for fn in sorted(os.listdir(archives)):
        m = re.search(r"_s(\d+)\.npz$", fn)
        if fn.startswith(prefix) and m:
            s = int(m.group(1))
            if s in seen:
                raise SystemExit("archive seed collision for seed %d under prefix %s: %s vs %s; "
                                 "tighten --prefix" % (s, prefix, seen[s], fn))
            seen[s] = fn
    keys = set()
    for fn in seen.values():
        z = np.load(os.path.join(archives, fn), allow_pickle=True)
        if "meta" in z.files:
            keys.add(config_key(json.loads(str(z["meta"]))))
    if len(keys) > 1:
        raise SystemExit("archives of different configurations under prefix %s: %s"
                         % (prefix, sorted(keys)))


def _option(name: str):
    for i, arg in enumerate(sys.argv):
        if arg == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def main() -> int:
    given = list(sys.argv[1:])
    prefix, out = _option("--prefix"), _option("--out")
    if not prefix or prefix == "arch_cifar100_s":
        raise SystemExit("give the --prefix of the new population (for the released one, run "
                         "limited_access_eval.py itself)")
    if not out or os.path.abspath(out) == os.path.abspath(
            os.path.join(_HERE, "..", "results", "limited_access_grid.json")):
        raise SystemExit("give an --out other than the released results/limited_access_grid.json")
    _check_population(_option("--archives") or os.path.join(_HERE, "..", "results", "archives"),
                      prefix)
    if _option("--parts") is None:
        sys.argv += ["--parts", os.path.join(_HERE, "..", ".reproduction",
                                             "limited_access_parts_" + prefix.strip("_"))]
    import limited_access_eval as le
    le.check_against_release = _no_release_check
    le.released_fused_check = _no_fused_check
    rc = le.main()
    if rc == 0 and out and os.path.exists(out) and "--no-write" not in sys.argv:
        with open(out, encoding="utf-8") as fh:
            payload = json.load(fh)
        payload["meta"]["command"] = "python experiments/limited_access_rev2.py " + " ".join(given)
        parts = os.path.abspath(_option("--parts"))
        try:
            parts = os.path.relpath(parts, os.path.join(_HERE, "..")).replace(os.sep, "/")
        except ValueError:
            pass
        payload["meta"]["parts"] = parts
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, default=float)
    return rc


if __name__ == "__main__":
    sys.exit(main())
