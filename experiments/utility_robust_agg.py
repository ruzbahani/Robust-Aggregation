from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import numpy as np

import utility_impact
from passportfl.fedproto import Client, Server

REPO = os.path.join(HERE, "..")
AGG_RULES = ("mean", "median", "trimmed")
TRIM = 0.2


def holder_rows(payloads: Sequence[np.ndarray], counts: Sequence[np.ndarray],
                rule: str) -> Tuple[np.ndarray, np.ndarray]:
    P = np.stack([np.asarray(p, dtype=np.float64) for p in payloads])
    H = np.stack([np.asarray(c) > 0 for c in counts])
    n_k = H.sum(axis=0)
    R = np.full(P.shape[1:], np.nan)
    for k in np.flatnonzero(n_k):
        V = P[H[:, k], k]
        if rule == "median":
            R[k] = np.median(V, axis=0)
        elif rule == "trimmed":
            n = len(V)
            c = int(np.floor(TRIM * n))
            R[k] = np.sort(V, axis=0)[c:n - c].mean(axis=0)
        else:
            raise ValueError("no holder rule %r" % rule)
    return R, n_k


class Trace:
    def __init__(self):
        self.round: Optional[int] = None
        self.agg: List[dict] = []
        self.evals: Dict[int, dict] = {}

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self

    def take(self) -> Tuple[List[dict], List[dict]]:
        out = (self.agg, [self.evals[r] for r in sorted(self.evals)])
        self.agg, self.evals = [], {}
        return out


class RobustServer(Server):
    def __init__(self, cfg, rule: str = "mean", trace: Optional[Trace] = None):
        super().__init__(cfg)
        if rule not in AGG_RULES:
            raise ValueError("unknown aggregation rule %r; known: %s" % (rule, ", ".join(AGG_RULES)))
        self.rule = rule
        self.trace = trace

    def aggregate(self, contribs, counts) -> np.ndarray:
        prev = None if self.global_C is None else self.global_C.copy()
        n0 = len(self.log)
        C_mean = super().aggregate(contribs, counts)
        entries = self.log[n0:]
        if (len(entries) != len(contribs)
                or any(e["round"] != c.round for e, c in zip(entries, contribs))):
            raise RuntimeError("Server.aggregate did not log one entry per contribution in "
                               "order, so the accepted set cannot be read from its log")
        acc = [i for i, e in enumerate(entries) if e["accepted"]]
        K = self.cfg.num_classes
        n_k = np.zeros(K, dtype=int)
        if acc:
            n_k = np.sum([np.asarray(counts[i]) > 0 for i in acc], axis=0).astype(int)
            den = np.sum([np.asarray(counts[i], dtype=np.float64) for i in acc], axis=0)
            if not np.array_equal(den > 0, n_k > 0):
                raise RuntimeError("holder set differs from the classes the mean aggregated")
        C = C_mean
        if self.rule != "mean" and acc:
            R, n_k = holder_rows([contribs[i].C for i in acc], [counts[i] for i in acc],
                                 self.rule)
            has = n_k > 0
            C = C_mean.copy()
            C[has] = R[has]
            self.global_C = C
        if self.trace is not None:
            rnd = int(contribs[0].round) if contribs else -1
            held = n_k[n_k > 0]
            gap = float(np.linalg.norm(C - C_mean))
            step = float(np.linalg.norm(C_mean - prev)) if prev is not None else float("nan")
            self.trace.round = rnd
            self.trace.agg.append({
                "round": rnd, "contributions": len(contribs), "accepted": len(acc),
                "classes_held": int(len(held)),
                "holders_min": int(held.min()) if len(held) else 0,
                "holders_median": float(np.median(held)) if len(held) else 0.0,
                "holders_max": int(held.max()) if len(held) else 0,
                "dev_from_mean_rel": gap / (float(np.linalg.norm(C_mean)) + 1e-12),
                "dev_from_mean_over_step": (gap / step if np.isfinite(step) and step > 0
                                            else float("nan")),
            })
        return C


class RecordingClient(Client):
    def __init__(self, *args, trace: Optional[Trace] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = trace

    def evaluate(self, test_ds, use_prototypes: bool = True) -> Dict[str, float]:
        out = super().evaluate(test_ds, use_prototypes)
        tr = self.trace
        if tr is not None and tr.round is not None:
            e = tr.evals.setdefault(tr.round, {"round": tr.round, "n_test": len(test_ds),
                                               "acc_prototype": {}, "acc_classifier": {}})
            if e["n_test"] != len(test_ds):
                raise RuntimeError("round %d evaluated on test sets of different sizes" % tr.round)
            e["acc_prototype"][int(self.cid)] = out.get("acc_prototype")
            e["acc_classifier"][int(self.cid)] = out["acc_classifier"]
        return out


@contextlib.contextmanager
def installed(rule: str, wrapper_args: Optional[Sequence[str]] = None):
    if rule not in AGG_RULES:
        raise ValueError("unknown aggregation rule %r" % rule)
    trace = Trace()
    names = ("Server", "Client", "run_scenario", "report")
    saved = {n: getattr(utility_impact, n) for n in names}
    run_scenario, report = saved["run_scenario"], saved["report"]
    prefix: Dict[str, list] = {}

    def traced_run_scenario(spec, *args, **kwargs):
        if "rounds" not in prefix:
            prefix["rounds"] = trace.take()[0]
        trace.take()
        trace.round = None
        res, cbar_hist = run_scenario(spec, *args, **kwargs)
        agg, evals = trace.take()
        res["aggregation"] = {"rule": rule, "rounds": agg}
        res["per_client_utility"] = evals
        return res, cbar_hist

    def report_with_rule(out, a):
        out["aggregation"] = {
            "rule": rule,
            "trim_fraction": TRIM if rule == "trimmed" else None,
            "wrapper": "experiments/utility_robust_agg.py",
            "wrapper_args": list(wrapper_args) if wrapper_args is not None else None,
            "prefix_rounds": prefix.get("rounds", []),
        }
        return report(out, a)

    utility_impact.Server = functools.partial(RobustServer, rule=rule, trace=trace)
    utility_impact.Client = functools.partial(RecordingClient, trace=trace)
    utility_impact.run_scenario = traced_run_scenario
    utility_impact.report = report_with_rule
    try:
        yield trace
    finally:
        for n, v in saved.items():
            setattr(utility_impact, n, v)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        description="utility_impact.py with a selectable server aggregation rule; every "
                    "other argument is forwarded to utility_impact.py",
        add_help=False, allow_abbrev=False)
    ap.add_argument("--agg", choices=AGG_RULES, default=None,
                    help="server aggregation rule (required)")
    ap.add_argument("--out", default="",
                    help="default results/utility_robust_<agg>_s<seed>.json")
    ap.add_argument("-h", "--help", action="store_true")
    a, rest = ap.parse_known_args(argv)
    if a.help:
        ap.print_help()
        print("\nForwarded arguments (utility_impact.py):\n")
        sys.argv = [os.path.join(HERE, "utility_impact.py"), "-h"]
        return utility_impact.main()
    if a.agg is None:
        ap.error("--agg is required")
    if any(x.startswith("--o") for x in rest):
        ap.error("give the output path once, as --out")
    sp = argparse.ArgumentParser(add_help=False)
    sp.add_argument("--seed", type=int, default=0)
    seed = sp.parse_known_args(rest)[0].seed
    path = os.path.abspath(a.out or os.path.join(REPO, "results", "utility_robust_%s_s%d.json"
                                                 % (a.agg, seed)))
    if os.path.basename(path).startswith("utility_impact"):
        raise SystemExit("refusing to write %s: utility_impact_* files hold the released runs "
                         "and those of other jobs" % path)

    sys.argv = [os.path.join(HERE, "utility_impact.py")] + rest + ["--out", path]
    print("[utility_robust_agg] aggregation rule %s -> %s" % (a.agg, path), flush=True)
    with installed(a.agg, argv):
        rc = utility_impact.main()

    with open(path, encoding="utf-8") as f:
        written = json.load(f)
    if (written.get("aggregation") or {}).get("rule") != a.agg:
        raise RuntimeError("%s lacks the aggregation block of rule %s" % (path, a.agg))
    return rc


if __name__ == "__main__":
    sys.exit(main())
