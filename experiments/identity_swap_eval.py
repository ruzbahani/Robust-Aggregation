from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from passportfl import identity as idm
from passportfl.passport import passport_bytes
from passportfl.protocol import PassportHolder, Receiver
from passportfl.reproducibility import environment

from negative_results import DEFAULT_ENTROPY, DIST_MATCHED, jsonable, seed_bytes
from passport_equivalence_eval import (EPS_IMPL, NET_DRAWS, Evaluator, as_float32, compare,
                                       configure, forged64, key_of, load_subject, per_branch_draw,
                                       rel, scaled, sha256_file, summary_of)


def protocol_head(S: dict, theta: dict) -> dict:
    m = copy.deepcopy(S["model"]).float()
    m.install_passport({k: v.float() for k, v in theta.items()})
    ck = S["ck"]
    scheme = ck.get("scheme", "Ed25519")
    M, T = ck.get("m_probes", 256), ck.get("t_reveal", 32)
    h = PassportHolder(m, scheme, M, T, "cpu")
    rec = dict(n_samples=len(ck.get("train_indices", [])), train_loss=0.0, local_acc=0.0,
               intra_compactness=0.0, inter_separation=0.0)
    contrib = h.issue(S["C"].numpy(), rec)
    res = Receiver(M, T).verify(contrib)
    return {"E": contrib.E_new, "G": h.G, "pk": h.pk, "scheme": scheme,
            "receiver_chain_ok": bool(res.chain_ok), "receiver_accepted": bool(res.accepted),
            "receiver_reason": res.reason}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(REPO, "checkpoints", "binding_a05_s0_clientA.pt"))
    ap.add_argument("--data-root", default=os.path.join(REPO, "data"))
    ap.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    ap.add_argument("--entropy", default=DEFAULT_ENTROPY, help="hex seed of the forgeries and the random head")
    ap.add_argument("--rescale-seed", type=int, default=0, help="seed of the per-branch scale draw")
    ap.add_argument("--batch-size", type=int, default=250)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=0, help="first N test images only (smoke tests); 0 = all")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "identity_swap.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    t_all = time.time()

    S = load_subject(a.checkpoint, a.data_root)
    if S["alpha"] == 0.0:
        raise SystemExit("%s was trained with alpha_dyn = 0; this measurement needs a checkpoint of "
                         "binding_eval.py --alpha-dyn > 0 --save-checkpoint" % a.checkpoint)
    if a.dtype == "float32":
        S["model"] = S["model"].float()
    if a.n_test:
        S["xs"], S["ys"] = S["xs"][:a.n_test], S["ys"][:a.n_test]
    S["entropy"] = bytes.fromhex(a.entropy)
    theta, keys, ck = S["theta"], sorted(S["theta"]), S["ck"]
    hist = list(ck.get("E_history", []))
    E_R = S["E"]
    if hist and hist[-1] != E_R:
        raise SystemExit("the checkpoint's E is not the last head of its history")
    configure(S["model"], theta, EPS_IMPL, S["alpha"], E_R)
    X = Evaluator(S, a.batch_size)

    draw = per_branch_draw(keys, *NET_DRAWS[0], np.random.default_rng(a.rescale_seed))
    forged = forged64(S)[0]
    forgeries = {"rescale_2": scaled(theta, {k: 2.0 for k in keys}),
                 "per_branch_draw_0": as_float32(scaled(theta, draw)),
                 "nullspace_min_norm": as_float32(forged[0.0]),
                 "nullspace_distribution_matched": as_float32(forged[DIST_MATCHED])}

    ref = X.ref("genuine_E_R")
    rows = {"forgery_genuine_head": [], "genuine_other_head": [], "forgery_own_head": []}
    for name, t in forgeries.items():
        ev = X.eval(t, label="forgery %s, genuine head E^R" % name)
        rows["forgery_genuine_head"].append(compare(name + "_E_R", ev, ref, "genuine_E_R", t, theta, **key_of(t)))

    heads = []
    for name, back in (("E_R_minus_1", 2), ("E_R_minus_2", 3), ("E_R_minus_5", 6)):
        if len(hist) >= back:
            heads.append((name, hist[-back]))
    if hist:
        heads.append(("E_1", hist[0]))
    heads += [("random_head", seed_bytes(S["entropy"], "identity-swap-random-E")), ("no_head_z0", None)]
    for name, E in heads:
        ev = X.eval(theta, E=E, label="genuine, " + name)
        rows["genuine_other_head"].append(compare("genuine_" + name, ev, ref, "genuine_E_R",
                                                  head=E.hex() if E is not None else None))

    protocol = {}
    pk0 = idm.derive_signer(passport_bytes(theta), "Ed25519").public_key
    own_heads = []
    for name in ("rescale_2", "nullspace_distribution_matched"):
        t = forgeries[name]
        protocol[name] = p = protocol_head(S, t)
        ev = X.eval(t, E=p["E"], label="forgery %s, its protocol head E'^1" % name)
        rows["forgery_own_head"].append(compare(name + "_protocol_head", ev, ref, "genuine_E_R", t, theta,
                                                head=p["E"].hex(), receiver_chain_ok=p["receiver_chain_ok"]))
        own_heads.append((name, "protocol_head", t, p["E"], ev))
        pk = idm.derive_signer(passport_bytes(t), "Ed25519").public_key
        ev = X.eval(t, E=idm.H(b"E0|" + pk), label="forgery %s, stand-in head H(E0|pk')" % name)
        rows["forgery_own_head"].append(compare(name + "_standin_head", ev, ref, "genuine_E_R", t, theta))
        own_heads.append((name, "standin_head", t, idm.H(b"E0|" + pk), ev))
    ev = X.eval(theta, E=idm.H(b"E0|" + pk0), label="genuine, stand-in head of its own key")
    rows["forgery_own_head"].append(compare("genuine_standin_head_control", ev, ref, "genuine_E_R"))
    rows["genuine_forgery_head"], rows["forgery_vs_genuine_same_head"] = [], []
    for name, kind, t, E, ev_f in own_heads:
        ev = X.eval(theta, E=E, label="genuine, %s of forgery %s" % (kind, name))
        rows["genuine_forgery_head"].append(compare("genuine_%s_of_%s" % (kind, name), ev, ref, "genuine_E_R",
                                                    head=E.hex()))
        rows["forgery_vs_genuine_same_head"].append(compare("%s_%s_vs_genuine_same_head" % (name, kind), ev_f, ev,
                                                            "genuine passport under the same head", t, theta,
                                                            head=E.hex()))

    def worst(rs):
        return {"max_abs_d_acc_prototype_pp": max(abs(r["d_acc_prototype_pp"]) for r in rs),
                "max_abs_d_acc_classifier_pp": max(abs(r["d_acc_classifier_pp"]) for r in rs),
                "max_predictions_changed": max(max(r["n_classifier_predictions_changed"],
                                                   r["n_prototype_predictions_changed"]) for r in rs),
                "max_abs_dlogit": max(r["max_abs_dlogit"] for r in rs)}

    def mean_drop(rs):
        return {"mean_d_acc_prototype_pp": float(np.mean([r["d_acc_prototype_pp"] for r in rs])),
                "min_d_acc_prototype_pp": min(r["d_acc_prototype_pp"] for r in rs),
                "max_d_acc_prototype_pp": max(r["d_acc_prototype_pp"] for r in rs),
                "mean_frac_prototype_predictions_changed": float(np.mean(
                    [r["n_prototype_predictions_changed"] / r["n_test"] for r in rs]))}

    cfg = ck.get("config", {})
    out = {"meta": {"environment": environment(), "threads": a.threads, "dtype": a.dtype,
                    "checkpoint": rel(a.checkpoint),
                    "checkpoint_sha256": sha256_file(a.checkpoint), "checkpoint_source": ck.get("source"),
                    "checkpoint_config": cfg, "alpha_dyn": S["alpha"], "E_R": E_R, "rounds": len(hist),
                    "dataset": S["dataset"], "n_test": int(len(S["ys"])), "batch_size": a.batch_size,
                    "entropy": a.entropy, "rescale_seed": a.rescale_seed, "per_branch_draw_0": draw,
                    "eps_impl": EPS_IMPL},
           "reference": summary_of(ref), "genuine_key": key_of(theta),
           "protocol_heads": protocol, "rows": rows,
           "summary": {"forgery_genuine_head": worst(rows["forgery_genuine_head"]),
                       "genuine_other_head": mean_drop(rows["genuine_other_head"]),
                       "forgery_own_head": mean_drop([r for r in rows["forgery_own_head"]
                                                      if r["name"] != "genuine_standin_head_control"]),
                       "genuine_standin_head_control": mean_drop([r for r in rows["forgery_own_head"]
                                                                  if r["name"] == "genuine_standin_head_control"]),
                       "genuine_forgery_head": mean_drop(rows["genuine_forgery_head"]),
                       "forgery_vs_genuine_same_head": worst(rows["forgery_vs_genuine_same_head"]),
                       "head_spread": mean_drop(rows["genuine_other_head"] + rows["genuine_forgery_head"] +
                                                [r for r in rows["forgery_own_head"]
                                                 if r["name"] == "genuine_standin_head_control"])}}
    out["meta"]["seconds"] = time.time() - t_all
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=jsonable)
    print("\nwrote %s  (%.1f s)" % (a.out, out["meta"]["seconds"]))
    print(json.dumps(out["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
