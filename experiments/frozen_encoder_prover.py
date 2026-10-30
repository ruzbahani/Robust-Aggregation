from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from passportfl.data import dirichlet_partition, load_dataset, load_synthetic
from passportfl.fedproto import Client, FedConfig
from passportfl.identity import dequantize
from passportfl.passport import build_model
from passportfl.probe import (MerkleTree, embed_probes, leaf_hash, mean_separation,
                              nearest_prototype, prototype_drift_ratio)
from passportfl.protocol import Receiver
from passportfl.reproducibility import environment

from negative_results import jsonable

VARIANTS = ("untrained", "stale_trained", "untrained_ckpt_global", "stale_trained_ckpt_global")
CHECKS = ("merkle", "subset", "proto", "stability", "drift")
POST_WARMUP_FROM = 6


class FrozenClient(Client):
    def round_step(self):
        stats = self.frozen_stats()
        C, counts, intra, inter = self.compute_prototypes()
        self._last_counts = counts
        contrib = self.holder.issue(C, dict(n_samples=self.n_samples, train_loss=stats["train_loss"],
                                            local_acc=stats["local_acc"], intra_compactness=intra,
                                            inter_separation=inter))
        self.calibrate()
        return contrib

    @torch.no_grad()
    def frozen_stats(self):
        self.model.eval()
        tot, n, correct = 0.0, 0, 0
        for x, y in self.loader:
            x, y = x.to(self.cfg.device), y.to(self.cfg.device)
            loss, logits = self._loss(x, y)
            tot += loss.item() * len(y)
            n += len(y)
            correct += (logits.argmax(1) == y).sum().item()
        return {"train_loss": tot / max(n, 1), "local_acc": correct / max(n, 1)}


def state_digest(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode() + b"\x00" + v.detach().cpu().contiguous().numpy().tobytes())
    for k, v in sorted(model.passport_tensors.items()):
        h.update(k.encode() + b"\x00" + v.detach().cpu().contiguous().numpy().tobytes())
    for pl in model.passport_layers:
        for t in (pl.theta_gamma, pl.theta_beta):
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def execution_check(model, probes, contrib, device) -> tuple:
    fresh = embed_probes(model, probes, device)
    idx = [rv.index for rv in contrib.reveals]
    opened = np.stack([rv.embedding for rv in contrib.reveals])
    single = np.concatenate([model.features(probes[i:i + 1].to(device)).cpu().numpy() for i in idx])
    rec = {"n_opened": len(idx),
           "root_recomputes_from_fresh_pass": MerkleTree([leaf_hash(e) for e in fresh]).root == contrib.F,
           "opened_equal_fresh_pass": bool(np.array_equal(opened, fresh[idx])),
           "opened_paths_verify_fresh_leaves": all(MerkleTree.verify(contrib.F, leaf_hash(fresh[rv.index]), rv.proof)
                                                   for rv in contrib.reveals),
           "single_probe_max_abs_dev": float(np.abs(single - opened).max()),
           "single_probe_quantized_mismatches": int((dequantize(single) != opened).sum())}
    return rec, fresh


def build_variant(name: str, a, K: int, D: int, W: int, shard, ck) -> FrozenClient:
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    cfg = FedConfig(experiment_seed=a.seed, num_classes=K, embed_dim=D, width=W, alpha_dyn=0.0,
                    m_probes=a.m_probes, t_reveal=a.t_reveal, delta_per_round=a.delta,
                    max_proto_ratio=a.max_proto_ratio, min_label_stability=a.min_label_stability,
                    scheme=a.scheme, device=a.device)
    model = build_model(K, D, W, passport=True, alpha=0.0)
    if name.startswith("stale_trained"):
        model.load_state_dict(ck["state"])
    if ck is not None:
        model.install_passport({k: v.detach().clone() for k, v in ck["passport"].items()})
    client = FrozenClient(a.client, model, shard, cfg)
    if name.endswith("_ckpt_global"):
        client.receive_global(ck["global_C"].detach().cpu().numpy())
    return client


def run_variant(name: str, a, K: int, D: int, W: int, shard, test, ck) -> dict:
    t0 = time.time()
    client = build_variant(name, a, K, D, W, shard, ck)
    model, holder = client.model, client.holder
    if client.cfg.alpha_dyn != 0.0 or any(pl.alpha != 0.0 for pl in model.passport_layers):
        raise AssertionError("alpha_dyn must be 0: calibrate() would otherwise train")
    digest0 = state_digest(model)
    rx = {w: Receiver(a.m_probes, a.t_reveal, a.delta, a.max_proto_ratio, a.min_label_stability, w)
          for w in a.warmups}
    w0 = a.warmups[0]
    emb_round1, seen, log = None, {}, []
    for r in range(1, a.rounds + 1):
        st = rx[w0].peers.get(holder.G)
        hist = dict(st.probe_history) if st is not None else {}
        C_prev = st.C_prev if st is not None else None
        contrib = client.round_step()
        res = {w: rx[w].verify(contrib) for w in a.warmups}

        if state_digest(model) != digest0 or len(client.opt.state) != 0 or model.training:
            raise AssertionError("round %d: the frozen model changed state" % r)
        ex, fresh = execution_check(model, holder.probes, contrib, a.device)
        emb_round1 = fresh if emb_round1 is None else emb_round1
        ex["probe_embeddings_equal_round1"] = bool(np.array_equal(fresh, emb_round1))

        row = {"round": r,
               "verdicts": {str(w): {"accepted": bool(res[w].accepted), "reason": res[w].reason,
                                     "chain_ok": bool(res[w].chain_ok),
                                     "checks": ({c: bool(getattr(res[w].probe, c + "_ok")) for c in CHECKS}
                                                if res[w].probe is not None else None)}
                            for w in a.warmups},
               "record": {"train_loss": contrib.record.train_loss, "local_acc": contrib.record.local_acc},
               "execution": ex, "state_unchanged": True}
        p = res[w0].probe
        if p is not None:
            for w in a.warmups:
                q = res[w].probe
                if (q.proto_drift_ratio, q.label_stability, q.max_drift) != (p.proto_drift_ratio, p.label_stability, p.max_drift):
                    raise AssertionError("round %d: receivers disagree on the statistics" % r)
            row.update({"proto_ratio": p.proto_drift_ratio, "label_stability": p.label_stability,
                        "max_drift": p.max_drift, "proto_velocity": p.proto_velocity,
                        "probe_velocity": p.probe_velocity,
                        "coupling": p.coupling if p.probe_velocity > 0 else None,
                        "drift_alignment": p.drift_alignment})

        C, pr = contrib.C, np.asarray(client._last_counts) > 0
        row["mean_separation"] = mean_separation(C)
        row["mean_separation_present"] = mean_separation(C[pr])
        if C_prev is not None:
            row["proto_ratio_present"] = prototype_drift_ratio(C[pr], C_prev[pr])
            row["stability_all_M_probes"] = float((nearest_prototype(fresh, C) == nearest_prototype(fresh, C_prev)).mean())
            if hist:
                stored = np.stack([e for (_, e) in hist.values()])
                row["stability_present"] = float((nearest_prototype(stored, C[pr]) ==
                                                  nearest_prototype(stored, C_prev[pr])).mean())
                row["frac_stored_nearest_absent_row"] = float((~pr[nearest_prototype(stored, C)]).mean())
        scale, rates = mean_separation(C) + 1e-12, []
        for rv in contrib.reveals:
            if rv.index in seen:
                r_seen, e_seen = seen[rv.index]
                rates.append(float(np.linalg.norm(rv.embedding - e_seen)) / scale / max(1, r - r_seen))
            seen[rv.index] = (r, rv.embedding)
        row["max_drift_rate"] = max(rates) if rates else 0.0
        log.append(row)
        print(json.dumps({"variant": name, "round": r, "elapsed_s": round(time.time() - t0, 1),
                          "accepted": {w: v["accepted"] for w, v in row["verdicts"].items()},
                          "ratio": round(row.get("proto_ratio", float("nan")), 4),
                          "stability": round(row.get("label_stability", float("nan")), 4),
                          "exec_ok": ex["root_recomputes_from_fresh_pass"] and ex["opened_equal_fresh_pass"]}),
              flush=True)

    evaluation, g_saved = {}, client.global_C
    client.receive_global(contrib.C)
    evaluation["own_published_prototypes_last_round"] = client.evaluate(test)
    if name.startswith("stale_trained"):
        client.receive_global(ck["global_C"].detach().cpu().numpy())
        evaluation["checkpoint_global_prototypes"] = client.evaluate(test)
    client.global_C = g_saved
    if state_digest(model) != digest0:
        raise AssertionError("evaluation changed the frozen model")

    counts = np.asarray(client._last_counts)
    return {"config": {"variant": name,
                       "model": (os.path.relpath(a.checkpoint, REPO).replace("\\", "/")
                                 if name.startswith("stale_trained") else "seed-%d random initialisation" % a.seed),
                       "passport": ("checkpoint passport" if ck is not None else
                                    "experiment_entropy(seed, client, 'federation-v1')"),
                       "alpha_dyn": 0.0, "optimizer_steps": 0, "scheme": a.scheme},
            "identity": {"G": holder.G.hex(), "pk_sha256": hashlib.sha256(holder.pk).hexdigest(),
                         "F0": holder.F0.hex(),
                         "equals_main_run_client_G": (holder.G == ck["G"]) if ck is not None else None},
            "shard": {"n_samples": int(client.n_samples), "classes_present": int((counts > 0).sum()),
                      "classes_absent": int((counts == 0).sum())},
            "state": {"digest": digest0, "unchanged_all_rounds": True, "optimizer_state_entries": len(client.opt.state)},
            "evaluation": evaluation,
            "summary": summarize(log, a.warmups),
            "seconds": time.time() - t0,
            "per_round": log}


def _median(log, key):
    vals = [l[key] for l in log if l.get(key) is not None and l["round"] >= POST_WARMUP_FROM]
    return float(np.median(vals)) if vals else None


def summarize(log: list, warmups: list) -> dict:
    out = {"rounds": len(log), "post_warmup_window": [POST_WARMUP_FROM, len(log)], "per_warmup": {}}
    for w in warmups:
        v = [l["verdicts"][str(w)] for l in log]
        failed = {c: [l["round"] for l, x in zip(log, v) if x["checks"] is not None and not x["checks"][c]]
                  for c in CHECKS}
        out["per_warmup"][str(w)] = {
            "accepted_rounds": sum(x["accepted"] for x in v),
            "identity_chain_passed_rounds": sum(x["chain_ok"] for x in v),
            "rejected_rounds": [l["round"] for l, x in zip(log, v) if not x["accepted"]],
            "reject_reasons": dict(Counter(x["reason"] for x in v if not x["accepted"])),
            "failed_checks_by_round": {c: rs for c, rs in failed.items() if rs}}
    out["medians_rounds_6_on"] = {k: _median(log, k) for k in (
        "proto_ratio", "label_stability", "proto_ratio_present", "stability_present",
        "stability_all_M_probes", "frac_stored_nearest_absent_row", "proto_velocity",
        "mean_separation", "mean_separation_present")}
    out["max_proto_ratio_rounds_2_on"] = max((l["proto_ratio"] for l in log if l["round"] >= 2 and "proto_ratio" in l),
                                             default=None)
    out["min_label_stability_rounds_2_on"] = min((l["label_stability"] for l in log
                                                  if l["round"] >= 2 and "label_stability" in l), default=None)
    out["max_max_drift"] = max(l.get("max_drift", 0.0) for l in log)
    out["max_drift_rate"] = max(l["max_drift_rate"] for l in log)
    ex = [l["execution"] for l in log]
    out["execution"] = {
        "rounds_root_recomputes_from_fresh_pass": sum(e["root_recomputes_from_fresh_pass"] for e in ex),
        "rounds_opened_equal_fresh_pass": sum(e["opened_equal_fresh_pass"] for e in ex),
        "rounds_opened_paths_verify_fresh_leaves": sum(e["opened_paths_verify_fresh_leaves"] for e in ex),
        "rounds_probe_embeddings_equal_round1": sum(e["probe_embeddings_equal_round1"] for e in ex),
        "single_probe_max_abs_dev": max(e["single_probe_max_abs_dev"] for e in ex),
        "single_probe_quantized_mismatches": sum(e["single_probe_quantized_mismatches"] for e in ex),
        "opened_embeddings_checked": sum(e["n_opened"] for e in ex)}
    out["rounds_state_unchanged"] = sum(l["state_unchanged"] for l in log)
    return out


def reference_block() -> dict:
    ref = {}
    p = os.path.join(REPO, "results", "fedproto_cifar100.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        w = d["config"]["warmup"]
        rows = [l for l in d["verify_log"] if l["round"] > w]
        ref["honest_cifar100"] = {"file": "results/fedproto_cifar100.json", "rounds_after_warmup": w,
                                  "client_rounds": len(rows),
                                  "median_proto_ratio": float(np.median([l["proto_ratio"] for l in rows])),
                                  "median_label_stability": float(np.median([l["stability"] for l in rows])),
                                  "max_max_drift": float(max(l["max_drift"] for l in rows))}
    p = os.path.join(REPO, "results", "negative_results.json")
    if os.path.exists(p):
        g = json.load(open(p, encoding="utf-8")).get("N3_ghost_prover")
        if g:
            rows = [l for l in g["per_round"] if l["round"] >= POST_WARMUP_FROM]
            ref["ghost_prover"] = {"file": "results/negative_results.json", "accepted_rounds": g["accepted_rounds"],
                                   "rounds": g["rounds"], "warmup_rounds": g["warmup_rounds"],
                                   "median_proto_ratio": float(np.median([l["proto_ratio"] for l in rows])),
                                   "median_label_stability": float(np.median([l["label_stability"] for l in rows])),
                                   "max_max_drift": g["max_max_drift"]}
    return ref


def verdicts(out: dict) -> dict:
    res = {}
    for name, v in out["variants"].items():
        s = v["summary"]
        acc = ", ".join("warm-up %s: %d/%d" % (w, x["accepted_rounds"], s["rounds"])
                        for w, x in s["per_warmup"].items())
        fails = "; ".join("warm-up %s fails %s" % (w, ", ".join("%s in rounds %s" % (c, rs)
                                                                for c, rs in x["failed_checks_by_round"].items()))
                          for w, x in s["per_warmup"].items() if x["failed_checks_by_round"])
        e = s["execution"]
        ev = v["evaluation"]["own_published_prototypes_last_round"]
        res[name] = ("accepted %s; %s; execution verified in %d/%d rounds (fresh-pass root and openings); "
                     "0 optimizer steps, state unchanged in %d/%d rounds; test prototype accuracy %.4f, "
                     "classifier %.4f"
                     % (acc, fails or "no check fails", min(e["rounds_root_recomputes_from_fresh_pass"],
                                                            e["rounds_opened_equal_fresh_pass"]),
                        s["rounds"], s["rounds_state_unchanged"], s["rounds"],
                        ev.get("acc_prototype", float("nan")), ev["acc_classifier"]))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--warmups", default="5,0", help="one receiver per warm-up value")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--client", type=int, default=0)
    ap.add_argument("--dirichlet", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--m-probes", type=int, default=256)
    ap.add_argument("--t-reveal", type=int, default=32)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--max-proto-ratio", type=float, default=1.0)
    ap.add_argument("--min-label-stability", type=float, default=0.5)
    ap.add_argument("--scheme", default="ML-DSA-44")
    ap.add_argument("--checkpoint", default=os.path.join(REPO, "checkpoints", "cifar100_client0.pt"))
    ap.add_argument("--data-root", default=os.path.join(REPO, "data"))
    ap.add_argument("--synthetic", action="store_true", help="tiny model and synthetic data; untrained only")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "frozen_encoder_prover.json"))
    a = ap.parse_args()
    a.warmups = [int(w) for w in a.warmups.split(",")]
    names = [v.strip() for v in a.variants.split(",") if v.strip()]
    if any(v not in VARIANTS for v in names):
        raise SystemExit("unknown variant in %r; choose from %s" % (a.variants, ",".join(VARIANTS)))
    t_all = time.time()

    if a.synthetic:
        K, D, W = 10, 32, 8
        train, test = load_synthetic(600 * a.clients, 300, K, a.seed)
        ck, names = None, ["untrained"]
    else:
        K, D, W = 100, 128, 64
        train, test, _ = load_dataset("cifar100", a.data_root)
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    shard = dirichlet_partition(train, a.clients, a.dirichlet, K, a.seed)[a.client]

    out = {}
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as fh:
            out = json.load(fh)
    out["meta"] = {"environment": environment(), "synthetic": bool(a.synthetic),
                   "dataset": "synthetic" if a.synthetic else "cifar100",
                   "num_classes": K, "embed_dim": D, "width": W, "clients": a.clients, "client": a.client,
                   "dirichlet": a.dirichlet, "seed": a.seed, "rounds": a.rounds, "warmups": a.warmups,
                   "receiver": {"m_probes": a.m_probes, "t_reveal": a.t_reveal, "delta_per_round": a.delta,
                                "max_proto_ratio": a.max_proto_ratio, "min_label_stability": a.min_label_stability},
                   "scheme": a.scheme,
                   "checkpoint": None if ck is None else os.path.relpath(a.checkpoint, REPO).replace("\\", "/"),
                   }
    out.setdefault("variants", {})
    for name in names:
        out["variants"][name] = run_variant(name, a, K, D, W, shard, test, ck)
    out["variants"] = {k: out["variants"][k] for k in VARIANTS if k in out["variants"]}
    out["reference"] = reference_block()
    out["verdicts"] = verdicts(out)
    out["meta"]["seconds_this_invocation"] = time.time() - t_all

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=jsonable)
    print("\nwrote %s  (%.1f s)" % (a.out, out["meta"]["seconds_this_invocation"]))
    for k, v in out["verdicts"].items():
        print("  %-26s %s" % (k, v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
