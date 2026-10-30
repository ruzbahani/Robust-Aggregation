from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import numpy as np
import torch

import passportfl.protocol as protocol_mod
from passportfl.data import dirichlet_partition, load_cifar100, load_synthetic
from passportfl.fedproto import Client, FedConfig, Server
from passportfl.identity import FIXED_POINT_SCALE
from passportfl.passport import build_model
from passportfl.probe import MerkleTree, leaf_hash, mean_separation
from passportfl.protocol import Receiver
from passportfl.reproducibility import environment

import ghost_extended_eval as gx

SERVER_WARMUP = 10 ** 6
LOG_WARMUPS = (5, 0)
STAT_FIELDS = ("proto_drift_ratio", "label_stability", "max_drift", "proto_velocity",
               "probe_velocity", "n_velocity_probes", "drift_alignment")


_LAST_TABLE: dict = {}
_ORIG_EMBED = protocol_mod.embed_probes


def _capturing_embed(model, probes, device="cpu", batch=64):
    embs = _ORIG_EMBED(model, probes, device, batch)
    _LAST_TABLE[id(model)] = embs
    return embs


def take_table(model, root: bytes) -> np.ndarray:
    Z = _LAST_TABLE.pop(id(model))
    if MerkleTree([leaf_hash(e) for e in Z]).root != root:
        raise AssertionError("captured probe table does not hash to the committed root")
    return Z


def to_codes(Z: np.ndarray) -> np.ndarray:
    q = np.round(np.asarray(Z, np.float64) * FIXED_POINT_SCALE)
    if np.abs(q).max() >= 2 ** 31 or not np.array_equal(q / FIXED_POINT_SCALE, Z):
        raise AssertionError("probe table is not an exact int32 fixed-point grid")
    return q.astype(np.int32)


def paper_bands(path: str) -> dict:
    g = json.load(open(path, encoding="utf-8"))
    return {s: tuple(float(x) for x in g["honest_reference"]["bands_p1_p99"][s]) for s in gx.STATS}


def logging_receivers(bands: dict) -> dict:
    rx = {"deployed_w%d" % w: Receiver(gx.M, gx.T, gx.DELTA, gx.MAX_RATIO, gx.MIN_STAB, w)
          for w in LOG_WARMUPS}
    rx.update({"two_sided_paper_w%d" % w: gx.ChallengeReceiver(w, bands) for w in LOG_WARMUPS})
    return rx


def opened_with_history(rx: Receiver, c) -> list:
    st = rx.peers.get(c.client_G)
    hist = st.probe_history if st is not None else {}
    scale = mean_separation(c.C) + 1e-12
    out = []
    for rv in c.reveals:
        if rv.index in hist:
            r0, e0 = hist[rv.index]
            el = max(1, c.round - r0)
            d = float(np.linalg.norm(rv.embedding - e0)) / scale
            out.append((int(rv.index), int(r0), int(el), d, d / el))
    return out


def log_client_round(rx: dict, c, cid: int, server_entry: dict, n_present: int) -> dict:
    first = rx[next(iter(rx))]
    st = first.peers.get(c.client_G)
    defined = {"proto_ratio": st is not None and st.C_prev is not None,
               "label_stability": st is not None and st.C_prev is not None and bool(st.probe_history)}
    probes = opened_with_history(first, c)
    defined["max_drift"] = bool(probes)
    verdicts, results = {}, {}
    for name, v in rx.items():
        res = v.verify(c)
        results[name] = res
        verdicts[name] = {"accepted": bool(res.accepted), "reason": res.reason}
        if isinstance(v, gx.ChallengeReceiver):
            verdicts[name]["band_fail"] = list(v.last.get("band_fail", []))
    row = {"round": int(c.round), "client": int(cid), "G": c.client_G.hex()[:8],
           "server_accepted": bool(server_entry["accepted"]), "server_reason": server_entry["reason"],
           "defined": defined, "mean_separation": mean_separation(c.C), "n_present": int(n_present),
           "verdicts": verdicts}
    if any(res.probe is None for res in results.values()):
        row["receiver_stat_mismatch"] = len(results)
        return row
    probe = results["deployed_w0"].probe
    ref = tuple(getattr(probe, k) for k in STAT_FIELDS)
    row["receiver_stat_mismatch"] = int(sum(tuple(getattr(res.probe, k) for k in STAT_FIELDS) != ref
                                            for res in results.values()))
    srv = (server_entry["proto_ratio"], server_entry["stability"], server_entry["max_drift"])
    row["server_stat_mismatch"] = int(srv != (probe.proto_drift_ratio, probe.label_stability,
                                              probe.max_drift))
    rates = [x[4] for x in probes]
    row.update({
        "proto_ratio": probe.proto_drift_ratio, "label_stability": probe.label_stability,
        "max_drift": probe.max_drift, "max_drift_rate": max(rates) if rates else None,
        "mean_drift_rate_sepC": float(np.mean(rates)) if rates else None,
        "mean_drift_rate_challenge_receiver": rx["two_sided_paper_w0"].last.get("mean_drift_rate"),
        "proto_velocity": probe.proto_velocity, "probe_velocity": probe.probe_velocity,
        "n_velocity_probes": probe.n_velocity_probes, "drift_alignment": probe.drift_alignment,
        "checks_no_warmup": {
            "proto": bool(probe.proto_drift_ratio <= gx.MAX_RATIO),
            "stability": bool(probe.label_stability >= gx.MIN_STAB),
            "drift": bool(all(x[3] <= gx.DELTA * x[2] for x in probes))},
        "probes": {"index": [x[0] for x in probes], "r0": [x[1] for x in probes],
                   "elapsed": [x[2] for x in probes], "drift": [x[3] for x in probes],
                   "rate": rates}})
    chk = row["checks_no_warmup"]
    row["recompute_mismatch"] = int((max((x[3] for x in probes), default=0.0) != probe.max_drift)
                                    + (chk["proto"] != probe.proto_ok)
                                    + (chk["stability"] != probe.stability_ok)
                                    + (chk["drift"] != probe.drift_ok))
    return row


def rel(path: str) -> str:
    try:
        r = os.path.relpath(path, REPO)
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")
    return (os.path.abspath(path) if r.startswith("..") else r).replace("\\", "/")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarise(rows: list, rounds: int) -> dict:
    out = {}
    for lo, hi in ((1, rounds), (6, rounds), (12, min(35, rounds))):
        sel = [x for x in rows if lo <= x["round"] <= hi]
        if not sel:
            continue
        out["rounds_%d_%d" % (lo, hi)] = {
            "client_rounds": len(sel),
            **{n: int(sum(x["verdicts"][n]["accepted"] for x in sel)) for n in sel[0]["verdicts"]}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--dirichlet", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--scheme", default="ML-DSA-44")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    ap.add_argument("--paper-bands", default=os.path.join(REPO, "results", "ghost_extended.json"))
    ap.add_argument("--reference", default=os.path.join(REPO, "results", "fedproto_cifar100.json"),
                    help="reference run whose client identities are compared (same seed only)")
    ap.add_argument("--archive-dir", default=os.path.join(REPO, "results", "paired_honest"))
    ap.add_argument("--no-probe-tables", action="store_true")
    ap.add_argument("--out", default="", help="default results/paired_honest_s<seed>.json")
    a = ap.parse_args()

    out_path = os.path.abspath(a.out or os.path.join(REPO, "results", "paired_honest_s%d.json" % a.seed))
    arch_dir = os.path.abspath(a.archive_dir)
    if os.path.basename(out_path).startswith("fedproto_"):
        raise SystemExit("refusing to write %s: fedproto_*.json are released runs" % out_path)
    if os.path.normcase(arch_dir) == os.path.normcase(os.path.abspath(os.path.join(REPO, "results", "archives"))):
        raise SystemExit("--archive-dir must not be results/archives, which existing scripts list")
    torch.set_num_threads(max(1, a.threads))
    t_all = time.time()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cfg = FedConfig(experiment_seed=a.seed, num_classes=a.num_classes, embed_dim=a.embed_dim, width=a.width,
                    rounds=a.rounds, local_epochs=a.local_epochs, lr=a.lr, lam=a.lam, alpha_dyn=0.0,
                    m_probes=gx.M, t_reveal=gx.T, delta_per_round=gx.DELTA, max_proto_ratio=gx.MAX_RATIO,
                    min_label_stability=gx.MIN_STAB, warmup_rounds=SERVER_WARMUP, scheme=a.scheme,
                    device=a.device)
    if a.synthetic:
        train, test = load_synthetic(600 * a.clients, 300, a.num_classes, a.seed)
    else:
        train, test = load_cifar100(os.path.join(REPO, "data"))
    parts = dirichlet_partition(train, a.clients, a.dirichlet, a.num_classes, a.seed)

    protocol_mod.embed_probes = _capturing_embed
    clients, tables, roots = [], [], []
    for i, p in enumerate(parts):
        m = build_model(a.num_classes, a.embed_dim, a.width, passport=True, alpha=0.0)
        c = Client(i, m, p, cfg)
        clients.append(c)
        tables.append([to_codes(take_table(c.model, c.holder.F0))])
        roots.append([c.holder.F0.hex()])
    server = Server(cfg)
    bands = paper_bands(a.paper_bands)
    rx = logging_receivers(bands)
    opened = [[] for _ in clients]

    ident = {"G": [c.holder.G.hex() for c in clients],
             "pk_sha256": [hashlib.sha256(c.holder.pk).hexdigest() for c in clients],
             "F0": [c.holder.F0.hex() for c in clients]}
    if os.path.exists(a.reference) and not a.synthetic:
        ref = json.load(open(a.reference, encoding="utf-8"))
        if int(ref["config"]["seed"]) == a.seed and ref["config"]["scheme"] == a.scheme:
            refG = [e["G"] for e in ref["verify_log"] if int(e["round"]) == 1]
            ident["equals_reference_run"] = refG == [g[:8] for g in ident["G"]]
            ident["reference"] = rel(a.reference)
            same = []
            for c in clients:
                ck = os.path.join(REPO, "checkpoints", "cifar100_client%d.pt" % c.cid)
                if os.path.exists(ck):
                    pp = torch.load(ck, map_location="cpu", weights_only=False)["passport"]
                    same.append(all(torch.equal(c.model.passport_tensors[k].cpu(), v.cpu()) for k, v in pp.items()))
            ident["passport_equals_reference_checkpoint"] = same

    arch = {"C": {i: {} for i in range(a.clients)}, "present": {i: {} for i in range(a.clients)}, "cbar": {}}
    hist, rows = [], []
    for r in range(1, a.rounds + 1):
        t0 = time.time()
        contribs, counts = [], []
        for c in clients:
            contribs.append(c.round_step()); counts.append(c._last_counts)
            Z = take_table(c.model, contribs[-1].F)
            idx = [rv.index for rv in contribs[-1].reveals]
            if not np.array_equal(np.stack([rv.embedding for rv in contribs[-1].reveals]), Z[idx]):
                raise AssertionError("opened embeddings differ from the captured table")
            if not a.no_probe_tables:
                tables[c.cid].append(to_codes(Z))
            roots[c.cid].append(contribs[-1].F.hex())
            opened[c.cid].append(idx)
        t_issue = time.time() - t0
        t1 = time.time()
        n_log = len(server.log)
        C = server.aggregate(contribs, counts)
        t_verify = time.time() - t1
        t2 = time.time()
        for c, contrib, cnt, entry in zip(clients, contribs, counts, server.log[n_log:]):
            rows.append(log_client_round(rx, contrib, c.cid, entry, int((np.asarray(cnt) > 0).sum())))
            arch["C"][c.cid][r] = contrib.C.copy()
            arch["present"][c.cid][r] = (np.asarray(cnt) > 0)
        arch["cbar"][r] = C.copy()
        t_log = time.time() - t2
        for c in clients:
            c.receive_global(C)
        t3 = time.time()
        accs = [c.evaluate(test) for c in clients]
        cur = [x for x in rows if x["round"] == r]
        row = {"round": r,
               "acc_classifier": float(np.mean([x["acc_classifier"] for x in accs])),
               "acc_prototype": float(np.mean([x["acc_prototype"] for x in accs])),
               "accepted": sum(1 for l in server.log if l["round"] == r and l["accepted"]),
               "wire_bytes_mean": float(np.mean([l["wire_bytes"] for l in server.log if l["round"] == r])),
               "logged_accepted": {n: int(sum(x["verdicts"][n]["accepted"] for x in cur)) for n in rx},
               "t_issue_s": t_issue, "t_verify_s": t_verify, "t_log_s": t_log,
               "t_eval_s": time.time() - t3}
        hist.append(row)
        print(json.dumps(row), flush=True)
    protocol_mod.embed_probes = _ORIG_EMBED

    os.makedirs(arch_dir, exist_ok=True)
    payload_path = os.path.join(arch_dir, "paired_honest_payload_s%d.npz" % a.seed)
    np.savez_compressed(payload_path, C=arch["C"], present=arch["present"], cbar=arch["cbar"])
    files = {"payload_archive": payload_path}
    if not a.no_probe_tables:
        probes_path = os.path.join(arch_dir, "paired_honest_probes_s%d.npz" % a.seed)
        np.savez_compressed(probes_path, Zq=np.stack([np.stack(t) for t in tables]),
                            opened=np.asarray(opened, np.int16), roots=np.asarray(roots),
                            G=np.asarray(ident["G"]), scale=np.int64(FIXED_POINT_SCALE))
        files["probe_tables"] = probes_path

    checks = {
        "client_rounds": len(rows),
        "server_rejections": int(sum(not x["server_accepted"] for x in rows)),
        "server_degenerate_rounds": list(server.degenerate_rounds),
        "receiver_stat_mismatches": int(sum(x["receiver_stat_mismatch"] for x in rows)),
        "server_stat_mismatches": int(sum(x.get("server_stat_mismatch", 0) for x in rows)),
        "recompute_mismatches": int(sum(x.get("recompute_mismatch", 0) for x in rows)),
        "probe_tables_hash_to_committed_roots": True,
        "opened_embeddings_equal_table": True,
        "reasons": {n: dict(Counter(x["verdicts"][n]["reason"] for x in rows if not x["verdicts"][n]["accepted"]))
                    for n in rx}}
    if checks["server_rejections"] or checks["receiver_stat_mismatches"] or checks["server_stat_mismatches"] \
            or checks["recompute_mismatches"]:
        print("[WARN] consistency checks failed: %s" % {k: v for k, v in checks.items() if k != "reasons"},
              file=sys.stderr)

    a_cfg = vars(a).copy()
    a_cfg.update({"out": rel(out_path), "archive_dir": rel(arch_dir), "paper_bands": rel(a.paper_bands),
                  "reference": rel(a.reference)})
    out = {
        "config": {**a_cfg, "warmup": SERVER_WARMUP, "alpha_dyn": 0.0, "passport": True,
                   "m_probes": gx.M, "t_reveal": gx.T, "delta": gx.DELTA,
                   "max_proto_ratio": gx.MAX_RATIO, "min_label_stability": gx.MIN_STAB,
                   "audit_min_history": cfg.audit_min_history, "audit_z_threshold": cfg.audit_z_threshold},
        "meta": {"script": "experiments/paired_honest_reference.py",
                 "command": "python experiments/paired_honest_reference.py " + " ".join(sys.argv[1:]),
                 "environment": environment(), "torch_threads": torch.get_num_threads(),
                 "total_seconds": time.time() - t_all,
                 "files": {k: {"path": rel(v), "sha256": sha256_file(v),
                               "bytes": os.path.getsize(v)} for k, v in files.items()},
                 },
        "receivers": {
            **{"deployed_w%d" % w: {"class": "passportfl.protocol.Receiver", "warmup": w,
                                    "limits": {"m": gx.M, "t": gx.T, "delta": gx.DELTA,
                                               "max_proto_ratio": gx.MAX_RATIO,
                                               "min_label_stability": gx.MIN_STAB}} for w in LOG_WARMUPS},
            **{"two_sided_paper_w%d" % w: {"class": "experiments/ghost_extended_eval.ChallengeReceiver",
                                           "warmup": w, "challenge": "chain_head",
                                           "bands": {s: list(b) for s, b in bands.items()},
                                           "bands_source": rel(a.paper_bands)
                                           + " honest_reference.bands_p1_p99"} for w in LOG_WARMUPS},
            "server": {"warmup": SERVER_WARMUP}},
        "identity": ident,
        "checks": checks,
        "summary": summarise(rows, a.rounds),
        "history": hist,
        "client_rounds": rows,
        "verify_log": server.log,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=gx.jsonable)
    print(json.dumps(out["summary"]))
    print("wrote %s and %s (%.0fs)" % (out_path, ", ".join(files.values()), out["meta"]["total_seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
