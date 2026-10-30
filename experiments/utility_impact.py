from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import Subset

import passportfl.adaptive_attack as adaptive_attack_mod
import passportfl.combined_attack as combined_attack_mod
from passportfl.adaptive_attack import (FIXED_POINT_SCALE, AttackConfig, WhiteBoxAttacker,
                                        _entry_from, _masked_cosine, _peer_round_stats,
                                        build_view)
from passportfl.baselines import BASELINES, score_round
from passportfl.combined_attack import (BASE_STATS, COMB_STATS, RULES, CombinedAttackConfig,
                                        CombinedAttacker, apply_norm, cohort_z_score, combine_one)
from passportfl.data import dirichlet_partition, labels_of
from passportfl.fedproto import Client, FedConfig, Server
from passportfl.innovation import EPS, InnovationAuditor
from passportfl.passport import build_model
from passportfl.reproducibility import environment

from online_eval import load_archive_lazily, roc_auc

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RULE_STATS = ["combined_" + r for r in RULES]
ALL_STATS = RULE_STATS + COMB_STATS


def parse_scenarios(names: Sequence[str], clients: int, victim: int) -> Dict[str, dict]:
    trio = sorted({(victim - k) % clients for k in range(3)})
    table = {
        "honest": dict(kind="honest", attackers=[], beta=0.0),
        "exclude": dict(kind="exclude", attackers=[victim], beta=0.0),
        "echo": dict(kind="echo", attackers=[victim], beta=0.0),
        "echo_x3": dict(kind="echo", attackers=trio, beta=0.0),
        "fused_b0": dict(kind="fused", attackers=[victim], beta=0.0),
        "fused_b0.25": dict(kind="fused", attackers=[victim], beta=0.25),
        "fused_b0.5": dict(kind="fused", attackers=[victim], beta=0.5),
        "fused_b1": dict(kind="fused", attackers=[victim], beta=1.0),
        "fused_b2": dict(kind="fused", attackers=[victim], beta=2.0),
        "fused_b4": dict(kind="fused", attackers=[victim], beta=4.0),
        "fused_b1_x3": dict(kind="fused", attackers=trio, beta=1.0),
        "fused_b0.5_x3": dict(kind="fused", attackers=trio, beta=0.5),
        "wb_b1": dict(kind="whitebox", attackers=[victim], beta=1.0),
        "wb_b2": dict(kind="whitebox", attackers=[victim], beta=2.0),
        "honest_rep": dict(kind="honest", attackers=[], beta=0.0),
    }
    for n, t in list(table.items()):
        if t["kind"] in ("fused", "whitebox"):
            table[n + "_lag"] = dict(t, knowledge="peer_lagged")
            if len(t["attackers"]) > 1:
                table[n + "_lag"]["coalition"] = "sequential"
                table[n + "_lag_nocoord"] = dict(t, knowledge="peer_lagged", coalition="none")
    out = {}
    for n in names:
        if n not in table:
            raise SystemExit("unknown scenario %r; known: %s" % (n, ",".join(table)))
        out[n] = dict(table[n], name=n)
    return out


def cap_shards(parts, cap: int, seed: int):
    if cap <= 0:
        return parts
    rng = np.random.default_rng(10_000 + seed)
    out = []
    for p in parts:
        idx = np.asarray(p.indices)
        if len(idx) > cap:
            idx = np.sort(rng.choice(idx, size=cap, replace=False))
        out.append(Subset(p.dataset, [int(i) for i in idx]))
    return out


def build_federation(a, cfg, parts, num_classes):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    clients = []
    for i, p in enumerate(parts):
        h = hashlib.sha256(f"passportfl|utility|seed{a.seed}|client{i}".encode()).digest()
        entropy = h + hashlib.sha256(h).digest()
        model = build_model(num_classes, a.embed_dim, a.width, True, 0.0)
        clients.append(Client(i, model, p, cfg, entropy=entropy))
    return clients, Server(cfg)


def snapshot_parameters(clients, server, auditor):
    return {
        "clients": [{
            "model": {k: v.detach().cpu().clone() for k, v in c.model.state_dict().items()},
            "opt": copy.deepcopy(c.opt.state_dict()),
            "holder_round": c.holder.round,
            "holder_E": c.holder.E,
            "holder_prev_C": None if c.holder.prev_C is None else c.holder.prev_C.copy(),
            "global_C": None if c.global_C is None else c.global_C.detach().cpu().clone(),
        } for c in clients],
        "server": copy.deepcopy(server),
        "auditor": copy.deepcopy(auditor),
        "torch_rng": torch.get_rng_state().clone(),
        "np_rng": np.random.get_state(),
    }


def restore_parameters(snap, clients):
    for c, s in zip(clients, snap["clients"]):
        c.model.load_state_dict(s["model"])
        c.opt.load_state_dict(copy.deepcopy(s["opt"]))
        c.holder.round = s["holder_round"]
        c.holder.E = s["holder_E"]
        c.holder.prev_C = None if s["holder_prev_C"] is None else s["holder_prev_C"].copy()
        c.global_C = None if s["global_C"] is None else s["global_C"].to(c.cfg.device)
        c.model.install_dynamic(s["holder_E"])
    torch.set_rng_state(snap["torch_rng"].clone())
    np.random.set_state(snap["np_rng"])
    return copy.deepcopy(snap["server"]), copy.deepcopy(snap["auditor"])


def label_counts(sub, K: int) -> np.ndarray:
    c = np.zeros(K)
    y = labels_of(sub.dataset)[np.asarray(sub.indices)]
    for k, n in zip(*np.unique(y, return_counts=True)):
        c[int(k)] = float(n)
    return c


def raw_round_stats(rs, payloads: Dict[int, tuple], Cbar_prev: Optional[np.ndarray],
                    cids: Sequence[int], mad_floor: float) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {
        "suspicion": {c: float(rs.per_client[c].get("suspicion", float("nan"))) for c in cids}}
    for b in BASE_STATS:
        out[b] = {c: float("nan") for c in cids}
    if Cbar_prev is not None and len(payloads) >= 3:
        innov = {cid: (payloads[cid][0] - Cbar_prev) for cid in payloads}
        base = score_round(innov)
        for b in BASE_STATS:
            out[b] = cohort_z_score(base.get(b, {}), mad_floor)
    return out


def combined_from_raw(raw, norm, cids) -> Dict[str, Dict[int, float]]:
    z = {c: {k: apply_norm(raw[k].get(c, float("nan")), norm[k]) for k in COMB_STATS}
         for c in cids}
    out = {k: {c: z[c][k] for c in cids} for k in COMB_STATS}
    for r in RULES:
        out["combined_" + r] = {c: combine_one(z[c], r) for c in cids}
    return out


class DelayedInformationLeak(AssertionError):
    pass


class AttackerMasks(dict):
    def __init__(self, cid: int, masks: Dict[int, np.ndarray]):
        super().__init__(masks)
        self.cid = cid

    def __missing__(self, key):
        raise DelayedInformationLeak("delayed attacker %d read the round-r present-mask of "
                                     "client %r" % (self.cid, key))

    def get(self, key, default=None):
        return self[key]


_library_build_view = build_view


def _x_const(aud: InnovationAuditor, view, lag0: Dict[int, object], lag_window: int) -> np.ndarray:
    K = view.K

    def past(j):
        h = aud.hist[j]
        es = (_entry_from(h[len(h) - lag], K) for lag in range(1, min(lag_window, len(h)) + 1))
        return [e for e in es if e is not None]

    pool = {j: ([lag0[j]] if lag0.get(j) is not None else []) + past(j) for j in view.peer_ids}
    own = past(view.victim)
    out = []
    for k, j in enumerate(view.peer_ids):
        ej = view.peer_I[k]
        best = float("nan")
        if ej is not None:
            for e in [e for j2 in view.peer_ids if j2 != j for e in pool[j2]] + own:
                c = _masked_cosine(ej.I, ej.mask, e.I, e.mask)
                if np.isfinite(c) and (not np.isfinite(best) or c > best):
                    best = c
        out.append(best)
    return np.asarray(out)


def coalition_lagged_view(aud: InnovationAuditor, r: int, victim: int, present_r,
                          coalition: Dict[int, np.ndarray], lag_window: int = 5):
    view = _library_build_view(aud, r, victim, present_r, None, "peer_lagged", lag_window)
    if view is None or not coalition:
        return view
    K, Cbar_prev = view.K, view.Cbar_prev
    cols = {f: np.array(getattr(view, f), dtype=np.float64)
            for f in ("peer_u1", "peer_u2", "peer_norm", "peer_step", "peer_rep_raw")}
    lag0 = {}
    for q in sorted(coalition):
        Cq = coalition[q]
        k = view.peer_ids.index(q)
        st = _peer_round_stats(aud, q, Cq, present_r[q], Cbar_prev, K, lag_window)
        view.peer_I[k] = st["entry"]
        view.peer_I_full[k] = Cq - Cbar_prev
        for f, s in (("peer_u1", "u1"), ("peer_u2", "u2"), ("peer_norm", "norm"),
                     ("peer_step", "step"), ("peer_rep_raw", "rep_raw")):
            cols[f][k] = st[s]
        lag0[q] = st["entry"]
        if st["entry"] is not None:
            view.pool_I.append(st["entry"])
    for f, v in cols.items():
        setattr(view, f, v)
    view.peer_x_const = _x_const(aud, view, lag0, lag_window)
    live = [v for v in view.peer_norm if np.isfinite(v) and v > EPS]
    view.cohort_scale = float(np.median(live)) if live else float("nan")
    return view


@contextlib.contextmanager
def coalition_knowledge(coalition: Dict[int, np.ndarray]):
    calls: List[int] = []
    if not coalition:
        yield calls
        return

    def view(aud, r, victim, present_r, peer_C_r, knowledge, lag_window=5):
        if knowledge != "peer_lagged" or peer_C_r is not None:
            raise DelayedInformationLeak("coalition view requested with knowledge=%r and "
                                         "peer payloads %s" % (knowledge, peer_C_r is not None))
        calls.append(r)
        return coalition_lagged_view(aud, r, victim, present_r, coalition, lag_window)

    saved = (combined_attack_mod.build_view, adaptive_attack_mod.build_view)
    combined_attack_mod.build_view = adaptive_attack_mod.build_view = view
    try:
        yield calls
    finally:
        combined_attack_mod.build_view, adaptive_attack_mod.build_view = saved


def check_delayed_inputs(aud: InnovationAuditor, r: int, aim_round: int) -> None:
    last = {"aggregate": max(aud.cbar, default=0),
            "payload": max((cr.round for h in aud.hist.values() for cr in h), default=0),
            "scores": aud.scores[-1].round if aud.scores else 0,
            "aim": aim_round}
    late = {k: v for k, v in last.items() if v > r - 1}
    if late:
        raise DelayedInformationLeak("round-%d delayed solve would read %s" % (r, late))


def run_scenario(spec, clients, snap, a, cfg, test_small, test_full, counts_fixed,
                 norm, honest_cbar, prev_peer_med0=float("nan")):
    K, d = cfg.num_classes, cfg.embed_dim
    kind = spec["kind"]
    knowledge = spec.get("knowledge", "peer_current")
    lagged = knowledge == "peer_lagged"
    coordinated = spec.get("coalition") == "sequential"
    attackers = sorted(spec["attackers"])
    excluded = set(attackers) if kind == "exclude" else set()
    active = [c for c in clients if c.cid not in excluded]
    honest_ids = [c.cid for c in active if c.cid not in attackers]
    cids = sorted(c.cid for c in active)

    server, aud = restore_parameters(snap, clients)

    atk_objs: Dict[int, object] = {}
    if kind in ("fused", "whitebox") and spec["beta"] > 0:
        for j, cid in enumerate(attackers):
            if kind == "fused":
                acfg = CombinedAttackConfig(
                    phi=0.0, beta=spec["beta"], rule=a.rule, knowledge=knowledge,
                    aim="mimic", basis_steps=a.basis_steps, full_steps=a.full_steps,
                    restarts=a.restarts, warm_families=bool(a.warm),
                    mad_floor=a.mad_floor, norm=norm, device=cfg.device)
                atk_objs[cid] = CombinedAttacker(acfg, seed=1000 * a.seed + cid)
            else:
                acfg = AttackConfig(
                    phi=0.0, beta=spec["beta"], knowledge=knowledge, aim="mimic",
                    basis_steps=a.basis_steps, full_steps=a.full_steps,
                    restarts=a.restarts, mad_floor=a.mad_floor, device=cfg.device)
                atk_objs[cid] = WhiteBoxAttacker(acfg, seed=1000 * a.seed + cid)

    last_meta = {c.cid: ({"train_loss": float("nan"), "local_acc": float("nan")}, 0.0, 0.0)
                 for c in clients}

    rows, util, statuses = [], [], {}
    cbar_hist: Dict[int, np.ndarray] = {}
    prev_peer_med = prev_peer_med0
    prev_peer_med_round = a.attack_start - 1
    t0 = time.time()

    for r in range(a.attack_start, a.rounds + 1):
        Cbar_prev = server.global_C.copy() if server.global_C is not None else None
        attacking = bool(attackers) and kind in ("echo", "fused", "whitebox")

        committed: Dict[int, tuple] = {}
        if attacking and lagged and spec["beta"] != 0.0:
            check_delayed_inputs(aud, r, prev_peer_med_round)
            C_honest = Cbar_prev if Cbar_prev is not None else np.zeros((K, d))
            for cid in attackers:
                coal = {}
                if coordinated:
                    for q in attackers:
                        if q == cid:
                            continue
                        coal[q] = (np.round(np.asarray(committed[q][0], dtype=np.float64)
                                            * FIXED_POINT_SCALE) / FIXED_POINT_SCALE
                                   if q in committed else C_honest)
                if not set(coal) <= set(attackers) - {cid}:
                    raise DelayedInformationLeak("coalition of %d holds %s" % (cid, sorted(coal)))
                masks = AttackerMasks(cid, {q: counts_fixed[q] > 0 for q in [cid] + sorted(coal)})
                with coalition_knowledge(coal) as used:
                    if kind == "fused":
                        aim = float(prev_peer_med) if np.isfinite(prev_peer_med) else 0.0
                        committed[cid] = atk_objs[cid].craft(aud, r, cid, C_honest, masks, None,
                                                             aim=aim)
                    else:
                        committed[cid] = atk_objs[cid].craft(aud, r, cid, C_honest, masks, None)
                if coal and not used and committed[cid][1].get("status") != "no_aggregate":
                    raise RuntimeError("attacker %d solved round %d without the coalition view"
                                       % (cid, r))

        true_C, counts, present = {}, {}, {}
        for c in active:
            if attacking and c.cid in attackers:
                counts[c.cid] = counts_fixed[c.cid]
                present[c.cid] = counts_fixed[c.cid] > 0
                continue
            stats = c.local_train()
            C, cnt, intra, inter = c.compute_prototypes()
            true_C[c.cid] = np.asarray(C, dtype=np.float64)
            counts[c.cid] = np.asarray(cnt)
            present[c.cid] = np.asarray(cnt) > 0
            last_meta[c.cid] = (stats, intra, inter)

        pub = {cid: true_C[cid] for cid in true_C}
        infos = {}
        if attacking:
            for cid in attackers:
                if kind == "echo" or spec["beta"] == 0.0:
                    Cp = (Cbar_prev.copy() if Cbar_prev is not None
                          else np.zeros((K, d)))
                    infos[cid] = {"status": "echo", "predicted": float("nan")}
                elif cid in committed:
                    Cp, info = committed[cid]
                    infos[cid] = info
                    statuses[info["status"]] = statuses.get(info["status"], 0) + 1
                else:
                    peer_C = {}
                    for j in cids:
                        if j == cid:
                            continue
                        peer_C[j] = pub.get(j, Cbar_prev)
                    C_honest = Cbar_prev if Cbar_prev is not None else np.zeros((K, d))
                    if kind == "fused":
                        aim = float(prev_peer_med) if np.isfinite(prev_peer_med) else 0.0
                        Cp, info = atk_objs[cid].craft(aud, r, cid, C_honest, present,
                                                       peer_C, aim=aim)
                    else:
                        Cp, info = atk_objs[cid].craft(aud, r, cid, C_honest, present, peer_C)
                    infos[cid] = info
                    statuses[info["status"]] = statuses.get(info["status"], 0) + 1
                pub[cid] = np.round(np.asarray(Cp, dtype=np.float64) * FIXED_POINT_SCALE) / FIXED_POINT_SCALE

        contribs, cnt_list = [], []
        for c in active:
            stats, intra, inter = last_meta[c.cid]
            contribs.append(c.holder.issue(
                pub[c.cid], dict(n_samples=c.n_samples, train_loss=stats["train_loss"],
                                 local_acc=stats["local_acc"], intra_compactness=intra,
                                 inter_separation=inter)))
            cnt_list.append(counts[c.cid])
        for c, ct in zip(active, contribs):
            pub[c.cid] = np.asarray(ct.C, dtype=np.float64)
        Cbar = server.aggregate(contribs, cnt_list)
        cbar_hist[r] = np.asarray(Cbar, dtype=np.float64).copy()

        rs = aud.observe_round(r, {cid: (pub[cid], present[cid]) for cid in cids})
        aud.set_aggregate(r, Cbar)
        payloads = {cid: (pub[cid], present[cid]) for cid in cids}
        raw = raw_round_stats(rs, payloads, Cbar_prev, cids, a.mad_floor)
        comb = combined_from_raw(raw, norm, cids)
        peer_vals = [comb["combined_" + a.rule][c] for c in cids
                     if c not in attackers and np.isfinite(comb["combined_" + a.rule][c])]
        prev_peer_med = float(np.median(peer_vals)) if peer_vals else float("nan")
        prev_peer_med_round = r

        for c in clients:
            c.receive_global(Cbar)

        ds = test_full if (r == a.rounds and test_full is not None) else test_small
        hon = [clients[i].evaluate(ds) for i in honest_ids]
        atk_acc = ([clients[i].evaluate(ds) for i in attackers]
                   if (attackers and kind != "exclude") else [])
        drift = float("nan")
        if honest_cbar is not None and r in honest_cbar:
            hc = honest_cbar[r]
            drift = float(np.linalg.norm(Cbar - hc) / (np.linalg.norm(hc) + 1e-12))

        inorm = {cid: float(np.linalg.norm(pub[cid] - Cbar_prev)) if Cbar_prev is not None
                 else float("nan") for cid in cids}
        hon_inorm = [inorm[c] for c in cids if c not in attackers and np.isfinite(inorm[c])]

        util.append({
            "round": r,
            "n_test": len(ds),
            "acc_prototype_honest": float(np.mean([x.get("acc_prototype", np.nan) for x in hon])),
            "acc_classifier_honest": float(np.mean([x["acc_classifier"] for x in hon])),
            "acc_prototype_honest_std": float(np.std([x.get("acc_prototype", np.nan) for x in hon])),
            "acc_classifier_honest_std": float(np.std([x["acc_classifier"] for x in hon])),
            "acc_prototype_attacker": (float(np.mean([x.get("acc_prototype", np.nan) for x in atk_acc]))
                                       if atk_acc else None),
            "acc_classifier_attacker": (float(np.mean([x["acc_classifier"] for x in atk_acc]))
                                        if atk_acc else None),
            "cbar_rel_drift_vs_honest": drift,
            "attacker_innov_norm": (float(np.mean([inorm[c] for c in attackers]))
                                    if attacking else None),
            "cohort_innov_norm_median": float(np.median(hon_inorm)) if hon_inorm else float("nan"),
        })

        row = {"round": r,
               "suspicion_raw": {c: float(raw["suspicion"][c]) for c in cids},
               "degenerate": {c: bool(rs.per_client[c].get("degenerate", False)) for c in cids}}
        for k in ALL_STATS:
            row[k] = {c: float(comb[k][c]) for c in cids}
        if infos:
            row["attack_info"] = {c: {kk: vv for kk, vv in i.items()
                                      if kk in ("status", "predicted_combined",
                                                "predicted_suspicion", "budget",
                                                "cohort_scale", "objective_gap", "aim")}
                                  for c, i in infos.items()}
        rows.append(row)
        print(json.dumps({"scen": spec["name"], "round": r,
                          "acc_proto_hon": round(util[-1]["acc_prototype_honest"], 4),
                          "acc_cls_hon": round(util[-1]["acc_classifier_honest"], 4),
                          "atk_suspicion": (round(float(np.mean([row["suspicion_raw"][c] for c in attackers])), 3)
                                            if attacking else None),
                          "atk_combined_max": (round(float(np.mean([row["combined_max"][c] for c in attackers])), 3)
                                               if attacking else None),
                          "drift": None if not np.isfinite(drift) else round(drift, 4),
                          "s": round(time.time() - t0, 1)}), flush=True)

    return ({"rows": rows, "utility": util, "statuses": statuses,
             "attackers": attackers, "excluded": sorted(excluded),
             "honest_ids": honest_ids, "elapsed_s": round(time.time() - t0, 1),
             "degenerate_rounds": server.degenerate_rounds},
            cbar_hist)


def summarise_scores(scen, a, taus):
    rows = [x for x in scen["rows"] if x["round"] >= a.eval_from]
    atk = set(scen["attackers"])
    out = {"n_rounds_scored": len(rows)}
    if not atk or scen["excluded"]:
        out["detection"] = None
    else:
        det = {}
        for k in ["suspicion_raw"] + ALL_STATS:
            pos = [x[k][c] for x in rows for c in atk if c in x[k]]
            neg = [v for x in rows for c, v in x[k].items() if c not in atk]
            tau = taus["suspicion_paper"] if k == "suspicion_raw" else taus.get(k)
            flag = (lambda v: np.isfinite(v) and v > tau) if tau is not None else None
            det[k] = {
                "auc": roc_auc(pos, neg),
                "pos_median": float(np.nanmedian(pos)) if pos else None,
                "neg_median": float(np.nanmedian(neg)) if neg else None,
                "tau": tau,
                "tpr_client_round": (float(np.mean([flag(v) for v in pos])) if (pos and flag) else None),
                "tpr_client_run": (float(any(flag(v) for v in pos)) if (pos and flag) else None),
                "fpr_client_round": (float(np.mean([flag(v) for v in neg])) if (neg and flag) else None),
            }
        det["structural_flag_rate"] = float(np.mean(
            [x["degenerate"][c] for x in rows for c in atk if c in x["degenerate"]])) if rows else None
        out["detection"] = det
    u = [x for x in scen["utility"] if x["round"] >= a.eval_from]
    last = scen["utility"][-1] if scen["utility"] else {}
    tail = [x for x in scen["utility"] if x["round"] > a.rounds - a.tail_rounds]
    out["utility"] = {
        "final_round": last.get("round"),
        "final_acc_prototype_honest": last.get("acc_prototype_honest"),
        "final_acc_classifier_honest": last.get("acc_classifier_honest"),
        "final_n_test": last.get("n_test"),
        "tail_acc_prototype_honest": float(np.mean([x["acc_prototype_honest"] for x in tail])) if tail else None,
        "tail_acc_classifier_honest": float(np.mean([x["acc_classifier_honest"] for x in tail])) if tail else None,
        "final_acc_prototype_attacker": last.get("acc_prototype_attacker"),
        "final_acc_classifier_attacker": last.get("acc_classifier_attacker"),
        "mean_cbar_rel_drift": (float(np.mean(_drift)) if (_drift := [x["cbar_rel_drift_vs_honest"]
                                                                     for x in u
                                                                     if np.isfinite(x["cbar_rel_drift_vs_honest"])])
                                else None),
        "final_cbar_rel_drift": last.get("cbar_rel_drift_vs_honest"),
        "attacker_innov_over_cohort": (
            float(np.nanmean([x["attacker_innov_norm"] / x["cohort_innov_norm_median"]
                              for x in u if x.get("attacker_innov_norm")]))
            if any(x.get("attacker_innov_norm") for x in u) else None),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--attack-start", type=int, default=9)
    ap.add_argument("--eval-from", type=int, default=0, help="0 means --attack-start")
    ap.add_argument("--tail-rounds", type=int, default=5)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--dirichlet", type=float, default=0.5)
    ap.add_argument("--shard-cap", type=int, default=2500, help="0 = uncapped")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--victim", type=int, default=7)
    ap.add_argument("--rule", default="max", choices=list(RULES))
    ap.add_argument("--mad-floor", type=float, default=1e-2)
    ap.add_argument("--basis-steps", type=int, default=40)
    ap.add_argument("--full-steps", type=int, default=80)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--warm", type=int, default=0)
    ap.add_argument("--eval-n", type=int, default=2000, help="per-round test subset size")
    ap.add_argument("--full-test-final", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenarios",
                    default="honest,exclude,echo,fused_b0.5,fused_b1,fused_b2,fused_b1_x3")
    ap.add_argument("--grid", default=os.path.join(REPO, "results", "combined_grid.json"))
    ap.add_argument("--proto", default=os.path.join(REPO, "results", "protocol_eval_d05.json"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if not a.eval_from:
        a.eval_from = a.attack_start

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    grid = json.load(open(os.path.abspath(a.grid), encoding="utf-8"))
    norm = {k: list(v) for k, v in grid["norm"].items()}
    taus = {k: float(v) for k, v in grid["thresholds"].items()}
    proto = json.load(open(os.path.abspath(a.proto), encoding="utf-8"))
    taus["suspicion_paper"] = float(proto["thresholds"]["suspicion"]["0.01"])
    taus.setdefault("suspicion", taus["suspicion_paper"])

    train, test, nc = load_archive_lazily(a.dataset, root=os.path.join(REPO, "data"))
    cfg = FedConfig(num_classes=nc, embed_dim=a.embed_dim, width=a.width, rounds=a.rounds,
                    local_epochs=a.local_epochs, warmup_rounds=10 ** 6,
                    audit_z_threshold=10 ** 6, scheme="Ed25519", device=a.device)
    parts = dirichlet_partition(train, a.clients, a.dirichlet, nc, a.seed)
    raw_sizes = [len(p) for p in parts]
    parts = cap_shards(parts, a.shard_cap, a.seed)
    sizes = [len(p) for p in parts]
    print("shard sizes (raw -> capped): %s -> %s" % (raw_sizes, sizes), flush=True)

    rng = np.random.default_rng(777 + a.seed)
    idx = np.sort(rng.choice(len(test), size=min(a.eval_n, len(test)), replace=False))
    test_small = Subset(test, [int(i) for i in idx])
    test_full = test if a.full_test_final else None

    clients, server = build_federation(a, cfg, parts, nc)
    counts_fixed = {c.cid: label_counts(parts[c.cid], nc) for c in clients}

    aud = InnovationAuditor(mad_floor=a.mad_floor)
    t0 = time.time()
    prev_peer_med0 = float("nan")
    all_cids = sorted(c.cid for c in clients)
    for r in range(1, a.attack_start):
        contribs, cnt_list, present, pub = [], [], {}, {}
        Cbar_prev = server.global_C.copy() if server.global_C is not None else None
        for c in clients:
            stats = c.local_train()
            C, cnt, intra, inter = c.compute_prototypes()
            present[c.cid] = np.asarray(cnt) > 0
            contribs.append(c.holder.issue(
                np.asarray(C, dtype=np.float64),
                dict(n_samples=c.n_samples, train_loss=stats["train_loss"],
                     local_acc=stats["local_acc"], intra_compactness=intra,
                     inter_separation=inter)))
            cnt_list.append(np.asarray(cnt))
        for c, ct in zip(clients, contribs):
            pub[c.cid] = np.asarray(ct.C, dtype=np.float64)
        Cbar = server.aggregate(contribs, cnt_list)
        payloads = {c.cid: (pub[c.cid], present[c.cid]) for c in clients}
        rs = aud.observe_round(r, payloads)
        aud.set_aggregate(r, Cbar)
        rawp = raw_round_stats(rs, payloads, Cbar_prev, all_cids, a.mad_floor)
        combp = combined_from_raw(rawp, norm, all_cids)
        vals = [combp["combined_" + a.rule][c] for c in all_cids
                if c != a.victim and np.isfinite(combp["combined_" + a.rule][c])]
        if vals:
            prev_peer_med0 = float(np.median(vals))
        for c in clients:
            c.receive_global(Cbar)
        print(json.dumps({"scen": "prefix", "round": r,
                          "s": round(time.time() - t0, 1)}), flush=True)
    snap = snapshot_parameters(clients, server, aud)
    print("[prefix] rounds 1..%d in %.0fs" % (a.attack_start - 1, time.time() - t0), flush=True)

    specs = parse_scenarios([s for s in a.scenarios.split(",") if s], a.clients, a.victim)
    if "honest" not in specs:
        raise SystemExit("the honest control must be in --scenarios: it is the reference")
    order = ["honest"] + [n for n in specs if n != "honest"]

    out = {"config": vars(a), "environment": environment(),
           "shard_sizes_raw": raw_sizes, "shard_sizes": sizes,
           "taus": taus, "norm": norm, "scenarios": {}, "summary": {}}
    path = os.path.abspath(a.out or os.path.join(REPO, "results",
                                                 "utility_impact_s%d.json" % a.seed))
    honest_cbar = None
    for name in order:
        spec = specs[name]
        res, cbar_hist = run_scenario(spec, clients, snap, a, cfg, test_small, test_full,
                                      counts_fixed, norm, honest_cbar, prev_peer_med0)
        out["scenarios"][name] = dict(res, spec={k: v for k, v in spec.items()})
        out["summary"][name] = summarise_scores(res, a, taus)
        if name == "honest":
            honest_cbar = cbar_hist
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, default=float)
        print("[%s] done in %.0fs -> %s" % (name, res["elapsed_s"], path), flush=True)

    report(out, a)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwrote %s" % path)
    return 0


def report(out: dict, a) -> None:
    S = out["summary"]
    hon = S.get("honest", {}).get("utility", {})
    exc = S.get("exclude", {}).get("utility", {})
    base_p = hon.get("tail_acc_prototype_honest")
    base_c = hon.get("tail_acc_classifier_honest")
    print("\n" + "=" * 132)
    print("HONEST-CLIENT utility and LIVE detection, rounds %d..%d, seed %d "
          "(tail = last %d rounds)" % (a.eval_from, a.rounds, a.seed, a.tail_rounds))
    print("=" * 132)
    hdr = ("%-14s %5s | %8s %8s %8s | %8s %8s | %7s %7s %7s | %7s"
           % ("scenario", "beta", "proto", "d_proto", "cls", "AUCmax", "TPRmax",
              "AUCsus", "TPRsus", "struct", "drift"))
    print(hdr)
    print("-" * len(hdr))
    for name, v in S.items():
        u, d = v["utility"], v["detection"]
        p = u["tail_acc_prototype_honest"]
        c = u["tail_acc_classifier_honest"]
        dp = (p - base_p) if (p is not None and base_p is not None) else float("nan")
        f = lambda x, fmt="%8.4f": (fmt % x) if isinstance(x, float) and np.isfinite(x) else "%8s" % "-"
        g = lambda x, fmt="%7.3f": (fmt % x) if isinstance(x, float) and np.isfinite(x) else "%7s" % "-"
        beta = out["scenarios"][name]["spec"].get("beta")
        if d is None:
            print("%-14s %5s | %s %s %s | %8s %8s | %7s %7s %7s | %s"
                  % (name, beta, f(p), f(dp), f(c), "-", "-", "-", "-", "-",
                     g(u.get("mean_cbar_rel_drift"))))
            continue
        print("%-14s %5s | %s %s %s | %s %s | %s %s %s | %s"
              % (name, beta, f(p), f(dp), f(c),
                 f(d["combined_max"]["auc"]), f(d["combined_max"]["tpr_client_round"]),
                 g(d["suspicion"]["auc"]), g(d["suspicion"]["tpr_client_round"]),
                 g(d.get("structural_flag_rate")),
                 g(u.get("mean_cbar_rel_drift"))))
    if exc:
        print("\nHonest utility with the attacker excluded:")
        print("  honest-10 control          proto %.4f  cls %.4f"
              % (base_p, base_c))
        print("  exclude (9 honest only)    proto %.4f  cls %.4f"
              % (exc["tail_acc_prototype_honest"], exc["tail_acc_classifier_honest"]))
        for name, v in S.items():
            if name in ("honest", "exclude"):
                continue
            u = v["utility"]
            print("  %-24s   proto %.4f  cls %.4f   excluding gains proto %+.4f"
                  % (name, u["tail_acc_prototype_honest"], u["tail_acc_classifier_honest"],
                     exc["tail_acc_prototype_honest"] - u["tail_acc_prototype_honest"]))


if __name__ == "__main__":
    sys.exit(main())
