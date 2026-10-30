import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from passportfl import identity as idm
from passportfl.encoders import (ENCODERS, architecture_descriptor, build_group_model,
                                 group_assignment, rebind_identity, self_check)
from passportfl.fedproto import Client, FedConfig
from passportfl.fedtgp import TGPConfig, TGPServer, TGPTrainer, class_gaps, cluster_centres
from passportfl.passport import build_model

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FA = load_module('fedtgp_attacks', 'experiments/fedtgp_attacks.py')


def random_payloads(K=6, d=5, n=4, seed=0):
    g = np.random.default_rng(seed)
    out = []
    for i in range(n):
        pres = np.ones(K, bool)
        pres[i % K] = False
        out.append((g.normal(size=(K, d)) + np.arange(K)[:, None], pres))
    return out


def test_encoders_interface_and_assignment():
    torch.set_num_threads(1)
    info = self_check(num_classes=10, embed_dim=16)
    assert set(info) == set(ENCODERS)
    names = group_assignment('hetero4', 10)
    assert [names.count(n) for n in ('resnet18_w64', 'resnet18_w32', 'cnn4', 'mobilenetv2')] == [3, 3, 2, 2]
    assert names[:4] == ['resnet18_w64', 'resnet18_w32', 'cnn4', 'mobilenetv2']


def test_homogeneous_group_is_build_model():
    torch.manual_seed(3)
    a = build_model(10, 16, 8, True, 0.0)
    torch.manual_seed(3)
    b = build_group_model('resnet18', 0, 10, 16, 8, True)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert architecture_descriptor(a) == idm.architecture_descriptor(a)
    try:
        build_group_model('hetero4', 0, 10, 16, 8, True)
    except ValueError:
        pass
    else:
        raise AssertionError('a passport request for hetero4 must be refused')


def test_rebind_identity_changes_descriptor_not_random_state():
    torch.set_num_threads(1)
    data = TensorDataset(torch.randn(8, 3, 32, 32), torch.arange(8) % 2)
    cfg = FedConfig(num_classes=2, embed_dim=4, local_epochs=1, batch_size=4, m_probes=4,
                    t_reveal=2, scheme='Ed25519', experiment_seed=0)

    def make(rebind):
        torch.manual_seed(5)
        c = Client(2, build_group_model('hetero4', 2, 2, 4, passport=False), data, cfg)
        G0 = c.holder.G
        if rebind:
            rebind_identity(c.holder)
        return c, G0, torch.rand(3)

    c1, g_plain, r1 = make(False)
    c2, _, r2 = make(True)
    assert torch.equal(r1, r2)
    assert c2.holder.G != g_plain and b'"CNN4"' in c2.holder.arch
    assert c1.holder.arch == idm.architecture_descriptor(c1.model)


def test_margin_matches_official_receive_protos():
    pl = random_payloads()
    pl[0][1][:] = True
    for C, p in pl[1:]:
        p[5] = False
    pl[0][1][4] = False
    for C, p in pl:
        p[4] = False
    K = 6
    Q, has = cluster_centres(pl, K)
    for k in range(K):
        held = [C[k] for C, p in pl if p[k]]
        assert has[k] == bool(held)
        if held:
            assert np.allclose(Q[k], np.mean(held, axis=0))
    gap = np.full(K, 1e9)
    ks = [k for k in range(K) if has[k]]
    for k1 in ks:
        for k2 in ks:
            if k1 > k2:
                dis = np.linalg.norm(Q[k1] - Q[k2])
                gap[k1] = min(gap[k1], dis)
                gap[k2] = min(gap[k2], dis)
    min_gap = gap.min()
    gap[gap > 1e8] = min_gap
    g = class_gaps(Q, has)
    assert np.allclose(g['gap'], gap) and np.isclose(g['max_gap'], gap.max())
    assert np.isclose(g['min_gap'], min_gap)


def test_trainer_deterministic_state_round_trip_and_rng_isolation():
    torch.set_num_threads(1)
    pl = random_payloads()
    cfg = TGPConfig(server_epochs=3, seed=7)
    torch.manual_seed(0)
    ref = torch.rand(4)
    torch.manual_seed(0)
    a = TGPTrainer(6, 5, cfg)
    s0 = a.state()
    ia = a.fit_round(pl, 1)
    assert torch.equal(torch.rand(4), ref)
    b = TGPTrainer(6, 5, cfg)
    assert all(np.array_equal(s0[k], b.state()[k]) for k in s0)
    ib = b.fit_round(pl, 1)
    assert np.array_equal(ia['P_hat'], ib['P_hat'])
    s1 = a.state()
    ia2 = a.fit_round(pl, 2)
    b.load_state(s1)
    ib2 = b.fit_round(pl, 2)
    assert np.array_equal(ia2['P_hat'], ib2['P_hat'])
    assert ia['margin'] == min(class_gaps(*cluster_centres(pl, 6))['max_gap'], cfg.margin_threshold)
    assert ia['n_pairs'] == sum(int(p.sum()) for _, p in pl)


def test_tgp_server_two_rounds():
    torch.set_num_threads(1)
    K, d = 2, 4
    cfg = FedConfig(num_classes=K, embed_dim=d, local_epochs=1, batch_size=4, m_probes=4,
                    t_reveal=2, scheme='Ed25519', experiment_seed=0, warmup_rounds=10 ** 6,
                    audit_z_threshold=10 ** 6, lam=10.0)
    torch.manual_seed(1)
    clients = []
    for i in range(3):
        data = TensorDataset(torch.randn(8, 3, 32, 32), torch.arange(8) % 2)
        c = Client(i, build_group_model('hetero4', 2, K, d, passport=False), data, cfg)
        rebind_identity(c.holder)
        clients.append(c)
    server = TGPServer(cfg, TGPConfig(server_epochs=2, seed=0))
    for r in (1, 2):
        contribs, counts = [], []
        for c in clients:
            contribs.append(c.round_step())
            counts.append(c._last_counts)
        P = server.aggregate(contribs, counts)
        assert len(server.log) == 3 * r and all(e['accepted'] for e in server.log)
        assert np.array_equal(P, server.global_C)
        assert np.array_equal(P, server.trainer.prototypes())
        Q, has = cluster_centres([(x.C, np.asarray(n) > 0) for x, n in zip(contribs, counts)], K)
        assert np.allclose(server.qbar[has], Q[has])
        assert server.tgp_log[-1]['round'] == r
        for c in clients:
            c.receive_global(P)


def toy_archive(clients=4, rounds=14, K=5, d=3, seed=0):
    g = np.random.default_rng(seed)
    arch = {"C": {i: {} for i in range(clients)}, "present": {i: {} for i in range(clients)},
            "cbar": {}}
    for r in range(1, rounds + 1):
        for i in range(clients):
            arch["C"][i][r] = g.normal(size=(K, d)) + 2.0
            pres = np.ones(K, bool)
            pres[(i + r) % K] = False
            arch["present"][i][r] = pres
        arch["cbar"][r] = g.normal(size=(K, d))
    arch["qbar"] = FA.recompute_qbar(arch)
    return arch


def test_free_rider_variants():
    arch = toy_archive()
    r, cid = 12, 1
    arch["published"] = {i: {t: arch["C"][i][t] for t in range(1, r)} for i in arch["C"]}
    rng = np.random.default_rng(0)
    P = arch["cbar"][r - 1]
    e = FA.attack_fedtgp_echo(arch, cid, r, rng)
    assert np.array_equal(e, P)
    pres = arch["present"][cid][r]
    num, cnt = np.zeros_like(P), np.zeros(len(P))
    for t in range(2, 12):
        p = arch["present"][cid][t]
        num[p] += (arch["C"][cid][t] - arch["cbar"][t - 1])[p]
        cnt[p] += 1
    o = FA.attack_fedtgp_offset_echo(arch, cid, r, rng)
    assert np.allclose(o[pres], (P + num / cnt[:, None])[pres])
    assert np.array_equal(o[~pres], P[~pres])
    q = FA.attack_fedtgp_oracle_echo(arch, cid, r, rng)
    assert np.array_equal(q[pres], arch["qbar"][r - 1][pres]) and np.array_equal(q[~pres], P[~pres])
    arch["published"][cid].update({t: arch["C"][cid][t] for t in range(12, 14)})
    oa = FA.attack_fedtgp_offset_echo(arch, cid, 14, rng)
    arch["published"][cid][13] = arch["published"][cid][13] + 100.0
    assert np.array_equal(oa, FA.attack_fedtgp_offset_echo(arch, cid, 14, rng))
    q_srv, has = cluster_centres([(arch["C"][i][5], arch["present"][i][5]) for i in arch["C"]], 5)
    assert np.allclose(arch["qbar"][5][has], q_srv[has])


def test_recent_offset_and_affine_variants():
    arch = toy_archive()
    r, cid = 13, 2
    arch["published"] = {i: {t: arch["C"][i][t] for t in range(1, r)} for i in arch["C"]}
    rng = np.random.default_rng(0)
    P = arch["cbar"][r - 1]
    pres = arch["present"][cid][r]
    want = P.copy()
    for k in np.flatnonzero(pres):
        t = max(t for t in range(2, 12) if arch["present"][cid][t][k])
        want[k] += arch["C"][cid][t][k] - arch["cbar"][t - 1][k]
    o = FA.attack_fedtgp_recent_offset_echo(arch, cid, r, rng)
    assert np.allclose(o, want) and np.array_equal(o[~pres], P[~pres])
    arch["published"][cid][12] = arch["published"][cid][12] + 100.0
    assert np.array_equal(o, FA.attack_fedtgp_recent_offset_echo(arch, cid, r, rng))
    beta, tau = 0.6, np.array([1.0, -2.0, 0.5])
    B10 = arch["cbar"][10]
    exact = beta * (B10 - B10.mean(axis=0)) + tau
    p11 = arch["present"][cid][11]
    arch["published"][cid][11] = np.where(p11[:, None], exact, arch["C"][cid][11])
    fit = FA.affine_fit(arch, cid, r)
    assert fit["round"] == 11 and fit["rows"] == int(p11.sum())
    assert np.isclose(fit["beta"], beta) and np.allclose(fit["tau"], tau) and fit["residual_fro"] < 1e-9
    a = FA.attack_fedtgp_affine_echo(arch, cid, r, rng)
    assert np.allclose(a[pres], (beta * (P - P.mean(axis=0)) + tau)[pres])
    assert np.array_equal(a[~pres], P[~pres])
    arch["present"][cid][11] = np.zeros_like(p11)
    arch["present"][cid][11][0] = True
    assert FA.affine_fit(arch, cid, r) is None
    assert np.array_equal(FA.attack_fedtgp_affine_echo(arch, cid, r, rng), P)
    assert FA.VARIANT_ORDER[:3] == ["fedtgp_echo", "fedtgp_offset_echo", "fedtgp_oracle_echo"]


def test_exact_zero_update_margins_match_replay_rank():
    from passportfl.baselines import bulyan_deviation, median_deviation, trimmed_mean_deviation
    g = np.random.default_rng(4)
    for trial in range(20):
        V = g.normal(size=(10, 6)) + g.normal(scale=2.0) * np.ones(6)
        ex = FA.zero_update_margins(V)
        for rule, f in (("median_dev", median_deviation), ("trimmed_mean_dev", trimmed_mean_deviation),
                        ("bulyan_dev", lambda x: bulyan_deviation(x, n_byzantine=2))):
            for v in range(10):
                innov = {j: (np.zeros(6) if j == v else V[j]) for j in range(10)}
                s = f(innov)
                rk = FA.attacker_rank([s[j] for j in range(10)], v)
                assert (rk["rank"] == 1) == (ex[rule]["m_min"][v] > 0)
                assert np.isclose(ex[rule]["centre_norm"][v], s[v])
            assert ex[rule]["inverts_frac_seats"] == np.mean(np.asarray(ex[rule]["m_min"]) > 0)


def test_attacker_rank_counts_ties_and_undefined_against_the_attacker():
    r = FA.attacker_rank([1.0, 0.5, 1.0, np.nan, 2.0], 0)
    assert r == {"rank": 4, "strict": 2, "ties": 1, "nan_honest": 1}
    assert FA.attacker_rank([np.nan, 1.0, 2.0], 0) is None
    first = FA.attacker_rank([0.1, 1.0, 2.0], 0)
    assert first == {"rank": 1, "strict": 1, "ties": 0, "nan_honest": 0}
    s = FA.rank_summary([r, None, first])
    assert s["n_rounds"] == 2 and s["n_attacker_undefined"] == 1
    assert s["rank_mean"] == 2.5 and s["rank_mean_strict"] == 1.5
    assert s["rank1_frac"] == 1 / 3 and s["rank1_frac_strict"] == 1 / 3
    assert s["n_rounds_with_ties"] == 1 and s["n_honest_undefined"] == 1


def test_inversion_condition_in_geometry():
    arch = toy_archive(clients=5, rounds=14)
    g = FA.innovation_geometry(arch, arch["cbar"], arch["qbar"], 14, 12)
    for row in g["per_round"]:
        r = row["round"]
        V = np.stack([(arch["C"][i][r] - arch["cbar"][r - 1]).ravel() for i in range(5)])
        c = np.median(V, axis=0)
        assert row["median_inverts_all"] == bool(np.linalg.norm(c) < np.linalg.norm(V - c, axis=1).min())
    assert g["median_over_rounds"]["median_inverts_all_frac"] == np.mean(
        [x["median_inverts_all"] for x in g["per_round"]])


def test_config_key_ignores_per_seed_fields():
    m = lambda seed, lam, dev: {"arch_group": "hetero4", "server": "fedtgp", "lam": lam, "seed": seed,
                                "tgp": {"server_epochs": 100, "seed": seed, "device": dev}}
    assert FA.config_key(m(0, 1.0, "cpu")) == FA.config_key(m(3, 1.0, "cuda"))
    assert FA.config_key(m(0, 1.0, "cpu")) != FA.config_key(m(0, 10.0, "cpu"))
    assert FA.config_key(None) is None


def test_lambda_arms_decision():
    from passportfl import fedtgp
    assert fedtgp.FEDTGP_DEFAULT_LAM == FedConfig().lam == 1.0
    assert fedtgp.LAMBDA_ARMS == {"lam1": 1.0, "lam10": 10.0, "lam01": 0.1}
    assert fedtgp.LAMBDA_ARMS[fedtgp.FEDTGP_PRIMARY_ARM] == fedtgp.FEDTGP_DEFAULT_LAM
