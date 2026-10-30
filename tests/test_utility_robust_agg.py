import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from passportfl.fedproto import Client, FedConfig, Server
from passportfl.passport import build_model
from passportfl.reproducibility import experiment_entropy

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


robust = load_module('utility_robust_agg', 'experiments/utility_robust_agg.py')


def reference_rows(payloads, counts, rule, prev):
    K, d = prev.shape
    out = prev.astype(np.float64).copy()
    for k in range(K):
        rows = [np.asarray(p, dtype=np.float64)[k] for p, c in zip(payloads, counts) if c[k] > 0]
        n = len(rows)
        if n == 0:
            continue
        for j in range(d):
            s = sorted(float(r[j]) for r in rows)
            if rule == 'median':
                out[k, j] = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
            else:
                t = n // 5
                core = s[t:n - t]
                out[k, j] = sum(core) / len(core)
    return out


def synthetic_cohort(seed, n=7, K=9, d=5):
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(n, K, d)).round(2)
    P[1] = P[0]
    P[3, :, 0] = P[2, :, 0]
    counts = np.zeros((n, K))
    for k in range(K):
        holders = rng.permutation(n)[:min(k, n)]
        counts[holders, k] = rng.integers(1, 50, size=len(holders))
    P[counts == 0] = 1e6
    return list(P), list(counts)


@pytest.mark.parametrize('rule', ['median', 'trimmed'])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_holder_rule_matches_independent_reference(rule, seed):
    payloads, counts = synthetic_cohort(seed)
    prev = np.random.default_rng(100 + seed).normal(size=payloads[0].shape)
    R, n_k = robust.holder_rows(payloads, counts, rule)
    expected = reference_rows(payloads, counts, rule, prev)
    has = n_k > 0
    assert list(n_k) == [min(k, len(payloads)) for k in range(len(n_k))]
    assert np.isnan(R[~has]).all()
    np.testing.assert_allclose(R[has], expected[has], rtol=0, atol=1e-12)
    assert np.abs(R[has]).max() < 1e5


def test_trim_count_is_floor_of_a_fifth():
    for n in range(1, 41):
        V = np.arange(n, dtype=float)[:, None, None]
        R, _ = robust.holder_rows(list(V), [np.ones(1)] * n, 'trimmed')
        t = n // 5
        assert R[0, 0] == pytest.approx(np.mean(np.arange(n)[t:n - t]))


def test_median_of_even_holders_averages_the_middle_pair():
    P = [np.array([[0.0, 1.0]]), np.array([[2.0, 1.0]])]
    R, _ = robust.holder_rows(P, [np.ones(1), np.ones(1)], 'median')
    np.testing.assert_array_equal(R, [[1.0, 1.0]])


def make_clients(n, K, d):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    data = TensorDataset(torch.zeros(4, 3, 32, 32), torch.arange(4) % K)
    cfg = FedConfig(num_classes=K, embed_dim=d, width=2, m_probes=4, t_reveal=2,
                    scheme='Ed25519')
    clients = [Client(i, build_model(K, d, 2, True, 0.0), data, cfg,
                      entropy=experiment_entropy(0, i, 'robust-agg-test')) for i in range(n)]
    return cfg, clients


@pytest.mark.parametrize('rule', ['mean', 'median', 'trimmed'])
def test_server_rule_on_signed_contributions(rule):
    n, K, d = 6, 7, 4
    cfg, clients = make_clients(n, K, d)
    base, robust_server = Server(cfg), robust.RobustServer(cfg, rule=rule, trace=robust.Trace())
    prev = np.random.default_rng(5).normal(size=(K, d))
    base.global_C = prev.copy()
    robust_server.global_C = prev.copy()
    held = []
    for rnd in (1, 2):
        payloads, counts = synthetic_cohort(10 + rnd, n=n, K=K, d=d)
        for i in range(1, n):
            counts[i][K - 1] = 0.0
            payloads[i][K - 1] = 1e6
        counts[0][K - 1] = 3.0
        contribs = [c.holder.issue(p, dict(n_samples=4, train_loss=0.0, local_acc=0.0,
                                           intra_compactness=0.0, inter_separation=0.0))
                    for c, p in zip(clients, payloads)]
        if rnd == 2:
            contribs[0] = copy.deepcopy(contribs[0])
            contribs[0].C = contribs[0].C + 1.0
        carried = robust_server.global_C.copy()
        out_base = base.aggregate(copy.deepcopy(contribs), counts)
        out_rob = robust_server.aggregate(copy.deepcopy(contribs), counts)
        accepted = [e['accepted'] for e in robust_server.log[-n:]]
        assert accepted == [rnd == 1 or i != 0 for i in range(n)]
        assert (json.dumps(base.log, sort_keys=True, default=str)
                == json.dumps(robust_server.log, sort_keys=True, default=str))
        if rule == 'mean':
            np.testing.assert_array_equal(out_rob, out_base)
        keep = [i for i in range(n) if accepted[i]]
        if rule != 'mean':
            expected = reference_rows([contribs[i].C for i in keep], [counts[i] for i in keep],
                                      rule, carried)
            np.testing.assert_allclose(out_rob, expected, rtol=0, atol=1e-12)
        no_holder = [k for k in range(K) if not any(counts[i][k] > 0 for i in keep)]
        assert 0 in no_holder and (K - 1 in no_holder) == (rnd == 2)
        np.testing.assert_array_equal(out_rob[no_holder], carried[no_holder])
        if rnd == 1 and rule != 'mean':
            np.testing.assert_array_equal(out_rob[K - 1], contribs[0].C[K - 1])
        assert robust_server.global_C is out_rob
        held.append(K - len(no_holder))
    trace = robust_server.trace.agg
    assert [t['accepted'] for t in trace] == [n, n - 1]
    assert [t['classes_held'] for t in trace] == held
    assert all((t['dev_from_mean_rel'] == 0.0) == (rule == 'mean') for t in trace)


def test_install_substitutes_only_the_module_attributes():
    ui = robust.utility_impact
    names = ('Server', 'Client', 'run_scenario', 'report')
    before = {k: getattr(ui, k) for k in names}
    original = Server.aggregate
    with robust.installed('median'):
        server = ui.Server(FedConfig(num_classes=3, embed_dim=2, scheme='Ed25519'))
        assert isinstance(server, robust.RobustServer) and server.rule == 'median'
        assert copy.deepcopy(server).trace is server.trace
        assert Server.aggregate is original
    assert {k: getattr(ui, k) for k in names} == before
