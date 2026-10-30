import copy
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from passportfl.effort_controls import (forward_only, hybrid_state, is_bn_statistic,
                                        recalibrated_state)
from passportfl.fedproto import Client, FedConfig
from passportfl.passport import build_model
from passportfl.reproducibility import experiment_entropy

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


effort = load_module('effort_bn', 'experiments/effort_decomposition.py')
reduced = load_module('reduced_modes', 'experiments/reduced_effort_eval.py')


def tiny_client(seed=12, n=12, cls=Client, **kw):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    data = TensorDataset(torch.randn(n, 3, 32, 32), torch.arange(n) % 2)
    cfg = FedConfig(num_classes=2, embed_dim=4, width=2, local_epochs=1,
                    batch_size=4, m_probes=4, t_reveal=2, scheme='Ed25519')
    c = cls(0, build_model(2, 4, 2, True, 0.0), data, cfg,
            entropy=experiment_entropy(0, 0, 'test'), **kw)
    c.receive_global(np.zeros((2, 4), dtype=np.float32))
    return c, data


def parameters_of(c):
    return {k: v.clone() for k, v in c.model.state_dict().items() if not is_bn_statistic(k)}


def statistics_of(c):
    return {k: v.clone() for k, v in c.model.state_dict().items() if is_bn_statistic(k)}


def test_state_split_and_hybrid_states():
    c, _ = tiny_client()
    keys = c.model.state_dict().keys()
    stats = [k for k in keys if is_bn_statistic(k)]
    assert stats and all(k.endswith(('running_mean', 'running_var', 'num_batches_tracked')) for k in stats)
    assert not any(is_bn_statistic(k) for k in keys if k.endswith(('weight', 'bias')))
    a = copy.deepcopy(c.model.state_dict())
    b = {k: v + 1 for k, v in a.items()}
    h = hybrid_state(a, b)
    assert all(torch.equal(h[k], b[k] if is_bn_statistic(k) else a[k]) for k in keys)


def test_forward_only_moves_statistics_and_nothing_else():
    c, _ = tiny_client()
    c.local_train()
    par, stats, opt = parameters_of(c), statistics_of(c), copy.deepcopy(c.opt.state_dict())
    out = forward_only(c, max_passes=2)
    assert c.forward_passes == 2 and np.isfinite(out['train_loss'])
    assert all(torch.equal(v, c.model.state_dict()[k]) for k, v in par.items())
    assert any(not torch.equal(v, c.model.state_dict()[k]) for k, v in stats.items())
    assert effort.same_optimizer_state(opt, c.opt.state_dict())
    before = copy.deepcopy(c.model.state_dict())
    forward_only(c, max_passes=0)
    assert effort.same_tensors(before, c.model.state_dict())


def test_recalibration_depends_on_parameters_only():
    c, data = tiny_client()
    c.local_train()
    state = copy.deepcopy(c.model.state_dict())
    other = hybrid_state(state, {k: (v * 3 + 1 if v.is_floating_point() else v) for k, v in state.items()})
    r1 = recalibrated_state(c, state, data)
    r2 = recalibrated_state(c, other, data)
    assert effort.same_tensors(r1, r2)
    assert all(torch.equal(r1[k], state[k]) for k in state if not is_bn_statistic(k))


def test_plain_restore_shares_momentum_and_exact_restore_does_not():
    for restore, shared in ((effort.restore_parameters, True), (effort.restore_exact, False)):
        c, _ = tiny_client()
        c.local_train()
        snap = effort.snapshot_parameters(c)
        kept = copy.deepcopy(snap['opt'])
        restore(c, snap)
        c.local_train(max_steps=1)
        assert (not effort.same_optimizer_state(kept, snap['opt'])) == shared


def test_movement_parts_add_up():
    rng = np.random.default_rng(0)
    C_pre, recal_pre = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))
    reads = {k: rng.normal(size=(3, 4)) for k in ('total', 'par', 'bn', 'recal', 'fwd')}
    p = effort.movement_parts(reads, C_pre, recal_pre, np.array([True, False, True]))
    np.testing.assert_allclose(p['par'] + p['bn'] + p['int'], p['move'])
    np.testing.assert_allclose(p['shap_par'] + p['shap_bn'], p['move'])
    np.testing.assert_allclose(p['par_rev'] + p['bn'], p['move'])
    np.testing.assert_allclose(p['recal'] + p['noise'], p['move'])


def test_split_probe_is_invisible_to_the_federation():
    c, data = tiny_client()
    c.local_train()
    reference = copy.deepcopy(c)
    rng = torch.get_rng_state()
    published, rec = effort.probe_victim_bn(c, np.zeros((2, 4)), [0, 1, 2], data)
    rng_probe = torch.get_rng_state()
    torch.set_rng_state(rng)
    expected = reference.round_step()
    np.testing.assert_array_equal(published.C, expected.C)
    assert effort.same_tensors(c.model.state_dict(), reference.model.state_dict())
    assert effort.same_optimizer_state(c.opt.state_dict(), reference.opt.state_dict())
    assert torch.equal(rng_probe, torch.get_rng_state())
    assert max(rec['by_steps']['0']['norm'].values()) < 1e-6
    full = rec['by_steps'][str(rec['full_steps'])]
    assert full['cos_full']['move'] > 0.999999
    one = rec['by_steps']['1']
    assert one['cos_bn_fwd'] > 0.999999 and np.isclose(one['norm']['bn'], one['norm']['fwd'], rtol=1e-6)
    for cap in ('1', '2', str(rec['full_steps'])):
        assert abs(sum(rec['by_steps'][cap]['share'].values()) - 1) < 1e-9


def test_live_forward_and_zero_modes():
    for mode, cap in (('forward', 3), ('zero', 0)):
        c, _ = tiny_client(cls=reduced.ReducedClient, start=2, max_steps=cap, mode=mode)
        c.round_step()
        par, state = parameters_of(c), copy.deepcopy(c.model.state_dict())
        c.round_step()
        assert c.steps_taken == [c.full_steps, 0]
        assert c.forward_taken == [c.full_steps, cap]
        assert all(torch.equal(v, c.model.state_dict()[k]) for k, v in par.items())
        assert effort.same_tensors(state, c.model.state_dict()) == (mode == 'zero')
        if mode == 'zero':
            assert c.published_stats[1] == c.published_stats[0]
