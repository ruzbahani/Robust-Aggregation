import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from passportfl.baselines import bulyan_deviation, median_deviation, trimmed_mean_deviation
from passportfl.fedproto import Client, FedConfig
from passportfl.passport import build_model
from passportfl.reproducibility import experiment_entropy

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


geometry = load_module('geometry', 'experiments/krum_margin.py')
tables = load_module('tables', 'analysis/result_tables.py')
effort = load_module('effort', 'experiments/effort_decomposition.py')


def test_center_uses_attacked_cohort_and_excludes_zero_from_honest_distances():
    original = np.array([[100.], [1.], [2.], [3.], [4.]])
    result = geometry.margin(original)
    assert result['centre_norm'] == 2.0
    assert result['honest_to_centre_median'] == 1.0
    np.testing.assert_array_equal(original[:, 0], [100, 1, 2, 3, 4])


@pytest.mark.parametrize('rule, scorer', [('median', median_deviation),
    ('trimmed_mean', trimmed_mean_deviation), ('bulyan', bulyan_deviation)])
def test_each_center_condition_matches_the_actual_detector(rule, scorer):
    points = np.random.default_rng(1).normal(size=(10, 11))
    points[0] = 0
    condition = geometry.centre_margin(points, np.arange(1, 10), rule)
    scores = scorer(dict(enumerate(points)))
    assert condition['centre_norm'] == pytest.approx(scores[0])
    assert condition['honest_to_centre_min'] == pytest.approx(min(scores[i] for i in range(1, 10)))
    assert condition['centre_inverts_all'] == (scores[0] < min(scores[i] for i in range(1, 10)))


def test_matching_moments_do_not_fix_krum_selection():
    collision = geometry.exact_collision()
    assert collision['moments_match'] and collision['outcomes_differ']


def test_missing_result_key_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, 'RESULTS', tmp_path)
    (tmp_path / 'incomplete.json').write_text('{}')
    with pytest.raises(KeyError):
        tables.load_json('incomplete.json')['required']
    with pytest.raises(FileNotFoundError):
        tables.load_json('missing.json')


def test_seeded_experiment_identity_has_separate_seed_client_and_domain():
    a = experiment_entropy(0, 0, 'control')
    assert a == experiment_entropy(0, 0, 'control') and len(a) == 64
    assert len({a, experiment_entropy(1, 0, 'control'), experiment_entropy(0, 1, 'control'),
                experiment_entropy(0, 0, 'other')}) == 4


@pytest.mark.parametrize('passport', [False, True])
def test_client_seed_controls_passport_and_external_key_identities(passport):
    torch.set_num_threads(1)
    data = TensorDataset(torch.zeros(4, 3, 32, 32), torch.arange(4) % 2)
    cfg = FedConfig(num_classes=2, embed_dim=4, width=2, m_probes=4,
                    t_reveal=2, scheme='Ed25519', experiment_seed=7)
    template = build_model(2, 4, 2, passport, 0.0)
    a = Client(0, copy.deepcopy(template), data, cfg)
    b = Client(0, copy.deepcopy(template), data, cfg)
    c = Client(1, copy.deepcopy(template), data, cfg)
    assert a.holder.pk == b.holder.pk
    assert a.holder.G == b.holder.G
    assert a.holder.pk != c.holder.pk


def test_fixed_readout_zero_step_and_honest_publication_are_preserved():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    data = TensorDataset(torch.randn(8, 3, 32, 32), torch.arange(8) % 2)
    cfg = FedConfig(num_classes=2, embed_dim=4, width=2, local_epochs=1,
                    batch_size=4, m_probes=4, t_reveal=2, scheme='Ed25519')
    client = Client(0, build_model(2, 4, 2, True, 0.0), data, cfg,
                    entropy=experiment_entropy(0, 0, 'test'))
    client.receive_global(np.zeros((2, 4), dtype=np.float32))
    reference = copy.deepcopy(client)
    snap = effort.snapshot_parameters(client)
    published, record = effort.probe_victim(client, np.zeros((2, 4)), [0, 1], data)
    effort.restore_parameters(reference, snap)
    expected = reference.round_step()
    assert record['by_steps']['0']['move'] < 1e-6
    np.testing.assert_array_equal(published.C, expected.C)
    for key, value in client.model.state_dict().items():
        assert torch.equal(value, reference.model.state_dict()[key])
