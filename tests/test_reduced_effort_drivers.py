import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from passportfl.innovation import InnovationAuditor

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reduced_effort_drivers', ROOT / 'experiments/reduced_effort_drivers.py')
drivers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drivers)

TAU = 3.7859094579543044
CLIENTS, K, D, ROUNDS, START = 10, 12, 6, 20, 8


def federation(seed, frozen):
    rng = np.random.default_rng(seed)
    heading = rng.normal(size=(CLIENTS, K, D))
    C = {i: {} for i in range(CLIENTS)}
    present = {i: {} for i in range(CLIENTS)}
    cbar, cur = {}, rng.normal(size=(CLIENTS, K, D))
    for r in range(1, ROUNDS + 1):
        for i in range(CLIENTS):
            if not (frozen and i == 0 and r >= START):
                cur[i] = cur[i] + 0.3 * heading[i] + 0.3 * rng.normal(size=(K, D))
            C[i][r] = (cur[i] + 0.01 * rng.normal(size=(K, D))).astype(np.float32)
            present[i][r] = np.ones(K, dtype=bool)
        cbar[r] = np.mean([C[i][r] for i in range(CLIENTS)], axis=0)
    return C, present, cbar


def online(C, present, cbar):
    aud = InnovationAuditor(mad_floor=drivers.MAD_FLOOR)
    pos, neg, Cbar = [], [], None
    for r in range(1, ROUNDS + 1):
        payloads = {i: (C[i][r], present[i][r]) for i in C}
        if Cbar is not None:
            aud.set_aggregate(r - 1, Cbar)
        rs = aud.observe_round(r, payloads)
        if r >= START and Cbar is not None:
            for i in payloads:
                (pos if i == 0 else neg).append(float(rs.per_client[i].get('suspicion', float('nan'))))
        Cbar = cbar[r]
    fp = [s for s in pos if np.isfinite(s)]
    fn = [s for s in neg if np.isfinite(s)]
    return {'auc': {'suspicion': drivers.roc_auc(pos, neg)},
            'tpr_client_round': sum(s > TAU for s in fp) / len(fp),
            'honest_fpr_client_round': sum(s > TAU for s in fn) / len(fn)}


def write_run(results, tag, seed, frozen, tamper=False):
    C, present, cbar = federation(seed, frozen)
    folder = 'archives' if tag.startswith('f0') else 'archives_rev2'
    (results / folder).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(results / folder / ('arch_reduced_%s_s%d.npz' % (tag, seed)), C=C, present=present, cbar=cbar)
    run = online(C, present, cbar)
    if tamper:
        run['tpr_client_round'] += 1.0 / 12
    run.update({'config': {'victim': 0, 'attack_start': START, 'rounds': ROUNDS, 'seed': seed}, 'tau': TAU})
    (results / ('reduced_effort_%s_s%d.json' % (tag, seed))).write_text(json.dumps(run), encoding='utf-8')
    return run


def write_control(results, seed, flip_class=False, **config):
    C, present, cbar = federation(seed, frozen=False)
    if flip_class:
        present[3][5] = present[3][5].copy()
        present[3][5][0] = False
    cfg = {'clients': CLIENTS, 'rounds': ROUNDS, 'attack_start': START, 'dirichlet': 0.5, 'seed': seed,
           'tag': 'cifar100_s%d' % seed}
    cfg.update(config)
    (results / 'archives').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(results / 'archives' / ('arch_cifar100_s%d_c%d_r%d_s%d.npz' % (seed, CLIENTS, ROUNDS, seed)),
                        C=C, present=present, cbar=cbar)
    (results / ('innovation_cifar100_s%d.json' % seed)).write_text(json.dumps({'config': cfg, 'results': {}}),
                                                                   encoding='utf-8')


def run_main(results, out, attackers, seeds, control=False):
    argv = sys.argv
    sys.argv = ['reduced_effort_drivers.py', '--results', str(results), '--out', str(out),
                '--attackers', *attackers, '--seeds', *map(str, seeds)] + ([] if control else ['--no-control'])
    try:
        return drivers.main()
    finally:
        sys.argv = argv


def test_replay_reproduces_online_scores_and_attributes_self_replay(tmp_path):
    for s in (0, 1):
        write_run(tmp_path, 'zero', s, frozen=True)
        write_run(tmp_path, 'f0.1', s, frozen=False)
    out = tmp_path / 'drivers.json'
    assert run_main(tmp_path, out, ['f0.1', 'zero'], [0, 1]) == 0
    d = json.loads(out.read_text(encoding='utf-8'))
    z = d['summary']['zero']
    n = ROUNDS - START + 1
    assert z['n_attacked'] == 2 * n and z['n_flagged'] >= 2 * (n - 2)
    assert z['argmax_flagged']['rep'] > z['n_flagged'] / 2
    v = d['summary']['zero']['per_seed_raw']['step_over_cohort_step']['victim']
    assert max(v) < 0.1
    assert d['summary']['f0.1']['argmax_flagged']['rep'] == 0
    for r in d['runs'].values():
        for row in r['per_round']:
            parts = [x for x in row['oriented'].values() if np.isfinite(x)]
            assert abs(max(parts) - row['suspicion']) < 1e-9
        assert sum(r['argmax_attacked'].values()) == r['n_attacked']
        assert all(row['absent_share'] == 0.0 and row['n_absent'] == 0 for row in r['per_round'])
        m = r['magnitude']
        assert m['baseline_rounds'] == [drivers.MAG_FIRST, START - 1] and len(m['honest_change']) == CLIENTS - 1
        assert m['victim_change_rank'] == 1 + sum(h < m['victim_change'] for h in m['honest_change'])
        assert abs(m['victim_change'] - (m['victim_g_attack'] - m['victim_g_baseline'])) < 1e-12
    mz = d['summary']['zero']['magnitude']
    assert mz['n_victim_largest_fall'] == sum(k == 1 for k in mz['per_seed_victim_change_rank'])
    assert 'control' not in d


def test_replay_refuses_a_run_whose_recorded_scores_differ(tmp_path):
    write_run(tmp_path, 'fwd', 0, frozen=True, tamper=True)
    with pytest.raises(AssertionError, match='replayed tpr'):
        run_main(tmp_path, tmp_path / 'drivers.json', ['fwd'], [0])


def test_control_scores_client_0_of_the_honest_federation(tmp_path):
    for s in (0, 1):
        write_run(tmp_path, 'zero', s, frozen=True)
        write_control(tmp_path, s)
    out = tmp_path / 'drivers.json'
    assert run_main(tmp_path, out, ['zero'], [0, 1], control=True) == 0
    d = json.loads(out.read_text(encoding='utf-8'))
    c = d['control']
    assert c['summary']['n_attacked'] == 2 * (ROUNDS - START + 1)
    assert c['summary']['argmax_flagged']['rep'] == 0
    assert min(c['summary']['per_seed_raw']['step_over_cohort_step']['victim']) > 0.5
    assert d['meta']['sources']['control_s1']['archive'] == 'archives/arch_cifar100_s1_c%d_r%d_s1.npz' % (CLIENTS, ROUNDS)
    cmp = c['comparison']['zero']
    for i, s in enumerate((0, 1)):
        v = d['runs']['zero_s%d' % s]['magnitude']['victim_change']
        k = c['runs']['s%d' % s]['magnitude']['victim_change']
        assert cmp['per_seed_victim_change'][i] == v and cmp['per_seed_control_change'][i] == k
        assert abs(cmp['per_seed_difference'][i] - (v - k)) < 1e-15
    assert cmp['n_victim_below_control'] == sum(x < 0 for x in cmp['per_seed_difference'])
    write_run(tmp_path, 'f0.1', 0, frozen=False)
    assert run_main(tmp_path, tmp_path / 'f.json', ['f0.1'], [0], control=True) == 0
    f = json.loads((tmp_path / 'f.json').read_text(encoding='utf-8'))
    assert f['control']['runs']['s0']['auc_single'] == f['runs']['f0.1_s0']['auc_single']


@pytest.mark.parametrize('change, match', [({'flip_class': True}, 'same partition'),
                                           ({'attack_start': START + 1}, 'differs from the run')])
def test_control_refused_when_partition_or_configuration_differs(tmp_path, change, match):
    write_run(tmp_path, 'zero', 0, frozen=True)
    write_control(tmp_path, 0, **change)
    with pytest.raises(ValueError, match=match):
        run_main(tmp_path, tmp_path / 'drivers.json', ['zero'], [0], control=True)
