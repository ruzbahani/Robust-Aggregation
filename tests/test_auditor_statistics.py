import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


AS = load_module('auditor_statistics', 'experiments/auditor_statistics.py')
P, T = AS.P, AS.T
SEEDS, N, ROUNDS, ONSET = (0, 1, 2, 3), 5, list(range(12, 18)), 12


@pytest.fixture(scope='module')
def scores(tmp_path_factory):
    g = np.random.default_rng(0)
    seed, victim, attack, pool, mats = [], [], [], [], []
    for s in SEEDS:
        for a in [-1] + list(range(len(P.ATTACK_ORDER))):
            for v in ([-1] if a < 0 else range(N)):
                m = g.normal(size=(len(ROUNDS), N))
                m[g.random(m.shape) < 0.05] = np.nan
                if a >= 0:
                    m[:, v] += 0.5 * (a + 1) + g.normal(size=len(ROUNDS))
                    if P.ATTACK_ORDER[a] == 'peer_copy':
                        m[:, P.accomplice('peer_copy', v, N)] = m[:, v]
                seed.append(s)
                victim.append(v)
                attack.append(a)
                pool.append('CAL' if s < 2 else 'TEST')
                mats.append(m)
    path = tmp_path_factory.mktemp('scores') / 'scores.npz'
    np.savez_compressed(
        path, columns=np.asarray(['det']), seed=np.asarray(seed), victim=np.asarray(victim),
        attack=np.asarray(attack), attack_names=np.asarray(P.ATTACK_ORDER), pool=np.asarray(pool),
        rounds=np.asarray(ROUNDS), clients=np.arange(N),
        meta_json=np.asarray(json.dumps({'attack_start': ONSET})), s__det=np.stack(mats))
    return AS.Scores(str(path))


def thresholds(sc):
    return P.fit_thresholds({'det': P.stack_scores(sc.passes(AS.CAL, ['det'], attacks=False), 'det')},
                            AS.TARGETS)['det']


def test_seat_decomposition_reproduces_per_seed_auc(scores):
    allp = scores.passes(SEEDS, ['det'], honest=False)
    for excl, key in ((False, 'U'), (True, 'Ux')):
        ref = T.auc_table(allp, list(SEEDS), ['det'], ONSET, excl_accomplice=excl)
        for s in SEEDS:
            st = AS.seat_statistics(scores, s, 'det', thresholds(scores))
            for a in P.ATTACK_ORDER:
                U, npos, nneg = st[key][a]
                want = ref[a]['det']['per_seed'][str(s)]
                assert U.sum() / (npos.sum() * nneg.sum()) == pytest.approx(want, abs=1e-12)
                w = AS.weighted_auc(U, npos, nneg, np.ones((1, N)))
                assert w[0] == pytest.approx(want, abs=1e-12)


def test_bootstrap_draw_without_resampling_is_the_point_estimate(scores):
    thr = thresholds(scores)
    st = {s: AS.seat_statistics(scores, s, 'det', thr) for s in SEEDS}
    point = AS.metrics(st, SEEDS, None, None, 1)
    same_seeds = AS.metrics(st, SEEDS, None, np.tile(np.arange(len(SEEDS)), (1, 1)), 1)
    unit_seats = AS.metrics(st, SEEDS, np.ones((1, len(SEEDS), N)), None, 1)
    for k, v in point.items():
        assert same_seeds[k][0] == pytest.approx(v[0], abs=1e-12, nan_ok=True)
        assert unit_seats[k][0] == pytest.approx(v[0], abs=1e-12, nan_ok=True)
    res = AS.bootstrap({'det': st}, SEEDS, 'two_stage', 50, np.random.default_rng(1))
    assert res['det']['worst_auc']['point'] == pytest.approx(point['worst_auc'][0], abs=1e-12)


def test_roc_rule_never_exceeds_nominal_and_is_monotone():
    g = np.random.default_rng(2)
    neg, pos = g.normal(size=997), g.normal(1.0, 1.0, size=300)
    neg[:20] = np.nan
    fpr, tpr = AS.roc_on_grid(pos, neg, AS.FPR_GRID)
    assert all(f <= nom + 1e-12 for f, nom in zip(fpr, AS.FPR_GRID))
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
    assert fpr[-1] == pytest.approx(977 / 997) and tpr[-1] == 1.0


def test_operational_auc_ranks_undefined_scores_lowest():
    pos = np.array([np.nan, 2.0, 3.0])
    neg = np.array([1.0, np.nan])
    assert AS.operational_auc(pos, neg) == pytest.approx(4.5 / 6)


def test_trajectory_tables_count_the_copied_client_as_honest(scores):
    thr = thresholds(scores)
    st = {s: AS.seat_statistics(scores, s, 'det', thr) for s in (2, 3)}
    tab = AS.trajectory_tables(st, [2, 3])
    for t in AS.TARGETS:
        th = thr[f'{t:g}']
        direct = [int(P.flag_scores(scores.mat('det', s, 'peer_copy', v)
                                    [:, P.accomplice('peer_copy', v, N)], th).sum())
                  for s in (2, 3) for v in range(N)]
        rec = tab[f'{t:g}']
        assert list(rec['copy_source']) == ['peer_copy']
        assert rec['copy_source']['peer_copy']['flags_total'] == sum(direct)
        for k, rule in rec['k_of_24_rules_post_hoc'].items():
            assert rule['copy_source_trajectories'] == 2 * N
            assert rule['copy_source_trajectories_flagged'] == sum(d >= int(k) for d in direct)
            assert rule['honest_trajectories_in_attacked_runs'] == (
                rule['uninvolved_trajectories'] + 2 * N)
            assert rule['honest_trajectories_in_attacked_runs_flagged'] == (
                rule['uninvolved_trajectories_flagged'] + rule['copy_source_trajectories_flagged'])
            for a, r in rule['attacked'].items():
                assert r['ci'][0] <= r['one_sided_95_lower'] <= r['rate']


def test_kth_flag_rounds_match_direct_flag_sequences(scores):
    assert AS.kth_flag_rounds(np.array([0, 1, 0, 1, 1, 1]), 3).tolist() == [1, 3, 4]
    assert AS.kth_flag_rounds(np.array([False, True, False]), 3).tolist() == [1, -1, -1]
    thr = thresholds(scores)
    st = {s: AS.seat_statistics(scores, s, 'det', thr) for s in (2, 3)}
    tab = AS.trajectory_tables(st, [2, 3])
    for t in AS.TARGETS:
        th = thr[f'{t:g}']
        for k, rule in tab[f'{t:g}']['k_of_24_rules_post_hoc'].items():
            k = int(k)
            for a, r in rule['attacked'].items():
                rounds = []
                for s in (2, 3):
                    for v in range(N):
                        f = P.flag_scores(scores.mat('det', s, a, v)[:, v], th)
                        idx = np.flatnonzero(f)
                        rounds.append(int(idx[k - 1]) if idx.size >= k else -1)
                reached = [x for x in rounds if x >= 0]
                assert r['round_of_kth_flag']['not_reached'] == rounds.count(-1)
                assert r['detected'] == len(reached)
                first_k = sum(bool(P.flag_scores(scores.mat('det', s, a, v)[:k, v], th).all())
                              for s in (2, 3) for v in range(N))
                assert r['round_of_kth_flag']['at_earliest_possible_round'] == first_k
                if reached:
                    assert r['round_of_kth_flag']['median'] == float(np.median(reached))
                    assert r['round_of_kth_flag']['max'] == max(reached)
            assert rule['attacked_all_attacks']['detected'] == sum(
                r['detected'] for r in rule['attacked'].values())
            assert rule['attacked_all_attacks']['kth_flag_at_earliest_possible_round'] == sum(
                r['round_of_kth_flag']['at_earliest_possible_round']
                for r in rule['attacked'].values())


def test_frozen_counts_equal_direct_flag_counts(scores):
    thr = thresholds(scores)
    ev = AS.evaluate_frozen(scores, (2, 3), 'det', thr, ONSET)
    for t in AS.TARGETS:
        th = thr[f'{t:g}']
        clean = sum(int(P.flag_scores(scores.mat('det', s, None, None), th).sum()) for s in (2, 3))
        assert ev[f'{t:g}']['clean_k'] == clean
        for a in P.ATTACK_ORDER:
            k = sum(int(P.flag_scores(scores.mat('det', s, a, v)[:, v], th).sum())
                    for s in (2, 3) for v in range(N))
            assert ev[f'{t:g}']['per_attack'][a]['k'] == k
