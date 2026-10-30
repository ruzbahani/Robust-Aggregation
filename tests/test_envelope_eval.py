import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


env = load_module('envelope_eval', 'experiments/envelope_eval.py')
P = env.P
CLIENTS, K, D, ROUNDS, ONSET = 10, 12, 6, 20, 8


def synthetic_archive(seed=0):
    g = np.random.default_rng(seed)
    C = {c: {} for c in range(CLIENTS)}
    present = {c: {} for c in range(CLIENTS)}
    cbar = {}
    for r in range(1, ROUNDS + 1):
        for c in range(CLIENTS):
            m = g.random(K) < 0.6
            m[g.integers(K, size=3)] = True
            present[c][r] = m
            X = g.normal(size=(K, D))
            if r - 1 in cbar:
                X[~m] = cbar[r - 1][~m]
            C[c][r] = X
        cbar[r] = np.mean([C[c][r] for c in range(CLIENTS)], axis=0)
    return {"C": C, "present": present, "cbar": cbar}


def replay(fn, victim, seed=0, tag=7):
    arch = dict(synthetic_archive(seed))
    arch["published"] = {c: {} for c in range(CLIENTS)}
    rng = np.random.default_rng([seed, victim + 1, tag])
    for r in range(1, ROUNDS + 1):
        for c in range(CLIENTS):
            X = arch["C"][c][r]
            if c == victim and r >= ONSET:
                X = fn(arch, c, r, rng)
            arch["published"][c][r] = X
    return arch


def test_lag3_factories_reduce_to_release_lagged_self():
    ref = replay(P.ATTACKS["lagged_self"], 2)["published"][2]
    lag = replay(env.make_lagged(3), 2)["published"][2]
    noise0 = replay(env.make_lag_noise(3, 0.0, env.pre_rounds(ONSET), ONSET), 2)["published"][2]
    for r in ref:
        assert np.array_equal(ref[r], lag[r])
        assert np.array_equal(ref[r], noise0[r])


@pytest.mark.parametrize("rel", env.ECHO_REL)
def test_echo_perturbation_size_and_support(rel):
    arch = replay(env.make_echo_eps(rel, env.pre_rounds(ONSET)), 4)
    target = rel * env.own_innovation_norm(arch, 4, env.pre_rounds(ONSET))
    for r in range(ONSET, ROUNDS + 1):
        E = arch["published"][4][r] - arch["cbar"][r - 1]
        assert abs(np.linalg.norm(E) - target) <= 1e-9 * target
        assert np.all(E[~arch["present"][4][r]] == 0.0)


def test_replay_noise_size_and_clean_chain():
    pre = env.pre_rounds(ONSET)
    arch = replay(env.make_lag_noise(6, 1.0, pre, ONSET), 5)
    clean = arch["_clean"][(5, 6)]
    step = env.own_payload_step(arch, 5, pre)
    for r in range(ONSET, ROUNDS + 1):
        assert np.array_equal(clean[r], clean[max(1, r - 6)])
        assert abs(np.linalg.norm(arch["published"][5][r] - clean[r]) - step) <= 1e-9 * step


def test_cayley_rotation_is_orthogonal_and_fresh():
    g = np.random.default_rng(0)
    R1, R2 = env.cayley_rotation(g, 16, 0.5), env.cayley_rotation(g, 16, 0.5)
    assert np.abs(R1.T @ R1 - np.eye(16)).max() < 1e-12
    assert np.abs(R1 - R2).max() > 1e-3
    assert np.abs(env.cayley_rotation(g, 16, 0.0) - np.eye(16)).max() == 0.0


@pytest.mark.parametrize("m", env.MOSAIC_M)
def test_mosaic_rows_come_from_declaring_peers(m):
    victim = 1
    arch = replay(env.make_mosaic(m), victim)
    peers = env.peers_of(arch, victim, m)
    used = set()
    for r in range(ONSET, ROUNDS + 1):
        X, rows = arch["published"][victim][r], arch["present"][victim][r]
        assert np.array_equal(X[~rows], arch["cbar"][r - 1][~rows])
        for k in np.flatnonzero(rows):
            src = [j for j in peers if arch["present"][j][r][k] and np.array_equal(X[k], arch["C"][j][r][k])]
            if src:
                used.update(src)
            else:
                assert not any(arch["present"][j][r][k] for j in peers)
                assert np.array_equal(X[k], arch["cbar"][r - 1][k])
    assert set(arch["_sources"]) == used and used <= set(peers) and victim not in used


@pytest.mark.parametrize("fill", ["echo", "lag3"])
def test_half_rows_copy_and_fill(fill):
    victim = 0
    arch = replay(env.make_rows_half(fill), victim)
    j = env.peers_of(arch, victim, 1)[0]
    total = 0
    for r in range(ONSET, ROUNDS + 1):
        X, rows = arch["published"][victim][r], arch["present"][victim][r]
        S = env.half_shared_rows(arch, victim, j, r)
        own = np.flatnonzero(rows)
        assert len(S) == min(int(round(0.5 * len(own))), int(arch["present"][j][r][own].sum()))
        assert np.array_equal(X[S], arch["C"][j][r][S])
        rest = np.setdiff1d(own, S)
        fill_src = (arch["cbar"][r - 1] if fill == "echo"
                    else arch["published"][victim][max(1, r - 3)])
        assert np.array_equal(X[rest], fill_src[rest])
        assert np.array_equal(X[~rows], arch["cbar"][r - 1][~rows])
        total += len(S)
    assert arch["_sources"] == {j: total}


def test_peer_average_and_sources():
    victim = 3
    arch = replay(env.make_peer_average(2), victim)
    a, b = env.peers_of(arch, victim, 2)
    assert a == P.accomplice("peer_copy", victim, CLIENTS)
    for r in range(ONSET, ROUNDS + 1):
        X, rows = arch["published"][victim][r], arch["present"][victim][r]
        assert np.allclose(X[rows], 0.5 * (arch["C"][a][r][rows] + arch["C"][b][r][rows]), atol=1e-12)
        assert np.array_equal(X[~rows], arch["cbar"][r - 1][~rows])
    assert set(arch["_sources"]) == {a, b}


def test_generator_tags():
    for i, a in enumerate(P.ATTACK_ORDER):
        assert env.rng_tag(a) == i + 1
    tags = [env.rng_tag(v) for v in env.VARIANT_ORDER if v not in P.ATTACK_ORDER]
    assert min(tags) >= 1000 and len(set(tags)) == len(tags)
    assert env.rng_tag(None) == 0
    assert set(env.DEFAULT_VARIANTS) == set(env.VARIANT_ORDER) - {f"lagged_{L}" for L in env.SENSITIVITY_LAGS}


def test_evaluate_variant_splits_bystanders():
    rounds = list(range(ONSET - 2, ONSET + 4))
    mat = np.zeros((len(rounds), CLIENTS))
    mat[:, 0] = 10.0
    mat[:, 4] = 10.0
    mat[-1, 7] = 10.0
    mat[0, 8] = 10.0
    p = {"key": {"seed": 3, "victim": 0}, "rounds": rounds, "mats": {"s": mat}, "sources": {4: 5}}
    e = env.evaluate_variant([p], "s", 1.0, ONSET)
    n = len(rounds) - 2
    assert e["tpr"] == [n, n] and e["onset"] == [1, 1] and e["run"] == [1, 1]
    assert e["src"] == [n, n]
    assert e["uninv"] == [1, n * (CLIENTS - 2)]
    assert e["byst"] == [n + 1, n * (CLIENTS - 1)]
    assert e["latency_median"] == 0.0


def _auc_entry(all_mean, excl_mean, pool="TEST"):
    return {"all": {pool: {"mean": all_mean}}, "excl_sources": {pool: {"mean": excl_mean}}}


def test_comparator_summary_counts_strict_wins_only():
    pr = {"honest": {c: {"0.01": {"client_round": [k, 100]}} for c, k in
                     (("suspicion", 2), ("ens_a", 1), ("ens_b", 3))},
          "attacks": {"x": {"suspicion": {"0.01": {"tpr": [5, 10]}}, "ens_a": {"0.01": {"tpr": [6, 10]}},
                            "ens_b": {"0.01": {"tpr": [5, 10]}}},
                      "y": {"suspicion": {"0.01": {"tpr": [9, 10]}}, "ens_a": {"0.01": {"tpr": [1, 10]}},
                            "ens_b": {"0.01": {"tpr": [0, 0]}}}}}
    pr["honest"]["suspicion"]["0.005"] = {"client_round": [1, 100]}
    pr["attacks"]["x"]["suspicion"]["0.005"] = {"tpr": [4, 10]}
    pr["attacks"]["y"]["suspicion"]["0.005"] = {"tpr": [8, 10]}
    auc = {"x": {"suspicion": _auc_entry(0.8, 0.9), "ens_a": _auc_entry(0.9, 0.85),
                 "ens_b": _auc_entry(0.8, None)},
           "y": {"suspicion": _auc_entry(0.7, None), "ens_a": _auc_entry(0.75, 0.9),
                 "ens_b": _auc_entry(0.6, 0.5)}}
    s = env.comparator_summary(pr, auc, "TEST", ["x", "y"], cols=("ens_a", "ens_b"))
    assert s["tpr_gt_reference"] == {"ens_a": ["x"], "ens_b": []}
    assert s["auc_gt_reference"] == {"ens_a": ["x", "y"], "ens_b": []}
    assert s["auc_excl_sources_gt_reference"] == {"ens_a": [], "ens_b": []}
    assert s["n_tpr_gt_reference"] == {"ens_a": 1, "ens_b": 0}
    assert s["tpr_lt_reference"] == {"ens_a": ["y"], "ens_b": []}
    assert s["tpr_tie"] == {"ens_a": [], "ens_b": ["x"]}
    assert s["auc_lt_reference"] == {"ens_a": [], "ens_b": ["y"]}
    assert s["auc_tie"] == {"ens_a": [], "ens_b": ["x"]}
    assert s["auc_excl_sources_lt_reference"] == {"ens_a": ["x"], "ens_b": []}
    assert s["n_auc_excl_sources_tie"] == {"ens_a": 0, "ens_b": 0}
    assert s["honest_client_round"]["ens_b"] == [3, 100]
    assert s["per_attack"]["y"]["ens_b"]["tpr"] is None
    assert "reference_at_lower_target" not in s
    s = env.comparator_summary(pr, auc, "TEST", ["x", "y"], cols=("ens_a", "ens_b"),
                               ref_low_target=0.005)
    low = s["reference_at_lower_target"]
    assert low["honest_client_round"] == [1, 100] and low["tpr"] == {"x": 0.4, "y": 0.8}
    assert low["still_gt"] == {"ens_a": ["y"], "ens_b": []} and low["n_still_gt"]["ens_a"] == 1
    assert s["comparators_at_lower_target"] == {"target": 0.005, "columns": {}}
    pr["honest"]["ens_b"]["0.005"] = {"client_round": [2, 100]}
    pr["attacks"]["x"]["ens_b"]["0.005"] = {"tpr": [6, 10]}
    pr["attacks"]["y"]["ens_b"]["0.005"] = {"tpr": [0, 0]}
    s = env.comparator_summary(pr, auc, "TEST", ["x", "y"], cols=("ens_a", "ens_b"),
                               ref_low_target=0.005)
    cl = s["comparators_at_lower_target"]["columns"]
    assert list(cl) == ["ens_b"] and cl["ens_b"]["honest_client_round"] == [2, 100]
    assert cl["ens_b"]["tpr"] == {"x": 0.6, "y": None}
    assert cl["ens_b"]["still_gt_reference"] == ["x"] and cl["ens_b"]["n_still_gt_reference"] == 1


def test_ranking_inversion_picks_the_largest_rule():
    auc = {"x": {c: _auc_entry(v, v) for c, v in zip(env.RANKING_RULES, (0.1, 0.4, 0.2, None))}}
    inv = env.ranking_inversion(auc, "TEST", ["x"])
    assert inv["x"]["max_auc"] == 0.4 and inv["x"]["rule"] == env.RANKING_RULES[1]
    assert set(inv["x"]["auc"]) == set(env.RANKING_RULES[:3])
    assert "max_seed_auc" not in inv["x"]
    assert env.RANKING_RULES == ("krum", "median_dev", "trimmed_mean_dev", "bulyan")
    assert env.RANKING_RULES_Z == tuple(c + "_z" for c in env.RANKING_RULES)


def test_inversion_block_with_seed_maxima():
    def entry(mean, per_seed):
        e = _auc_entry(mean, mean)
        e["all"]["per_seed"] = per_seed
        return e
    rules = env.RANKING_RULES
    auc = {"x": {c: entry(m, {"3": m - 0.1, "4": m + 0.1, "9": 0.99}) for c, m in zip(rules, (0.1, 0.2, 0.3, 0.45))},
           "y": {c: entry(m, {"3": m, "4": m}) for c, m in zip(rules, (0.6, 0.0, 0.0, 0.0))}}
    inv = env.ranking_inversion(auc, "TEST", ["x", "y"], rules, seeds=[3, 4])
    assert inv["x"]["max_seed_auc"] == pytest.approx(0.55)
    b = env.inversion_block(inv, ["x", "y", "absent"], rules)
    fa = b["families_a_d"]
    assert fa["attacks"] == ["x", "y"] and fa["attack"] == "y" and fa["rule"] == "krum"
    assert fa["max_auc"] == 0.6 and fa["n_all_below_half"] == 1
    assert fa["max_seed_auc"] == pytest.approx(0.6)
    assert env.inversion_block(inv, ["absent"], rules)["families_a_d"] is None


def test_member_breakdown_attributes_rule_and_sign():
    members = ("m_a_z", "m_b_z")
    rounds = list(range(ONSET, ONSET + 3))
    za = np.array([[5.0, 0.0, 0.0], [0.0, -7.0, 0.0], [1.0, 0.0, 0.0]])
    zb = np.array([[-6.0, 0.0, 0.0], [2.0, 0.0, 0.0], [np.nan, 0.0, 4.0]])
    ens = np.maximum(np.abs(za), np.where(np.isfinite(zb), np.abs(zb), -np.inf))
    mats = {"m_a_z": za, "m_b_z": zb, "ens": ens}
    att = {"key": {"victim": 0}, "rounds": rounds, "mats": mats}
    assert env.member_breakdown([att], "ens", members, 3.0, ONSET) == {"m_b_z-": 1}
    hon = {"key": {"victim": None}, "rounds": rounds, "mats": mats}
    assert env.member_breakdown([hon], "ens", members, 3.0, ONSET) == {"m_b_z-": 1, "m_a_z-": 1, "m_b_z+": 1}
    hon2 = {"key": {"victim": None}, "rounds": [ONSET - 1] + rounds[1:], "mats": dict(mats, ens=ens + 0.5)}
    assert env.member_breakdown([hon2], "ens", members, 3.0, ONSET) == {"m_a_z-": 1, "m_b_z+": 1, "mismatch": 2}
    assert env.MEMORYLESS_MEMBERS == tuple(f"{b}_z" for b in env.BASELINES)


def test_command_line_is_relative(monkeypatch):
    monkeypatch.setattr(env.sys, "argv", [r"C:\somewhere\experiments\envelope_eval.py", "--workers", "4"])
    assert env.command_line() == "python experiments/envelope_eval.py --workers 4"
    monkeypatch.setattr(env.sys, "argv", ["envelope_eval.py"])
    assert env.command_line() == "python experiments/envelope_eval.py"
