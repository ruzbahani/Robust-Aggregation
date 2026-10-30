import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from passportfl.passport import PassportNorm, build_model, make_passport_tensors

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


pe = load_module('passport_equivalence', 'experiments/passport_equivalence_eval.py')
ENTROPY = b'test-passport-equivalence'


def tiny_model():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = build_model(10, 16, 8, passport=True).double().eval()
    theta = {k: v.double() for k, v in make_passport_tensors(model.passport_spec(), ENTROPY).items()}
    pe.configure(model, theta)
    return model, theta


def test_unit_rms_override_matches_library():
    v = torch.randn(37, dtype=torch.float64)
    assert torch.equal(pe.unit_rms_with(pe.EPS_IMPL)(v), PassportNorm._unit_rms(v))


def test_per_branch_scales_exact_for_eps0_and_analytic_for_eps_impl():
    model, theta = tiny_model()
    keys = sorted(theta)
    assert len(keys) == 2 * len(model.passport_layers) == 24
    a = pe.per_branch_draw(keys, 1e-2, 1e2, np.random.default_rng(3))
    pe.configure(model, theta, eps=0.0)
    base = pe.affine_of(model)
    pe.configure(model, pe.scaled(theta, a), eps=0.0)
    got = pe.affine_of(model)
    assert max(float((g - g0).abs().max()) for (g, _), (g0, _) in zip(got, base)) < 1e-12
    assert max(float((b - b0).abs().max()) for (_, b), (_, b0) in zip(got, base)) < 1e-12
    rms = pe.branch_rms(model, theta)
    two = {k: 2.0 for k in keys}
    pe.configure(model, theta)
    base = pe.affine_of(model)
    pe.configure(model, pe.scaled(theta, two))
    got = pe.affine_of(model)
    r, m = rms['p0_gamma']['rms'], rms['p0_gamma']['max_abs']
    pred = m * pe.EPS_IMPL / ((2 * r + pe.EPS_IMPL) * (r + pe.EPS_IMPL))
    meas = float((got[0][0] - base[0][0]).abs().max())
    assert meas == pytest.approx(pred, rel=1e-5)
    assert pe.eps_bound(2.0, r) < pe.EPS_IMPL / r


def test_nullspace_forgery_exact_and_cast_equals_forge():
    model, theta = tiny_model()
    forged, _, resid, resid32, _ = pe.forge64(model, theta, (0.0, 1.0), entropy=ENTROPY)
    ref32, _, _, _ = pe.forge(model, theta, (0.0, 1.0), entropy=ENTROPY)
    for v in ref32:
        assert resid[v] < 1e-12
        for k in ref32[v]:
            assert torch.equal(forged[v][k].float(), ref32[v][k])
    x = torch.randn(4, 3, 32, 32, dtype=torch.float64)
    with torch.no_grad():
        pe.configure(model, theta)
        lg0 = model(x)[0]
        pe.configure(model, forged[pe.DIST_MATCHED])
        assert float((model(x)[0] - lg0).abs().max()) < 1e-10


def test_wellformed_forgery_passes_range_and_lattice():
    model, theta = tiny_model()
    start = {k: v.double() for k, v in make_passport_tensors(model.passport_spec(), b'other-draw').items()}
    wf, diag = pe.wellformed_forgery(model, theta, start)
    s = pe.wellformedness(wf)
    assert s['frac_outside_generator_range'] == 0.0 and s['frac_on_generator_lattice'] == 1.0
    assert diag['max_rel_residual_A_theta'] < 1e-5
    assert pe.wellformedness(theta)['frac_on_generator_lattice'] == 1.0


def test_rankmatched_forgery_keeps_the_map_and_takes_the_draw_law():
    model, theta = tiny_model()
    start = {k: v.double() for k, v in make_passport_tensors(model.passport_spec(), b'other-draw').items()}
    wf, _ = pe.wellformed_forgery(model, theta, start)
    rm, diag = pe.wellformed_forgery(model, theta, start, rank_iters=20)
    assert diag['rank_matching_rounds'] == 20 and diag['max_rel_residual_A_theta'] < 1e-5
    s, s0 = pe.wellformedness(rm), pe.wellformedness(start)
    assert s['frac_outside_generator_range'] == 0.0 and s['frac_on_generator_lattice'] == 1.0
    assert abs(s['ks_vs_uniform'] - s0['ks_vs_uniform']) < pe.wellformedness(wf)['ks_vs_uniform']
    wf0, _ = pe.wellformed_forgery(model, theta, start, rank_iters=0)
    assert all(torch.equal(wf0[k], wf[k]) for k in wf)


def test_shaped_start_keeps_values_and_shapes_neighbours():
    model, theta = tiny_model()
    start = {k: v.double() for k, v in make_passport_tensors(model.passport_spec(), b'other-draw').items()}
    st = pe.shaped_start(start, 0.2, 0)
    assert all(torch.equal(torch.sort(st[k].flatten()).values, torch.sort(start[k].flatten()).values) for k in start)
    z, z0 = pe.serial_z(st), pe.serial_z(start)
    assert z['serial_z_width'] < -10 and z['serial_z_height'] < -10
    assert abs(z0['serial_z_width']) < 4
    assert pe.shaped_start(start, 0.2, 0)['p0_gamma'].equal(st['p0_gamma'])
    sh, diag = pe.wellformed_forgery(model, theta, pe.shaped_start(start, pe.SHAPE_BETA, 0), rank_iters=5)
    s = pe.wellformedness(sh)
    assert diag['max_rel_residual_A_theta'] < 1e-5
    assert s['frac_outside_generator_range'] == 0.0 and s['frac_on_generator_lattice'] == 1.0


def test_unlinked_forgery_zeroes_the_w_free_comparisons():
    model, theta = tiny_model()
    start = {k: v.double() for k, v in make_passport_tensors(model.passport_spec(), b'other-draw').items()}
    rm, _ = pe.wellformed_forgery(model, theta, start, rank_iters=5)
    ul, diag = pe.wellformed_forgery(model, theta, start, rank_iters=5, unlink=True)
    assert diag['max_rel_residual_A_theta'] < 1e-5 and diag['forced_shift_ratio_max'] > 0
    s = pe.wellformedness(ul)
    assert s['frac_outside_generator_range'] == 0.0 and s['frac_on_generator_lattice'] == 1.0
    lk, lk0 = pe.linkage_stats(theta, ul), pe.linkage_stats(theta, rm)
    assert abs(lk['z_cos']) < 0.05 and abs(lk['z_chan_means']) < 0.05
    assert lk0['z_cos'] > abs(lk['z_cos'])
    assert pe.linkage_stats(theta, theta)['z_chan_means'] == pytest.approx(math.sqrt(lk['n_channels']))


def test_chanvar_z_detects_channel_levels():
    g = torch.Generator().manual_seed(1)
    iid = {'a': torch.rand(1, 64, 8, 8, generator=g, dtype=torch.float64) * 2 - 1}
    assert abs(pe.chanvar_z(iid)) < 4
    lev = torch.linspace(-0.2, 0.2, 64, dtype=torch.float64).reshape(1, 64, 1, 1)
    assert pe.chanvar_z({'a': iid['a'] + lev}) > 10
    assert pe.chanvar_z({'a': torch.tensor([-0.5, 0.5], dtype=torch.float64)}) is None


def test_extra_screens_are_reported_outside_the_battery():
    model, theta = tiny_model()
    s = pe.wellformedness(theta)
    assert all(k in s for k in pe.EXTRA_SCREENS)
    assert pe.passes_screens({**s, **{k: 1e3 for k in pe.EXTRA_SCREENS}}) == pe.passes_screens(s)
    assert set(pe.serial_z(theta)) == {'serial_z_width', 'serial_z_height', 'serial_z_channel'}
    assert 'serial_z_lag3_channel' in pe.serial_z(theta, 3)


def test_fine_linkage_statistics_and_channel_alignment_reference():
    model, theta = tiny_model()
    lk = pe.linkage_stats(theta, theta)
    n_rows = sum(v.shape[1] * v.shape[2] for v in theta.values())
    assert lk['z_chan_row_means'] == pytest.approx(math.sqrt(n_rows))
    other = {k: v.double() for k, v in make_passport_tensors(model.passport_spec(), b'other-draw').items()}
    assert abs(pe.linkage_stats(theta, other)['z_chan_row_means']) < 5
    ca = pe.channel_alignment(model, theta, {'self': theta, 'other': other})
    assert ca['others']['self']['share_of_chan_mean_energy_cos_gt_0.9'] == pytest.approx(
        ca['share_of_chan_mean_energy_cos_gt_0.9'])
    assert ca['n_directions_all_layers'] >= ca['n_cos_gt_0.9_all_layers']
    assert 'others' not in pe.channel_alignment(model, theta)


def test_pick_shaping_seed_minimises_the_gap():
    model, _ = tiny_model()
    start = {k: v.double() for k, v in make_passport_tensors(model.passport_spec(), b'other-draw').items()}
    seed, gap = pe.pick_shaping_seed(start, 0.05, 0.5, pool=6)
    gaps = [abs(pe.serial_z(pe.shaped_start(start, 0.05, s))['serial_z_channel'] - 0.5) for s in range(6)]
    assert gap == min(gaps) and gaps[seed] == gap
    assert pe.UNLINK_SECANT_BETA == pytest.approx(0.1555188, abs=1e-6)


def test_passes_screens_battery():
    ok = {'frac_outside_generator_range': 0.0, 'frac_on_generator_lattice': 1.0, 'ks_vs_uniform': 0.001,
          'ks_critical_1pct': 0.002, 'branches_over_ks_1pct': 0, 'serial_z_width': 1.0, 'serial_z_height': -2.0,
          'serial_z_channel': 0.0}
    assert pe.passes_screens(ok)
    for k, v in (('frac_outside_generator_range', 1e-6), ('frac_on_generator_lattice', 0.99),
                 ('ks_vs_uniform', 0.003), ('branches_over_ks_1pct', 1), ('serial_z_channel', -2.6)):
        assert not pe.passes_screens({**ok, k: v})


def test_serial_z_detects_neighbour_correlation():
    g = torch.Generator().manual_seed(0)
    iid = {'a': torch.rand(1, 32, 8, 8, generator=g, dtype=torch.float64) * 2 - 1}
    z = pe.serial_z(iid)
    assert all(abs(z['serial_z_' + ax]) < 4 for ax in ('width', 'height', 'channel'))
    smooth = {'a': iid['a'] + iid['a'].roll(1, -1)}
    assert pe.serial_z(smooth)['serial_z_width'] > 10


def test_ks_statistic_on_a_small_example():
    s = pe.wellformedness({'a': torch.tensor([-0.5, 0.5], dtype=torch.float64)})
    assert s['ks_vs_uniform'] == pytest.approx(0.25)
    assert s['ks_critical_1pct'] == pytest.approx(pe.KS_C01 / math.sqrt(2))


def test_merge_refuses_other_settings(tmp_path):
    meta = {k: 1 for k in pe.MERGE_KEYS}
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text(json.dumps({'meta': meta, 'family': {}}))
    b.write_text(json.dumps({'meta': {**meta, 'checkpoint_sha256': 2}, 'affine': {}}))
    with pytest.raises(SystemExit):
        pe.merge([str(a), str(b)], str(tmp_path / 'out.json'))
