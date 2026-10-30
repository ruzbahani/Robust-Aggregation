from pathlib import Path
import argparse
import json
import re
import statistics
from decimal import Decimal, ROUND_HALF_UP

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
SOURCES = set()
ATTACKS = ['global_echo', 'peer_copy', 'lagged_self', 'ar2_replay', 'noise', 'shrink']


def load_json(name):
    SOURCES.add(name)
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))


def format_number(x, places=3):
    value = Decimal(str(x)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return f'{value:.{places}f}'


def format_percent(x, places=1):
    return format_number(100 * x, places) + '%'


def format_milliseconds(x):
    return format_number(x[0]) + ' ± ' + format_number(x[1])


def load_innovation_results(pattern, seeds, attack, rule):
    return statistics.mean(load_json(pattern % s)['results'][attack][rule]['auc'] for s in range(seeds))


def expected_tables(include_effort=True):
    tables = {}
    rows = {}
    for label, rule in [('Krum', 'krum'), ('Coordinate-wise median', 'median_dev'),
                        ('Trimmed mean', 'trimmed_mean_dev'), ('Two-sided norm bound', 'norm_bound')]:
        values = [load_innovation_results('innovation_cifar100_s%d.json', 5, a, rule) for a in ATTACKS]
        values += [load_innovation_results('innovation_cifar100_d01_s%d.json', 3, 'global_echo', rule),
                   load_innovation_results('innovation_cifar10_s%d.json', 3, 'global_echo', rule)]
        rows[label] = list(map(format_number, values))
    b = load_json('bulyan_per_seed.json')['configs']
    rows['Bulyan'] = list(map(format_number, [b['cifar100_d05'][a]['auc_mean'] for a in ATTACKS] +
                                    [b['cifar100_d01']['global_echo']['auc_mean'], b['cifar10']['global_echo']['auc_mean']]))
    for label, rule in [('FLTrust, server root set', 'fltrust_mean'),
                        ('FoolsGold, full history', 'foolsgold_mean')]:
        values = [load_json('faithful_baselines_cifar100_d05.json')['summary'][a][rule] for a in ATTACKS]
        values += [load_json(f'faithful_baselines_{cfg}.json')['summary']['global_echo'][rule]
                   for cfg in ('cifar100_d01', 'cifar10')]
        rows[label] = list(map(format_number, values))
    tables['inversion'] = rows

    det = load_json('auditor_statistics.json')['frozen_threshold']['HELDOUT']['detectors']
    rows = {}
    for label, key in [('Payload auditor', 'suspicion'), ('Without structural tier', 'abl_full_nostruct'),
                       ('Stateful ensemble, 24 families', 'ens_stateful'), ('Causal fits only', 'ens_stateful_causal'),
                       ('Burn-in fits only', 'ens_stateful_burnin'), ('Single stateful, CAL-selected', 'tb_maha_pca_burnin_zabs'),
                       ('AR(1) residual', 'tb_ar1_burnin_zabs'), ('Six memoryless rules', 'ens_stateless'),
                       ('Stateful + memoryless', 'ens_all'), ('Stateful + FoolsGold', 'ens_stateful_fg'),
                       ('Stateful + memoryless + FoolsGold', 'ens_all_fg'), ('Stateful + x', 'ens_stateful_x'),
                       ('Stateful + x + rep', 'ens_stateful_x_rep')]:
        r = det[key]
        cells = [f"{format_number(Decimal(str(r['operating_points'][t]['worst_tpr'])) * 100, 1)} "
                 f"({format_number(Decimal(str(r['operating_points'][t]['clean_fpr'])) * 100, places)})"
                 for t, places in (('0.1', 1), ('0.05', 1), ('0.01', 2), ('0.005', 2))]
        rows[label] = cells + [format_number(r['worst_auc']), format_number(r['worst_auc_excl_copy_source'])]
    tables['operating'] = rows
    d = load_json('operating_table.json')
    pc = d['rows'][0]['per_attack']['peer_copy']
    false_all = (pc['flag_rate_copy_source'] + 8 * pc['fpr_uninvolved']) / 9
    precision = pc['tpr'] / (pc['tpr'] + pc['flag_rate_copy_source'] + 8 * pc['fpr_uninvolved'])
    tables['peercopy'] = {
        'Copier detected': [format_percent(pc['tpr'])],
        'Copied honest client accused': [format_percent(pc['flag_rate_copy_source'])],
        'False alarms on the other eight honest clients': [format_percent(pc['fpr_uninvolved'], 2)],
        'False alarms over all nine honest clients': [format_percent(false_all, 2)],
        'Share of all flags falling on a real attacker': [format_percent(precision)]}

    pb = load_json('passport_equivalence_binding.json')['table2']
    sw = load_json('identity_swap.json')
    fw, inv, ns = pb['forward'], pb['inverse_dimensions'], pb['inverse_nullspace']
    head = {r['name']: r for r in sw['rows']['genuine_other_head']}
    older = [head[k]['acc_prototype'] for k in ('genuine_E_R_minus_2', 'genuine_E_R_minus_5', 'genuine_E_1')]
    pairs = sw['rows']['forgery_genuine_head'] + sw['rows']['forgery_vs_genuine_same_head']
    rows = {
        'Prototype accuracy, correct passport': [format_number(fw['correct']['acc_prototype'], 4)],
        'random passport (three draws)': [format_number(fw['random']['acc_prototype']['max'], 4)],
        'zero passport': [format_number(fw['zero']['acc_prototype'], 4)],
        "the other client's passport": [format_number(fw['foreign']['acc_prototype']['max'], 4)],
        'chance at K=100': [format_number(fw['chance'], 4)],
        'Ambiguity attack, 20% of data, 2000 steps':
            [format_number(load_json('binding_eval_ckpt.json')['ambiguity_attack']['curve'][-1][1], 4)],
        'Passport entries': [f"{inv['passport_entries']:,}"],
        'Independent constraints, without / with the 24 branch scales':
            [f"{inv['numeric_rank_total']:,} / "
             f"{load_json('passport_equivalence_binding.json')['family']['family_constraints_eps0']:,}"],
        'Nullspace of the installation map': [format_percent(inv['nullspace_fraction'], 2)],
        'Invariant family (eps=0): nullspace and 24 positive scales': [format_percent(inv['family_fraction_eps0'], 2)],
        'Prototype accuracy, original then distribution-matched forgery':
            [', '.join(format_number(x, 4) for x in ns['acc_prototype_original_forged'])],
        'Predictions changed by any nullspace forgery, of 10,000': [str(ns['predictions_changed_any_nullspace_row'])],
        'Max logit shift in float64, forgery; floor': [f"{ns['distribution_matched_float64_max_abs_dlogit']:.1e}; "
                                                       f"{pb['floor_float64']['max_abs_dlogit']:.1e}"],
        'Max logit shift, forgery stored in float32': [f"{ns['distribution_matched_float32_stored_max_abs_dlogit']:.1e}"],
        'Accuracy under the current head E^R': [format_number(sw['reference']['acc_prototype'], 4)],
        'the head of the last training round, E^(R-1)': [format_number(head['genuine_E_R_minus_1']['acc_prototype'], 4)],
        'older genuine heads': [format_number(min(older), 4) + '-' + format_number(max(older), 4)],
        'a random head': [format_number(head['genuine_random_head']['acc_prototype'], 4)],
        'the term removed (z=0)': [format_number(head['genuine_no_head_z0']['acc_prototype'], 4)],
        'a forged passport under E^R': [format_number(min(r['acc_prototype'] for r in sw['rows']['forgery_genuine_head']), 4)],
        'Predictions changed, forged vs genuine passport under one head (8 pairs)':
            [str(max(r['n_prototype_predictions_changed'] for r in pairs))]}
    tables['passport'] = rows

    cp = load_json('combination_power.json')
    modes = ('max_raw', 'any_raw', 'max_z')
    rows = {'realized honest FPR': [format_percent(cp['fpr'][m], 2) for m in modes]}
    for label, attack in zip(('global echo', 'peer copy', 'lagged self', 'AR(2) replay', 'noise', 'shrink'), ATTACKS):
        rows[label] = [format_percent(cp['tpr'][attack][m]) for m in modes]
    rows['worst case'] = [format_percent(cp['worst_tpr'][m]) for m in modes]
    tables['combination'] = rows

    fc = load_json('feature_combination.json')['results']
    rows = {}
    labels = [('maximum of oriented robust', 'max_z'), ('sum, or mean', 'sum_z'),
              ('maximum after mean', 'max_zsd'), ('logistic regression', 'logreg_z'),
              ('Mahalanobis distance', 'mahal_raw'), ('isolation forest', 'iforest_raw')]
    for label, key in labels:
        r = fc[key]
        rows[label] = [format_percent(r['clean_fpr'], 2), format_percent(r['worst_tpr']), format_number(r['worst_auc'])]
    tables['combiners'] = rows

    betas = (0, .5, 1, 2)
    alone = load_json('adaptive_costcurve.json')['cells']
    fused = load_json('combined_grid.json')['cells']
    beacon = load_json('unpredictable_eval.json')['adaptive']
    def a(beta):
        return alone[f'suspicion/mimic|peer_current|phi=0|beta={beta:g}']['stats']['suspicion']['auc']
    def f(beta, rule):
        return fused[f'max|mimic|flat|peer_current|phi=0|beta={beta:g}']['stats'][rule]
    rows = {'Payload auditor alone': [format_number(a(beta)) for beta in betas]}
    for label, rule in [('Fused, maximum of seven', 'combined_max'), ('Fused, weighted sum', 'combined_sum'),
                        ('Fused, two-of-seven', 'combined_two')]:
        rows[label] = [format_number(f(beta, rule)['auc']) for beta in betas]
    rows['Beacon, three of five'] = [format_number(a(0))] + [format_number(beacon[f'sub3|worst|peer_current|phi=0|beta={beta:g}']['auc']) for beta in betas[1:]]
    rows['TPR'] = [format_percent(f(beta, 'combined_max')['tpr_client_round_TEST'], 0) for beta in betas]
    tables['adaptive'] = rows

    effort = load_json('reduced_effort_summary.json')
    full_steps = effort['0.1']['full_steps']
    rows = {'shrink, scale 0.1': [f'{full_steps}/{full_steps}', '0.100',
            format_number(load_innovation_results('innovation_cifar100_s%d.json', 5, 'shrink', 'suspicion')),
            format_number(d['rows'][0]['per_attack']['shrink']['tpr'])] +
            [format_number(load_innovation_results('innovation_cifar100_s%d.json', 5, 'shrink', rule))
             for rule in ('norm_bound', 'krum', 'median_dev')]}
    modes = load_json('reduced_effort_summary_modes.json')
    for label, key, r in [('trains a tenth', '0.1', None), ('trains half', '0.5', None),
                          ('forward passes only', None, modes['fwd']), ('no computation', None, modes['zero'])]:
        r = effort[key] if r is None else r
        rows[label] = [f"{r['victim_steps']}/{r['full_steps']}", format_number(r['innovation_over_cohort'][0]),
                       format_milliseconds(r['auc']['suspicion']), format_milliseconds(r['tpr']), format_milliseconds(r['auc']['norm_bound']),
                       format_milliseconds(r['auc']['krum']), format_milliseconds(r['auc']['median_dev'])]
    tables['effort'] = rows

    if not include_effort:
        return tables
    ed = load_json('effort_decomposition_bn.json')
    if ed.get('schema') != 'fixed-readout-bn-split-v1':
        raise ValueError('effort decomposition must be the parameter/statistics split run')
    undefined_or = lambda x: 'undefined' if x is None else format_milliseconds(x)
    rows, split = {}, {}
    for label in ('0', '0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1'):
        r = ed['by_fraction']['full' if label == '1' else label]
        rows[label] = [format_milliseconds(r['move_over_move_full']), undefined_or(r['cos_move_full']),
                       format_milliseconds(r['innov_over_innov_full'])]
        split[label] = [undefined_or(r['share_par']), format_milliseconds(r['bn_over_move_full']),
                        undefined_or(r['cos_recal_full']), format_milliseconds(r['fwd_over_move_full']),
                        format_milliseconds(r['innov_fwd_over_innov_full'])]
    tables['effortdecomp'] = rows
    tables['effortsplit'] = split

    dr = load_json('reduced_effort_drivers.json')
    if dr.get('schema') != 'reduced-effort-drivers-v3':
        raise ValueError('reduced-effort drivers must include the attack-free control')
    names = ('u1', 'u2', 'g', 'x', 'rep')
    columns = [dr['summary'][t] for t in ('fwd', 'zero', 'f0.1', 'f0.5')] + [dr['control']['summary']]
    span = lambda v, p: format_number(min(v), p) + '–' + format_number(max(v), p)
    def top(c):
        counts = sorted(((n, -names.index(k), k) for k, n in c['argmax_flagged'].items() if n), reverse=True)
        return ', '.join(f'{k} {n}' for n, _, k in counts) or 'none'
    def change(c):
        rel = [float(Decimal(str(x)).exp()) - 1 for x in c['magnitude']['per_seed_victim_change']]
        signed = lambda x: ('+' if x > 0 else '') + format_percent(x)
        return signed(min(rel)) + ' to ' + signed(max(rel))
    raw = lambda c, q: c['per_seed_raw'][q]['victim']
    rows = {'flagged / client-rounds': [f"{c['n_flagged']}/{c['n_attacked']}" for c in columns],
            'statistic attaining the score': [top(c) for c in columns]}
    for k in names:
        rows['AUC of ' + k + ' alone'] = [format_number(c['auc_single'][k][0]) for c in columns]
    rows.update({
        'payload step / cohort step': [span(raw(c, 'step_over_cohort_step'), 2) for c in columns],
        'self-replay distance / cohort step': [span(raw(c, 'replay_dist_over_cohort_step'), 2) for c in columns],
        'absent-row share of squared step': [span(raw(c, 'absent_row_share_of_step'), 3) for c in columns],
        'magnitude change': [change(c) for c in columns],
        'rank of the fall among 10 clients': [', '.join(map(str, c['magnitude']['per_seed_victim_change_rank']))
                                              for c in columns]})
    tables['effortdrivers'] = rows
    return tables
