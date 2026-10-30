from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import html
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

GROUPS = {
    'archives': 'Honest federations and architecture controls',
    'geometry': 'Robust rankings, exact comparisons and clustering',
    'references': 'FLTrust and FoolsGold with their required inputs',
    'identity': 'Behavioral commitment and passport constructions',
    'support': 'Declared masks and class-support sweeps',
    'auditor': 'Calibration, temporal baselines and feature combinations',
    'effort': 'Real optimizer budgets and fixed-readout controls',
    'adaptive': 'Adaptive payloads, fusion and randomized verification',
    'utility': 'Paired utility losses and computation cost',
}


@dataclass
class Job:
    name: str
    group: str
    kind: str
    script: str
    args: list
    outputs: list
    needs: list

    def command(self):
        return [sys.executable, '-u', self.script, *map(str, self.args)]


def plan(device='cpu', workers=4):
    jobs = []
    def add(name, group, kind, script, args=(), outputs=(), needs=()):
        jobs.append(Job(name, group, kind, 'experiments/' + script + '.py',
                        list(args), list(outputs), list(needs)))
    def innov(tag, seed, clients=10, rounds=35, delta=.5, dataset='cifar100', extra=()):
        add(tag, 'archives', 'archive', 'innovation_eval',
            ['--tag', tag, '--seed', seed, '--clients', clients, '--rounds', rounds,
             '--dirichlet', delta, '--dataset', dataset, '--num-classes', 10 if dataset == 'cifar10' else 100,
             '--attack-start', 12, '--eval-from', 12, '--device', device, *extra],
            [f'results/innovation_{tag}.json'],
            [f'results/archives/arch_{tag}_c{clients}_r{rounds}*_s{seed}.npz'])
    for s in range(8):
        innov(f'cifar100_s{s}', s)
    for s in range(3):
        innov(f'cifar100_d01_s{s}', s, delta=.1)
        innov(f'cifar10_s{s}', s, dataset='cifar10')
        innov(f'late70_s{s}', s, rounds=70)
        for n in (5, 20, 40):
            innov(f'cohort{n:02d}_s{s}', s, clients=n)
        innov(f'nopp_s{s}', s, extra=['--no-passport'])
    add('honest-behavior-and-checkpoints', 'archives', 'train', 'train_federation',
        ['--rounds', 50, '--warmup', 28, '--seed', 0, '--scheme', 'ML-DSA-44',
         '--save-models', '--tag', 'cifar100', '--device', device],
        ['results/fedproto_cifar100.json', *[f'checkpoints/cifar100_client{i}.pt' for i in range(10)]])
    for s in range(3):
        for passport in (True, False):
            tag = f'abl_{"pp" if passport else "nopp"}_s{s}'
            add(tag, 'archives', 'train', 'train_federation',
                ['--rounds', 35, '--warmup', 40, '--seed', s, '--scheme', 'Ed25519',
                 '--tag', tag, '--device', device, *([] if passport else ['--no-passport'])],
                [f'results/fedproto_{tag}.json'])
    for s in range(2):
        add(f'mixed-support-{s}', 'archives', 'train', 'mask_archive',
            ['--partition', 'mixed', '--seed', s, '--tag', 'maskmix', '--device', device],
            [f'results/archives/arch_maskmix_c10_r35_s{s}.npz'])
    add('support-10', 'archives', 'train', 'mask_archive',
        ['--partition', 'shard', '--classes-per-client', 10, '--seed', 0,
         '--tag', 'maskshard10', '--device', device], ['results/archives/arch_maskshard10_c10_r35_s0.npz'])
    for k in (5, 10, 20, 40):
        for s in (0, 1):
            add(f'shard{k:02d}-{s}', 'archives', 'train', 'mask_archive',
                ['--partition', 'shard', '--classes-per-client', k, '--seed', s,
                 '--tag', f'shard{k:02d}', '--device', device],
                [f'results/archives/arch_shard{k:02d}_c10_r35_s{s}.npz'])
    for script, out in [('krum_margin', 'krum_margin'),
                        ('sweep_summary', 'sweep_summary'), ('parameter_sweep', 'sweeps')]:
        add(script, 'geometry', 'replay', script, ['--seeds', '0,1,2'] if script == 'parameter_sweep' else [],
            [f'results/{out}.json'], ['results/archives/arch_cifar100_s0_*.npz'])
    add('bulyan_eval', 'geometry', 'replay', 'bulyan_eval',
        ['--per-seed-out', 'results/bulyan_per_seed.json', '--workers', workers],
        ['results/bulyan_eval.json', 'results/bulyan_per_seed.json'], ['results/archives/arch_cifar100_s0_*.npz'])
    add('inversion-condition', 'geometry', 'replay', 'inversion_condition', ['--workers', workers],
        ['results/inversion_condition.json'],
        ['results/archives/arch_cifar100_s0_*.npz', 'results/archives/arch_cohort40_s*_c40_r35*.npz',
         'results/krum_margin.json'])
    add('cohort05-krum-f1', 'geometry', 'replay', 'cohort05_krum_f1', (), ['results/cohort05_f1.json'],
        ['results/archives/arch_cohort05_s*_c5_r35*.npz', 'results/innovation_cohort05_s*.json',
         'results/krum_margin.json'])
    add('aggregate-impact', 'geometry', 'replay', 'aggregate_impact', ['--workers', workers],
        ['results/aggregate_impact.json'], ['results/archives/arch_cifar100_s0_*.npz'])
    for cfg, seeds in [('cifar100_d05', range(5)), ('cifar100_d01', range(3)), ('cifar10', range(3))]:
        add('faithful-' + cfg, 'references', 'root', 'faithful_baselines_eval',
            ['--config', cfg, '--seeds', *seeds, '--device', device],
            [f'results/faithful_baselines_{cfg}.json'],
            [f'results/archives/fltrust_root_{cfg}_r100_s{s}.npz' for s in seeds])
    add('fltrust-references', 'references', 'replay', 'fltrust_reference_eval', (),
        ['results/fltrust_reference_eval.json'], ['results/faithful_baselines_cifar100_d05.json'])
    std_needs = ['results/archives/arch_cifar100_s0_*.npz', 'results/archives/arch_cifar100_d01_s*_c10_r35*.npz',
                 'results/archives/arch_cifar10_s*_c10_r35*.npz', 'results/innovation_cifar100_s0.json']
    add('std-baseline', 'references', 'replay', 'std_baseline_eval', (), ['results/std_baseline.json'], std_needs)
    add('std-dagmm-within', 'references', 'replay', 'std_baseline_eval', ['--dagmm', 'within', '--workers', workers],
        ['results/std_dagmm_within.json'], std_needs)
    add('passport-checkpoint', 'identity', 'checkpoint', 'passport_maps_eval',
        ['--device', device, '--across-clients', '--negative-out', 'results/negative_results.json',
         '--out', 'results/passport_maps_device.json'],
        ['results/negative_results.json', 'results/passport_maps_device.json'],
        [f'checkpoints/cifar100_client{i}.pt' for i in range(4)])
    add('passport-cpu-rescale', 'identity', 'checkpoint', 'passport_maps_eval',
        ['--device', 'cpu', '--out', 'results/n1_with_rescale.json'],
        ['results/n1_with_rescale.json'], ['checkpoints/cifar100_client0.pt'])
    add('ghost-40-rounds', 'identity', 'replay', 'ghost_client_eval', (), ['results/negative_results.json'])
    add('forward-and-version-binding', 'identity', 'train', 'binding_eval',
        ['--rounds', 30, '--alpha-dyn', .5, '--calib-steps', 50, '--attack-steps', 2000,
         '--warmup', 30, '--device', device], ['results/binding_eval.json'])
    add('ghost-extended', 'identity', 'replay', 'ghost_extended_eval', ['--workers', workers],
        ['results/ghost_extended.json'], ['results/fedproto_cifar100.json'])
    add('frozen-encoder-prover', 'identity', 'checkpoint', 'frozen_encoder_prover', ['--device', device],
        ['results/frozen_encoder_prover.json'], ['checkpoints/cifar100_client0.pt'])
    PE = '.reproduction/passport_equivalence/'
    PE_GROUPS = (('g1', 'family,affine,floor,rescale'), ('g2', 'nullspace,forward'), ('g3', 'eps0,dynamic'),
                 ('g4', 'wellformed'), ('g5', 'linkage'))
    for g, secs in PE_GROUPS:
        add('passport-equivalence-' + g, 'identity', 'checkpoint', 'passport_equivalence_eval',
            ['--sections', secs, '--threads', 4, '--out', PE + g + '.json'], [PE + g + '.json'],
            [f'checkpoints/cifar100_client{i}.pt' for i in range(10)] if g == 'g2' else ['checkpoints/cifar100_client0.pt'])
    add('passport-equivalence', 'identity', 'summary', 'passport_equivalence_eval',
        ['--merge', ','.join(PE + g + '.json' for g, _ in PE_GROUPS)],
        ['results/passport_equivalence.json'], [PE + g + '.json' for g, _ in PE_GROUPS])
    add('shard-summary', 'identity', 'summary', 'shard_summary', (), ['results/shard_summary.json'])
    add('ambiguity-attack-archived', 'identity', 'checkpoint', 'ambiguity_attack_eval', ['--device', device],
        ['results/ambiguity_attack_archived.json'], ['checkpoints/cifar100_client0.pt'])
    add('binding-checkpoint', 'identity', 'train', 'binding_eval',
        ['--rounds', 30, '--alpha-dyn', .5, '--calib-steps', 50, '--attack-steps', 2000, '--warmup', 30,
         '--device', device, '--random-theta-seed', 0, '--save-checkpoint', 'checkpoints/binding_a05_s0_clientA.pt',
         '--out', 'results/binding_eval_ckpt.json'],
        ['results/binding_eval_ckpt.json', 'checkpoints/binding_a05_s0_clientA.pt'])
    add('identity-swap', 'identity', 'checkpoint', 'identity_swap_eval', ['--threads', 4],
        ['results/identity_swap.json'], ['checkpoints/binding_a05_s0_clientA.pt'])
    PEB = '.reproduction/passport_equivalence_binding/'
    PEB_GROUPS = (('b1', 'family,affine,floor,rescale'), ('b2', 'nullspace,forward'), ('b3', 'wellformed'),
                  ('b4', 'linkage'))
    for g, secs in PEB_GROUPS:
        add('passport-equivalence-binding-' + g, 'identity', 'checkpoint', 'passport_equivalence_eval',
            ['--checkpoint', 'checkpoints/binding_a05_s0_clientA.pt', '--sections', secs, '--threads', 4,
             '--out', PEB + g + '.json'], [PEB + g + '.json'], ['checkpoints/binding_a05_s0_clientA.pt'])
    add('passport-equivalence-binding', 'identity', 'summary', 'passport_equivalence_eval',
        ['--merge', ','.join(PEB + g + '.json' for g, _ in PEB_GROUPS), '--out', 'results/passport_equivalence_binding.json'],
        ['results/passport_equivalence_binding.json'], [PEB + g + '.json' for g, _ in PEB_GROUPS])
    for s in range(5):
        add(f'paired-honest-{s}', 'identity', 'train', 'paired_honest_reference',
            ['--seed', s, '--rounds', 40 if s < 3 else 35, '--device', device],
            [f'results/paired_honest_s{s}.json', f'results/paired_honest/paired_honest_payload_s{s}.npz',
             f'results/paired_honest/paired_honest_probes_s{s}.npz'],
            ['results/ghost_extended.json'])
    add('paired-honest-summary', 'identity', 'summary', 'paired_honest_summary', ['--workers', workers],
        ['results/paired_honest_summary.json'],
        [*[f'results/paired_honest_s{s}.json' for s in range(5)],
         *[f'results/paired_honest/paired_honest_{k}_s{s}.npz' for k in ('payload', 'probes') for s in range(5)],
         'results/ghost_extended.json', 'results/combined_grid.json', 'results/adaptive_costcurve.json',
         'results/published_predicates.json', 'results/limited_access_grid.json',
         'results/archives/arch_cifar100_s3_*.npz', 'results/fedproto_abl_pp_s0.json', 'results/fedproto_abl_pp_s1.json'])
    add('paired-fig3a', 'identity', 'summary', 'paired_fig3a_data', (), ['results/paired_fig3a.json'],
        ['results/paired_honest_s0.json', 'results/paired_honest_s1.json', 'results/paired_honest_s2.json',
         'results/paired_honest_summary.json', 'results/negative_results.json', 'results/ghost_extended.json',
         'results/fedproto_cifar100.json'])
    add('published-predicates', 'identity', 'replay', 'published_predicates_eval', (),
        ['results/published_predicates.json'],
        ['results/archives/arch_cifar100_s0_*.npz', 'results/archives/fltrust_root_cifar100_d05_r100_s3.npz'])
    add('passport-ablation', 'identity', 'summary', 'passport_ablation_summary', (),
        ['results/passport_ablation.json'], [f'results/fedproto_abl_{arm}_s{s}.json' for arm in ('pp', 'nopp') for s in range(3)])
    for pool in ('cifar100_d05', 'shard05', 'shard10', 'shard20', 'shard40'):
        cfgs = 'paper,carry,support,full,derived,hardened' if pool == 'cifar100_d05' else 'paper,derived,hardened'
        add('mask-' + pool, 'support', 'replay', 'mask_eval',
            ['--pool', pool, '--configs', cfgs, '--workers', workers], [f'results/mask_eval_{pool}.json'],
            ['results/archives/arch_cifar100_s0_*.npz' if pool == 'cifar100_d05' else f'results/archives/arch_{pool}_*.npz'])
    add('absent-row-energy', 'support', 'replay', 'absent_row_probe', (), ['results/absent_row_probe.json'])
    add('score-range', 'support', 'replay', 'score_range', ['--workers', workers], ['results/score_range.json'],
        ['results/archives/arch_cifar100_s0_*.npz', 'results/archives/arch_cifar100_s7_*.npz',
         'results/utility_impact_s1.json', 'results/utility_robust_summary.json'])
    add('calibration', 'auditor', 'replay', 'protocol_eval',
        ['--workers', workers, '--no-sensitivity', '--out', 'results/protocol_eval_d05.json'],
        ['results/protocol_eval_d05.json'], ['results/archives/arch_cifar100_s0_*.npz'])
    for policy in ('declared', 'full'):
        out = 'temporal_eval' if policy == 'declared' else 'temporal_eval_fullmask'
        add(out, 'auditor', 'replay', 'temporal_eval',
            ['--workers', workers, '--mask-policy', policy, '--out', f'results/{out}.json'],
            [f'results/{out}.json'], ['results/protocol_eval_d05.json'])
    for script, deps in [
        ('operating_table', ['temporal_eval']), ('fresh_seed_transfer', ['temporal_eval']),
        ('held_out_summary', ['temporal_eval', 'fresh_seed_transfer']),
        ('combination_power', ['temporal_eval']), ('mask_policy_delta', ['temporal_eval', 'temporal_eval_fullmask']),
        ('feature_combination_eval', ['temporal_eval'])]:
        out = {'feature_combination_eval': 'feature_combination'}.get(script, script)
        add(script, 'auditor', 'replay', script, (), [f'results/{out}.json'], [f'results/{d}.json' for d in deps])
    c100 = ['results/archives/arch_cifar100_s0_*.npz', 'results/archives/arch_cifar100_s7_*.npz']
    add('temporal-eval-v2', 'auditor', 'replay', 'temporal_eval',
        ['--workers', workers, '--test', '3,4,5,6,7', '--peer-aware',
         '--save-scores', '.reproduction/scores/temporal_eval_v2_scores.npz',
         '--out', 'results/temporal_eval_v2.json'],
        ['results/temporal_eval_v2.json', '.reproduction/scores/temporal_eval_v2_scores.npz'],
        c100 + ['results/protocol_eval_d05.json'])
    add('auditor-statistics', 'auditor', 'replay', 'auditor_statistics', (),
        ['results/auditor_statistics.json'],
        ['.reproduction/scores/temporal_eval_v2_scores.npz', 'results/temporal_eval_v2.json',
         'results/temporal_eval.json', 'results/fresh_seed_transfer.json', 'results/held_out_summary.json'])
    for cfg, prefix in (('cifar10', 'arch_cifar10_s'), ('cifar100_d01', 'arch_cifar100_d01_s')):
        add(f'temporal-eval-{cfg}', 'auditor', 'replay', 'temporal_eval',
            ['--archive-prefix', prefix, '--cal', '0,1', '--dev', '2', '--test=',
             '--workers', workers, '--out', f'results/temporal_eval_{cfg}.json'],
            [f'results/temporal_eval_{cfg}.json'], [f'results/archives/{prefix}*_c10_r35*.npz'])
    add('eta-sensitivity', 'auditor', 'replay', 'eta_sensitivity', ['--workers', workers],
        ['results/eta_sensitivity.json'],
        c100 + ['results/archives/arch_cifar10_s*_c10_r35*.npz', 'results/temporal_eval.json',
                'results/temporal_eval_cifar10.json'])
    add('eta-convention', 'auditor', 'replay', 'eta_convention', ['--workers', workers],
        ['results/eta_convention.json'],
        ['results/archives/arch_cifar100_s4_*.npz', 'results/archives/arch_cifar100_d01_s*_c10_r35*.npz',
         'results/archives/arch_cifar10_s*_c10_r35*.npz', 'results/archives/arch_late70_s*_c10_r70*.npz',
         'results/archives/arch_cohort05_s*_c5_r35*.npz', 'results/archives/arch_cohort20_s*_c20_r35*.npz',
         'results/archives/arch_cohort40_s*_c40_r35*.npz', 'results/archives/arch_nopp_s*_c10_r35*.npz',
         'results/innovation_cohort40_s2.json', 'results/sweeps.json'])
    env_needs = ([f'results/archives/arch_cifar100_s{s}_*.npz' for s in range(8)]
                 + ['results/temporal_eval.json', 'results/temporal_eval_v2.json'])
    add('envelope-eval', 'auditor', 'replay', 'envelope_eval',
        ['--workers', workers, '--save-scores', '.reproduction/scores/envelope_scores_w5.pkl'],
        ['results/envelope_eval.json'], env_needs)
    add('envelope-longlags-w5', 'auditor', 'replay', 'envelope_eval',
        ['--auditor-only', '--variants', 'lagged_12,lagged_16', '--workers', workers,
         '--out', 'results/envelope_eval_longlags_w5.json'],
        ['results/envelope_eval_longlags_w5.json'], env_needs)
    add('envelope-w34', 'auditor', 'replay', 'envelope_eval',
        ['--auditor-only', '--lag-window', 34, '--variants', 'all,lagged_12,lagged_16', '--workers', workers,
         '--out', 'results/envelope_eval_w34.json'],
        ['results/envelope_eval_w34.json'], env_needs)
    for f in (.1, .5):
        for s in range(3):
            add(f'effort-{f}-{s}', 'effort', 'train', 'reduced_effort_eval',
                ['--fraction', f, '--seed', s, '--device', device],
                [f'results/reduced_effort_f{f}_s{s}.json', f'results/archives/arch_reduced_f{f}_s{s}.npz'])
    add('reduced-effort-summary', 'effort', 'summary', 'reduced_effort_summary', (),
        ['results/reduced_effort_summary.json'], [f'results/reduced_effort_f{f}_s{s}.json' for f in (.1, .5) for s in range(3)])
    for s in range(3):
        add(f'fixed-readout-{s}', 'effort', 'train', 'effort_decomposition',
            ['--seeds', s, '--device', device], [f'results/effort_decomposition_s{s}.json'])
    add('fixed-readout-summary', 'effort', 'summary', 'effort_decomposition',
        ['--seeds', 0, 1, 2, '--summarize-only'], ['results/effort_decomposition.json'],
        [f'results/effort_decomposition_s{s}.json' for s in range(3)])
    for s in range(3):
        add(f'bn-split-{s}', 'effort', 'train', 'effort_decomposition',
            ['--seeds', s, '--bn-split', '--device', device], [f'results/effort_decomposition_bn_s{s}.json'])
    add('bn-split-summary', 'effort', 'summary', 'effort_decomposition',
        ['--seeds', 0, 1, 2, '--bn-split', '--summarize-only'], ['results/effort_decomposition_bn.json'],
        [f'results/effort_decomposition_bn_s{s}.json' for s in range(3)])
    for mode, tag in (('forward', 'fwd'), ('zero', 'zero')):
        for s in range(3):
            add(f'effort-{tag}-{s}', 'effort', 'train', 'reduced_effort_eval',
                ['--mode', mode, '--seed', s, '--device', device],
                [f'results/reduced_effort_{tag}_s{s}.json', f'results/archives_rev2/arch_reduced_{tag}_s{s}.npz'])
    add('reduced-effort-modes-summary', 'effort', 'summary', 'reduced_effort_summary', ['--modes-only'],
        ['results/reduced_effort_summary_modes.json'],
        [f'results/reduced_effort_{t}_s{s}.json' for t in ('f0.1', 'f0.5', 'fwd', 'zero') for s in range(3)])
    add('reduced-effort-drivers', 'effort', 'replay', 'reduced_effort_drivers', (),
        ['results/reduced_effort_drivers.json'],
        [f'results/reduced_effort_{t}_s{s}.json' for t in ('f0.1', 'f0.5', 'fwd', 'zero') for s in range(3)]
        + [f'results/archives/arch_reduced_f{f}_s{s}.npz' for f in (0.1, 0.5) for s in range(3)]
        + [f'results/archives_rev2/arch_reduced_{t}_s{s}.npz' for t in ('fwd', 'zero') for s in range(3)]
        + [f'results/innovation_cifar100_s{s}.json' for s in range(3)]
        + [f'results/archives/arch_cifar100_s{s}_c10_r35_s{s}.npz' for s in range(3)])
    curves = [('A', '0,0.1,0.25,0.5,1', '0,0.25,0.5,1,2', 'peer_current', 'mimic', 'suspicion'),
              ('B', '0,0.5,1', '0.5,1,2', 'peer_lagged', 'mimic', 'suspicion'),
              ('C', '0,0.5', '1,2', 'peer_current', 'mimic', 'suspicion_g2s,suspicion_cos'),
              ('D', '0', '1', 'peer_current', 'min', 'suspicion')]
    for label, phis, betas, knowledge, aim, targets in curves:
        add('adaptive-curve-' + label, 'adaptive', 'replay', 'adaptive_eval',
            ['--phis', phis, '--betas', betas, '--knowledge', knowledge, '--aims', aim,
             '--targets', targets, '--basis-steps', 80, '--full-steps', 80, '--restarts', 3,
             '--victims', '0,2,4,6,8', '--workers', workers, '--out', f'results/adaptive_curve_{label}.json'],
            [f'results/adaptive_curve_{label}.json'], ['results/protocol_eval_d05.json'])
    add('adaptive-costcurve-merge', 'adaptive', 'summary', 'merge_adaptive_curves',
        [*[f'results/adaptive_curve_{v}.json' for v in 'ABCD'], '--out', 'results/adaptive_costcurve.json'],
        ['results/adaptive_costcurve.json'], [f'results/adaptive_curve_{v}.json' for v in 'ABCD'])
    add('fused-grid', 'adaptive', 'replay', 'combined_eval',
        ['--basis-steps', 40, '--full-steps', 80, '--restarts', 1, '--no-warm',
         '--victims', '0,1', '--workers', workers, '--out', 'results/combined_grid.json'],
        ['results/combined_grid.json'], ['results/protocol_eval_d05.json'])
    add('beacon-grid', 'adaptive', 'replay', 'unpredictable_eval',
        ['--adaptive-mechs', 'det,sub3,wsum3', '--phis', '0,0.25,0.5,1', '--betas', '0,0.5,1,2',
         '--basis-steps', 40, '--full-steps', 80, '--restarts', 1, '--victims', '0,1',
         '--workers', workers, '--out', 'results/unpredictable_eval.json'],
        ['results/unpredictable_eval.json'], ['results/protocol_eval_d05.json'])
    add('foolsgold-observer', 'adaptive', 'replay', 'adaptive_eval',
        ['--phis', '0', '--betas', '0,1', '--aims', 'mimic', '--knowledge', 'peer_current',
         '--workers', workers, '--out', 'results/adaptive_fg_observer.json'],
        ['results/adaptive_fg_observer.json'], ['results/protocol_eval_d05.json'])
    add('optimizer-effort-sweep', 'adaptive', 'replay', 'adaptive_converge',
        ['--victims', '0,7', '--workers', workers], ['results/adaptive_converge.json'])
    add('limited-access-grid', 'adaptive', 'replay', 'limited_access_eval', ['--workers', workers],
        ['results/limited_access_grid.json'], ['results/archives/arch_cifar100_s0_*.npz'])
    add('limited-access-checks', 'adaptive', 'replay', 'limited_access_checks', ['--workers', workers],
        ['results/limited_access_checks.json'], ['results/archives/arch_cifar100_s0_*.npz'])
    joint_needs = ['results/limited_access_grid.json', 'results/published_predicates.json', 'results/ghost_extended.json',
                   'results/combined_grid.json', 'results/adaptive_costcurve.json',
                   'results/archives/arch_cifar100_s0_*.npz', 'results/archives/fltrust_root_cifar100_d05_r100_s3.npz']
    add('joint-ghost', 'adaptive', 'replay', 'joint_ghost_eval', ['--workers', workers], ['results/joint_ghost.json'],
        joint_needs + ['results/fedproto_abl_pp_s2.json'])
    add('joint-ghost-paired', 'adaptive', 'replay', 'joint_ghost_eval',
        ['--workers', workers, '--bands-file', 'results/paired_honest_summary.json',
         '--honest-dev', 'results/paired_honest_s2.json', '--out', 'results/joint_ghost_paired.json'],
        ['results/joint_ghost_paired.json'],
        joint_needs + ['results/paired_honest_summary.json', 'results/paired_honest_s2.json', 'results/joint_ghost.json',
                       'results/fedproto_abl_pp_s0.json', 'results/fedproto_abl_pp_s1.json'])
    add('wall-clock-cost', 'utility', 'train', 'attacker_cost',
        ['--rounds', 3, '--device', device], ['results/attacker_cost.json'])
    for dev in ['cpu'] + ([device] if device != 'cpu' else []):
        add(f'wall-clock-cost-{dev}', 'utility', 'train', 'attacker_cost',
            ['--rounds', 5, '--device', dev, '--out', f'results/attacker_cost_{dev}.json'],
            [f'results/attacker_cost_{dev}.json'])
    for s in range(5):
        out = 'results/utility_impact_gpu_s0.json' if s == 0 else f'results/utility_impact_s{s}.json'
        add(f'utility-{s}', 'utility', 'train', 'utility_impact',
            ['--seed', s, '--shard-cap', 0, '--local-epochs', 2, '--rounds', 26, '--attack-start', 10,
             '--victim', 7, '--rule', 'max', '--scenarios', 'honest,echo,fused_b1,fused_b2,fused_b1_x3',
             '--device', device, '--out', out], [out], ['results/combined_grid.json', 'results/protocol_eval_d05.json'])
    add('utility-summary', 'utility', 'summary', 'utility_seed_spread', (),
        ['results/utility_seed_spread.json'], ['results/utility_impact_gpu_s0.json', *[f'results/utility_impact_s{s}.json' for s in range(1, 5)]])
    for s in range(5):
        out = f'results/utility_impact_lagged_s{s}.json'
        add(f'utility-lagged-{s}', 'utility', 'train', 'utility_impact',
            ['--seed', s, '--shard-cap', 0, '--local-epochs', 2, '--rounds', 26, '--attack-start', 10,
             '--victim', 7, '--rule', 'max', '--threads', 16, '--scenarios',
             'honest,honest_rep,fused_b1,fused_b2,fused_b1_x3,fused_b1_lag,fused_b2_lag,fused_b1_x3_lag',
             '--device', device, '--out', out], [out], ['results/combined_grid.json', 'results/protocol_eval_d05.json'])
    add('utility-lagged-summary', 'utility', 'summary', 'utility_lagged_summary', (),
        ['results/utility_lagged_summary.json'], [f'results/utility_impact_lagged_s{s}.json' for s in range(5)])
    for rule in ('median', 'mean'):
        for s in range(5):
            out = f'results/utility_robust_{rule}_s{s}.json'
            add(f'utility-robust-{rule}-{s}', 'utility', 'train', 'utility_robust_agg',
                ['--agg', rule, '--seed', s, '--shard-cap', 0, '--local-epochs', 2, '--rounds', 26,
                 '--attack-start', 10, '--victim', 7, '--rule', 'max', '--threads', 16,
                 '--scenarios', 'honest,honest_rep,echo,exclude', '--device', device, '--out', out],
                [out], ['results/combined_grid.json', 'results/protocol_eval_d05.json'])
    add('utility-robust-summary', 'utility', 'summary', 'utility_robust_summary', (),
        ['results/utility_robust_summary.json'],
        [f'results/utility_robust_{rule}_s{s}.json' for rule in ('median', 'mean') for s in range(5)])
    add('zero-innovation-cost', 'utility', 'replay', 'zero_innovation_cost', (), ['results/zero_innovation_cost.json'])
    rev2 = 'results/archives_rev2'
    hetero = ['--clients', 10, '--rounds', 35, '--dirichlet', .5, '--dataset', 'cifar100',
              '--num-classes', 100, '--attack-start', 12, '--eval-from', 12, '--no-passport',
              '--arch-group', 'hetero4', '--archive-dir', rev2, '--device', device]
    tgp = ['--server', 'fedtgp', '--tgp-device', 'cpu', '--tgp-threads', 1]
    arms = {'lam1': 1, 'lam10': 10, 'lam01': 0.1}
    cfgs = ['hetero'] + [f'fedtgp_{arm}' for arm in arms]
    for s in range(5):
        add(f'hetero_s{s}', 'archives', 'archive', 'innovation_eval',
            ['--tag', f'hetero_s{s}', '--seed', s, *hetero],
            [f'results/innovation_hetero_s{s}.json'],
            [f'{rev2}/arch_hetero_s{s}_c10_r35*_s{s}.npz'])
        for arm, lam in arms.items():
            tag = f'fedtgp_{arm}_s{s}'
            add(tag, 'archives', 'archive', 'innovation_eval',
                ['--tag', tag, '--seed', s, *hetero, *tgp, '--lam', lam],
                [f'results/innovation_{tag}.json'],
                [f'{rev2}/arch_{tag}_c10_r35*_s{s}.npz'])
    need_all = [f'{rev2}/arch_{c}_s{s}_c10_r35*_s{s}.npz' for c in cfgs for s in range(5)]
    rev2_cfgs = 'hetero_fedproto,hetero_fedtgp_lam1,hetero_fedtgp_lam10,hetero_fedtgp_lam01'
    add('fedtgp-pilot', 'geometry', 'replay', 'fedtgp_pilot', (), ['results/fedtgp_pilot.json'],
        ['results/archives/arch_nopp_s0_*.npz'])
    add('krum-margin-rev2', 'geometry', 'replay', 'krum_margin',
        ['--configs', rev2_cfgs, '--out', 'results/krum_margin_rev2.json'],
        ['results/krum_margin_rev2.json'], need_all)
    add('bulyan-rev2', 'geometry', 'replay', 'bulyan_eval',
        ['--configs', rev2_cfgs, '--workers', workers, '--out', 'results/bulyan_eval_rev2.json',
         '--per-seed-out', 'results/bulyan_per_seed_rev2.json'],
        ['results/bulyan_eval_rev2.json', 'results/bulyan_per_seed_rev2.json'], need_all)
    add('aggregate-impact-rev2', 'geometry', 'replay', 'aggregate_impact',
        ['--configs', rev2_cfgs, '--workers', workers, '--out', 'results/aggregate_impact_rev2.json'],
        ['results/aggregate_impact_rev2.json'],
        need_all + [f'results/innovation_{c}_s{s}.json' for c in cfgs for s in range(5)])
    for arm in arms:
        pre = f'arch_fedtgp_{arm}_s'
        add(f'fedtgp-live-impact-{arm}', 'geometry', 'replay', 'fedtgp_live_impact',
            ['--archives', rev2, '--archive-prefix', pre, '--workers', workers,
             '--out', f'results/fedtgp_live_impact_{arm}.json'],
            [f'results/fedtgp_live_impact_{arm}.json'],
            [f'{rev2}/{pre}{s}_c10_r35*_s{s}.npz' for s in range(5)])
    for cfg in cfgs:
        pre = f'arch_{cfg}_s'
        need = [f'{rev2}/{pre}{s}_c10_r35*_s{s}.npz' for s in range(5)]
        common = ['--archives', rev2, '--archive-prefix', pre, '--workers', workers]
        add(f'protocol-{cfg}', 'auditor', 'replay', 'protocol_eval',
            common + ['--out', f'results/protocol_eval_{cfg}.json'], [f'results/protocol_eval_{cfg}.json'], need)
        add(f'temporal-{cfg}', 'auditor', 'replay', 'temporal_eval',
            common + ['--out', f'results/temporal_eval_{cfg}.json'], [f'results/temporal_eval_{cfg}.json'], need)
        add(f'fedtgp-echo-{cfg}', 'auditor', 'replay', 'fedtgp_echo_eval',
            common + ['--out', f'results/fedtgp_echo_{cfg}.json'], [f'results/fedtgp_echo_{cfg}.json'], need)
        add(f'limited-access-{cfg}', 'adaptive', 'replay', 'limited_access_rev2',
            ['--archives', rev2, '--prefix', pre, '--defenses', 'P,F', '--victims', '0,1',
             '--levels', 'current,delayed:1,query:16,decision:16,feedback,pseudo', '--betas', 1,
             '--workers', workers, '--out', f'results/limited_access_{cfg}.json'],
            [f'results/limited_access_{cfg}.json'], need)
    for cfg, floor in (('hetero', 0.1), ('fedtgp_lam1', 0.0003), ('fedtgp_lam01', 0.003)):
        pre = f'arch_{cfg}_s'
        add(f'limited-access-{cfg}-calfloor', 'adaptive', 'replay', 'limited_access_rev2',
            ['--archives', rev2, '--prefix', pre, '--defenses', 'P,F', '--victims', '0,1',
             '--levels', 'current,delayed:1,query:16,decision:16,feedback,pseudo', '--betas', 1,
             '--mad-floor', floor, '--parts', f'.reproduction/limited_access_parts_{pre}_mf{floor}',
             '--workers', workers, '--out', f'results/limited_access_{cfg}_calfloor.json'],
            [f'results/limited_access_{cfg}_calfloor.json'],
            [f'{rev2}/{pre}{s}_c10_r35*_s{s}.npz' for s in range(5)])
    h60 = list(hetero)
    h60[h60.index('--rounds') + 1] = 60
    for s in range(5):
        for arm, rounds, extra in (('r60', 60, []), ('euc', 35, ['--penalty', 'euclid'])):
            tag = f'fedtgp_lam01_{arm}_s{s}'
            add(tag, 'archives', 'archive', 'innovation_eval',
                ['--tag', tag, '--seed', s, *(h60 if arm == 'r60' else hetero), *tgp, '--lam', 0.1, *extra],
                [f'results/innovation_{tag}.json'], [f'{rev2}/arch_{tag}_c10_r{rounds}*_s{s}.npz'])
    for arm, rounds in (('r60', 60), ('euc', 35)):
        cfg, pre = f'fedtgp_lam01_{arm}', f'arch_fedtgp_lam01_{arm}_s'
        need = [f'{rev2}/{pre}{s}_c10_r{rounds}*_s{s}.npz' for s in range(5)]
        common = ['--archives', rev2, '--archive-prefix', pre, '--workers', workers]
        add(f'fedtgp-echo-{cfg}', 'auditor', 'replay', 'fedtgp_echo_eval',
            common + ['--out', f'results/fedtgp_echo_{cfg}.json'], [f'results/fedtgp_echo_{cfg}.json'], need)
        add(f'protocol-{cfg}', 'auditor', 'replay', 'protocol_eval',
            common + ['--out', f'results/protocol_eval_{cfg}.json'], [f'results/protocol_eval_{cfg}.json'], need)
    xc = cfgs + ['fedtgp_lam01_r60', 'fedtgp_lam01_euc']
    add('rev2-auc-cross-check', 'auditor', 'replay', 'rev2_auc_cross_check',
        ['--protocol', *[f'results/protocol_eval_{c}.json' for c in xc],
         '--out', 'results/rev2_auc_cross_check.json'],
        ['results/rev2_auc_cross_check.json'],
        [f'results/protocol_eval_{c}.json' for c in xc]
        + [f'results/innovation_{c}_s{s}.json' for c in xc for s in (3, 4)])
    return jobs


CODE_DIRS = ('passportfl', 'experiments', 'analysis', 'notebooks')


def prepare_workdir(root, mode='archived', workdir=None):
    root = Path(root).resolve()
    if mode not in ('archived', 'replay', 'fresh'):
        raise ValueError(mode)
    if workdir:
        work = Path(workdir).resolve()
        work.mkdir(parents=True, exist_ok=True)
        if any(work.iterdir()):
            raise RuntimeError('Choose an empty working directory; an existing run is never overwritten.')
    else:
        (root / '.reproduction').mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='run_', dir=root / '.reproduction')).resolve()
    for folder in CODE_DIRS:
        shutil.copytree(root / folder, work / folder, ignore=shutil.ignore_patterns('__pycache__', '.ipynb_checkpoints'))
    for folder in ('results', 'results/archives', 'checkpoints', 'figures', 'tables', 'data'):
        (work / folder).mkdir(parents=True, exist_ok=True)
    if mode != 'fresh':
        if not (root / 'results/temporal_eval.json').is_file():
            raise FileNotFoundError('The released results are missing; archived and replay mode need results/*.json.')
        for pattern, folder in (('*.json', 'results'), ('*.pdf', 'figures'), ('*.values.json', 'figures')):
            for p in (root / folder).glob(pattern):
                shutil.copy2(p, work / folder / p.name)
    if mode == 'replay':
        for folder, ext in (('results/archives', '*.npz'), ('checkpoints', '*.pt')):
            assets = list((root / folder).glob(ext))
            if not assets:
                raise FileNotFoundError(f'Extract the data asset at the repository root: missing {folder}/{ext}')
            for p in assets:
                shutil.copy2(p, work / folder / p.name)
        for folder in ('results/archives_rev2', 'results/paired_honest'):
            (work / folder).mkdir(parents=True, exist_ok=True)
            for p in (root / folder).glob('*.npz'):
                shutil.copy2(p, work / folder / p.name)
    if mode != 'archived' and (root / 'data').is_dir():
        shutil.copytree(root / 'data', work / 'data', dirs_exist_ok=True)
    return work


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Runner:
    def __init__(self, work, mode='archived', device='cpu', workers=4, selected=None):
        self.work = Path(work).resolve()
        if mode not in ('archived', 'replay', 'fresh'):
            raise ValueError(mode)
        self.mode, self.jobs = mode, plan(device, workers)
        self.selected = set(selected or GROUPS)
        if self.selected - set(GROUPS):
            raise ValueError(f'Unknown groups: {self.selected - set(GROUPS)}')
        self.events = []
        (self.work / 'run_logs').mkdir(exist_ok=True)

    def save(self):
        data = {'mode': self.mode, 'python': sys.version, 'platform': platform.platform(),
                'work': str(self.work), 'jobs': [asdict(j) for j in self.jobs], 'events': self.events}
        (self.work / 'execution_manifest.json').write_text(json.dumps(data, indent=2), encoding='utf-8')

    def execute(self, name, script, args=(), outputs=(), needs=()):
        for pattern in needs:
            if not list(self.work.glob(pattern)):
                raise FileNotFoundError(f'{name} needs {pattern}; run its producer or supply the data release.')
        log = self.work / 'run_logs' / (name + '.log')
        cmd = [sys.executable, '-u', str(self.work / script), *map(str, args)]
        print(subprocess.list2cmdline(cmd), flush=True)
        start = time.time()
        before = {out: (self.work / out).stat().st_mtime_ns if (self.work / out).exists() else None for out in outputs}
        env = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
        with log.open('w', encoding='utf-8') as f:
            p = subprocess.run(cmd, cwd=self.work, env=env, stdout=f, stderr=subprocess.STDOUT)
        event = {'name': name, 'command': cmd, 'returncode': p.returncode,
                 'seconds': round(time.time() - start, 3), 'log': str(log), 'outputs': {}}
        for out in outputs:
            path = self.work / out
            if not path.is_file():
                event['missing_output'] = out
                break
            if before[out] is not None and path.stat().st_mtime_ns == before[out]:
                event['stale_output'] = out
                break
            event['outputs'][out] = sha256(path)
        self.events.append(event)
        self.save()
        lines = log.read_text(encoding='utf-8', errors='replace').splitlines()
        print('\n'.join(lines[-8:]), flush=True)
        if p.returncode or 'missing_output' in event or 'stale_output' in event:
            raise RuntimeError(f'{name} failed; see {log}')
        print(f"Completed in {event['seconds']:.1f}s; full output: {log}")

    def group(self, group):
        for j in (j for j in self.jobs if j.group == group):
            if self.mode == 'archived' or group not in self.selected or (self.mode == 'replay' and j.kind == 'train'):
                self.events.append({'name': j.name, 'status': 'archived' if self.mode != 'fresh' else 'not-selected',
                                    'command': j.command()})
                continue
            needs = j.needs
            if self.mode == 'fresh' and j.kind in ('train', 'archive', 'root'):
                needs = []
            self.execute(j.name, j.script, j.args, j.outputs, needs)
        self.save()
        print(f'{group}: mode={self.mode}; consult execution_manifest.json for run/cached status.')


def show_tables(work):
    from IPython.display import HTML, display
    sys.path.insert(0, str(Path(work) / 'analysis'))
    from export_tables import export_content, render_table
    tables, _ = export_content()
    for name, rows in tables.items():
        display(HTML(render_table(name, rows)))
