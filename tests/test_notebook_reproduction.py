from pathlib import Path
import json
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
sys.path.insert(0, str(ROOT / 'experiments'))
from reproduction_workflow import Runner, plan, prepare_workdir
from merge_adaptive_curves import merge_curves


def test_replay_refuses_to_train_when_archive_missing(tmp_path, monkeypatch):
    import subprocess
    def no_launch(*args, **kwargs):
        pytest.fail('Missing archive must fail before subprocess launch')
    monkeypatch.setattr(subprocess, 'run', no_launch)
    runner = Runner(tmp_path, mode='replay')
    with pytest.raises(FileNotFoundError, match='needs results/archives'):
        runner.group('archives')


def test_adaptive_merge_reproduces_released_cells():
    result = merge_curves([ROOT / f'results/adaptive_curve_{s}.json' for s in 'ABCD'])
    published = json.loads((ROOT / 'results/adaptive_costcurve.json').read_text())
    for key in ('cells', 'thresholds', 'honest_fpr_TEST'):
        assert json.dumps(result[key], sort_keys=True) == json.dumps(published[key], sort_keys=True)


@pytest.mark.parametrize('inconsistency', ['duplicate', 'threshold'])
def test_adaptive_merge_rejects_inconsistent_shards(tmp_path, inconsistency):
    a = {'cells': {'a': 1}, 'thresholds': {'s': 3.786}, 'honest_fpr_TEST': {}}
    b = dict(a, cells={'b': 2})
    if inconsistency == 'duplicate':
        b['cells'] = a['cells']
    else:
        b['thresholds'] = {'s': 9}
    paths = [tmp_path / 'a.json', tmp_path / 'b.json']
    for path, data in zip(paths, (a, b)):
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        merge_curves(paths)


def test_notebook_recipes_preserve_distinct_adaptive_populations():
    recipes = {j.name: j for j in plan()}
    def value(job, flag):
        args = recipes[job].args
        return args[args.index(flag) + 1]
    assert value('adaptive-curve-A', '--victims') == '0,2,4,6,8'
    assert value('fused-grid', '--victims') == '0,1'
    assert value('fused-grid', '--basis-steps') == 40
    assert value('fused-grid', '--restarts') == 1
    assert '--no-warm' in recipes['fused-grid'].args
    assert recipes['beacon-grid'].script == 'experiments/unpredictable_eval.py'
    assert recipes['calibration'].outputs == ['results/protocol_eval_d05.json']


def test_success_exit_does_not_mislabel_stale_result_as_recomputed(tmp_path):
    (tmp_path / 'noop.py').write_text('pass\n')
    (tmp_path / 'old.json').write_text('{"old": true}\n')
    runner = Runner(tmp_path)
    with pytest.raises(RuntimeError, match='failed'):
        runner.execute('no-output-written', 'noop.py', outputs=['old.json'])
    assert runner.events[-1]['stale_output'] == 'old.json'


def test_archive_jobs_list_only_outputs_a_replay_rewrites():
    for job in plan():
        if job.kind == 'archive':
            assert all(Path(out).name.startswith('innovation_') and 'innovation_run_' not in out
                       for out in job.outputs), job.name


def test_replay_workdir_carries_the_revision_archives(tmp_path, monkeypatch):
    import subprocess
    import reproduction_workflow as rw
    monkeypatch.setattr(rw, 'CODE_DIRS', ())
    def no_launch(*args, **kwargs):
        pytest.fail('A missing revision archive must fail before subprocess launch')
    monkeypatch.setattr(subprocess, 'run', no_launch)
    root = tmp_path / 'root'
    for rel in ('results/temporal_eval.json', 'results/archives/a.npz', 'checkpoints/c.pt'):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text('x')
    job = {j.name: j for j in plan()}['hetero_s0']
    bare = prepare_workdir(root, 'replay', tmp_path / 'w1')
    with pytest.raises(FileNotFoundError, match='needs results/archives_rev2'):
        Runner(bare, mode='replay').execute(job.name, job.script, job.args, job.outputs, job.needs)
    rev2 = root / 'results/archives_rev2/arch_hetero_s0_c10_r35_d0.5_s0.npz'
    rev2.parent.mkdir(parents=True)
    rev2.write_text('x')
    work = prepare_workdir(root, 'replay', tmp_path / 'w2')
    assert (work / 'results/archives_rev2' / rev2.name).is_file()
    assert all(list(work.glob(p)) for p in job.needs)


def test_prepare_workdir_copies_code_and_results_without_touching_the_repository(tmp_path):
    work = prepare_workdir(ROOT, 'archived', tmp_path / 'run')
    assert (work / 'passportfl/innovation.py').is_file()
    assert (work / 'results/temporal_eval.json').is_file()
    assert (work / 'figures/fig_adaptive_summary.values.json').is_file()
    assert (work / 'run_logs').is_dir() is False
    with pytest.raises(RuntimeError, match='empty'):
        prepare_workdir(ROOT, 'archived', tmp_path / 'run')
