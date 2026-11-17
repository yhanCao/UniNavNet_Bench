"""Regression tests for replay errors, exit codes and reproducible versions."""
import importlib.util
import json
from pathlib import Path
import queue
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import yaml

import uninavbench.challenge as challenge
from uninavbench.challenge import (ProcthorChallenge, DynamicReplayError,
                                  PROCTHOR_REVISION, AI2THOR_COMMIT_ID, DEFAULT_CONFIG)
from uninavbench.cli import batch


class Simulator:
    def __init__(self, fail_at=None, raises=False):
        self.calls = []
        self.fail_at = fail_at
        self.raises = raises

    def step(self, **kwargs):
        self.calls.append(kwargs)
        failed = len(self.calls) == self.fail_at
        if failed and self.raises:
            raise OSError('simulator disconnected')
        return SimpleNamespace(metadata={'lastActionSuccess': not failed, 'errorMessage': 'object not found' if failed else ''})


def scheduled(kind):
    event = {'action': kind, 'objectId': 'Cup|1', 'time': 20}
    if kind == 'PlaceObjectAtPoint':
        event.update(target_position={'x': 1, 'y': 0, 'z': 2}, target_rotation={'x': 0, 'y': 0, 'z': 0})
    if kind == 'InitialRandomSpawn':
        event['rectangleobj'] = 'CounterTop|1'
    if kind == 'resolved':
        event['action'] = 'InitialRandomSpawn'
        event['resolved'] = [{'objectId': oid, 'position': {'x': 1, 'y': 0, 'z': 2},
                              'rotation': {'x': 0, 'y': 0, 'z': 0}} for oid in ['Cup|1', 'Plate|2']]
    return [(20, 7, [event])]


@pytest.mark.parametrize('kind', ['ToggleObjectOn', 'PlaceObjectAtPoint', 'InitialRandomSpawn', 'resolved'])
@pytest.mark.parametrize('raises', [False, True])
def test_replay_failure_aborts_with_context(kind, raises):
    sim = Simulator(fail_at=2 if kind == 'resolved' else 1, raises=raises)
    with pytest.raises(DynamicReplayError, match=r'time 20 \(group 7\)') as error:
        ProcthorChallenge.apply_dynamic_events(sim, scheduled(kind), 0, 20)
    assert ('Plate|2' if kind == 'resolved' else 'Cup|1') in str(error.value)
    assert ('simulator disconnected' if raises else 'object not found') in str(error.value)


@pytest.mark.parametrize('kind', ['ToggleObjectOn', 'PlaceObjectAtPoint', 'InitialRandomSpawn', 'resolved'])
def test_successful_replay_only_advances_due_events(kind):
    sim = Simulator()
    assert ProcthorChallenge.apply_dynamic_events(sim, scheduled(kind), 0, 19) == (0, set())
    assert not sim.calls
    index, moved = ProcthorChallenge.apply_dynamic_events(sim, scheduled(kind), 0, 20)
    assert index == 1
    assert moved == ({'Cup|1', 'Plate|2'} if kind == 'resolved' else {'Cup|1'})
    calls = len(sim.calls)
    assert ProcthorChallenge.apply_dynamic_events(sim, scheduled(kind), index, 40) == (1, set())
    assert len(sim.calls) == calls


def test_spawn_is_seeded_and_unknown_actions_are_rejected():
    a, b = Simulator(), Simulator()
    for sim in [a, b]:
        ProcthorChallenge.apply_dynamic_events(sim, scheduled('InitialRandomSpawn'), 0, 20)
    assert a.calls == b.calls and isinstance(a.calls[0]['randomSeed'], int)
    with pytest.raises(DynamicReplayError, match='Unsupported'):
        ProcthorChallenge.apply_dynamic_events(a, scheduled('TypoAction'), 0, 20)


def test_recorded_pose_snapshots_are_stable_without_disabling_global_physics():
    sim = Simulator()
    ProcthorChallenge.apply_dynamic_events(sim, scheduled('resolved'), 0, 20)
    assert len(sim.calls) == 2
    assert all(call['action'] == 'TeleportObject' and call['forceKinematic'] is True
               for call in sim.calls)
    # Fresh placements retain the simulator's physical placement behavior.
    sim = Simulator()
    ProcthorChallenge.apply_dynamic_events(sim, scheduled('PlaceObjectAtPoint'), 0, 20)
    assert 'forceKinematic' not in sim.calls[0]


@pytest.mark.parametrize('success', [False, 'False', None])
def test_replay_requires_boolean_success(success):
    sim = SimpleNamespace(step=lambda **kw: SimpleNamespace(metadata={'lastActionSuccess': success}))
    with pytest.raises(DynamicReplayError):
        ProcthorChallenge.apply_dynamic_events(sim, scheduled('ToggleObjectOn'), 0, 20)


def test_controller_stops_when_agent_initialization_fails(tmp_path):
    bench = object.__new__(ProcthorChallenge)
    bench.config, bench.dataset = {}, {'test': [{}]}
    incoming = queue.Queue()
    incoming.put({'scene_type': 'test', 'scene_id': 0})
    controller = SimpleNamespace(stop=Mock())
    def broken_agent():
        raise RuntimeError('agent initialization failed')
    with patch.object(challenge, 'Controller', return_value=controller):
        with pytest.raises(RuntimeError, match='agent initialization failed'):
            bench.inference_worker(0, incoming, queue.Queue(), broken_agent, {}, {}, 1, str(tmp_path), True, 0)
    controller.stop.assert_called_once()


def batch_args(tmp_path):
    return SimpleNamespace(scenes=['0', '1'], data_split='test', out=str(tmp_path/'out'),
                           logs=str(tmp_path/'logs'), levels=['static'], tasks=['nav'],
                           agent='unused', cfg='unused', data_pth=str(tmp_path),
                           start_episode_id=0, end_episode_id=1, result_path=None, nprocesses=1)


def test_batch_failed_worker_makes_whole_command_fail(tmp_path):
    args = batch_args(tmp_path)
    def child(cmd, **kwargs):
        if cmd[cmd.index('-i')+1] == '0':
            return 2
        Path(cmd[cmd.index('-o')+1]).write_bytes(b'metrics fixture')
        return 0
    with patch('uninavbench.cli.subprocess.call', side_effect=child) as run:
        assert batch(args) == 1
        assert run.call_count == 2
    assert (Path(args.out)/'test_1_static_nav.json.gz').exists()


def test_batch_success_and_skipped_work_exit_zero(tmp_path):
    args = batch_args(tmp_path)
    Path(args.out).mkdir()
    for scene in args.scenes:
        (Path(args.out)/f'test_{scene}_static_nav.json.gz').write_bytes(b'existing metrics')
    with patch('uninavbench.cli.subprocess.call') as run:
        assert batch(args) == 0
        run.assert_not_called()


@pytest.mark.parametrize('module', ['uninavbench', 'uninavbench.cli'])
def test_actual_module_process_propagates_batch_exit(tmp_path, module):
    script = '''import sys,runpy
from unittest.mock import patch
module = sys.argv[3]
sys.argv = ['uninavbench','batch','--scenes','0','--levels','static','--tasks','nav','--out',sys.argv[1], '--logs',sys.argv[2]]
with patch('subprocess.call', return_value=1):
    runpy.run_module(module, run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path/'out'), str(tmp_path/'logs'), module], capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert 'FAILED' in result.stdout


def test_both_version_pins_are_forwarded_including_offline_fallback(tmp_path):
    (tmp_path/'attributes').mkdir()
    dataset = {'test': [{}]}
    with patch.object(ProcthorChallenge, 'setup_env'), patch.object(challenge.prior, 'load_dataset', side_effect=[OSError('offline'), dataset]) as load:
        bench = ProcthorChallenge(data_pth=str(tmp_path), judge=object())
    assert load.call_args_list[0].kwargs == {'revision': PROCTHOR_REVISION}
    assert load.call_args_list[1].kwargs == {'revision': PROCTHOR_REVISION, 'offline': True}
    assert bench.controller_kwargs['commit_id'] == AI2THOR_COMMIT_ID
    assert bench.environment['procthor_revision'] == PROCTHOR_REVISION
    with patch.object(bench, 'inference_worker'):
        assert bench.inference([], test=True)['environment'] == bench.environment


def config_file(tmp_path, **overrides):
    cfg = yaml.safe_load(Path(DEFAULT_CONFIG).read_text())
    cfg.update(overrides)
    p = tmp_path/'config.yaml'
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.mark.parametrize('key,value', [('procthor_revision', 'main'), ('ai2thor_commit_id', None),
                                       ('thor_build_id', 'ab3cacd0fc17754d4c080a3fd50b18395fae8647')])
def test_ambiguous_or_mutable_version_configuration_fails(tmp_path, key, value):
    with pytest.raises(ValueError, match=key):
        ProcthorChallenge.load_config(config_file(tmp_path, **{key: value}), True)


def test_custom_pins_are_respected_and_cannot_be_shadowed(tmp_path):
    p = config_file(tmp_path, procthor_revision='1'*40, ai2thor_commit_id='2'*40)
    cfg = ProcthorChallenge.load_config(p, False)
    assert cfg['procthor_revision'] == '1'*40 and cfg['ai2thor_commit_id'] == '2'*40
    cfg['initialize']['branch'] = 'main'
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='bypasses'):
        ProcthorChallenge.load_config(p, False)


@pytest.fixture
def verifier():
    # The existing simulator driver installs oracle hooks on import. Restore them
    # immediately so importing it for release-gate tests cannot affect other tests.
    controller, target = challenge.Controller, challenge.get_target_answer
    apply = ProcthorChallenge.__dict__['apply_dynamic_events']
    path = Path(__file__).with_name('verify_dynamic.py')
    spec = importlib.util.spec_from_file_location('dynamic_verifier', path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        challenge.Controller, challenge.get_target_answer = controller, target
        ProcthorChallenge.apply_dynamic_events = apply
    return module


def validation_row(**overrides):
    row = dict(level='small', task='nav', agent='oracle', subtasks=10, success=1., n_event_failures=0)
    row.update(overrides)
    return row


@pytest.mark.parametrize('overrides', [{'n_event_failures': 1}, {'target_never_visible': 1},
                                        {'success': .99999}, {'subtasks': 0}])
def test_verifier_rejects_invalid_positive_runs(verifier, overrides):
    assert verifier.validation_errors({'runs': [validation_row(**overrides)]})


@pytest.mark.parametrize('agent', ['stop', 'random', 'stale_oracle'])
def test_negative_control_task_failures_are_expected(verifier, agent):
    assert verifier.validation_errors({'runs': [validation_row(agent=agent, success=0.)]}) == []


def test_verifier_execution_failure_writes_report_and_returns_one(verifier, tmp_path, monkeypatch):
    def broken(*args):
        raise DynamicReplayError('scheduled replay failed')
    monkeypatch.setattr(verifier, 'evaluate', broken)
    out = tmp_path/'report.json'
    code = verifier.main([str(tmp_path), '--out', str(out), '--levels', 'small', '--tasks', 'nav', '--agents', 'oracle'])
    report = json.loads(out.read_text())
    assert code == 1 and not report['validation']['passed']
    assert 'scheduled replay failed' in report['execution_error']


def test_verifier_success_writes_report_and_returns_zero(verifier, tmp_path, monkeypatch):
    def evaluate(data, cls, kw, split, scene, level, task, episodes, steps, out_dir, judge):
        p = Path(out_dir)/task/split/scene/level
        p.mkdir(parents=True)
        (p/'episode_results_1.json').write_text(json.dumps({'0': {'stopped': True, 'success': 1, 'task_type': 1}}))
        return {'num_subtasks': 1, 'success': 1., 'spl': 1., 'per_task_type': {'1': {'success': 1.}},
                'environment': {'procthor_revision': PROCTHOR_REVISION, 'ai2thor_commit_id': AI2THOR_COMMIT_ID}}
    monkeypatch.setattr(verifier, 'evaluate', evaluate)
    out = tmp_path/'report.json'
    code = verifier.main([str(tmp_path), '--out', str(out), '--result_dir', str(tmp_path/'results'),
                          '--levels', 'static', '--tasks', 'nav', '--agents', 'oracle', '--episodes', '1'])
    assert code == 0 and json.loads(out.read_text())['validation']['passed']


def test_memory_captioning_target_need_not_be_visible(verifier):
    logs = [dict(task_type=11, goal_poses=0), dict(task_type=7, goal_poses=3), dict(task_type=7, goal_poses=0),
            dict(task_type=9, planned=False)]
    assert verifier.count_never_visible(logs) == 1


class GridWorld:
    """Floor cells on a 0.25 m grid; Teleport fails onto a blocked cell, reachability is a flood fill."""

    def __init__(self, cells, blocked, at):
        self.cells, self.blocked, self.at = set(cells), set(blocked), at
        self.last_event = self.event(True)

    def event(self, success, value=None):
        x, z = self.at
        return SimpleNamespace(metadata={'lastActionSuccess': success, 'actionReturn': value, 'agent': {
            'position': {'x': x * .25, 'y': .9, 'z': z * .25}, 'rotation': {'x': 0, 'y': 90, 'z': 0}, 'cameraHorizon': 0}})

    def reachable(self):
        if self.at in self.blocked:
            return []
        seen, todo = {self.at}, [self.at]
        while todo:
            x, z = todo.pop()
            for n in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)):
                if n in self.cells and n not in self.blocked and n not in seen:
                    seen.add(n)
                    todo.append(n)
        return [{'x': x * .25, 'y': .9, 'z': z * .25} for x, z in seen]

    def step(self, action, position=None, forceAction=False, **kwargs):
        if action == 'Teleport':
            cell = (round(position['x'] / .25), round(position['z'] / .25))
            ok = forceAction or (cell in self.cells and cell not in self.blocked)
            if ok:
                self.at = cell
            self.last_event = self.event(ok)
        elif action == 'GetReachablePositions':
            self.last_event = self.event(True, self.reachable())
        return self.last_event


ROOM = [(x, z) for x in range(10) for z in range(4)]


def floor_of(world):
    return world.step('GetReachablePositions').metadata['actionReturn']


def test_agent_under_a_placed_object_is_moved_to_the_nearest_free_cell():
    world = GridWorld(ROOM, [], at=(5, 2))
    floor = floor_of(world)
    world.blocked = {(5, 2)}
    moved = challenge.free_agent(world, floor)
    assert moved['from'] == {'x': 1.25, 'y': .9, 'z': .5}
    assert abs(moved['to']['x'] - 1.25) + abs(moved['to']['z'] - .5) == .25


def test_agent_walled_into_a_pocket_is_moved_out():
    world = GridWorld(ROOM, [], at=(0, 0))
    floor = floor_of(world)
    world.blocked = {(1, 0), (0, 1)}  # (0, 0) is free but cut off
    moved = challenge.free_agent(world, floor)
    assert moved is not None and world.at not in {(0, 0)} and len(world.reachable()) == len(floor) - 3


def test_drifted_camera_horizon_is_rounded_for_teleport():
    world = GridWorld(ROOM, [], at=(5, 2))
    floor = floor_of(world)
    world.last_event.metadata['agent']['cameraHorizon'] = 60.00002
    calls = []
    step = world.step
    world.step = lambda action, **kw: (calls.append(kw), step(action, **kw))[1]
    assert challenge.free_agent(world, floor) is None
    assert calls[0]['horizon'] == 60


def test_free_agent_is_left_in_place():
    world = GridWorld(ROOM, [], at=(5, 2))
    floor = floor_of(world)
    world.blocked = {(9, 3)}
    assert challenge.free_agent(world, floor) is None and world.at == (5, 2)


def test_only_relocations_can_trap_the_agent():
    events = [(1, 0, [{'action': 'OpenObject'}]), (2, 1, [{'action': 'PlaceObjectAtPoint'}])]
    assert not challenge.relocates(events, 0, 1)
    assert challenge.relocates(events, 0, 2)

