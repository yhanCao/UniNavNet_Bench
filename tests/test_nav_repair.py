import importlib.util
from pathlib import Path
from copy import deepcopy
from collections import Counter
import pytest

spec = importlib.util.spec_from_file_location('repair_nav', Path(__file__).resolve().parents[1] / 'scripts/repair_nav_completeness.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(level='static'):
    types = [1]*4 + [2] + [3]*4 + [4]*4 + [5]*4
    episodes = []
    for offset in [0, 10]:
        part = types[offset:offset+10]
        n = len(part)
        e = {'instruction': ['Find the Chair in the LivingRoom' if t == 2 else f'task {offset+i}' for i, t in enumerate(part)],
             'id': ['Chair|0' if t == 2 else None for t in part],
             'task_type': part, 'image_name': [None]*n, 'object_type': [None]*n,
             'scene_type': 'train', 'scene_id': 220, 'level': level,
             'initial_position': {'x': offset, 'y': .9, 'z': 0}, 'initial_orientation': 0, 'initial_horizon': 0}
        if level != 'static':
            e['time'] = 500
        episodes.append(e)
    cache = {'types': {'Chair|0': 'Chair'}, 'timeline': {'Chair|0': [[0, True], [100, True]]}}
    return episodes, cache


@pytest.mark.parametrize('level', ['static', 'small', 'all'])
def test_repair_preserves_prefix_and_balances_counts(level):
    episodes, cache = fixture(level)
    original = deepcopy(episodes)
    repaired, report = module.repair(episodes, cache, {}, 220, level, expected_episodes=2)
    assert episodes == original
    assert report['added_subtasks'] == 3
    assert all(len(e['instruction']) == 10 for e in repaired)
    assert Counter(t for e in repaired for t in e['task_type']) == Counter({t: 4 for t in range(1, 6)})
    assert repaired[0] == original[0]
    assert repaired[1]['instruction'][:7] == original[1]['instruction']
    assert repaired[1]['initial_position'] == original[1]['initial_position']
    second, report = module.repair(repaired, cache, {}, 220, level, expected_episodes=2)
    assert second == repaired
    assert report['added_subtasks'] == 0


def test_reject_invisible_or_moved_relation_targets():
    episodes, cache = fixture()
    cache['timeline']['Chair|0'].append([200, False])
    with pytest.raises(ValueError, match='No certified'):
        module.repair(episodes, cache, {}, 220, 'static', expected_episodes=2)
    cache['timeline']['Chair|0'] = [[0, True]]
    schedule = {'0': [{'objectId': 'Chair|0', 'action': 'PlaceObjectAtPoint', 'time': 200}]}
    with pytest.raises(ValueError, match='No certified'):
        module.repair(episodes, cache, schedule, 220, 'static', expected_episodes=2)


def test_relation_reference_class_must_stay_unchanged():
    ep = {'task_type': [2], 'image_name': [None], 'id': ['Cup|0'],
          'instruction': ['Find the Cup which is on the SideTable']}
    cache = {'types': {'Cup|0': 'Cup'}, 'timeline': {'Cup|0': [[0, True]]}}
    assert module.relation_is_safe(ep, 0, cache, set())
    assert not module.relation_is_safe(ep, 0, cache, {'SideTable|1'})


def test_refuse_other_missing_task_types():
    episodes, cache = fixture()
    episodes[0]['task_type'][0] = 2
    with pytest.raises(ValueError, match='shortfall confined'):
        module.repair(episodes, cache, {}, 220, 'static', expected_episodes=2)
