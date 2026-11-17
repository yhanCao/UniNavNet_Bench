"""Contract tests for ST-Map without detector weights or a GPU."""
import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

# ST-Map is optional: skip without its extra (pip install '.[st-map]')
nx = pytest.importorskip("networkx", reason="needs the st-map extra")
pytest.importorskip("scipy", reason="needs the st-map extra")

from st_map.config import load_config, resource_path
from st_map.graph import QueryByLLM, find_nearby_objects_2d, select_subgraph_nodes, format_nearby_objects
from st_map.memory import TemporalMemoryMixin

ROOT = Path(__file__).resolve().parents[1]


def record(name='Cup'):
    return {'class_name': [name], 'state_dict': {0: 1}, 'exists_in_SG': True,
            'caption': [{'start_time': 0, 'end_time': None, 'state': 1, 'caption': None}],
            'voxel_index': np.array([0]), 'image_idx': [0]}


def memory(objects):
    m = TemporalMemoryMixin()
    m.objects = objects
    m.target_similarity_with_objects = [(0.9, 1)] * len(objects)
    m.history_archive = []
    m.history_caption = False
    m.size, m.min_bound = (2, 2, 2), (0, 0, 0)
    m.textmodel = SimpleNamespace(query_multiple=lambda names: (0.9 if names[0] == 'Cup' else 0.1, 1))
    return m


@pytest.mark.parametrize('depth,mismatch,detected,expected', [
    (2.0, True, False, 2), (2.0, False, False, 1),
    (1.0, True, False, 1), (0.5, True, False, 0),
    (2.0, True, True, 1),
])
def test_depth_and_appearance_control_archival(depth, mismatch, detected, expected):
    obj = record()
    obj['pcd'] = SimpleNamespace(points=np.repeat([[0., 1., 0.]], 110, axis=0))
    if detected:
        obj['state_dict'][1] = 1
    m = memory([obj])
    m.screen_h = m.screen_w = 10
    m.agent_history = {1: {'angle': 0, 'roll': 0, 'pos': {'x': 0, 'y': 0, 'z': 0}}}
    m.env_camera_matrix = np.eye(4)
    m.appearanceMismatch = lambda *args: mismatch
    m.updateObjectState(1, '', None, None, np.full((10, 10), depth))
    assert obj['state_dict'][1] == expected
    assert bool(m.history_archive) == (expected == 2)
    assert bool(m.objects) == (expected != 2)
    if expected == 2:
        assert obj['caption'][-1] == {'start_time': 1, 'end_time': 1, 'state': 2, 'caption': None}
        assert obj['caption'][0]['end_time'] == 0
        assert not obj['exists_in_SG']
        assert m.target_similarity_with_objects == []


def test_archive_retrieval_filters_and_preserves_history():
    m = memory([record('Chair'), record('Cup'), record('Table')])
    originals = m.objects[:]
    for obj in originals:
        obj['state_dict'][2] = 2
        m.updateStateSegment(obj, 2)
    m.archiveObjects([0, 1], 2)
    assert m.objects == [originals[2]]
    assert len(m.target_similarity_with_objects) == 1
    context = m.retrieveHistoryContext('/nonexistent')
    assert len(context) == 1
    assert context[0]['name'] == 'Cup'
    assert context[0]['archived_time'] == 2
    assert 'appears in view' in context[0]['captions'][0]
    assert 'disappears from its recorded location' in context[0]['captions'][1]


def test_histogram_match_and_mismatch(tmp_path):
    cv2 = pytest.importorskip('cv2')
    m = memory([])
    m.appearance_sim_thresh = 0.5
    red = np.zeros((12, 12, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    blue = red[:, :, ::-1].copy()
    reference = tmp_path / 'red.png'
    cv2.imwrite(str(reference), cv2.cvtColor(red, cv2.COLOR_RGB2BGR))
    obj = {'color_path': [str(reference)], 'mask': [np.ones((12, 12), np.uint8)]}
    uv = (np.arange(12), np.arange(12))
    assert not m.appearanceMismatch(obj, uv, red)
    assert m.appearanceMismatch(obj, uv, blue)


def graph():
    g = nx.Graph()
    for i, pos in enumerate([(0, 0), (1, 0), (2, 0), (20, 0)]):
        g.add_node(i, pos=pos)
    g.add_edges_from([(0, 1), (1, 2), (2, 3)])
    return g


def test_subgraph_keeps_relevance_and_nearest_frontier():
    g = graph()
    keep = select_subgraph_nodes(g, 0, [2, 3], {1: [0], 3: [1]}, [.1, .9], 1, 1)
    assert keep == {2, 3}
    assert select_subgraph_nodes(g, 0, [], None, [], 1, 0) == {1}
    prompt, candidates = QueryByLLM('Find Cup', 0, g, [0], [2, 3], None,
                                    [{'name': 'Cup', 'center': np.array([1, 0]),
                                      'archived_time': 5, 'captions': ['Cup disappeared.']}], 6, keep)
    assert set(candidates) == keep
    assert 'Node 0 :' in prompt
    assert 'Node 1 :' not in prompt
    assert 'Cup disappeared.' in prompt
    assert 'current timestep 6' in prompt


def test_each_nearby_object_is_listed_and_last_seen_is_in_prompt():
    objects = [record('Cup'), record('Book')]
    idxs = find_nearby_objects_2d({0: np.array([.125, .125])}, objects, .5,
                                 (2, 2, 2), (0, 0, 0), .25, return_index=True)
    assert set(idxs[0]) == {0, 1}
    names = format_nearby_objects(idxs, objects, [.9, .1], 2)
    assert names[0] == ['Cup (last seen at timestep 0)', 'Book']


def test_configuration_and_packaged_paths(tmp_path, monkeypatch):
    monkeypatch.delenv('ST_MAP_CONFIG', raising=False)
    monkeypatch.delenv('ST_MAP_OPENSEED_CHECKPOINT', raising=False)
    path = tmp_path / 'config.yaml'
    path.write_text('openseed_checkpoint: model.pt\nmemory:\n  subgraph_nodes: 2\n')
    cfg = load_config(path)
    assert cfg['openseed_checkpoint'] == str(tmp_path / 'model.pt')
    assert cfg['memory']['subgraph_frontiers'] == 4
    monkeypatch.setenv('ST_MAP_LLM_MODEL', 'local-text')
    assert load_config(path)['llm']['model'] == 'local-text'
    assert load_config()['memory']['subgraph_nodes'] == 8
    assert Path(resource_path('configs/panoptic_categories_nomerge.txt')).is_file()
    with pytest.raises(ValueError):
        load_config({'memory': {'subgraph_nodes': -1}})


def test_clean_library_import_has_no_models():
    command = 'import st_map; from st_map import STMap; import sys; assert "torch" not in sys.modules; assert "ai2thor" not in sys.modules'
    subprocess.run([sys.executable, '-c', command], check=True, cwd=ROOT)


class Backend:
    def __init__(self):
        self.sem_map_module = SimpleNamespace(history_archive=['remembered'])
        self.calls = []

    def init(self, positions, path, instruction):
        self.calls.append(('init', instruction))

    def reset(self, positions, path, instruction):
        self.calls.append(('reset', instruction))
        return ['LookUp']

    def update(self, positions):
        self.calls.append(('update', positions))

    def act(self, obs):
        return 'MoveAhead', None


def test_agent_lifecycle_persists_memory_across_repeated_tasks(tmp_path):
    from st_map.agent import STMapAgent
    backend = Backend()
    controller = SimpleNamespace(initialization_parameters={})
    with patch.object(STMapAgent, '_make_backend', return_value=backend):
        agent = STMapAgent()
        agent.bind_environment(controller, tmp_path)
    obs = {'event': SimpleNamespace(depth_frame=np.ones((1, 1)))}
    with pytest.raises(RuntimeError, match='start_task'):
        agent.act(obs)
    agent.start_task('Find Cup', [])
    assert agent.act(obs) == ('MoveAhead', None)
    agent.start_task('Find Cup', [])
    assert agent.memory.history_archive == ['remembered']
    assert backend.calls == [('init', 'Find Cup'), ('reset', 'Find Cup')]
    assert agent.act(obs) == ('LookUp', None)
    assert agent.act(obs) == ('MoveAhead', None)
    agent.update_environment([{'x': 1, 'y': 0, 'z': 0}])
    assert backend.calls[-1][0] == 'update'
    assert backend.actions is None and backend.replan
    assert agent.memory.history_archive == ['remembered']
    agent.reset()
    assert agent.memory is None
    assert not agent._task_started


def test_all_agent_entrypoints_agree_without_loading_models():
    from uninavbench.cli import load_agent
    for path in ['st_map', 'uninavbench.agents.st_map', 'examples/agents/st_map.py']:
        cls, kwargs, depth = load_agent(path)
        assert cls.__name__ == 'STMapAgent'
        assert depth is True
        assert cls(**kwargs).backend is None
    assert 'torch' not in sys.modules


def test_missing_checkpoint_explained(tmp_path):
    from st_map.config import validate_runtime
    with pytest.raises(RuntimeError, match='ST_MAP_OPENSEED_CHECKPOINT'):
        validate_runtime(load_config({'openseed_checkpoint': str(tmp_path / 'missing.pt')}))


def test_local_imports_resolve_and_no_embedded_credentials():
    for path in (ROOT / 'st_map').rglob('*.py'):
        source = path.read_text()
        assert 'sk-' not in source
        assert 'sys.path.insert' not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and not node.level and node.module and node.module.startswith('st_map._vendor'):
                base = ROOT.joinpath(*node.module.split('.'))
                assert base.with_suffix('.py').is_file() or (base / '__init__.py').is_file(), (path, node.module)


@pytest.mark.parametrize('use_hooks', [False, True])
@pytest.mark.parametrize('dynamic', [False, True])
def test_benchmark_calls_optional_hooks_per_episode_and_task(tmp_path, use_hooks, dynamic):
    import queue
    import uninavbench.challenge as challenge
    calls = []
    event = SimpleNamespace(metadata={
        'agent': {'position': {'x': 0, 'y': 0, 'z': 0},
                  'rotation': {'x': 0, 'y': 0, 'z': 0}, 'cameraHorizon': 0},
        'actionReturn': [{'x': 0, 'y': 0, 'z': 0}], 'objects': [],
        'lastActionSuccess': True,
    })

    class Controller:
        def __init__(self, **kwargs):
            self.last_event = event
        def step(self, *args, **kwargs):
            return self.last_event
        def stop(self):
            calls.append('stop-controller')

    class LegacyAgent:
        def act(self, obs):
            calls.append(('act', obs['instruction']))
            return ('MoveAhead' if dynamic and obs['step'] == 0 else 'Stop'), None

    class HookAgent(LegacyAgent):
        def bind_environment(self, controller, output_dir):
            calls.append('bind')
        def start_task(self, instruction, reachable_positions):
            assert reachable_positions == event.metadata['actionReturn']
            calls.append(('start', instruction))
        def update_environment(self, reachable_positions):
            assert reachable_positions == event.metadata['actionReturn']
            calls.append('update')

    bench = object.__new__(challenge.ProcthorChallenge)
    bench.config = {}
    bench.dataset = {'test': [{}]}
    incoming, outgoing = queue.Queue(), queue.Queue()
    incoming.put({'scene_type': 'test', 'scene_id': 0, 'level': 'static',
                  'instruction': ['Find Cup', 'Find Cup'],
                  'initial_position': {'x': 0, 'y': 0, 'z': 0},
                  'initial_orientation': 0, 'initial_horizon': 0})
    changes = []
    def apply_changes(controller, events, next_event, scene_time):
        if dynamic and scene_time == 1 and not changes:
            changes.append(True)
            return 1, {'moved-object'}
        return next_event, set()

    with patch.object(challenge, 'Controller', Controller), patch.object(challenge, 'get_target_answer', return_value=([], None)), patch.object(bench, 'apply_dynamic_events', side_effect=apply_changes):
        bench.inference_worker(0, incoming, outgoing, HookAgent if use_hooks else LegacyAgent,
                               {}, {}, 1, str(tmp_path), True, 0)
    if use_hooks:
        expected = ['bind', ('start', 'Find Cup'), ('act', 'Find Cup')]
        if dynamic:
            expected += ['update']
        expected += [('start', 'Find Cup'), ('act', 'Find Cup'), 'stop-controller']
        assert calls == expected
    else:
        assert calls == [('act', 'Find Cup'), ('act', 'Find Cup'), 'stop-controller']
    assert outgoing.qsize() == 1


def test_memory_stays_bounded_over_a_long_episode(tmp_path):
    (tmp_path / 'image').mkdir()
    m = TemporalMemoryMixin()
    m.objects, m.history_archive, m.agent_history, m.saved_frames = [record()], [], {}, set()
    m.target_similarity_with_objects = []
    obj = m.objects[0]
    obj.update(pixel_area=[1], mask=[None], xyxy=[(0, 0, 1, 1)], color_path=['0.jpg'], class_id=[0])
    rng = np.random.default_rng(0)
    for step in range(1, 1000):
        (tmp_path / 'image' / f'{step}.jpg').write_bytes(b'x')
        m.saved_frames.add(step)
        m.agent_history[step] = {}
        for key, value in (('image_idx', step), ('pixel_area', int(rng.integers(1, 1000))), ('mask', None),
                           ('xyxy', (0, 0, 1, 1)), ('color_path', f'{step}.jpg'), ('class_name', 'Cup'), ('class_id', 0)):
            obj[key].append(value)
        obj['state_dict'][step] = step % 2
        m.updateStateSegment(obj, step)
        if step % 50 == 0:  # an object leaves the active graph now and then
            m.history_archive.append({**record('Book'), 'pcd': object(), 'archived_time': step, 'voxel_index': np.array([0])})
        m.trimMemory(step, str(tmp_path))
    assert len(obj['image_idx']) == len(obj['xyxy']) == len(obj['pixel_area']) == 3
    assert obj['image_idx'][-1] == 999 and len(obj['class_name']) <= 20
    assert len(obj['state_dict']) <= 2 and obj['state_dict'][999] == 1
    assert len(obj['caption']) <= 8 and len(m.history_archive) <= 32
    assert all('pcd' not in a for a in m.history_archive)
    assert len(list((tmp_path / 'image').iterdir())) <= 4 and len(m.agent_history) <= 4


def test_verifier_images_are_cropped_and_small(tmp_path):
    import base64, io
    from PIL import Image
    from st_map._vendor.vlm.client import image_url
    Image.new('RGB', (640, 480)).save(tmp_path / 'f.jpg')
    def size(**kw):
        url = image_url(str(tmp_path / 'f.jpg'), **kw)['image_url']['url']
        return Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).size
    assert max(size()) == 320
    assert size(box=(100, 100, 160, 140)) == (128, 96)  # 60x40 box + (30% + 16 px) margin each side, not scaled up


def test_planner_prompt_is_one_line_per_node():
    g = graph()
    prompt, candidates = QueryByLLM('Find Cup', 0, g, [0], [3], {1: ['Sofa', 'Cup (last seen at timestep 3)']})
    lines = prompt.splitlines()
    assert lines[2] == 'Node 0 : explored'
    assert lines[3].startswith('Node 1 : unexplored, 1.0m') and lines[3].endswith('| Sofa, Cup (last seen at timestep 3)')
    assert lines[5].startswith('Node 3 : frontier, unexplored, 20.0m')
    assert lines[-1] == 'Candidates: [1, 2, 3]' and 'Path' not in prompt
