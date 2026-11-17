"""UniNavBench lifecycle adapter for the pinned ST-CogNav implementation."""
import json
from pathlib import Path
from collections import deque
from uninavbench.agent import Agent
from .config import load_config, validate_runtime


class STMapAgent(Agent):
    """Navigation baseline with persistent spatial and temporal memory per episode.

    Bind a simulator once, then call start_task for every subtask. Direct users can
    pass ``controller`` to the constructor and use the same lifecycle methods.
    """

    def __init__(self, config=None, controller=None):
        self.config = load_config(config)
        self.backend = None
        self._pending_actions = deque()
        self._task_started = False
        self._instruction, self._task_steps = None, 0
        if controller is not None:
            self.bind_environment(controller, self.config['output_dir'])

    def _make_backend(self, controller):
        validate_runtime(self.config)
        try:
            from ._vendor.agents.random_agent import SimpleRandomAgent
        except ImportError as exc:
            raise RuntimeError('Missing ST-Map model dependency: ' + str(exc) +
                               '. Install the GPU environment described in docs/st-map.md.') from exc
        return SimpleRandomAgent(controller, memory_cfg=self.config['memory'], runtime_cfg=self.config)

    def bind_environment(self, controller, output_dir=None):
        if self.backend is not None:
            raise RuntimeError('Use a new STMapAgent for each simulator episode')
        # The upstream projection and detector assume 640x480 images and 0.25 m grid.
        params = controller.initialization_parameters
        if params.get('width', 640) != 640 or params.get('height', 480) != 480:
            raise ValueError('ST-Map requires simulator width=640 and height=480')
        if params.get('gridSize', 0.25) != 0.25 or params.get('rotateStepDegrees', 90) != 90:
            raise ValueError('ST-Map requires gridSize=0.25 and rotateStepDegrees=90')
        self.output_dir = Path(output_dir or self.config['output_dir']).resolve() / 'ST-Map'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = self._make_backend(controller)

    def start_task(self, instruction, reachable_positions):
        if self.backend is None:
            raise RuntimeError('Call bind_environment(controller) before start_task')
        # Upstream uses string concatenation for some artifact directories.
        path = str(self.output_dir) + '/'
        self._instruction, self._task_steps = instruction, 0
        if not self._task_started:
            self.backend.init(reachable_positions, path, instruction)
            self._task_started = True
        else:
            corrections = self.backend.reset(reachable_positions, path, instruction)
            self._pending_actions = deque(corrections or [])

    def reset(self):
        """Clear episode memory explicitly; task changes use start_task instead."""
        self.backend = None
        self._task_started = False
        self._pending_actions.clear()

    def update_environment(self, reachable_positions):
        """Refresh planning geometry after scheduled object/furniture changes."""
        if self._task_started:
            self.backend.update(reachable_positions)
            self.backend.actions = None
            self.backend.replan = True

    @property
    def memory(self):
        return self.backend.sem_map_module if self.backend is not None else None

    def act(self, observations):
        if not self._task_started:
            raise RuntimeError('Call start_task before act (UniNavBench does this automatically)')
        if observations['event'].depth_frame is None:
            raise ValueError('ST-Map requires renderDepthImage=True')
        if self._pending_actions:
            action, answer = self._pending_actions.popleft(), None
        else:
            action, answer = self.backend.act(observations)
        self._task_steps += 1
        # Active-EQA: one VLM answer when the agent stops, or on the last step it has
        if answer is None and self._instruction.strip().endswith('?') and \
                (action == 'Stop' or self._task_steps >= self.config['max_task_steps']):
            answer = self._answer(observations)
        return action, answer

    def _answer(self, observations):
        from PIL import Image
        from ._vendor.vlm.client import USAGE
        backend, mapper = self.backend, self.backend.sem_map_module
        frame = self.output_dir / 'answer_view.jpg'  # overwritten: constant disk use
        Image.fromarray(observations['event'].frame).save(frame)
        target = backend.found_goal
        sims = [s[0] for s in mapper.target_similarity_with_objects]
        if target is None and len(sims) == len(mapper.objects) and sims and max(sims) > 0.5:
            target = mapper.objects[max(range(len(sims)), key=sims.__getitem__)]
        views = backend.object_views(target, target['image_idx'][-2:]) if target is not None else []
        answer = mapper.vlm.answer(self._instruction, str(frame), [v for v in views if Path(v[0]).exists()])
        (self.output_dir / 'usage.json').write_text(json.dumps(dict(USAGE)))
        return answer


def build():
    return STMapAgent, {'config': load_config()}, True
