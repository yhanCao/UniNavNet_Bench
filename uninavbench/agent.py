from abc import ABC, abstractmethod


class Agent(ABC):
    """Base class of the agents evaluated by the benchmark.

    ``act`` receives a dict with
      - ``instruction``: the sub-task instruction or question (str)
      - ``image_task``: path of the goal image for image-goal navigation, else None
      - ``event``: the latest ai2thor Event (RGB in ``event.frame``, depth in ``event.depth_frame``)
      - ``step``: number of steps taken so far in the episode
    and returns ``(action, prediction)``: an action from ``uninavbench.ALLOWED_ACTIONS`` and, for
    Active-EQA, the answer (None otherwise). The prediction returned with the last action of a sub-task
    is scored.

    Agents may additionally implement ``bind_environment(controller, output_dir)``
    (once per episode) and ``start_task(instruction, reachable_positions)`` (once
    per sub-task) when they need simulator geometry or persistent episode memory.
    ``update_environment(reachable_positions)`` refreshes optional planning geometry
    after scheduled scene changes during a task.
    These optional hooks expose privileged simulator access, as used by ST-Map.
    """

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self, observations):
        pass
