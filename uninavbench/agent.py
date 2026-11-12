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
    """

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self, observations):
        pass
