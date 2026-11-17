"""A random agent: the smallest working example of the UniNavBench agent interface.

    uninavbench run -a examples/agents/random_agent.py -p test -l all -i 0 -t nav -es 0 -ee 10

Every step it returns a random movement action; with probability ``stop_prob`` it stops the sub-task
instead. For Active-EQA sub-tasks it answers when it stops (the answer is scored by the judge).
"""
import random

from uninavbench import ALLOWED_ACTIONS, Agent

MOVES = [a for a in ALLOWED_ACTIONS if a != "Stop"]  # MoveAhead, RotateLeft, RotateRight, LookUp, LookDown


class RandomAgent(Agent):
    def __init__(self, stop_prob=0.02, seed=None):
        self.stop_prob = stop_prob
        self.rng = random.Random(seed)

    def reset(self):
        pass

    def act(self, observations):
        # observations["instruction"]: instruction or question of the current sub-task
        # observations["image_task"]:  goal image path for image-goal navigation, else None
        # observations["event"]:       ai2thor Event (RGB: event.frame, depth: event.depth_frame)
        # observations["step"]:        steps taken so far in the episode
        if self.rng.random() < self.stop_prob:
            return "Stop", self.answer(observations["instruction"])
        return self.rng.choice(MOVES), None

    def answer(self, question):
        """A (random) answer to an Active-EQA question; ignored for navigation sub-tasks."""
        return self.rng.choice(["yes", "no", "1", "2", "red", "on the table"])


def build():
    """Called by `uninavbench run -a <this file>`: (agent class, constructor kwargs, render depth?)."""
    return RandomAgent, {"stop_prob": 0.02}, False
