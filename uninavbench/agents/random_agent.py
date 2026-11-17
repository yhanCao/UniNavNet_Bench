import random

from uninavbench.agent import Agent
from uninavbench.challenge import ALLOWED_ACTIONS


class RandomAgent(Agent):
    """Random movement actions; never stops (the default agent of `uninavbench run`)."""

    def reset(self):
        pass

    def act(self, observations):
        return random.choice(ALLOWED_ACTIONS[1:]), None


SimpleRandomAgent = RandomAgent  # previous name


def build():
    return RandomAgent, {}, False
