import random

from uninavbench.agent import Agent
from uninavbench.challenge import ALLOWED_ACTIONS


class SimpleRandomAgent(Agent):

    def reset(self):
        pass

    def act(self, observations):
        action = random.choice(ALLOWED_ACTIONS[1:])
        predict = None
        return action, predict


def build():
    agent_class = SimpleRandomAgent
    agent_kwargs = {}
    # resembles SimpleRandomAgent(**{})
    render_depth = False
    return agent_class, agent_kwargs, render_depth
