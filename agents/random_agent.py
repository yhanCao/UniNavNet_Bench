from procthor_challenge.agent import Agent
from procthor_challenge.challenge import ALLOWED_ACTIONS
import random


class SimpleRandomAgent(Agent):

    def reset(self):
        pass

    def act(self, action_return):
        instruction = action_return["instruction"]           # np.uint8 : 480 x 640 x 3
        image_task = action_return["image_task"]       # np.float32 : 480 x 640 (default: None)
        event = action_return["event"]  # str : e.g. "AlarmClock"
        step = action_return["step"]
        action = random.choice(ALLOWED_ACTIONS[1:])
        predict = None
        return action,predict


def build():
    agent_class = SimpleRandomAgent
    agent_kwargs = {}
    # resembles SimpleRandomAgent(**{})
    render_depth = False
    return agent_class, agent_kwargs, render_depth