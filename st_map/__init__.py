"""ST-Map: ST-CogNav mapping, temporal memory and navigation baseline.

Importing this package does not import CUDA, detection models or the simulator.
"""
from .config import UPSTREAM_COMMIT, load_config

__all__ = ['STMap', 'STMapAgent', 'build', 'load_config', 'UPSTREAM_COMMIT']


def __getattr__(name):
    if name == 'STMapAgent':
        from .agent import STMapAgent
        return STMapAgent
    if name == 'STMap':
        from .mapping import STMap
        return STMap
    raise AttributeError(name)


def build():
    from .agent import STMapAgent
    return STMapAgent, {'config': load_config()}, True
