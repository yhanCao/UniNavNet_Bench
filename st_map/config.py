"""Configuration and packaged resources for ST-Map (no model imports)."""
from copy import deepcopy
import os
from pathlib import Path

UPSTREAM_COMMIT = '5ac93986c2db14d9f25143d86ed0b55ad7f8735c'
DEFAULTS = {
    'output_dir': 'results',
    'openseed_checkpoint': '',
    'embedding_model': 'mixedbread-ai/mxbai-embed-large-v1',
    'clip_model': 'ViT-H-14',
    'clip_pretrained': 'laion2b_s32b_b79k',
    'llm': {'base_url': 'http://127.0.0.1:8000/v1', 'model': 'default', 'api_key': 'EMPTY',
            'enable_thinking': False, 'max_tokens': 64},
    'vlm': {'base_url': 'http://127.0.0.1:8000/v1', 'model': 'default', 'api_key': 'EMPTY',
            'enable_thinking': False, 'max_tokens': 8, 'image_max_side': 320, 'crop_margin': 0.3},
    'memory': {'subgraph_nodes': 8, 'subgraph_frontiers': 4,
               'subgraph_objects_per_node': 6, 'history_caption': False,
               'caption_vlm': {},
               # bounded memory: per-object views and votes, state segments, archived objects
               'views_per_object': 3, 'class_votes': 20, 'segments_per_object': 8, 'archive_limit': 32},
    # per-step RGB-D dumps, detection overlays and path plots (grow with the episode)
    'debug_artifacts': False,
    # Active-EQA answers are given on Stop or at this step of a sub-task (the benchmark's max_steps)
    'max_task_steps': 500,
}


def resource_path(relative):
    """Absolute filesystem path to an installed algorithm resource."""
    return str(Path(__file__).resolve().parent / '_vendor' / relative)


def _merge(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def load_config(config=None):
    """Merge defaults, optional YAML/dict, then ST_MAP_* environment overrides.

    Relative asset/output paths in YAML are resolved against that YAML's directory.
    """
    cfg = deepcopy(DEFAULTS)
    config = config if config is not None else os.environ.get('ST_MAP_CONFIG')
    base_dir = Path.cwd()
    if isinstance(config, (str, os.PathLike)):
        import yaml
        path = Path(config).expanduser().resolve()
        base_dir = path.parent
        with path.open(encoding='utf-8') as stream:
            config = yaml.safe_load(stream) or {}
    if config is not None:
        if not isinstance(config, dict):
            raise ValueError('ST-Map configuration must be a YAML mapping or dictionary')
        _merge(cfg, config)
    for key in ('openseed_checkpoint', 'embedding_model', 'clip_pretrained', 'output_dir'):
        env = os.environ.get('ST_MAP_' + key.upper())
        if env is not None:
            cfg[key] = env
    for kind in ('llm', 'vlm'):
        for key in ('base_url', 'model', 'api_key'):
            env = os.environ.get('ST_MAP_' + kind.upper() + '_' + key.upper())
            if env is not None:
                cfg[kind][key] = env
    for key in ('openseed_checkpoint', 'output_dir', 'mapping_config', 'openseed_config'):
        if cfg.get(key):
            value = Path(os.path.expandvars(cfg[key])).expanduser()
            cfg[key] = str((base_dir / value).resolve())
    for key in ('subgraph_nodes', 'subgraph_frontiers', 'subgraph_objects_per_node',
                'views_per_object', 'class_votes', 'segments_per_object', 'archive_limit'):
        value = cfg['memory'][key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError('memory.' + key + ' must be a nonnegative integer')
    return cfg


def validate_runtime(config):
    checkpoint = Path(config['openseed_checkpoint'])
    if not config['openseed_checkpoint'] or not checkpoint.is_file():
        raise RuntimeError('ST-Map requires an OpenSeeD checkpoint: set ST_MAP_OPENSEED_CHECKPOINT '
                           'or openseed_checkpoint in ST_MAP_CONFIG. See docs/st-map.md.')
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('ST-Map requires the optional CUDA/model environment; see docs/st-map.md.') from exc
    if not torch.cuda.is_available():
        raise RuntimeError('ST-Map at the pinned upstream version requires CUDA; CPU execution is unsupported.')
