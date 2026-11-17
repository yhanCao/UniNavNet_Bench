"""Standalone mapper facade; perception models load on construction."""
from .config import load_config, validate_runtime


class STMap:
    """Own a mapper with the original mapping/memory API.

    ``size`` and ``min_bound`` use upstream's (x, z, y) voxel convention.
    Use ``mapper.backend`` to access the original torch module directly.
    ``getGlobalMap`` takes the original RGB-D observations and map tensor.
    """

    def __init__(self, fov, size, min_bound, config=None):
        cfg = load_config(config)
        validate_runtime(cfg)
        from ._vendor.model.Semantic_Mapping_SG import Semantic_Mapping_SG
        self.backend = Semantic_Mapping_SG(fov, size, min_bound, cfg['memory'], cfg).eval()

    def __getattr__(self, name):
        return getattr(self.backend, name)
