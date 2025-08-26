# Losses package
from .heatmap_utils import make_heatmaps_torch, heatmaps_to_coords_argmax, softargmax_2d

__all__ = ["make_heatmaps_torch", "heatmaps_to_coords_argmax", "softargmax_2d"]
