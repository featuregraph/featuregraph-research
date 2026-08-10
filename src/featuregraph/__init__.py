"""
FeatureGraph public API.
"""

from .behaviors import oscillation, accumulation, spatial_sketch
from . import datasets

from .utils._plot import plot
from .utils._arc_agi import get_block_coordinates, get_output_cells, get_candidate_transformations, get_instruction_layout

__version__ = "0.1.0a1"

__all__ = [
    "oscillation",
    "accumulation",
    "spatial_sketch",
    "datasets",
    "plot",
    "get_block_coordinates",
    "get_output_cells",
    "get_candidate_transformations",
    "get_instruction_layout"
]