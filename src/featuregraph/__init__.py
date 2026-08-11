"""
FeatureGraph public API.
"""

from .behaviors import oscillation, accumulation, spatial_sketch
from . import datasets

from .utils._plot import plot
from .utils._arc_agi import training_cycle, test_cycle, get_block_coordinates, get_output_cells, get_valid_transformations, get_candidate_transformations, get_instruction_candidates, plot_arc_agi

__version__ = "0.1.0a1"

__all__ = [
    "oscillation",
    "accumulation",
    "spatial_sketch",
    "datasets",
    "plot",
    "training_cycle",
    "test_cycle",
    "get_block_coordinates",
    "get_output_cells",
    "get_valid_transformations",
    "get_candidate_transformations",
    "get_instruction_candidates",
    "plot_arc_agi"
]