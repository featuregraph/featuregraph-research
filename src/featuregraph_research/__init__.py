"""
FeatureGraph public API.
"""

from . import datasets
from .behaviors import accumulation, oscillation, spatial_sketch
from .utils._arc_agi import (
    get_block_coordinates,
    get_candidate_transformations,
    get_instruction_candidates,
    get_output_cells,
    get_training_pairs,
    get_valid_transformations,
    plot_arc_agi,
    test_cycle,
    training_cycle,
)
from .utils._array_axes import coordinate_arrays, plot_array_axes
from .utils._plot import plot

__version__ = "0.1.0a1"

__all__ = [
    "oscillation",
    "accumulation",
    "spatial_sketch",
    "datasets",
    "plot",
    "coordinate_arrays",
    "plot_array_axes",
    "training_cycle",
    "test_cycle",
    "get_block_coordinates",
    "get_output_cells",
    "get_valid_transformations",
    "get_candidate_transformations",
    "get_instruction_candidates",
    "get_training_pairs",
    "plot_arc_agi",
]
