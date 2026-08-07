"""
FeatureGraph public API.
"""

from .behaviors import oscillation, accumulation, spatial_sketch
from . import datasets

from .utils._plot import plot

__version__ = "0.1.0a1"

__all__ = [
    "oscillation",
    "accumulation",
    "spatial_sketch",
    "datasets",
    "plot",
]