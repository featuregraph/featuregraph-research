"""Deterministic operators for two-dimensional grid observations.

These are shared by every grid behavior so that the definition of "a grid" and
"a region" exists in exactly one place. A construction whose object boundaries
drift between behaviors cannot support a provenance claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

__all__ = [
    "CONNECTIVITY",
    "REGION_DEFINITIONS",
    "grid_from_cells",
    "hole_count",
    "label_regions",
]

#: Neighbourhood structures, keyed by the number of neighbours considered.
CONNECTIVITY: dict[int, np.ndarray] = {
    4: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    8: np.ones((3, 3), dtype=int),
}

#: How a region is delimited.
REGION_DEFINITIONS = ("uniform_color", "foreground_mask")


def grid_from_cells(cells: pd.DataFrame, signal: str) -> np.ndarray:
    """Rebuild a dense grid from its cell observations.

    Raises when the observations do not cover a complete rectangle, because a
    partially observed grid has no well-defined region structure.
    """
    rows = cells["row"].to_numpy(dtype=int)
    columns = cells["column"].to_numpy(dtype=int)

    if rows.min() < 0 or columns.min() < 0:
        raise ValueError("negative_grid_coordinates")

    height = int(rows.max()) + 1
    width = int(columns.max()) + 1

    if len(cells) != height * width:
        raise ValueError("incomplete_grid_observations")

    grid = np.zeros((height, width), dtype=int)
    grid[rows, columns] = cells[signal].to_numpy(dtype=int)
    return grid


def label_regions(
    grid: np.ndarray,
    *,
    background_color: int = 0,
    connectivity: int = 4,
    definition: str = "uniform_color",
    include_background: bool = True,
) -> np.ndarray:
    """Label the connected regions of ``grid``.

    Returns an integer array the shape of ``grid``, where ``0`` marks a cell
    belonging to no region and every other value identifies one region.

    ``definition`` selects what holds a region together: ``"uniform_color"``
    groups adjacent cells of the same colour, ``"foreground_mask"`` groups
    adjacent non-background cells whatever their colours. Both are used in
    practice, so the choice is declared rather than assumed.
    """
    if connectivity not in CONNECTIVITY:
        raise ValueError(f"connectivity must be one of: {sorted(CONNECTIVITY)}")
    if definition not in REGION_DEFINITIONS:
        raise ValueError(f"definition must be one of: {list(REGION_DEFINITIONS)}")

    values = np.asarray(grid)
    structure = CONNECTIVITY[connectivity]
    labels = np.zeros(values.shape, dtype=int)
    offset = 0

    if definition == "uniform_color":
        masks = [
            (color, values == color)
            for color in np.unique(values)
            if include_background or color != background_color
        ]
    else:
        masks = [(None, values != background_color)]
        if include_background:
            masks.append((background_color, values == background_color))

    for _, mask in masks:
        if not mask.any():
            continue
        component, count = ndimage.label(mask, structure=structure)
        labels = np.where(mask, component + offset, labels)
        offset += count

    return labels


def hole_count(
    rows: np.ndarray,
    columns: np.ndarray,
    connectivity: int = 4,
) -> int:
    """Count enclosed empty areas inside the region spanned by these cells.

    Holes are counted with 4-connectivity regardless of how the region itself
    was labelled, so that a diagonally connected ring still encloses its hole.
    """
    rows = np.asarray(rows, dtype=int)
    columns = np.asarray(columns, dtype=int)

    top, left = rows.min(), columns.min()
    mask = np.zeros((rows.max() - top + 1, columns.max() - left + 1), dtype=bool)
    mask[rows - top, columns - left] = True

    holes = ndimage.binary_fill_holes(mask) & ~mask
    if not holes.any():
        return 0

    _, count = ndimage.label(holes, structure=CONNECTIVITY[connectivity])
    return int(count)
