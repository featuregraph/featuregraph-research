"""Inspectable NumPy coordinate representations."""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np


def coordinate_arrays(
    shape: tuple[int, int],
    *,
    block_shape: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Return named coordinate arrays for a two-dimensional shape.

    When ``block_shape`` is supplied, the result also describes which block
    contains each cell and where the cell lies inside that block.
    """
    if len(shape) != 2 or any(size <= 0 for size in shape):
        raise ValueError("shape must contain two positive dimensions")

    row, column = np.indices(shape)
    coordinates = {"row": row, "column": column}

    if block_shape is None:
        return coordinates

    if len(block_shape) != 2 or any(size <= 0 for size in block_shape):
        raise ValueError("block_shape must contain two positive dimensions")

    block_height, block_width = block_shape
    coordinates.update(
        {
            "block_row": row // block_height,
            "block_column": column // block_width,
            "within_block_row": row % block_height,
            "within_block_column": column % block_width,
        }
    )
    return coordinates


def plot_array_axes(
    arrays: Mapping[str, np.ndarray],
    *,
    annotate: bool = True,
    cmap: str = "viridis",
):
    """Plot equally shaped two-dimensional arrays as labeled heatmaps."""
    if not arrays:
        raise ValueError("arrays must contain at least one named array")

    normalized = {name: np.asarray(values) for name, values in arrays.items()}
    shapes = {values.shape for values in normalized.values()}
    if len(shapes) != 1:
        raise ValueError("all arrays must have the same shape")

    shape = next(iter(shapes))
    if len(shape) != 2:
        raise ValueError("each array must be two-dimensional")

    count = len(normalized)
    columns = min(3, count)
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 3.5 * rows),
        constrained_layout=True,
        squeeze=False,
    )

    flat_axes = axes.ravel()
    for axis, (name, values) in zip(
        flat_axes,
        normalized.items(),
        strict=False,
    ):
        axis.imshow(values, cmap=cmap)
        axis.set_title(name.replace("_", " ").title())
        axis.set_xticks(np.arange(shape[1]))
        axis.set_yticks(np.arange(shape[0]))

        if annotate:
            for row, column in np.ndindex(shape):
                axis.text(
                    column,
                    row,
                    str(values[row, column]),
                    ha="center",
                    va="center",
                    color="white",
                )

    for axis in flat_axes[count:]:
        axis.set_visible(False)

    return fig, axes
