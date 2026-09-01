"""Declared grid operators for block-composition representations.

Each operator is an :class:`~featuregraph.operators.registry.OperatorRecord`,
so the vocabulary carries its own provenance: a stable name, the function that
produces the candidate grid, and whether the operator is shape preserving.

The vocabulary is deliberately small and explicit. Measuring where it runs out
is the point; silently widening it is not.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from featuregraph.operators.registry import OperatorRecord

__all__ = [
    "BLOCK_OPERATOR_NAMES",
    "block_operators",
    "candidate_grid",
    "shape_valid",
]

#: Canonical ordering of the block-composition vocabulary.
BLOCK_OPERATOR_NAMES: tuple[str, ...] = (
    "copy",
    "flip_horizontal",
    "flip_vertical",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "background",
)


def block_operators(background_color: int = 0) -> tuple[OperatorRecord, ...]:
    """Return the block-composition operator vocabulary.

    ``background_color`` parameterizes the constant-fill operator so that a
    task whose background is not zero is described with its own background,
    rather than silently filled with zeros.
    """
    if not 0 <= int(background_color) <= 9:
        raise ValueError("background_color must be an integer from 0 through 9")

    fill = int(background_color)

    return (
        OperatorRecord(
            name="copy",
            function=lambda grid: grid,
            preserves_shape=True,
        ),
        OperatorRecord(
            name="flip_horizontal",
            function=np.fliplr,
            preserves_shape=True,
        ),
        OperatorRecord(
            name="flip_vertical",
            function=np.flipud,
            preserves_shape=True,
        ),
        # Quarter turns preserve shape only on square grids, so they do not
        # declare the invariant. Shape validity is checked per grid instead.
        OperatorRecord(
            name="rotate_90",
            function=lambda grid: np.rot90(grid, k=1),
        ),
        OperatorRecord(
            name="rotate_180",
            function=lambda grid: np.rot90(grid, k=2),
            preserves_shape=True,
        ),
        OperatorRecord(
            name="rotate_270",
            function=lambda grid: np.rot90(grid, k=3),
        ),
        OperatorRecord(
            name="background",
            function=lambda grid: np.full_like(grid, fill),
            preserves_shape=True,
        ),
    )


def shape_valid(operator: OperatorRecord, grid: np.ndarray) -> bool:
    """Report whether ``operator`` returns a grid shaped like ``grid``."""
    source = np.asarray(grid)
    return np.asarray(operator.function(source)).shape == source.shape


def candidate_grid(
    operator: OperatorRecord,
    grid: np.ndarray,
) -> np.ndarray | None:
    """Apply ``operator`` when it is shape valid, otherwise return ``None``.

    Returning ``None`` keeps a shape-invalid operator distinguishable from an
    operator that applied and simply did not match the observed output.
    """
    source = np.asarray(grid)
    if not shape_valid(operator, source):
        return None
    return operator.apply(source)


def operator_names(operators: Sequence[OperatorRecord]) -> list[str]:
    """Return the declared names of ``operators`` in order."""
    return [operator.name for operator in operators]
