from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from featuregraph.operators.grids import (
    grid_from_cells,
    hole_count,
    label_regions,
)


def cells(grid, signal="color"):
    grid = np.asarray(grid)
    return pd.DataFrame(
        [
            {"row": row, "column": column, signal: int(grid[row, column])}
            for row, column in np.ndindex(grid.shape)
        ]
    )


def test_grid_from_cells_round_trips():
    grid = np.array([[1, 2, 3], [4, 5, 6]])
    assert grid_from_cells(cells(grid), "color").tolist() == grid.tolist()


def test_incomplete_observations_are_rejected():
    frame = cells([[1, 2], [3, 4]]).drop(index=0)
    with pytest.raises(ValueError, match="incomplete_grid_observations"):
        grid_from_cells(frame, "color")


def test_negative_coordinates_are_rejected():
    frame = cells([[1, 2], [3, 4]])
    frame.loc[0, "row"] = -1
    with pytest.raises(ValueError, match="negative_grid_coordinates"):
        grid_from_cells(frame, "color")


def test_uniform_colour_regions_split_adjacent_colours():
    grid = np.array([[1, 2], [1, 2]])
    labels = label_regions(grid, definition="uniform_color")

    assert len(np.unique(labels)) == 2
    assert labels[0, 0] == labels[1, 0]
    assert labels[0, 0] != labels[0, 1]


def test_foreground_mask_regions_join_adjacent_colours():
    grid = np.array([[1, 2], [1, 2]])
    labels = label_regions(grid, definition="foreground_mask")

    assert len(np.unique(labels)) == 1
    assert (labels == labels[0, 0]).all()


def test_background_can_be_excluded():
    grid = np.array([[1, 0], [0, 0]])

    with_background = label_regions(grid, include_background=True)
    without_background = label_regions(grid, include_background=False)

    assert (with_background > 0).all()
    assert without_background[0, 0] > 0
    assert (without_background[grid == 0] == 0).all()


def test_connectivity_changes_what_joins():
    grid = np.array([[1, 0], [0, 1]])

    four = label_regions(grid, connectivity=4, include_background=False)
    eight = label_regions(grid, connectivity=8, include_background=False)

    assert four[0, 0] != four[1, 1]
    assert eight[0, 0] == eight[1, 1]


def test_labels_partition_the_grid_when_background_is_included():
    grid = np.array([[1, 0, 2], [0, 0, 2]])
    labels = label_regions(grid, include_background=True)

    assert (labels > 0).all()


def test_invalid_construction_parameters_are_rejected():
    grid = np.array([[1]])

    with pytest.raises(ValueError, match="connectivity"):
        label_regions(grid, connectivity=6)
    with pytest.raises(ValueError, match="definition"):
        label_regions(grid, definition="nonsense")


def test_hole_count_finds_an_enclosed_area():
    ring = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    rows, columns = np.nonzero(ring)

    assert hole_count(rows, columns) == 1


def test_hole_count_is_zero_for_a_solid_region():
    rows, columns = np.nonzero(np.ones((3, 3), dtype=int))

    assert hole_count(rows, columns) == 0


def test_hole_count_counts_separate_holes():
    shape = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    rows, columns = np.nonzero(shape)

    assert hole_count(rows, columns) == 2
