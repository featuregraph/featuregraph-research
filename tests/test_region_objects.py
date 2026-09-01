from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from featuregraph.behaviors.objects import BehaviorObjects
from featuregraph.behaviors.regions import RegionObjects, edit_alignment

GROUP = ["task_id", "pair_type", "pair_index", "grid_role"]


def observations(grids, task_id="deadbeef"):
    """Build a cell-level observation table from role-keyed grids."""
    records = []
    for pair_type, pair_index, roles in grids:
        for grid_role, grid in roles.items():
            grid = np.asarray(grid, dtype=int)
            for row, column in np.ndindex(grid.shape):
                records.append(
                    {
                        "task_id": task_id,
                        "pair_type": pair_type,
                        "pair_index": pair_index,
                        "grid_role": grid_role,
                        "row": row,
                        "column": column,
                        "color": int(grid[row, column]),
                    }
                )
    return pd.DataFrame.from_records(records)


def single_grid(grid):
    return observations([("train", 0, {"input": grid})])


def build(frame, **kwargs):
    builder = RegionObjects(signals="color", group=GROUP, **kwargs)
    return builder, builder.summarize(builder.fit_transform(frame))


def test_one_object_per_region():
    grid = [[1, 1, 0], [1, 0, 0], [0, 0, 2]]
    _, objects = build(single_grid(grid))

    assert isinstance(objects, BehaviorObjects)
    assert objects.behavior_type == "region"
    # Three colour components plus the background component.
    assert objects.count == 3
    table = objects.to_pandas()
    assert sorted(table["size"]) == [1, 3, 5]


def test_background_regions_can_be_excluded():
    grid = [[1, 1, 0], [1, 0, 0], [0, 0, 2]]
    _, objects = build(single_grid(grid), include_background=False)

    table = objects.to_pandas()
    assert objects.count == 2
    assert not table["is_background"].any()


def test_region_properties_describe_geometry():
    # A ring: eight cells around one enclosed hole.
    grid = [[3, 3, 3], [3, 0, 3], [3, 3, 3]]
    _, objects = build(single_grid(grid), include_background=False)

    region = objects.to_pandas().iloc[0]
    assert region["size"] == 8
    assert region["height"] == 3
    assert region["width"] == 3
    assert region["bbox_area"] == 9
    assert region["fill_ratio"] == pytest.approx(8 / 9)
    assert region["hole_count"] == 1
    assert region["touches_border"]


def test_interior_region_does_not_touch_the_border():
    grid = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
    _, objects = build(single_grid(grid), include_background=False)

    region = objects.to_pandas().iloc[0]
    assert not region["touches_border"]
    assert region["centroid_row"] == pytest.approx(1.0)
    assert region["centroid_column"] == pytest.approx(1.0)


def test_connectivity_is_a_declared_construction_parameter():
    grid = [[1, 0], [0, 1]]

    _, four = build(single_grid(grid), connectivity=4, include_background=False)
    _, eight = build(single_grid(grid), connectivity=8, include_background=False)

    assert four.count == 2
    assert eight.count == 1
    assert four.construction["connectivity"] == 4
    assert eight.construction["connectivity"] == 8


def test_region_definition_is_a_declared_construction_parameter():
    grid = [[1, 2], [1, 2]]

    _, by_color = build(single_grid(grid), definition="uniform_color")
    _, by_mask = build(single_grid(grid), definition="foreground_mask")

    assert by_color.count == 2
    assert by_mask.count == 1
    assert by_mask.construction["definition"] == "foreground_mask"


def test_background_color_is_a_declared_construction_parameter():
    grid = [[5, 5], [1, 1]]
    _, objects = build(single_grid(grid), background_color=5)

    table = objects.to_pandas()
    background = table[table["is_background"]]
    assert len(background) == 1
    assert background.iloc[0]["color"] == 5


def test_regions_are_constructed_per_grid_not_per_pair():
    frame = observations(
        [("train", 0, {"input": [[1, 1], [0, 0]], "output": [[2, 0], [0, 3]]})]
    )
    _, objects = build(frame)

    table = objects.to_pandas()
    assert set(table["grid_role"]) == {"input", "output"}
    assert len(table[table["grid_role"] == "input"]) == 2


def test_objects_are_queryable_by_property():
    grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    _, objects = build(single_grid(grid), include_background=False)

    with_holes = objects.query().where(hole_count__ge=1).collect()
    assert len(with_holes) == 1


def test_incomplete_grid_is_declined_with_a_reason():
    frame = single_grid([[1, 2], [3, 4]]).drop(index=0).reset_index(drop=True)
    builder, objects = build(frame)

    assert objects.count == 0
    assert builder.declined_["reason"].tolist() == ["incomplete_grid_observations"]


def test_missing_coordinate_columns_are_rejected():
    frame = single_grid([[1, 2]]).drop(columns=["column"])
    builder = RegionObjects(signals="color", group=GROUP)

    with pytest.raises(ValueError, match="Required columns are missing"):
        builder.fit_transform(frame)


def test_invalid_construction_parameters_are_rejected():
    with pytest.raises(ValueError, match="connectivity"):
        RegionObjects(connectivity=6)
    with pytest.raises(ValueError, match="definition"):
        RegionObjects(definition="nonsense")


def test_edit_alignment_reports_an_edit_inside_one_region():
    # One cell of a solid block changes: the edit sits inside that block.
    frame = observations(
        [
            (
                "train",
                0,
                {
                    "input": [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                    "output": [[1, 2, 0], [1, 1, 0], [0, 0, 0]],
                },
            )
        ]
    )
    edits = edit_alignment(frame)

    assert len(edits) == 1
    assert edits.iloc[0]["is_aligned"]
    assert edits.iloc[0]["regions_spanned"] == 1
    assert edits.iloc[0]["size"] == 1


def test_edit_alignment_reports_an_edit_spanning_regions():
    # The changed cells run from inside the object out into the background.
    frame = observations(
        [
            (
                "train",
                0,
                {
                    "input": [[1, 1, 0], [0, 0, 0], [0, 0, 0]],
                    "output": [[2, 2, 2], [0, 0, 0], [0, 0, 0]],
                },
            )
        ]
    )
    edits = edit_alignment(frame)

    assert len(edits) == 1
    assert not edits.iloc[0]["is_aligned"]
    assert edits.iloc[0]["regions_spanned"] == 2
    assert pd.isna(edits.iloc[0]["region_id"])


def test_edit_alignment_ignores_pairs_that_do_not_match_in_shape():
    frame = observations(
        [("train", 0, {"input": [[1, 1]], "output": [[1, 1], [1, 1]]})]
    )

    assert edit_alignment(frame).empty


def test_edit_alignment_returns_nothing_when_grids_are_identical():
    frame = observations(
        [("train", 0, {"input": [[1, 0], [0, 1]], "output": [[1, 0], [0, 1]]})]
    )

    assert edit_alignment(frame).empty


def test_construction_records_every_declared_parameter():
    _, objects = build(single_grid([[1, 0]]))

    assert objects.construction["background_color"] == 0
    assert objects.construction["connectivity"] == 4
    assert objects.construction["definition"] == "uniform_color"
    assert objects.construction["include_background"] is True
