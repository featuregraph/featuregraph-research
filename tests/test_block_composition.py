from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from featuregraph.behaviors.composition import BlockComposition, resolve_layout
from featuregraph.behaviors.objects import BehaviorObjects

GROUP = ["task_id", "pair_type", "pair_index"]


def observations(pairs, task_id="deadbeef"):
    """Build a cell-level observation table from (input, output) grids."""
    records = []
    for pair_type, pair_index, grids in pairs:
        for grid_role, grid in grids.items():
            if grid is None:
                continue
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


def build(frame, **kwargs):
    builder = BlockComposition(signals="color", group=GROUP, **kwargs)
    objects = builder.summarize(builder.fit_transform(frame))
    return builder, objects


def test_one_object_per_output_block():
    grid = [[1, 2], [3, 4]]
    output = [[1, 2, 2, 1], [3, 4, 4, 3]]
    _, objects = build(observations([("train", 0, {"input": grid, "output": output})]))

    assert isinstance(objects, BehaviorObjects)
    assert objects.behavior_type == "block_composition"
    assert objects.count == 2
    assert objects.to_pandas()["cell_count"].tolist() == [4, 4]
    assert objects.to_pandas()["operator"].tolist() == ["copy", "flip_horizontal"]


def test_candidate_sets_are_retained_rather_than_collapsed():
    # A left-right symmetric grid cannot distinguish copy from flip_horizontal.
    grid = [[1, 1], [2, 2]]
    output = [[1, 1], [2, 2]]
    _, objects = build(observations([("train", 0, {"input": grid, "output": output})]))

    block = objects.to_pandas().iloc[0]
    assert block["candidates"] == frozenset({"copy", "flip_horizontal"})
    assert block["candidate_count"] == 2
    assert not block["is_determined"]
    assert pd.isna(block["operator"])


def test_unmatched_block_is_recorded_not_raised():
    grid = [[1, 2], [3, 4]]
    output = [[9, 9], [9, 9]]
    _, objects = build(observations([("train", 0, {"input": grid, "output": output})]))

    block = objects.to_pandas().iloc[0]
    assert block["is_unmatched"]
    assert block["candidate_count"] == 0
    assert block["candidates"] == frozenset()


def test_unmatched_blocks_are_queryable():
    grid = [[1, 2], [3, 4]]
    output = [[1, 2, 9, 9], [3, 4, 9, 9]]
    _, objects = build(observations([("train", 0, {"input": grid, "output": output})]))

    unmatched = objects.query().where(is_unmatched=True).collect()
    assert len(unmatched) == 1
    assert unmatched["block_column"].tolist() == [1]


def test_pair_that_does_not_tile_is_declined_with_a_reason():
    grid = [[1, 2], [3, 4]]
    output = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    builder, objects = build(
        observations([("train", 0, {"input": grid, "output": output})])
    )

    assert objects.count == 0
    assert builder.declined_["reason"].tolist() == ["output_not_block_multiple"]


def test_pair_without_an_output_grid_is_declined():
    grid = [[1, 2], [3, 4]]
    builder, _ = build(
        observations(
            [
                ("train", 0, {"input": grid, "output": [[1, 2], [3, 4]]}),
                ("test", 0, {"input": grid, "output": None}),
            ]
        )
    )

    declined = builder.declined_
    assert declined["reason"].tolist() == ["no_output_grid"]
    assert declined["pair_type"].tolist() == ["test"]


def test_shape_invalid_operators_never_become_candidates():
    # rotate_90 changes the shape of a non-square grid, so it cannot describe
    # any block of this pair even though the block happens to match.
    grid = [[1, 1, 1], [1, 1, 1]]
    output = [[1, 1, 1], [1, 1, 1]]
    _, objects = build(observations([("train", 0, {"input": grid, "output": output})]))

    candidates = objects.to_pandas().iloc[0]["candidates"]
    assert "rotate_90" not in candidates
    assert "rotate_270" not in candidates
    assert "copy" in candidates


def test_background_operator_follows_the_declared_background_color():
    grid = [[1, 2], [3, 4]]
    output = [[1, 2, 5, 5], [3, 4, 5, 5]]
    _, objects = build(
        observations([("train", 0, {"input": grid, "output": output})]),
        background_color=5,
    )

    assert objects.to_pandas()["operator"].tolist() == ["copy", "background"]


def test_resolve_layout_by_block_coordinate_intersects_across_pairs():
    # Pair 0 is ambiguous on its single block; pair 1 resolves it to copy.
    pairs = [
        ("train", 0, {"input": [[1, 1], [2, 2]], "output": [[1, 1], [2, 2]]}),
        ("train", 1, {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}),
    ]
    _, objects = build(observations(pairs))

    resolved = resolve_layout(objects, by=("block_row", "block_column"))
    assert resolved["operator"].tolist() == ["copy"]
    assert resolved["is_determined"].all()


def test_resolve_layout_by_state_describes_an_input_dependent_layout():
    # The output tiles a copy of the grid where the input cell is non-zero and
    # background where it is zero. No fixed layout describes this.
    grid = np.array([[1, 2], [0, 3]])
    output = np.block(
        [
            [grid if grid[r, c] else np.zeros_like(grid) for c in range(2)]
            for r in range(2)
        ]
    )
    _, objects = build(
        observations([("train", 0, {"input": grid, "output": output})])
    )

    by_state = resolve_layout(objects, by=("block_state",))
    assert dict(zip(by_state["block_state"], by_state["operator"], strict=True)) == {
        "background": "background",
        "foreground": "copy",
    }


def test_resolve_layout_returns_empty_intersection_when_pairs_disagree():
    pairs = [
        ("train", 0, {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}),
        ("train", 1, {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]}),
    ]
    _, objects = build(observations(pairs))

    resolved = resolve_layout(objects, by=("block_row", "block_column"))
    assert resolved["candidate_count"].tolist() == [0]
    assert not resolved["is_determined"].any()


def test_resolve_layout_can_hold_tasks_apart():
    frame = pd.concat(
        [
            observations(
                [("train", 0, {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]})],
                task_id="aaaaaaaa",
            ),
            observations(
                [("train", 0, {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]})],
                task_id="bbbbbbbb",
            ),
        ],
        ignore_index=True,
    )
    _, objects = build(frame)

    resolved = resolve_layout(
        objects,
        by=("block_row", "block_column"),
        within=("task_id",),
    )
    assert dict(zip(resolved["task_id"], resolved["operator"], strict=True)) == {
        "aaaaaaaa": "copy",
        "bbbbbbbb": "flip_horizontal",
    }


def test_resolve_layout_rejects_unknown_grouping_columns():
    _, objects = build(
        observations([("train", 0, {"input": [[1, 2]], "output": [[1, 2]]})])
    )

    with pytest.raises(KeyError, match="not_a_column"):
        resolve_layout(objects, by=("not_a_column",))


def test_block_state_is_undefined_when_the_layout_is_not_input_shaped():
    # A 2 x 2 input tiled into a 1 x 2 block layout leaves no input cell for a
    # block to correspond to, so no state-derived layout is available.
    grid = [[1, 2], [3, 4]]
    output = [[1, 2, 1, 2], [3, 4, 3, 4]]
    _, objects = build(observations([("train", 0, {"input": grid, "output": output})]))

    table = objects.to_pandas()
    assert table["block_rows"].tolist() == [1, 1]
    assert table["block_columns"].tolist() == [2, 2]
    assert table["block_state"].isna().all()


def test_construction_records_the_declared_vocabulary():
    _, objects = build(
        observations([("train", 0, {"input": [[1, 2]], "output": [[1, 2]]})])
    )

    assert objects.construction["background_color"] == 0
    assert objects.construction["operators"][0] == "copy"
    assert "background" in objects.construction["operators"]


def test_missing_coordinate_columns_are_rejected():
    frame = observations(
        [("train", 0, {"input": [[1, 2]], "output": [[1, 2]]})]
    ).drop(columns=["row"])

    builder = BlockComposition(signals="color", group=GROUP)
    with pytest.raises(ValueError, match="Required columns are missing"):
        builder.fit_transform(frame)


def test_summarize_rejects_an_unconfigured_signal():
    frame = observations([("train", 0, {"input": [[1, 2]], "output": [[1, 2]]})])
    builder = BlockComposition(signals="color", group=GROUP)
    cells = builder.fit_transform(frame)

    with pytest.raises(ValueError, match="was not configured"):
        builder.summarize(cells, signal="other")
