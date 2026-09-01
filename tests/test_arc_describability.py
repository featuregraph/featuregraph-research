from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.arc.describability import main, scan_split, scan_task, summarize

TILING_TASK = {
    "train": [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 2, 1], [3, 4, 4, 3]],
        },
        {
            "input": [[5, 6], [7, 8]],
            "output": [[5, 6, 6, 5], [7, 8, 8, 7]],
        },
    ],
    "test": [{"input": [[9, 1], [2, 3]]}],
}

SAME_SHAPE_TASK = {
    "train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}],
    "test": [{"input": [[5, 6], [7, 8]]}],
}

NOT_TILING_TASK = {
    "train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}],
    "test": [{"input": [[5, 6], [7, 8]]}],
}


def _state_layout_pair(grid):
    grid = np.asarray(grid)
    output = np.block(
        [
            [grid if grid[r, c] else np.zeros_like(grid) for c in range(grid.shape[1])]
            for r in range(grid.shape[0])
        ]
    )
    return {"input": grid.tolist(), "output": output.tolist()}


def state_layout_task():
    # Two demonstrations whose background cells sit in different places, so no
    # layout fixed to block position can describe both.
    return {
        "train": [
            _state_layout_pair([[1, 2], [0, 3]]),
            _state_layout_pair([[0, 4], [5, 6]]),
        ],
        "test": [{"input": [[7, 0], [8, 9]]}],
    }


def test_tiling_task_is_in_frame_and_fixed_layout_determined():
    record = scan_task(TILING_TASK, task_id="00000001", split="training")

    assert record["in_frame"]
    assert record["block_rows"] == 1
    assert record["block_columns"] == 2
    assert not record["identity_layout"]
    assert record["blocks_unmatched"] == 0
    assert record["fixed_layout_determined"]


def test_same_shape_task_is_marked_as_an_identity_layout():
    record = scan_task(SAME_SHAPE_TASK, task_id="00000002", split="training")

    assert record["in_frame"]
    assert record["identity_layout"]
    assert record["block_rows"] == 1
    assert record["block_columns"] == 1


def test_task_whose_output_does_not_tile_is_out_of_frame():
    record = scan_task(NOT_TILING_TASK, task_id="00000003", split="training")

    assert not record["in_frame"]
    assert record["declined_reason"] == "output_not_block_multiple"


def test_state_layout_task_resolves_by_state_not_by_coordinate():
    record = scan_task(state_layout_task(), task_id="00000004", split="training")

    assert record["in_frame"]
    assert record["state_layout_determined"]
    assert not record["fixed_layout_determined"]


def test_layout_inference_ignores_test_pairs():
    # A test pair whose output contradicts the demonstrations must not change
    # the inferred layout, because layouts are resolved from demonstrations.
    task = {
        "train": TILING_TASK["train"],
        "test": [{"input": [[9, 1], [2, 3]], "output": [[1, 9, 9, 1], [3, 2, 2, 3]]}],
    }
    record = scan_task(task, task_id="00000005", split="training")

    assert record["fixed_layout_determined"]
    assert record["n_blocks"] == 4  # two demonstration pairs, two blocks each


def test_scan_split_and_summary_roll_up(tmp_path):
    split_dir = tmp_path / "training"
    split_dir.mkdir()
    for task_id, task in (
        ("00000001", TILING_TASK),
        ("00000002", SAME_SHAPE_TASK),
        ("00000003", NOT_TILING_TASK),
    ):
        (split_dir / f"{task_id}.json").write_text(json.dumps(task), encoding="utf-8")

    table = scan_split(tmp_path, "training")
    assert len(table) == 3

    summary = summarize(table).iloc[0]
    assert summary["tasks"] == 3
    assert summary["in_frame"] == 2
    assert summary["identity_layout"] == 1
    assert summary["non_trivial_tiling"] == 1
    assert summary["fixed_layout_determined"] == 1


def test_missing_split_directory_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="split directory not found"):
        scan_split(tmp_path, "training")


def test_main_writes_a_table_and_a_summary(tmp_path):
    for split in ("training", "evaluation"):
        split_dir = tmp_path / split
        split_dir.mkdir()
        (split_dir / "00000001.json").write_text(
            json.dumps(TILING_TASK), encoding="utf-8"
        )

    output = tmp_path / "out" / "describability.csv"
    assert main(["--data-dir", str(tmp_path), "--output", str(output)]) == 0

    assert output.exists()
    summary_path = output.with_name("describability_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["arc_agi_version"] == "2"
    assert summary["operators"][0] == "copy"
    assert {row["split"] for row in summary["splits"]} == {"training", "evaluation"}
