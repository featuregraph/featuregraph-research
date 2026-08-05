from __future__ import annotations

import json

import pandas as pd
import pytest

from featuregraph.datasets import arc_agi


@pytest.fixture
def arc_data_dir(tmp_path):
    task = {
        "train": [
            {
                "input": [[0, 1], [1, 0]],
                "output": [[2, 1], [1, 2]],
            }
        ],
        "test": [{"input": [[0, 1, 0]]}],
    }
    split_dir = tmp_path / "training"
    split_dir.mkdir()
    (split_dir / "00ff00ff.json").write_text(json.dumps(task), encoding="utf-8")
    return tmp_path


def test_arc_agi_returns_one_row_per_cell(arc_data_dir):
    observations = arc_agi("00ff00ff", data_dir=arc_data_dir)

    assert isinstance(observations, pd.DataFrame)
    assert len(observations) == 11
    assert observations.columns.tolist() == [
        "task_id",
        "split",
        "pair_type",
        "pair_index",
        "grid_role",
        "row",
        "column",
        "color",
        "grid_height",
        "grid_width",
    ]
    assert observations.attrs["arc_agi_version"] == "2"
    assert observations.attrs["task_id"] == "00ff00ff"


def test_arc_agi_preserves_grid_coordinates(arc_data_dir):
    observations = arc_agi("00ff00ff", data_dir=arc_data_dir)
    test_input = observations.query(
        "pair_type == 'test' and pair_index == 0 and grid_role == 'input'"
    )

    assert test_input[["row", "column", "color"]].to_dict("records") == [
        {"row": 0, "column": 0, "color": 0},
        {"row": 0, "column": 1, "color": 1},
        {"row": 0, "column": 2, "color": 0},
    ]
    assert test_input["grid_height"].eq(1).all()
    assert test_input["grid_width"].eq(3).all()


@pytest.mark.parametrize(
    ("task_id", "split", "message"),
    [
        ("../secret", "training", "task_id"),
        ("00ff00ff", "private", "split"),
    ],
)
def test_arc_agi_validates_request(arc_data_dir, task_id, split, message):
    with pytest.raises(ValueError, match=message):
        arc_agi(task_id, split=split, data_dir=arc_data_dir)
