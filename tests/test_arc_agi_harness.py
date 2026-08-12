import json
from pathlib import Path

import numpy as np

from featuregraph.utils._arc_agi import (
    fallback_predictions,
    get_test_grids,
    get_training_pairs,
    load_challenges,
    predictions_to_attempts,
    predictions_to_lists,
    run_harness,
    solve_challenges,
    solve_fixed_layout_task,
    solve_state_layout_task,
    solve_task,
    write_submission,
)


def supported_task():
    return {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[1, 2, 2, 1], [3, 4, 4, 3]],
            }
        ],
        "test": [
            {
                "input": [[5, 6], [7, 8]],
            }
        ],
    }


def unsupported_task():
    return {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[9, 9], [9, 9]],
            }
        ],
        "test": [
            {
                "input": [[5, 6], [7, 8]],
            }
        ],
    }


def expected_prediction():
    return np.array(
        [
            [5, 6, 6, 5],
            [7, 8, 8, 7],
        ]
    )


def test_get_training_pairs_converts_native_grids_to_arrays():
    pairs = get_training_pairs(supported_task())

    assert len(pairs) == 1
    assert isinstance(pairs[0]["grid"], np.ndarray)
    assert isinstance(pairs[0]["output"], np.ndarray)

    assert np.array_equal(
        pairs[0]["grid"],
        np.array([[1, 2], [3, 4]]),
    )

    assert np.array_equal(
        pairs[0]["output"],
        np.array([[1, 2, 2, 1], [3, 4, 4, 3]]),
    )


def test_get_test_grids_converts_every_test_input():
    task = supported_task()
    task["test"].append(
        {
            "input": [[9, 8], [7, 6]],
        }
    )

    grids = get_test_grids(task)

    assert len(grids) == 2
    assert all(isinstance(grid, np.ndarray) for grid in grids)

    assert np.array_equal(
        grids[0],
        np.array([[5, 6], [7, 8]]),
    )

    assert np.array_equal(
        grids[1],
        np.array([[9, 8], [7, 6]]),
    )


def test_solve_task_predicts_every_test_grid():
    predictions = solve_task(supported_task())

    assert len(predictions) == 1
    assert np.array_equal(predictions[0], expected_prediction())


def test_predictions_to_lists_removes_numpy_objects():
    serialized = predictions_to_lists([expected_prediction()])

    assert serialized == [
        [
            [5, 6, 6, 5],
            [7, 8, 8, 7],
        ]
    ]

    assert isinstance(serialized[0][0][0], int)


def test_predictions_to_attempts_builds_two_attempt_contract():
    attempts = predictions_to_attempts([expected_prediction()])

    expected = expected_prediction().tolist()

    assert attempts == [
        {
            "attempt_1": expected,
            "attempt_2": expected,
        }
    ]


def test_fallback_predictions_copy_test_inputs():
    predictions = fallback_predictions(unsupported_task())

    assert len(predictions) == 1

    assert np.array_equal(
        predictions[0],
        np.array([[5, 6], [7, 8]]),
    )


def test_solve_challenges_isolates_unsupported_tasks():
    challenges = {
        "supported": supported_task(),
        "unsupported": unsupported_task(),
    }

    submission, failures = solve_challenges(challenges)

    assert set(submission) == set(challenges)
    assert set(failures) == {"unsupported"}

    failure = failures["unsupported"]
    assert failure.startswith("No solver family matched the task.")
    assert "fixed: No operator matched block (0, 0)." in failure
    assert "state: No operator matched block (0, 0)." in failure

    assert submission["supported"][0]["attempt_1"] == (
        expected_prediction().tolist()
    )

    assert submission["unsupported"][0]["attempt_1"] == [
        [5, 6],
        [7, 8],
    ]


def test_write_submission_round_trip(tmp_path):
    submission = {
        "example": [
            {
                "attempt_1": [[1, 2], [3, 4]],
                "attempt_2": [[1, 2], [3, 4]],
            }
        ]
    }

    path = tmp_path / "submission.json"
    written_path = write_submission(submission, path)

    assert written_path == path
    assert written_path.exists()

    with written_path.open(encoding="utf-8") as file:
        loaded = json.load(file)

    assert loaded == submission


def test_load_challenges_round_trip(tmp_path):
    challenges = {
        "example": supported_task(),
    }

    path = tmp_path / "challenges.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(challenges, file)

    loaded = load_challenges(path)

    assert loaded == challenges


def test_run_harness_reports_supported_and_fallback_counts(tmp_path):
    challenges = {
        "supported": supported_task(),
        "unsupported": unsupported_task(),
    }

    challenges_path = tmp_path / "challenges.json"
    submission_path = tmp_path / "submission.json"

    with challenges_path.open("w", encoding="utf-8") as file:
        json.dump(challenges, file)

    report = run_harness(
        challenges_path,
        submission_path,
    )

    assert report["submission_path"] == submission_path
    assert report["number_of_tasks"] == 2
    assert report["number_supported"] == 1
    assert report["number_fallback"] == 1
    assert set(report["failures"]) == {"unsupported"}
    assert submission_path.exists()

    with submission_path.open(encoding="utf-8") as file:
        submission = json.load(file)

    assert set(submission) == set(challenges)


def load_official_arc_task(task_id):
    repository_root = Path(__file__).resolve().parents[1]
    task_path = (
        repository_root
        / "tests"
        / "fixtures"
        / "arc-agi-2"
        / f"{task_id}.json"
    )

    assert task_path.exists(), (
        f"Official ARC integration fixture is missing: {task_path}"
    )

    with task_path.open(encoding="utf-8") as file:
        return json.load(file)


def assert_test_predictions_exact(task, predictions):
    assert len(predictions) == len(task["test"])

    for prediction, test_pair in zip(
        predictions,
        task["test"],
        strict=True,
    ):
        assert "output" in test_pair

        assert np.array_equal(
            prediction,
            np.asarray(test_pair["output"], dtype=int),
        )


def test_official_arc_task_00576224_fixed_layout_exact_match():
    task = load_official_arc_task("00576224")

    predictions = solve_fixed_layout_task(task)

    assert_test_predictions_exact(task, predictions)


def test_official_arc_task_007bbfb7_state_layout_exact_match():
    task = load_official_arc_task("007bbfb7")

    predictions = solve_state_layout_task(task)

    assert_test_predictions_exact(task, predictions)


def test_solver_dispatches_both_official_layout_families():
    for task_id in ("00576224", "007bbfb7"):
        task = load_official_arc_task(task_id)

        assert_test_predictions_exact(task, solve_task(task))
