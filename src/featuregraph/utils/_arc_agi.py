"""NumPy utilities for block-composition ARC-AGI tasks."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def fixed_layout_training_cycle(training_pairs):
    """Infer one block-coordinate-to-operator layout shared by all pairs."""
    _validate_training_pairs(training_pairs)
    shared_candidates = None

    for training_pair in training_pairs:
        _, pair_candidates = _get_pair_evidence(training_pair)
        if shared_candidates is None:
            shared_candidates = _copy_candidate_sets(pair_candidates)
            continue
        if pair_candidates.shape != shared_candidates.shape:
            raise ValueError(
                "Training pairs produced different instruction-layout shapes."
            )
        for index in np.ndindex(shared_candidates.shape):
            shared_candidates[index] &= pair_candidates[index]

    return _resolve_candidate_layout(shared_candidates)


def state_training_cycle(training_pairs, background_color=0):
    """Infer a state-to-operator mapping shared by all observations."""
    _validate_training_pairs(training_pairs)
    state_candidates = {"background": None, "foreground": None}

    for training_pair in training_pairs:
        states, pair_candidates = _get_pair_evidence(
            training_pair,
            background_color=background_color,
        )
        if states.shape != pair_candidates.shape:
            raise ValueError(
                "State-derived layouts require the output block layout "
                "to have the same shape as the input grid."
            )

        for state in state_candidates:
            state_mask = states == state
            for candidates in pair_candidates[state_mask]:
                if state_candidates[state] is None:
                    state_candidates[state] = set(candidates)
                else:
                    state_candidates[state] &= candidates

    state_to_operator = {}
    for state, candidates in state_candidates.items():
        if candidates is None:
            raise ValueError(
                f"Training pairs contain no observations for state {state!r}."
            )
        if len(candidates) == 0:
            raise ValueError(
                f"Training pairs have no shared operator for state {state!r}."
            )
        if len(candidates) > 1:
            raise ValueError(
                "Multiple operators remain after all training observations "
                f"for state {state!r}: {sorted(candidates)}"
            )
        state_to_operator[state] = next(iter(candidates))

    return state_to_operator


def training_cycle(training_pairs):
    """Backward-compatible name for fixed-layout inference."""
    return fixed_layout_training_cycle(training_pairs)


def derive_state_instruction_layout(
    grid,
    state_to_operator,
    background_color=0,
):
    """Derive an input-specific layout from background/foreground states."""
    missing_states = (
        {"background", "foreground"} - set(state_to_operator)
    )
    if missing_states:
        raise ValueError(
            "State-to-operator mapping is missing states: "
            f"{sorted(missing_states)}"
        )

    states = np.where(
        np.asarray(grid) == background_color,
        "background",
        "foreground",
    )
    return np.where(
        states == "background",
        state_to_operator["background"],
        state_to_operator["foreground"],
    )


def test_cycle(test_grid, instruction_layout, background_color=0):
    """Compose a prediction from a concrete block-operator layout."""
    grid = np.asarray(test_grid)
    instruction_layout = np.asarray(instruction_layout)
    grid_height, grid_width = grid.shape
    predicted_shape = (
        instruction_layout.shape[0] * grid_height,
        instruction_layout.shape[1] * grid_width,
    )
    block_coordinates = get_block_coordinates(grid.shape, predicted_shape)
    valid_transformations = get_valid_transformations(
        grid,
        background_color=background_color,
    )
    name_to_column = {
        transformation["name"]: column
        for column, transformation in enumerate(valid_transformations)
    }

    missing_names = set(instruction_layout.ravel()) - set(name_to_column)
    if missing_names:
        raise ValueError(
            f"Test grid does not support operators: {sorted(missing_names)}"
        )

    candidate_transformations = np.array([
        get_candidate_transformations(
            transformation["value"],
            block_coordinates,
        )
        for transformation in valid_transformations
    ]).T
    numeric_instruction_layout = np.array([
        [name_to_column[name] for name in row]
        for row in instruction_layout
    ])
    cell_indices = np.arange(candidate_transformations.shape[0])
    cell_instructions = numeric_instruction_layout[
        block_coordinates[:, 0],
        block_coordinates[:, 1],
    ]
    return candidate_transformations[
        cell_indices,
        cell_instructions,
    ].reshape(predicted_shape)


def solve_fixed_layout_task(task):
    training_pairs = get_training_pairs(task)
    instruction_layout = fixed_layout_training_cycle(training_pairs)
    return [
        test_cycle(test_grid, instruction_layout)
        for test_grid in get_test_grids(task)
    ]


def solve_state_layout_task(task, background_color=0):
    training_pairs = get_training_pairs(task)
    state_to_operator = state_training_cycle(
        training_pairs,
        background_color=background_color,
    )
    predictions = []
    for test_grid in get_test_grids(task):
        instruction_layout = derive_state_instruction_layout(
            test_grid,
            state_to_operator,
            background_color=background_color,
        )
        predictions.append(
            test_cycle(
                test_grid,
                instruction_layout,
                background_color=background_color,
            )
        )
    return predictions


def solve_task(task, background_color=0):
    """Try the supported solver families in deterministic order."""
    failures = []
    solvers = (
        ("fixed", lambda: solve_fixed_layout_task(task)),
        (
            "state",
            lambda: solve_state_layout_task(
                task,
                background_color=background_color,
            ),
        ),
    )
    for solver_name, solve in solvers:
        try:
            return solve()
        except ValueError as error:
            failures.append(f"{solver_name}: {error}")
    raise ValueError(
        "No solver family matched the task. " + " | ".join(failures)
    )


def _get_pair_evidence(training_pair, background_color=0):
    grid = np.asarray(training_pair["grid"])
    output = np.asarray(training_pair["output"])
    grid_height, grid_width = grid.shape
    output_height, output_width = output.shape
    if output_height % grid_height or output_width % grid_width:
        raise ValueError("Output cannot be divided into input-shaped blocks.")

    block_rows = output_height // grid_height
    block_columns = output_width // grid_width
    block_coordinates = get_block_coordinates(grid.shape, output.shape)
    valid_transformations = get_valid_transformations(
        grid,
        background_color=background_color,
    )
    operator_names = [item["name"] for item in valid_transformations]
    candidate_transformations = np.array([
        get_candidate_transformations(item["value"], block_coordinates)
        for item in valid_transformations
    ]).T
    pair_candidates = get_instruction_candidates(
        grid_height,
        grid_width,
        output,
        block_rows,
        block_columns,
        candidate_transformations,
        operator_names,
    )
    states = np.where(
        grid == background_color,
        "background",
        "foreground",
    )
    return states, pair_candidates


def _copy_candidate_sets(candidate_layout):
    copied = np.empty(candidate_layout.shape, dtype=object)
    for index in np.ndindex(candidate_layout.shape):
        copied[index] = set(candidate_layout[index])
    return copied


def _resolve_candidate_layout(shared_candidates):
    instruction_layout = np.empty(shared_candidates.shape, dtype=object)
    for index in np.ndindex(shared_candidates.shape):
        candidates = shared_candidates[index]
        if len(candidates) == 0:
            raise ValueError(
                f"Training pairs have no shared operator for block {index}."
            )
        if len(candidates) > 1:
            raise ValueError(
                "Multiple operators remain after all training pairs "
                f"for block {index}: {sorted(candidates)}"
            )
        instruction_layout[index] = next(iter(candidates))
    return instruction_layout


def _validate_training_pairs(training_pairs):
    if len(training_pairs) == 0:
        raise ValueError("Training cycle requires at least one pair.")


def get_known_transformations(grid, background_color=0):
    transformations = {
        "copy": grid,
        "flip_horizontal": np.fliplr(grid),
        "flip_vertical": np.flipud(grid),
        "rotate_90": np.rot90(grid, k=1),
        "rotate_180": np.rot90(grid, k=2),
        "rotate_270": np.rot90(grid, k=3),
        "background": np.full_like(grid, background_color),
    }
    return {
        name: {
            "value": value,
            "shape": value.shape,
            "valid": value.shape == grid.shape,
        }
        for name, value in transformations.items()
    }


def get_valid_transformations(grid, background_color=0):
    return [
        {"name": name, **transformation}
        for name, transformation in get_known_transformations(
            grid,
            background_color=background_color,
        ).items()
        if transformation["valid"]
    ]


def get_block_coordinates(block_shape, output_shape):
    grid_height, grid_width = block_shape
    row, column = np.indices(output_shape)
    return np.array([
        row.ravel() // grid_height,
        column.ravel() // grid_width,
        row.ravel() % grid_height,
        column.ravel() % grid_width,
    ]).T


def get_output_cells(output):
    row, column = np.indices(output.shape)
    return np.array([row.ravel(), column.ravel(), output.ravel()]).T


def get_candidate_transformations(grid, block_coordinates):
    return grid[block_coordinates[:, 2], block_coordinates[:, 3]]


def get_instruction_candidates(
    grid_height,
    grid_width,
    output,
    number_of_block_rows,
    number_of_block_columns,
    candidate_transformations,
    operator_names,
):
    output_cells = get_output_cells(output)
    number_of_operators = candidate_transformations.shape[1]
    cell_matches = output_cells[:, 2:3] == candidate_transformations
    block_matches = cell_matches.reshape(
        number_of_block_rows,
        grid_height,
        number_of_block_columns,
        grid_width,
        number_of_operators,
    ).all(axis=(1, 3))
    instruction_candidates = np.empty(
        (number_of_block_rows, number_of_block_columns),
        dtype=object,
    )

    for block_row in range(number_of_block_rows):
        for block_column in range(number_of_block_columns):
            matching_positions = np.flatnonzero(
                block_matches[block_row, block_column]
            )
            matching_names = {
                operator_names[position]
                for position in matching_positions
            }
            if not matching_names:
                raise ValueError(
                    "No operator matched block "
                    f"({block_row}, {block_column})."
                )
            instruction_candidates[block_row, block_column] = matching_names
    return instruction_candidates


def plot_arc_agi(grid, output=None, manual_output=None):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(grid, vmin=0, vmax=9)
    axes[0].set_title("grid object")
    if output is not None:
        axes[1].imshow(output, vmin=0, vmax=9)
        axes[1].set_title("Expected output")
    if manual_output is not None:
        axes[2].imshow(manual_output, vmin=0, vmax=9)
        axes[2].set_title("Reconstructed output")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    plt.tight_layout()
    plt.show()


def get_training_pairs(task):
    return [
        {
            "grid": np.asarray(pair["input"], dtype=int),
            "output": np.asarray(pair["output"], dtype=int),
        }
        for pair in task["train"]
    ]


def get_test_grids(task):
    return [
        np.asarray(pair["input"], dtype=int)
        for pair in task["test"]
    ]


def predictions_to_lists(predictions):
    return [prediction.tolist() for prediction in predictions]


def predictions_to_attempts(predictions):
    return [
        {
            "attempt_1": prediction.tolist(),
            "attempt_2": prediction.tolist(),
        }
        for prediction in predictions
    ]


def fallback_predictions(task):
    return get_test_grids(task)


def solve_challenges(challenges):
    submission = {}
    failures = {}
    for task_id, task in challenges.items():
        try:
            predictions = solve_task(task)
        except ValueError as error:
            predictions = fallback_predictions(task)
            failures[task_id] = str(error)
        submission[task_id] = predictions_to_attempts(predictions)
    return submission, failures


def load_challenges(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def write_submission(submission, path):
    path = Path(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(submission, file)
    return path


def run_harness(challenges_path, submission_path):
    challenges = load_challenges(challenges_path)
    submission, failures = solve_challenges(challenges)
    written_path = write_submission(submission, submission_path)
    return {
        "submission_path": written_path,
        "number_of_tasks": len(challenges),
        "number_supported": len(challenges) - len(failures),
        "number_fallback": len(failures),
        "failures": failures,
    }