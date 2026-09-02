import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def training_cycle(training_pairs):
    if len(training_pairs) == 0:
        raise ValueError(
            "Training cycle requires at least one pair."
        )

    shared_candidates = None

    state_stack = []
    candidate_stack = []

    for training_pair in training_pairs:
        grid = training_pair['grid']
        output = training_pair['output']

        grid_height, grid_width = grid.shape
        output_height, output_width = output.shape

        number_of_block_rows = output_height // grid_height
        number_of_block_cols = output_width // grid_width
        row, col = np.indices(output.shape)

        block_row_indices = row.ravel() // grid_height
        block_col_indices = col.ravel() // grid_width
        within_block_row_indices = row.ravel() % grid_height
        within_block_col_indices = col.ravel() % grid_width

        block_coordinates = np.array([block_row_indices, block_col_indices, within_block_row_indices, within_block_col_indices]).T

        valid_transformations = get_valid_transformations(grid)

        operator_names = [
            transformation["name"]
            for transformation in valid_transformations
        ]

        number_of_operators = len(operator_names)

        candidate_transformations = np.array([get_candidate_transformations(transformation["value"], block_coordinates) for transformation in valid_transformations]).T
        output_cells = np.array([row.ravel(), col.ravel(), output.ravel()]).T
        states = np.where(grid == 0, 'background', 'foreground')
        candidates = (output_cells[:,2:3] == candidate_transformations).reshape(number_of_block_rows, grid_height, number_of_block_cols, grid_width, number_of_operators).all(axis=(1, 3))

        state_stack.append(states)
        candidate_stack.append(candidates)

    state_array = np.array(state_stack)
    candidate_array = np.array(candidate_stack)

    flat_states = state_array.reshape(-1)
    flat_candidates = candidate_array.reshape(-1, 7)
    background_mask = (flat_states == 'background')
    foreground_mask = (flat_states == 'foreground')

    background_survivors = (
        flat_candidates[background_mask]
        .all(axis=0)
    )

    foreground_survivors = (
        flat_candidates[foreground_mask]
        .all(axis=0)
    )

    operator_names = np.asarray(operator_names)

    background_operator = operator_names[
        background_survivors
    ]

    foreground_operator = operator_names[
        foreground_survivors
    ]

    background_operator = background_operator.item()
    foreground_operator = foreground_operator.item()

    state_to_operator =  {
        "background": background_operator,
        "foreground": foreground_operator,
    }

    instruction_layout = np.empty(
        shared_candidates.shape,
        dtype=object,
    )

    for index in np.ndindex(shared_candidates.shape):
        candidates = shared_candidates[index]
        instruction_layout[index] = next(iter(candidates))

    return instruction_layout, state_to_operator

def test_cycle(test_grid, instruction_layout):
    grid = test_grid

    grid_height, grid_width = grid.shape

    predicted_height = instruction_layout.shape[0] * grid_height
    predicted_width = instruction_layout.shape[1] * grid_width
    predicted_shape = (predicted_height, predicted_width)

    block_coordinates = get_block_coordinates(
        grid.shape,
        predicted_shape,
    )

    valid_transformations = get_valid_transformations(grid)

    name_to_column = {
        transformation["name"]: column
        for column, transformation in enumerate(valid_transformations)
    }

    required_names = set(instruction_layout.ravel())
    available_names = set(name_to_column)
    missing_names = required_names - available_names

    if missing_names:
        raise ValueError(
            f"Test grid does not support operators: "
            f"{sorted(missing_names)}"
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

    prediction = candidate_transformations[
        cell_indices,
        cell_instructions,
    ].reshape(predicted_shape)

    return prediction

def get_known_transformations(grid):
    return {
        'copy': {
            'value': grid,
            'shape': grid.shape,
            'valid': grid.shape == grid.shape
        },
        'flip_horizontal': {
            'value': np.fliplr(grid),
            'shape': np.fliplr(grid).shape,
            'valid': np.fliplr(grid).shape == grid.shape
        },
        "flip_vertical": {
            "value": np.flipud(grid),
            "shape": np.flipud(grid).shape,
            "valid": np.flipud(grid).shape == grid.shape,
        },
        "rotate_90": {
            "value": np.rot90(grid, k=1),
            "shape": np.rot90(grid, k=1).shape,
            "valid": np.rot90(grid, k=1).shape == grid.shape,
        },
        "rotate_180": {
            "value": np.rot90(grid, k=2),
            "shape": np.rot90(grid, k=2).shape,
            "valid": np.rot90(grid, k=2).shape == grid.shape,
        },
        "rotate_270": {
            "value": np.rot90(grid, k=3),
            "shape": np.rot90(grid, k=3).shape,
            "valid": np.rot90(grid, k=3).shape == grid.shape,
        },
        'background': {
            'value': np.full_like(grid, 0),
            'shape': grid.shape,
            'valid': grid.shape == grid.shape
        }
    }

def get_valid_transformations(grid):
    known_transformations = get_known_transformations(grid)

    return [
        {"name": name, **transformation}
        for name, transformation in known_transformations.items()
        if transformation["valid"]
    ]

def get_block_coordinates(block_shape, output_shape):
    grid_height, grid_width = block_shape
    row, column = np.indices(output_shape) 
    
    block_row_indices = row.ravel() // grid_height
    block_col_indices = column.ravel() // grid_width
    within_block_row_indices = row.ravel() % grid_height
    within_block_col_indices = column.ravel() % grid_width
    
    return np.array([block_row_indices, block_col_indices, within_block_row_indices, within_block_col_indices]).T

def get_output_cells(output):
    row, column = np.indices(output.shape) 
    return np.array([row.ravel(), column.ravel(), output.ravel()]).T

def get_candidate_transformations(grid, block_coordinates):
    return grid[block_coordinates[:,2], block_coordinates[:,3]]

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

    cell_matches = (
        output_cells[:, 2:3] == candidate_transformations
    )

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

            instruction_candidates[
                block_row,
                block_column,
            ] = matching_names

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

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

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

def solve_task(task):
    training_pairs = get_training_pairs(task)
    instruction_layout = training_cycle(training_pairs)

    test_grids = get_test_grids(task)

    return [
        test_cycle(test_grid, instruction_layout)
        for test_grid in test_grids
    ]

def predictions_to_lists(predictions):
    return [
        prediction.tolist()
        for prediction in predictions
    ]

def predictions_to_attempts(predictions):
    return [
        {
            "attempt_1": prediction.tolist(),
            "attempt_2": prediction.tolist(),
        }
        for prediction in predictions
    ]

def fallback_predictions(task):
    return [
        np.asarray(pair["input"], dtype=int)
        for pair in task["test"]
    ]

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

def write_submission(submission, path):
    path = Path(path)

    with path.open("w", encoding="utf-8") as file:
        json.dump(submission, file)

    return path

def load_challenges(path):
    path = Path(path)

    with path.open(encoding="utf-8") as file:
        return json.load(file)

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