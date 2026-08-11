import numpy as np
import matplotlib.pyplot as plt

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
    }

def get_valid_transformations(grid):
    known_transformations = get_known_transformations(grid)
    return [t['value'] for t in known_transformations.values() if t['valid']]

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

def get_instruction_layout(grid_height, grid_width, output, number_of_block_rows, number_of_block_columns, candidate_transformations):
    output_cells = get_output_cells(output)
    number_of_operators = candidate_transformations.shape[1]
    cell_matches = output_cells[:, 2:3] == candidate_transformations
    block_matches = cell_matches.reshape(number_of_block_rows, grid_height, number_of_block_columns, grid_width, number_of_operators).all(axis=(1,3))
    return np.argmax(block_matches, axis=2)

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