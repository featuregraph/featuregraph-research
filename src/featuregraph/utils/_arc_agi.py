import numpy as np

def get_block_coordinates(block_shape, output_shape):
    input_height, input_width = block_shape
    row, column = np.indices(output_shape) 
    
    block_row_indices = row.ravel() // input_height
    block_col_indices = column.ravel() // input_width
    within_block_row_indices = row.ravel() % input_height
    within_block_col_indices = column.ravel() % input_width
    
    return np.array([block_row_indices, block_col_indices, within_block_row_indices, within_block_col_indices]).T

def get_output_cells(output):
    row, column = np.indices(output.shape) 
    return np.array([row.ravel(), column.ravel(), output.ravel()]).T

def get_candidate_transformations(input, block_coordinates):
    return input[block_coordinates[:,2], block_coordinates[:,3]]

def get_instruction_layout(input_height, input_width, output, number_of_block_rows, number_of_block_columns, candidate_transformations):
    output_cells = get_output_cells(output)
    number_of_operators = candidate_transformations.shape[1]
    cell_matches = output_cells[:, 2:3] == candidate_transformations
    block_matches = cell_matches.reshape(number_of_block_rows, input_height, number_of_block_columns, input_width, number_of_operators).all(axis=(1,3))
    return np.argmax(block_matches, axis=2)
