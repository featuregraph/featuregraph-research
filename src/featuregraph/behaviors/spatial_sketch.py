# spatial.py

import numpy as np
import pandas as pd

def partition_blocks(object_grid, output_grid, background_color):

    reflected_object = np.fliplr(object_grid)
    object_height, object_width = object_grid.shape
    output_height, output_width = output_grid.shape

    assert output_height % object_height == 0
    assert output_width % object_width == 0

    number_of_block_rows = output_height // object_height
    number_of_block_columns = output_width // object_width

    row, column = np.indices(output_grid.shape)

    cells = pd.DataFrame({
        "row": row.ravel(),
        "column": column.ravel(),
        "value": output_grid.ravel(),
    })

    # Identify the block containing each output cell
    cells["block_row"] = cells["row"] // object_height
    cells["block_column"] = cells["column"] // object_width

    cells["block_id"] = (
        cells["block_row"] * number_of_block_columns
        + cells["block_column"]
    )

    # Coordinate of each cell within its block
    cells["within_block_row"] = cells["row"] % object_height
    cells["within_block_column"] = cells["column"] % object_width

    cells["original_value"] = object_grid[
    cells["within_block_row"],
    cells["within_block_column"],
    ]

    cells["reflected_value"] = reflected_object[
        cells["within_block_row"],
        cells["within_block_column"],
    ]

    cells["matches_original"] = (
        cells["value"] == cells["original_value"]
    )

    cells["matches_reflected"] = (
        cells["value"] == cells["reflected_value"]
    )

    cells["matches_background"] = cells["value"].eq(background_color)
    # cells['rotated'] = ?

    return cells

def compare_blocks(cells):
    blocks = (
        cells
        .groupby(
            ["block_id", "block_row", "block_column"],
            as_index=False,
        )
        .agg(
            equals_original=("matches_original", "all"),
            equals_reflected=("matches_reflected", "all"),
            equals_background=('matches_background', 'all'),
            cell_count=("value", "size"),
        )
    )

    blocks["instruction"] = np.select(
        [
            blocks["equals_original"],
            blocks["equals_reflected"],
            blocks['equals_background']
        ],
        [
            "copy",
            "reflect_horizontal",
            'fill_background'
        ],
        default="unknown",
    )

    return blocks

def build_instruction_layout(blocks, background_color):
    layout = blocks.pivot(
    index="block_row",
    columns="block_column",
    values="instruction",
    )

    operators = {
        "copy": lambda grid: grid.copy(),
        "reflect_horizontal": np.fliplr,
        "fill_background": lambda grid: np.full_like(grid, background_color)
        # 'rotate': # rotate 90, 180, 270
    }

    if blocks["instruction"].eq("unknown").any():
        raise ValueError("At least one block does not match a known operator")

    return layout, operators


def block_compose(layout, object_grid, operators):
    manual_output = np.block([
    [
        operators[layout.loc[block_row, block_column]](
            object_grid
        )
        for block_column in layout.columns
    ]
    for block_row in layout.index
    ])

    return manual_output

def exact_match(manual_output, output_grid):
    exact_match = np.array_equal(manual_output, output_grid)
    return exact_match