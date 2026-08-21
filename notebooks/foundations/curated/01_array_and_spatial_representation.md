# Array and Spatial Representation

## Question

How can a two-dimensional array carry both observed values and an explicit
coordinate system without converting the computation to a DataFrame?

## Contract

- **Input:** a two-dimensional shape and, optionally, a block shape.
- **Operator:** construct semantic coordinate arrays from integer indices.
- **Output:** named arrays for row, column, block row, block column, and
  within-block position.
- **Invariant:** every coordinate array has exactly the same shape as the
  represented grid.
- **Validation:** quotient and remainder coordinates reconstruct each cell's
  block membership and local position.

## Representation

For an output divided into input-shaped blocks, each output cell has two
coordinate systems:

```text
global position:         row, column
block-relative position: block_row, block_column,
                         within_block_row, within_block_column
```

The coordinate arrays play the role that helper columns play in a DataFrame.
They contain indexing information rather than observed values, but remain
ordinary NumPy arrays that can participate in masking, reshaping, and advanced
indexing.

```python
from featuregraph import coordinate_arrays, plot_array_axes

axes = coordinate_arrays((6, 9), block_shape=(2, 3))
figure, panels = plot_array_axes(axes)
```

The plotting function is deliberately a renderer. Computation remains in
NumPy; Matplotlib is introduced only to make each semantic axis inspectable.

## Spatial operators

Copy, horizontal reflection, vertical reflection, and rotation are candidate
operators over a grid representation. An operator may enter a candidate set
only when its output satisfies the required shape contract.

For block composition, the instruction layout selects one named operator for
each block position. Coordinate arrays expand that block-level instruction to
the cells belonging to the block. The selected candidate cell values can then
be reshaped into the composed output.

## Connection

This representation is used by the ARC solver to keep four concerns separate:

```text
observed grid values
coordinate and block indices
candidate operator values
instruction layout
```

That separation prevents the known output from leaking into prediction and
makes the inferred instruction layout transferable to an unknown test grid.

The promoted ideas came from the archived array plotting, reflection,
rotation, Kronecker-product, neighborhood, and NumPy workshop records.
