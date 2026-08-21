import matplotlib.pyplot as plt
import numpy as np
import pytest

from featuregraph import coordinate_arrays, plot_array_axes


def test_coordinate_arrays_make_cell_and_block_axes() -> None:
    axes = coordinate_arrays((4, 6), block_shape=(2, 3))

    assert set(axes) == {
        "row",
        "column",
        "block_row",
        "block_column",
        "within_block_row",
        "within_block_column",
    }
    assert all(values.shape == (4, 6) for values in axes.values())
    assert np.array_equal(axes["block_row"][:, 0], [0, 0, 1, 1])
    assert np.array_equal(axes["block_column"][0], [0, 0, 0, 1, 1, 1])
    assert np.array_equal(axes["within_block_row"][:, 0], [0, 1, 0, 1])
    assert np.array_equal(
        axes["within_block_column"][0],
        [0, 1, 2, 0, 1, 2],
    )


def test_coordinate_arrays_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="two positive dimensions"):
        coordinate_arrays((2, 0))


def test_plot_array_axes_returns_one_visible_panel_per_array() -> None:
    arrays = coordinate_arrays((2, 3))

    figure, axes = plot_array_axes(arrays)

    assert sum(axis.get_visible() for axis in axes.ravel()) == len(arrays)
    assert axes.ravel()[0].get_title() == "Row"
    plt.close(figure)


def test_plot_array_axes_requires_matching_two_dimensional_arrays() -> None:
    with pytest.raises(ValueError, match="same shape"):
        plot_array_axes({"a": np.zeros((2, 2)), "b": np.zeros((3, 2))})

    with pytest.raises(ValueError, match="two-dimensional"):
        plot_array_axes({"a": np.zeros(3)})
