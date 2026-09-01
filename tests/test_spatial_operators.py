from __future__ import annotations

import numpy as np
import pytest

from featuregraph.operators.spatial import (
    BLOCK_OPERATOR_NAMES,
    block_operators,
    candidate_grid,
    shape_valid,
)


def test_vocabulary_is_declared_in_canonical_order():
    names = [operator.name for operator in block_operators()]
    assert tuple(names) == BLOCK_OPERATOR_NAMES


def test_quarter_turns_are_shape_valid_only_on_square_grids():
    square = np.array([[1, 2], [3, 4]])
    rectangle = np.array([[1, 2, 3], [4, 5, 6]])
    operators = {operator.name: operator for operator in block_operators()}

    assert shape_valid(operators["rotate_90"], square)
    assert shape_valid(operators["rotate_270"], square)
    assert not shape_valid(operators["rotate_90"], rectangle)
    assert not shape_valid(operators["rotate_270"], rectangle)


def test_half_turn_and_flips_are_shape_valid_on_rectangles():
    rectangle = np.array([[1, 2, 3], [4, 5, 6]])
    operators = {operator.name: operator for operator in block_operators()}

    for name in ("copy", "flip_horizontal", "flip_vertical", "rotate_180"):
        assert shape_valid(operators[name], rectangle)


def test_candidate_grid_returns_none_when_shape_invalid():
    rectangle = np.array([[1, 2, 3], [4, 5, 6]])
    rotate_90 = next(
        operator
        for operator in block_operators()
        if operator.name == "rotate_90"
    )

    assert candidate_grid(rotate_90, rectangle) is None


def test_background_operator_uses_the_declared_background_color():
    grid = np.array([[1, 2], [3, 4]])

    zero_fill = next(
        operator for operator in block_operators() if operator.name == "background"
    )
    five_fill = next(
        operator
        for operator in block_operators(background_color=5)
        if operator.name == "background"
    )

    assert candidate_grid(zero_fill, grid).tolist() == [[0, 0], [0, 0]]
    assert candidate_grid(five_fill, grid).tolist() == [[5, 5], [5, 5]]


def test_background_color_is_validated():
    with pytest.raises(ValueError, match="background_color"):
        block_operators(background_color=10)


def test_flip_operators_produce_expected_grids():
    grid = np.array([[1, 2, 3], [4, 5, 6]])
    operators = {operator.name: operator for operator in block_operators()}

    assert candidate_grid(operators["flip_horizontal"], grid).tolist() == [
        [3, 2, 1],
        [6, 5, 4],
    ]
    assert candidate_grid(operators["flip_vertical"], grid).tolist() == [
        [4, 5, 6],
        [1, 2, 3],
    ]
    assert candidate_grid(operators["rotate_180"], grid).tolist() == [
        [6, 5, 4],
        [3, 2, 1],
    ]
