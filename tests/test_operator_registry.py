import numpy as np
import pytest

from featuregraph_research.operators import OperatorRecord


def test_operator_record_applies_named_shape_preserving_operator() -> None:
    operator = OperatorRecord(
        name="flip_horizontal",
        function=np.fliplr,
        preserves_shape=True,
    )
    values = np.array([[1, 2], [3, 4]])

    assert np.array_equal(operator.apply(values), [[2, 1], [4, 3]])


def test_operator_record_enforces_declared_shape_invariant() -> None:
    operator = OperatorRecord(
        name="flatten",
        function=np.ravel,
        preserves_shape=True,
    )

    with pytest.raises(ValueError, match="declared shape preservation"):
        operator.apply(np.zeros((2, 2)))
