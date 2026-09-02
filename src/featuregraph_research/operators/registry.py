"""Typed provenance records for deterministic operators."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OperatorRecord:
    """Describe a named array operator and its representation contract."""

    name: str
    function: Callable[[np.ndarray], np.ndarray]
    input_kind: str = "array"
    output_kind: str = "array"
    preserves_shape: bool = False

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Apply the operator and enforce its declared shape invariant."""
        source = np.asarray(values)
        result = np.asarray(self.function(source))
        if self.preserves_shape and result.shape != source.shape:
            raise ValueError(
                f"Operator {self.name!r} declared shape preservation but "
                f"changed {source.shape} to {result.shape}."
            )
        return result
