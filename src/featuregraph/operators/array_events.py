"""NumPy transition operators for one-dimensional state arrays."""

import numpy as np


def enter_mask(state: np.ndarray) -> np.ndarray:
    """Mark false-to-true transitions without changing array length."""
    values = np.asarray(state, dtype=bool)
    if values.ndim != 1:
        raise ValueError("state must be one-dimensional")
    if values.size == 0:
        return values.copy()
    return np.concatenate(([values[0]], ~values[:-1] & values[1:]))


def exit_mask(state: np.ndarray) -> np.ndarray:
    """Mark true-to-false transitions without changing array length."""
    values = np.asarray(state, dtype=bool)
    if values.ndim != 1:
        raise ValueError("state must be one-dimensional")
    if values.size == 0:
        return values.copy()
    return np.concatenate(([False], values[:-1] & ~values[1:]))


def between_masks(
    enter: np.ndarray,
    exit: np.ndarray,
) -> np.ndarray:
    """Materialize the intervals opened by enter and closed by exit events."""
    enters = np.asarray(enter, dtype=bool)
    exits = np.asarray(exit, dtype=bool)
    if enters.ndim != 1 or exits.ndim != 1:
        raise ValueError("enter and exit must be one-dimensional")
    if enters.shape != exits.shape:
        raise ValueError("enter and exit must have the same shape")
    return np.cumsum(enters) - np.cumsum(exits) > 0
