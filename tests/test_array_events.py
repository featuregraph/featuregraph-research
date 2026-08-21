import numpy as np
import pytest

from featuregraph.operators import between_masks, enter_mask, exit_mask


def test_numpy_enter_exit_and_between_masks() -> None:
    state = np.array([False, True, True, False, True, False])

    enters = enter_mask(state)
    exits = exit_mask(state)

    assert np.array_equal(
        enters,
        [False, True, False, False, True, False],
    )
    assert np.array_equal(
        exits,
        [False, False, False, True, False, True],
    )
    assert np.array_equal(
        between_masks(enters, exits),
        [False, True, True, False, True, False],
    )


def test_numpy_event_operators_validate_dimensions_and_alignment() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        enter_mask(np.zeros((2, 2)))

    with pytest.raises(ValueError, match="same shape"):
        between_masks(np.zeros(2), np.zeros(3))


def test_enter_mask_reconstructs_state_that_starts_active() -> None:
    state = np.array([True, True, False])

    assert np.array_equal(
        between_masks(enter_mask(state), exit_mask(state)),
        state,
    )


def test_event_operators_accept_an_empty_state() -> None:
    state = np.array([], dtype=bool)

    assert enter_mask(state).shape == (0,)
    assert exit_mask(state).shape == (0,)
