"""Construct block-composition objects from grid cell observations.

A block-composition task presents an output grid that tiles an input grid: the
output is divided into blocks the size of the input, and each block is some
declared operator applied to that input. This module turns that structure into
the same four-stage representation the temporal behaviors use.

.. code-block:: text

    grid cell observations
        → per-cell operator match states
        → block boundaries and identities
        → one row per block object
        → computational queries

The object table records every block's full candidate set rather than
collapsing it to a single answer or raising on the first block that fails.
A block with no candidate is a measured property of the declared vocabulary,
not an error.

Resolving an instruction layout is then a grouped intersection over that table,
and the choice of grouping *is* the hypothesis class:

``by=("block_row", "block_column")``
    A layout fixed to block position, shared by every training pair.

``by=("block_state",)``
    A layout derived from the state of the input cell each block corresponds
    to, so the layout varies with the input.

Both read the same objects. Neither needs its own solver.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from featuregraph.behaviors.base import Behavior, Group, Signals
from featuregraph.behaviors.objects import BehaviorObjects
from featuregraph.operators.registry import OperatorRecord
from featuregraph.operators.spatial import block_operators, candidate_grid

__all__ = ["BlockComposition", "resolve_layout"]

_REQUIRED_COLUMNS = ("row", "column", "grid_role")

_DEFAULT_GROUP = ("task_id", "pair_type", "pair_index")


class BlockComposition(Behavior):
    """Construct one object per output block of a grid pair.

    Parameters
    ----------
    signals:
        Column holding the observed cell value. ``"color"`` for ARC-AGI-2
        observation tables.
    group:
        Columns that together identify a single input/output grid pair.
    background_color:
        Cell value treated as background, both for the constant-fill operator
        and for the ``block_state`` property.
    operators:
        Optional explicit vocabulary. Defaults to
        :func:`~featuregraph.operators.spatial.block_operators`.
    """

    def __init__(
        self,
        signals: Signals = "color",
        group: Group = _DEFAULT_GROUP,
        background_color: int = 0,
        operators: Sequence[OperatorRecord] | None = None,
    ) -> None:
        super().__init__(signals=signals, group=group)

        if len(self.signals) != 1:
            raise ValueError("BlockComposition accepts exactly one signal.")

        self.background_color = int(background_color)
        self.operators: tuple[OperatorRecord, ...] = tuple(
            operators
            if operators is not None
            else block_operators(self.background_color)
        )

        if not self.operators:
            raise ValueError("At least one operator is required.")

        self.declined_: pd.DataFrame = _empty_declined(self.group_columns)

    @property
    def signal(self) -> str:
        """Return the single configured observation column."""
        return self.signals[0]

    @property
    def operator_names(self) -> list[str]:
        """Return the declared operator names in vocabulary order."""
        return [operator.name for operator in self.operators]

    def validate(self, df: pd.DataFrame) -> None:
        """Validate observation, coordinate, and grouping columns."""
        super().validate(df)

        missing = [
            column for column in _REQUIRED_COLUMNS if column not in df.columns
        ]
        if missing:
            raise ValueError(f"Required columns are missing: {missing}")

    def add_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Project observations onto output cells aligned to input blocks.

        This is the one stage that changes the row level: the constructed
        signal is the correspondence between an output cell and the input cell
        it could have come from, which exists only on output cells. Pairs whose
        output does not tile their input are declined and recorded in
        :attr:`declined_` rather than raising.
        """
        signal = self.signal
        group_columns = self.group_columns
        frames: list[pd.DataFrame] = []
        declined: list[dict[str, Any]] = []

        for key, part in _iter_groups(df, group_columns):
            frame, reason = self._pair_frame(part, signal)

            if frame is None:
                declined.append({**_key_record(group_columns, key), "reason": reason})
                continue

            for column, value in _key_record(group_columns, key).items():
                frame[column] = value

            frames.append(frame)

        self.declined_ = (
            pd.DataFrame(declined)
            if declined
            else _empty_declined(group_columns)
        )

        if not frames:
            return _empty_cells(group_columns, signal, self.operator_names)

        return pd.concat(frames, ignore_index=True)

    def add_primitives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add one boolean match state per declared operator, per cell."""
        signal = self.signal

        for name in self.operator_names:
            # A shape-invalid operator leaves its candidate column empty, so
            # matching is evaluated only where the operator actually applied.
            valid = df[f"operator_valid_{name}"].astype(bool)
            matches = pd.Series(False, index=df.index, dtype=bool)

            if valid.any():
                candidate = pd.to_numeric(
                    df.loc[valid, f"candidate_{name}"],
                    errors="coerce",
                )
                matches.loc[valid] = (
                    candidate.eq(df.loc[valid, signal]).fillna(False)
                )

            df[f"matches_{name}"] = matches

        df["is_background"] = df["source_color"].eq(self.background_color)
        return df

    def add_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign one identifier to each output block."""
        df["block_id"] = (
            df["block_row"] * df["block_columns"] + df["block_column"]
        ).astype(int)
        return df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lift per-cell match states to whole-block match states."""
        object_group = [*self.group_columns, "block_id"]

        if df.empty:
            for name in self.operator_names:
                df[f"block_matches_{name}"] = pd.Series(dtype=bool)
            df["block_candidate_count"] = pd.Series(dtype=int)
            return df

        grouped = df.groupby(object_group, sort=False)

        block_match_columns = []
        for name in self.operator_names:
            column = f"block_matches_{name}"
            # A block matches an operator only if every one of its cells does.
            df[column] = grouped[f"matches_{name}"].transform("all")
            block_match_columns.append(column)

        df["block_candidate_count"] = (
            df[block_match_columns].sum(axis=1).astype(int)
        )
        return df

    def summarize(
        self,
        df: pd.DataFrame,
        signal: str | None = None,
    ) -> BehaviorObjects:
        """Return one row per block object."""
        signal = signal or self.signal
        if signal != self.signal:
            raise ValueError(
                f"Signal {signal!r} was not configured for this "
                "BlockComposition constructor."
            )

        object_group = [
            *self.group_columns,
            "block_id",
            "block_row",
            "block_column",
            "block_rows",
            "block_columns",
            "block_source_color",
            "block_state",
        ]

        names = self.operator_names
        aggregations = {
            f"block_matches_{name}": (f"block_matches_{name}", "first")
            for name in names
        }

        if df.empty:
            table = _empty_objects(self.group_columns)
        else:
            summary = (
                df.groupby(object_group, sort=False, dropna=False)
                .agg(cell_count=(signal, "size"), **aggregations)
                .reset_index()
            )

            match_columns = [f"block_matches_{name}" for name in names]
            matches = summary[match_columns].to_numpy(dtype=bool)
            vocabulary = np.array(names, dtype=object)

            summary["candidates"] = [
                frozenset(vocabulary[row]) for row in matches
            ]
            summary["candidate_count"] = matches.sum(axis=1).astype(int)
            summary["is_determined"] = summary["candidate_count"].eq(1)
            summary["is_unmatched"] = summary["candidate_count"].eq(0)
            summary["operator"] = [
                next(iter(candidates)) if len(candidates) == 1 else pd.NA
                for candidates in summary["candidates"]
            ]

            table = summary.drop(columns=match_columns)

        properties = (
            "block_id",
            "block_row",
            "block_column",
            "block_rows",
            "block_columns",
            "block_source_color",
            "block_state",
            "cell_count",
            "candidates",
            "candidate_count",
            "is_determined",
            "is_unmatched",
            "operator",
        )

        return BehaviorObjects(
            behavior_type="block_composition",
            signal=signal,
            table=table[[*self.group_columns, *properties]],
            features=df,
            group=tuple(self.group_columns),
            properties=properties,
            construction={
                "background_color": self.background_color,
                "operators": tuple(self.operator_names),
                "declined": self.declined_.to_dict("records"),
            },
        )

    def _pair_frame(
        self,
        part: pd.DataFrame,
        signal: str,
    ) -> tuple[pd.DataFrame | None, str | None]:
        """Build the output-cell correspondence table for one grid pair."""
        inputs = part[part["grid_role"] == "input"]
        outputs = part[part["grid_role"] == "output"]

        if inputs.empty:
            return None, "no_input_grid"
        if outputs.empty:
            # Held-out test pairs carry no output grid to describe.
            return None, "no_output_grid"

        try:
            grid = _grid_from_cells(inputs, signal)
            output = _grid_from_cells(outputs, signal)
        except ValueError as error:
            return None, str(error)

        grid_height, grid_width = grid.shape
        output_height, output_width = output.shape

        if output_height % grid_height or output_width % grid_width:
            return None, "output_not_block_multiple"

        block_rows = output_height // grid_height
        block_columns = output_width // grid_width

        row, column = np.indices(output.shape)
        within_row = row % grid_height
        within_column = column % grid_width
        block_row = row // grid_height
        block_column = column // grid_width

        frame = pd.DataFrame(
            {
                "row": row.ravel(),
                "column": column.ravel(),
                signal: output.ravel(),
                "block_row": block_row.ravel(),
                "block_column": block_column.ravel(),
                "within_block_row": within_row.ravel(),
                "within_block_column": within_column.ravel(),
                "block_rows": block_rows,
                "block_columns": block_columns,
            }
        )

        frame["source_color"] = grid[
            frame["within_block_row"], frame["within_block_column"]
        ]

        # A block layout shaped like the input grid lets every block be tied to
        # one input cell, which is what a state-derived layout groups on.
        if (block_rows, block_columns) == grid.shape:
            frame["block_source_color"] = grid[
                frame["block_row"], frame["block_column"]
            ]
            frame["block_state"] = np.where(
                frame["block_source_color"] == self.background_color,
                "background",
                "foreground",
            )
        else:
            frame["block_source_color"] = pd.NA
            frame["block_state"] = pd.NA

        for operator in self.operators:
            candidate = candidate_grid(operator, grid)
            valid_column = f"operator_valid_{operator.name}"
            candidate_column = f"candidate_{operator.name}"

            if candidate is None:
                # Shape invalid for this grid: distinguishable from a mismatch.
                frame[candidate_column] = pd.NA
                frame[valid_column] = False
            else:
                frame[candidate_column] = candidate[
                    frame["within_block_row"], frame["within_block_column"]
                ]
                frame[valid_column] = True

        return frame, None


def resolve_layout(
    objects: BehaviorObjects | pd.DataFrame,
    by: Sequence[str] = ("block_row", "block_column"),
    within: Sequence[str] = (),
) -> pd.DataFrame:
    """Intersect block candidate sets over a chosen grouping.

    The grouping selects the hypothesis class. Grouping by block coordinate
    asks for a layout fixed to position; grouping by ``block_state`` asks for a
    layout derived from the input. ``within`` names columns that must not be
    intersected across, such as ``("task_id",)``.
    """
    table = objects.table if isinstance(objects, BehaviorObjects) else objects
    keys = [*within, *by]

    missing = [column for column in keys if column not in table.columns]
    if missing:
        raise KeyError(f"Unknown object properties: {missing}")

    if table.empty:
        return pd.DataFrame(
            columns=[
                *keys,
                "candidates",
                "candidate_count",
                "is_determined",
                "operator",
            ]
        )

    resolved = (
        table.groupby(list(keys), sort=True, dropna=False)["candidates"]
        .agg(_intersect)
        .reset_index()
    )

    resolved["candidate_count"] = resolved["candidates"].map(len)
    resolved["is_determined"] = resolved["candidate_count"].eq(1)
    resolved["operator"] = [
        next(iter(candidates)) if len(candidates) == 1 else pd.NA
        for candidates in resolved["candidates"]
    ]
    return resolved


def _intersect(candidate_sets: Iterable[frozenset[str]]) -> frozenset[str]:
    """Return the intersection of every observed candidate set."""
    result: frozenset[str] | None = None
    for candidates in candidate_sets:
        result = (
            frozenset(candidates)
            if result is None
            else result & frozenset(candidates)
        )
    return result if result is not None else frozenset()


def _grid_from_cells(part: pd.DataFrame, signal: str) -> np.ndarray:
    """Rebuild a dense grid from its cell observations."""
    rows = part["row"].to_numpy(dtype=int)
    columns = part["column"].to_numpy(dtype=int)
    height = int(rows.max()) + 1
    width = int(columns.max()) + 1

    if len(part) != height * width:
        raise ValueError("incomplete_grid_observations")

    grid = np.zeros((height, width), dtype=int)
    grid[rows, columns] = part[signal].to_numpy(dtype=int)
    return grid


def _iter_groups(
    df: pd.DataFrame,
    group_columns: Sequence[str],
) -> Iterable[tuple[Any, pd.DataFrame]]:
    """Iterate grid pairs, tolerating an ungrouped single-pair frame."""
    if not group_columns:
        return [((), df)]
    return list(df.groupby(list(group_columns), sort=False))


def _key_record(
    group_columns: Sequence[str],
    key: Any,
) -> dict[str, Any]:
    """Pair group column names with one group key."""
    if not group_columns:
        return {}
    values = key if isinstance(key, tuple) else (key,)
    return dict(zip(group_columns, values, strict=True))


def _empty_declined(group_columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=[*group_columns, "reason"])


def _empty_cells(
    group_columns: Sequence[str],
    signal: str,
    names: Sequence[str],
) -> pd.DataFrame:
    columns = [
        *group_columns,
        "row",
        "column",
        signal,
        "block_row",
        "block_column",
        "within_block_row",
        "within_block_column",
        "block_rows",
        "block_columns",
        "source_color",
        "block_source_color",
        "block_state",
    ]
    for name in names:
        columns += [f"candidate_{name}", f"operator_valid_{name}"]
    return pd.DataFrame(columns=columns)


def _empty_objects(group_columns: Sequence[str]) -> pd.DataFrame:
    columns = [
        *group_columns,
        "block_id",
        "block_row",
        "block_column",
        "block_rows",
        "block_columns",
        "block_source_color",
        "block_state",
        "cell_count",
        "candidates",
        "candidate_count",
        "is_determined",
        "is_unmatched",
        "operator",
    ]
    return pd.DataFrame(columns=columns)
