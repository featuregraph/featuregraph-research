"""Construct region objects from grid cell observations.

A region is a connected group of cells that belong together under a declared
rule. This module builds one object per region through the same four-stage
construction the temporal behaviors use:

.. code-block:: text

    grid cell observations
        → background and foreground states
        → connected-component identities
        → one row per region object
        → computational queries

Unlike :mod:`~featuregraph.behaviors.composition`, this construction describes
a single grid. It does not need a corresponding output grid, does not need the
grid to be part of a task, and does not assume any relationship between grids.
That makes it applicable to every grid in a dataset rather than to the subset
whose structure a particular hypothesis happens to fit.

What holds a region together is declared, not assumed. Connectivity, the
background colour, whether a region is a uniform-colour component or a
non-background component, and whether background cells form objects of their
own are all construction parameters, recorded on the result in the same way
``Oscillation`` records its smoothing window and difference lag.

:func:`edit_alignment` measures whether the differences between two grids
respect these boundaries. It answers whether regions are the right object
boundary for a transformation, which is a separate question from whether the
transformation itself can be predicted.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage

from featuregraph.behaviors.base import Behavior, Group, Signals
from featuregraph.behaviors.objects import BehaviorObjects
from featuregraph.operators.grids import (
    CONNECTIVITY,
    REGION_DEFINITIONS,
    grid_from_cells,
    hole_count,
    label_regions,
)

__all__ = ["RegionObjects", "edit_alignment"]

_REQUIRED_COLUMNS = ("row", "column")

_DEFAULT_GROUP = ("task_id", "pair_type", "pair_index", "grid_role")

_PAIR_GROUP = ("task_id", "pair_type", "pair_index")


class RegionObjects(Behavior):
    """Construct one object per connected region of a grid.

    Parameters
    ----------
    signals:
        Column holding the observed cell value.
    group:
        Columns that together identify a single grid. For ARC-AGI-2 observation
        tables this must include ``grid_role``, since an input and its output
        are different grids.
    background_color:
        Cell value treated as background.
    connectivity:
        ``4`` or ``8``. Whether diagonal neighbours join a region.
    definition:
        ``"uniform_color"`` groups adjacent cells of the same colour;
        ``"foreground_mask"`` groups adjacent non-background cells whatever
        their colours.
    include_background:
        Whether background cells form regions of their own. When ``True`` the
        regions partition the grid, which is what :func:`edit_alignment`
        requires.
    """

    def __init__(
        self,
        signals: Signals = "color",
        group: Group = _DEFAULT_GROUP,
        background_color: int = 0,
        connectivity: int = 4,
        definition: str = "uniform_color",
        include_background: bool = True,
    ) -> None:
        super().__init__(signals=signals, group=group)

        if len(self.signals) != 1:
            raise ValueError("RegionObjects accepts exactly one signal.")
        if connectivity not in CONNECTIVITY:
            raise ValueError(f"connectivity must be one of: {sorted(CONNECTIVITY)}")
        if definition not in REGION_DEFINITIONS:
            raise ValueError(f"definition must be one of: {list(REGION_DEFINITIONS)}")

        self.background_color = int(background_color)
        self.connectivity = int(connectivity)
        self.definition = definition
        self.include_background = bool(include_background)
        self.declined_: pd.DataFrame = _empty_declined(self.group_columns)

    @property
    def signal(self) -> str:
        """Return the single configured observation column."""
        return self.signals[0]

    def validate(self, df: pd.DataFrame) -> None:
        """Validate observation, coordinate, and grouping columns."""
        super().validate(df)

        missing = [
            column for column in _REQUIRED_COLUMNS if column not in df.columns
        ]
        if missing:
            raise ValueError(f"Required columns are missing: {missing}")

    def add_primitives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the background state each region rule is defined against."""
        df["is_background"] = df[self.signal].eq(self.background_color)
        return df

    def add_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign one identifier to each connected region.

        Grids whose observations do not form a complete rectangle are declined
        and recorded in :attr:`declined_` rather than raising.
        """
        signal = self.signal
        group_columns = self.group_columns
        labelled: list[pd.DataFrame] = []
        declined: list[dict[str, Any]] = []

        for key, part in _iter_groups(df, group_columns):
            try:
                grid = grid_from_cells(part, signal)
            except ValueError as error:
                declined.append(
                    {**_key_record(group_columns, key), "reason": str(error)}
                )
                continue

            labels = label_regions(
                grid,
                background_color=self.background_color,
                connectivity=self.connectivity,
                definition=self.definition,
                include_background=self.include_background,
            )

            part = part.copy()
            part["region_id"] = labels[
                part["row"].to_numpy(dtype=int),
                part["column"].to_numpy(dtype=int),
            ]
            # Recorded before excluded cells are dropped, so border contact is
            # measured against the real grid rather than the surviving cells.
            part["grid_max_row"] = grid.shape[0] - 1
            part["grid_max_column"] = grid.shape[1] - 1
            labelled.append(part)

        self.declined_ = (
            pd.DataFrame(declined) if declined else _empty_declined(group_columns)
        )

        if not labelled:
            # Every grid was declined, so no cell belongs to a region.
            empty = df.iloc[0:0].copy()
            for column in ("region_id", "grid_max_row", "grid_max_column"):
                empty[column] = pd.Series(dtype=int)
            return empty

        result = pd.concat(labelled, ignore_index=True)
        # Label 0 marks a cell excluded from every region.
        return result[result["region_id"] > 0].reset_index(drop=True)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add row-aligned region measurements."""
        object_group = [*self.group_columns, "region_id"]

        if df.empty:
            for column in _FEATURE_COLUMNS:
                df[column] = pd.Series(dtype=float)
            return df

        grouped = df.groupby(object_group, sort=False)

        df["region_size"] = grouped["row"].transform("size")
        df["region_min_row"] = grouped["row"].transform("min")
        df["region_max_row"] = grouped["row"].transform("max")
        df["region_min_column"] = grouped["column"].transform("min")
        df["region_max_column"] = grouped["column"].transform("max")
        df["region_centroid_row"] = grouped["row"].transform("mean")
        df["region_centroid_column"] = grouped["column"].transform("mean")

        df["region_height"] = df["region_max_row"] - df["region_min_row"] + 1
        df["region_width"] = df["region_max_column"] - df["region_min_column"] + 1
        df["region_bbox_area"] = df["region_height"] * df["region_width"]
        # How completely the region fills its own bounding box.
        df["region_fill_ratio"] = df["region_size"] / df["region_bbox_area"]

        # Counted per region rather than through a grouped apply, which keeps
        # the construction working across the supported pandas range.
        counts = [
            (
                *(key if isinstance(key, tuple) else (key,)),
                hole_count(part["row"], part["column"]),
            )
            for key, part in df.groupby(object_group, sort=False)
        ]
        holes = pd.DataFrame(counts, columns=[*object_group, "region_hole_count"])

        return df.merge(holes, on=object_group, how="left")

    def summarize(
        self,
        df: pd.DataFrame,
        signal: str | None = None,
    ) -> BehaviorObjects:
        """Return one row per region object."""
        signal = signal or self.signal
        if signal != self.signal:
            raise ValueError(
                f"Signal {signal!r} was not configured for this "
                "RegionObjects constructor."
            )

        object_group = [*self.group_columns, "region_id"]
        properties = (
            "region_id",
            "color",
            "is_background",
            "size",
            "min_row",
            "max_row",
            "min_column",
            "max_column",
            "height",
            "width",
            "bbox_area",
            "fill_ratio",
            "hole_count",
            "centroid_row",
            "centroid_column",
            "touches_border",
        )

        if df.empty:
            table = pd.DataFrame(columns=[*self.group_columns, *properties])
            return self._objects(table, df, properties, signal)

        summary = (
            df.groupby(object_group, sort=False)
            .agg(
                color=(signal, "first"),
                is_background=("is_background", "first"),
                size=("region_size", "first"),
                min_row=("region_min_row", "first"),
                max_row=("region_max_row", "first"),
                min_column=("region_min_column", "first"),
                max_column=("region_max_column", "first"),
                height=("region_height", "first"),
                width=("region_width", "first"),
                bbox_area=("region_bbox_area", "first"),
                fill_ratio=("region_fill_ratio", "first"),
                hole_count=("region_hole_count", "first"),
                centroid_row=("region_centroid_row", "first"),
                centroid_column=("region_centroid_column", "first"),
            )
            .reset_index()
        )

        # A region touching the grid edge may be clipped by the frame rather
        # than bounded by the phenomenon, which changes how its shape reads.
        extent = (
            df.groupby(self.group_columns, sort=False)
            .agg(
                grid_max_row=("grid_max_row", "first"),
                grid_max_column=("grid_max_column", "first"),
            )
            .reset_index()
        )
        summary = summary.merge(extent, on=list(self.group_columns), how="left")
        summary["touches_border"] = (
            summary["min_row"].eq(0)
            | summary["min_column"].eq(0)
            | summary["max_row"].eq(summary["grid_max_row"])
            | summary["max_column"].eq(summary["grid_max_column"])
        )

        table = summary[[*self.group_columns, *properties]]
        return self._objects(table, df, properties, signal)

    def _objects(
        self,
        table: pd.DataFrame,
        features: pd.DataFrame,
        properties: tuple[str, ...],
        signal: str,
    ) -> BehaviorObjects:
        return BehaviorObjects(
            behavior_type="region",
            signal=signal,
            table=table,
            features=features,
            group=tuple(self.group_columns),
            properties=properties,
            construction={
                "background_color": self.background_color,
                "connectivity": self.connectivity,
                "definition": self.definition,
                "include_background": self.include_background,
                "declined": self.declined_.to_dict("records"),
            },
        )


def edit_alignment(
    observations: pd.DataFrame,
    signal: str = "color",
    group: Sequence[str] = _PAIR_GROUP,
    background_color: int = 0,
    connectivity: int = 4,
    definition: str = "foreground_mask",
) -> pd.DataFrame:
    """Report whether the differences between paired grids respect regions.

    For each pair of grids sharing a group, the cells that differ are grouped
    into connected edit components, and each component is checked against the
    partition its input grid produces. A component contained in one region is
    aligned; one spanning several is not.

    This measures whether regions are the right object boundary for a
    transformation. It says nothing about whether the transformation can be
    predicted, which is a separate question.

    Returns one row per edit component.
    """
    group_columns = list(group)
    structure = CONNECTIVITY[connectivity]
    records: list[dict[str, Any]] = []

    for key, part in _iter_groups(observations, group_columns):
        inputs = part[part["grid_role"] == "input"]
        outputs = part[part["grid_role"] == "output"]
        if inputs.empty or outputs.empty:
            continue

        try:
            grid = grid_from_cells(inputs, signal)
            output = grid_from_cells(outputs, signal)
        except ValueError:
            continue

        if grid.shape != output.shape:
            continue

        # include_background is required: an aligned edit may lie inside a
        # background area rather than inside an object.
        partition = label_regions(
            grid,
            background_color=background_color,
            connectivity=connectivity,
            definition=definition,
            include_background=True,
        )

        change = grid != output
        if not change.any():
            continue

        components, count = ndimage.label(change, structure=structure)
        for index in range(1, count + 1):
            cells = components == index
            parts = np.unique(partition[cells])
            records.append(
                {
                    **_key_record(group_columns, key),
                    "edit_id": index,
                    "size": int(cells.sum()),
                    "regions_spanned": int(len(parts)),
                    "region_id": int(parts[0]) if len(parts) == 1 else pd.NA,
                    "is_aligned": bool(len(parts) == 1),
                }
            )

    columns = [
        *group_columns,
        "edit_id",
        "size",
        "regions_spanned",
        "region_id",
        "is_aligned",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


_FEATURE_COLUMNS = (
    "region_size",
    "region_min_row",
    "region_max_row",
    "region_min_column",
    "region_max_column",
    "region_centroid_row",
    "region_centroid_column",
    "region_height",
    "region_width",
    "region_bbox_area",
    "region_fill_ratio",
    "region_hole_count",
)


def _iter_groups(
    df: pd.DataFrame,
    group_columns: Sequence[str],
) -> Iterable[tuple[Any, pd.DataFrame]]:
    """Iterate grids, tolerating an ungrouped single-grid frame."""
    if not group_columns:
        return [((), df)]
    return list(df.groupby(list(group_columns), sort=False))


def _key_record(group_columns: Sequence[str], key: Any) -> dict[str, Any]:
    """Pair group column names with one group key."""
    if not group_columns:
        return {}
    values = key if isinstance(key, tuple) else (key,)
    return dict(zip(group_columns, values, strict=True))


def _empty_declined(group_columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=[*group_columns, "reason"])
