"""Are connected regions the right object boundary for same-shape ARC-AGI-2?

Block composition described 5.1% of the ARC-AGI-2 training split and 0.0% of
evaluation. Same-shape tasks — output dimensions equal to input dimensions —
are roughly two thirds of both splits and sit entirely outside that frame.
This experiment asks three questions of that population:

1. Does the background partition survive the transformation?
2. Is a region's output colour a function of simple region properties?
3. Do the edits respect region boundaries at all?

Question 3 decides the object boundary. A construction can be the right
boundary even when the edit it contains is not predictable; those are separate
claims and this experiment keeps them separate.

Every measurement runs through ``RegionObjects`` and ``edit_alignment`` so the
definition of a region here is the same one the library constructs.

Usage
-----
    git clone --depth 1 https://github.com/arcprize/ARC-AGI-2 /tmp/ARC-AGI-2
    python -m experiments.arc.region_probe \\
        --data-dir /tmp/ARC-AGI-2/data \\
        --describability artifacts/arc/describability.csv \\
        --output artifacts/arc/region_probe.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from featuregraph.behaviors.regions import RegionObjects, edit_alignment

GRID_GROUP = ["task_id", "pair_type", "pair_index", "grid_role"]
PAIR_KEYS = ["task_id", "pair_type", "pair_index"]

_DETERMINANTS = {
    "colour_determines_output": "in_color",
    "size_determines_output": "size",
    "size_rank_determines_output": "size_rank",
    "bbox_determines_output": "bbox",
}

_COLUMNS = [
    "task_id",
    "split",
    "connectivity",
    "regions_median",
    "regions_max",
    "mask_preserved",
    "regions_uniform_out",
    "mean_fraction_changed",
    "change_components",
    "aligned_components",
    "aligned_fraction",
    *_DETERMINANTS,
]


def task_observations(task: dict[str, Any], task_id: str, split: str) -> pd.DataFrame:
    """Cell-level observations for the demonstration pairs of one task."""
    records = []
    for pair_index, pair in enumerate(task.get("train", [])):
        for grid_role in ("input", "output"):
            grid = pair.get(grid_role)
            if grid is None:
                continue
            records.extend(
                {
                    "task_id": task_id,
                    "split": split,
                    "pair_type": "train",
                    "pair_index": pair_index,
                    "grid_role": grid_role,
                    "row": row,
                    "column": column,
                    "color": color,
                }
                for row, values in enumerate(grid)
                for column, color in enumerate(values)
            )
    return pd.DataFrame.from_records(records)


def paired_cells(observations: pd.DataFrame) -> pd.DataFrame:
    """Join every input cell to the output cell at the same coordinates."""
    keys = [*PAIR_KEYS, "row", "column"]
    inputs = observations[observations["grid_role"] == "input"]
    outputs = observations[observations["grid_role"] == "output"]
    return inputs.merge(
        outputs[[*keys, "color"]].rename(columns={"color": "output_color"}),
        on=keys,
        how="inner",
    )


def probe_task(
    task: dict[str, Any],
    task_id: str,
    split: str,
    background_color: int = 0,
    connectivity: int = 4,
) -> dict[str, Any] | None:
    """Measure region structure and edit alignment for one same-shape task."""
    observations = task_observations(task, task_id=task_id, split=split)
    if observations.empty:
        return None

    pairs = paired_cells(observations)
    if pairs.empty:
        return None

    record: dict[str, Any] = {
        "task_id": task_id,
        "split": split,
        "connectivity": connectivity,
        "mean_fraction_changed": float(
            pairs.groupby(PAIR_KEYS)
            .apply(lambda part: part["color"].ne(part["output_color"]).mean())
            .mean()
        ),
        "mask_preserved": bool(
            pairs["color"]
            .eq(background_color)
            .eq(pairs["output_color"].eq(background_color))
            .all()
        ),
    }

    # Regions of the input grids, under the declared construction.
    builder = RegionObjects(
        signals="color",
        group=GRID_GROUP,
        background_color=background_color,
        connectivity=connectivity,
        definition="foreground_mask",
        include_background=False,
    )
    cells = builder.fit_transform(observations)
    objects = builder.summarize(cells)

    inputs = objects.query().where(grid_role="input").collect()
    counts = (
        inputs.groupby(PAIR_KEYS).size()
        if not inputs.empty
        else pd.Series(dtype=int)
    )
    record["regions_median"] = float(counts.median()) if len(counts) else 0.0
    record["regions_max"] = int(counts.max()) if len(counts) else 0

    # Each input region's output colours, read from the retained cell table.
    region_cells = cells[cells["grid_role"] == "input"].merge(
        observations[observations["grid_role"] == "output"][
            [*PAIR_KEYS, "row", "column", "color"]
        ].rename(columns={"color": "output_color"}),
        on=[*PAIR_KEYS, "row", "column"],
        how="inner",
    )
    out_colors = region_cells.groupby([*PAIR_KEYS, "region_id"])["output_color"]
    uniform = bool((out_colors.nunique() == 1).all()) if len(region_cells) else False
    record["regions_uniform_out"] = uniform

    determinants = dict.fromkeys(_DETERMINANTS, False)
    if uniform and record["mask_preserved"] and not inputs.empty:
        properties = inputs.rename(columns={"color": "in_color"}).merge(
            out_colors.first().rename("out_color").reset_index(),
            on=[*PAIR_KEYS, "region_id"],
            how="inner",
        )
        properties["size_rank"] = properties["size"].rank(method="dense").astype(int)
        properties["bbox"] = list(
            zip(properties.height, properties.width, properties["size"], strict=True)
        )
        for column, key in _DETERMINANTS.items():
            determinants[column] = bool(
                (properties.groupby(key)["out_color"].nunique() == 1).all()
            )
    record.update(determinants)

    edits = edit_alignment(
        observations,
        group=PAIR_KEYS,
        background_color=background_color,
        connectivity=connectivity,
        definition="foreground_mask",
    )
    record["change_components"] = int(len(edits))
    record["aligned_components"] = int(edits["is_aligned"].sum()) if len(edits) else 0
    record["aligned_fraction"] = (
        record["aligned_components"] / record["change_components"]
        if record["change_components"]
        else np.nan
    )
    return record


def probe_split(
    data_dir: str | Path,
    tasks: pd.DataFrame,
    background_color: int = 0,
    connectivity: int = 4,
) -> pd.DataFrame:
    records = []
    for _, row in tasks.iterrows():
        path = Path(data_dir) / row["split"] / f"{row['task_id']}.json"
        record = probe_task(
            json.loads(path.read_text(encoding="utf-8")),
            task_id=row["task_id"],
            split=row["split"],
            background_color=background_color,
            connectivity=connectivity,
        )
        if record:
            records.append(record)
    return pd.DataFrame.from_records(records, columns=_COLUMNS)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, connectivity), part in table.groupby(["split", "connectivity"]):
        edited = part[part["change_components"] > 0]
        components = int(part["change_components"].sum())
        rows.append(
            {
                "split": split,
                "connectivity": connectivity,
                "tasks": len(part),
                "median_regions": part["regions_median"].median(),
                "max_regions": int(part["regions_max"].max()),
                "mask_preserved": int(part["mask_preserved"].sum()),
                "output_colour_determined": int(
                    part[list(_DETERMINANTS)].any(axis=1).sum()
                ),
                "edit_components": components,
                "edits_region_aligned": (
                    part["aligned_components"].sum() / components
                    if components
                    else np.nan
                ),
                "tasks_fully_aligned": int((edited["aligned_fraction"] >= 1.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--describability", default="artifacts/arc/describability.csv")
    parser.add_argument("--output", default="artifacts/arc/region_probe.csv")
    parser.add_argument("--background-color", type=int, default=0)
    arguments = parser.parse_args(argv)

    describability = pd.read_csv(arguments.describability)
    same_shape = describability[
        describability["in_frame"] & describability["identity_layout"].astype(bool)
    ][["task_id", "split"]]

    table = pd.concat(
        [
            probe_split(
                arguments.data_dir,
                same_shape,
                background_color=arguments.background_color,
                connectivity=connectivity,
            )
            for connectivity in (4, 8)
        ],
        ignore_index=True,
    )

    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    print(summarize(table).to_string(index=False))
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
