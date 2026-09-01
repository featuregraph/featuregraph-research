"""Scoping probe: are connected regions the right object boundary for the
same-shape ARC-AGI-2 population?

Block composition described 5.1% of the training split and 0% of evaluation.
Same-shape tasks — where the output has the same dimensions as the input — are
roughly two thirds of both splits, and block decomposition says nothing about
them. Before building a region behavior, this probe asks three questions of
that population:

1. Does the background/foreground partition survive the transformation?
2. Is a region's output colour a function of simple region properties?
3. Do the edits respect region boundaries at all?

Question 3 is the one that decides whether regions are the right object
boundary. A construction can be the correct boundary even when the edit it
contains is not predictable — those are separate claims, and this probe keeps
them separate.

Usage
-----
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
from scipy import ndimage

CONNECTIVITY = {
    4: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    8: np.ones((3, 3), dtype=int),
}


def _demonstration_pairs(task: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for pair in task.get("train", []):
        if "output" not in pair:
            continue
        grid = np.asarray(pair["input"], dtype=int)
        output = np.asarray(pair["output"], dtype=int)
        if grid.shape != output.shape:
            return []
        pairs.append((grid, output))
    return pairs


def _input_partition(
    grid: np.ndarray,
    background_color: int,
    structure: np.ndarray,
) -> np.ndarray:
    """Label every cell by the input part it belongs to.

    Foreground components are labelled first; background components continue
    the numbering, so a change crossing from an object into its surrounding
    background counts as unaligned.
    """
    foreground, _ = ndimage.label(grid != background_color, structure=structure)
    background, _ = ndimage.label(grid == background_color, structure=structure)
    offset = int(foreground.max())
    return np.where(foreground > 0, foreground, offset + background)


def probe_task(
    task: dict[str, Any],
    task_id: str,
    split: str,
    background_color: int = 0,
    connectivity: int = 4,
) -> dict[str, Any] | None:
    """Measure region structure and edit alignment for one same-shape task."""
    pairs = _demonstration_pairs(task)
    if not pairs:
        return None

    structure = CONNECTIVITY[connectivity]

    mask_preserved = True
    region_counts: list[int] = []
    changed_fractions: list[float] = []
    property_rows: list[dict[str, Any]] = []
    aligned = components = 0

    for grid, output in pairs:
        if not np.array_equal(grid != background_color, output != background_color):
            mask_preserved = False

        changed_fractions.append(float((grid != output).mean()))

        labels, count = ndimage.label(grid != background_color, structure=structure)
        region_counts.append(count)

        for index in range(1, count + 1):
            cells = labels == index
            rows, columns = np.nonzero(cells)
            output_colors = np.unique(output[cells])
            property_rows.append(
                {
                    "size": int(cells.sum()),
                    "height": int(rows.max() - rows.min() + 1),
                    "width": int(columns.max() - columns.min() + 1),
                    "in_color": int(np.unique(grid[cells])[0]),
                    "out_color": (
                        int(output_colors[0]) if len(output_colors) == 1 else -1
                    ),
                    "uniform_out": len(output_colors) == 1,
                }
            )

        change = grid != output
        if not change.any():
            continue

        partition = _input_partition(grid, background_color, structure)
        change_labels, change_count = ndimage.label(change, structure=structure)
        for index in range(1, change_count + 1):
            components += 1
            if len(np.unique(partition[change_labels == index])) == 1:
                aligned += 1

    properties = pd.DataFrame(property_rows)
    uniform = bool(properties["uniform_out"].all()) if len(properties) else False

    record: dict[str, Any] = {
        "task_id": task_id,
        "split": split,
        "connectivity": connectivity,
        "regions_median": float(np.median(region_counts)) if region_counts else 0.0,
        "regions_max": int(max(region_counts)) if region_counts else 0,
        "mask_preserved": mask_preserved,
        "regions_uniform_out": uniform,
        "mean_fraction_changed": float(np.mean(changed_fractions)),
        "change_components": components,
        "aligned_components": aligned,
        "aligned_fraction": (aligned / components) if components else np.nan,
    }

    # Is the output colour a function of a simple region property?
    if uniform and mask_preserved and len(properties):
        properties["size_rank"] = properties["size"].rank(method="dense").astype(int)
        properties["bbox"] = list(
            zip(properties.height, properties.width, properties["size"], strict=True)
        )
        for key, column in (
            ("in_color", "colour_determines_output"),
            ("size", "size_determines_output"),
            ("size_rank", "size_rank_determines_output"),
            ("bbox", "bbox_determines_output"),
        ):
            record[column] = bool(
                (properties.groupby(key)["out_color"].nunique() == 1).all()
            )
    else:
        for column in (
            "colour_determines_output",
            "size_determines_output",
            "size_rank_determines_output",
            "bbox_determines_output",
        ):
            record[column] = False

    return record


def probe_split(
    data_dir: str | Path,
    task_ids: pd.DataFrame,
    background_color: int = 0,
    connectivity: int = 4,
) -> pd.DataFrame:
    records = []
    for _, row in task_ids.iterrows():
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
    return pd.DataFrame(records)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, connectivity), part in table.groupby(["split", "connectivity"]):
        edited = part[part["change_components"] > 0]
        determines = part[
            [
                "colour_determines_output",
                "size_determines_output",
                "size_rank_determines_output",
                "bbox_determines_output",
            ]
        ].any(axis=1)
        rows.append(
            {
                "split": split,
                "connectivity": connectivity,
                "tasks": len(part),
                "median_regions": part["regions_median"].median(),
                "max_regions": int(part["regions_max"].max()),
                "mask_preserved": int(part["mask_preserved"].sum()),
                "output_colour_determined": int(determines.sum()),
                "edit_components": int(part["change_components"].sum()),
                "edits_region_aligned": (
                    part["aligned_components"].sum() / part["change_components"].sum()
                    if part["change_components"].sum()
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

    tables = [
        probe_split(
            arguments.data_dir,
            same_shape,
            background_color=arguments.background_color,
            connectivity=connectivity,
        )
        for connectivity in (4, 8)
    ]
    table = pd.concat(tables, ignore_index=True)

    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    summary = summarize(table)
    print(summary.to_string(index=False))
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
