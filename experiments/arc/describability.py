"""Measure where the block-composition vocabulary applies across ARC-AGI-2.

This is a representation study, not a solver benchmark. For every task it asks
a narrower and more answerable question than "was it solved":

* Is the task *in frame* at all — does the output tile the input?
* If so, how many of its blocks does the declared operator vocabulary describe?
* Does a layout grouped by block coordinate resolve to one operator per block?
* Does a layout grouped by input-cell state resolve instead?

Layouts are resolved from demonstration pairs only. Test pairs are described
but never intersected, so a resolved layout never sees its own answer.

Usage
-----
Clone the public dataset, then scan it::

    git clone --depth 1 https://github.com/arcprize/ARC-AGI-2 /tmp/ARC-AGI-2
    python -m experiments.arc.describability \\
        --data-dir /tmp/ARC-AGI-2/data \\
        --output artifacts/arc/describability.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from featuregraph.behaviors.composition import BlockComposition, resolve_layout

SPLITS = ("training", "evaluation")

_TASK_COLUMNS = [
    "task_id",
    "split",
    "in_frame",
    "declined_reason",
    "block_rows",
    "block_columns",
    "identity_layout",
    "n_blocks",
    "blocks_described",
    "blocks_determined",
    "blocks_ambiguous",
    "blocks_unmatched",
    "fixed_layout_determined",
    "state_layout_determined",
]


def scan_task(
    task: dict[str, Any],
    task_id: str,
    split: str,
    background_color: int = 0,
) -> dict[str, Any]:
    """Describe one ARC task in terms of block objects."""
    observations = _task_observations(task, task_id=task_id, split=split)

    builder = BlockComposition(
        signals="color",
        group=["task_id", "pair_type", "pair_index"],
        background_color=background_color,
    )
    objects = builder.summarize(builder.fit_transform(observations))

    record: dict[str, Any] = {
        "task_id": task_id,
        "split": split,
        "in_frame": False,
        "declined_reason": None,
        "block_rows": pd.NA,
        "block_columns": pd.NA,
        "identity_layout": pd.NA,
        "n_blocks": pd.NA,
        "blocks_described": pd.NA,
        "blocks_determined": pd.NA,
        "blocks_ambiguous": pd.NA,
        "blocks_unmatched": pd.NA,
        "fixed_layout_determined": pd.NA,
        "state_layout_determined": pd.NA,
    }

    declined = builder.declined_
    demonstrations_declined = declined[declined["pair_type"].eq("train")]
    if not demonstrations_declined.empty:
        # A demonstration that cannot be tiled puts the whole task out of frame.
        record["declined_reason"] = (
            demonstrations_declined["reason"].iloc[0]
        )
        return record

    # Layout inference reads demonstrations only.
    train = objects.query().where(pair_type="train").collect()
    if train.empty:
        record["declined_reason"] = "no_demonstration_blocks"
        return record

    layouts = set(
        zip(train["block_rows"], train["block_columns"], strict=True)
    )
    if len(layouts) > 1:
        record["declined_reason"] = "inconsistent_block_layout"
        return record

    block_rows, block_columns = layouts.pop()

    fixed = resolve_layout(train, by=("block_row", "block_column"))
    state = (
        resolve_layout(train, by=("block_state",))
        if train["block_state"].notna().all()
        else None
    )

    record.update(
        {
            "in_frame": True,
            "block_rows": int(block_rows),
            "block_columns": int(block_columns),
            "identity_layout": bool(block_rows == 1 and block_columns == 1),
            "n_blocks": int(len(train)),
            "blocks_described": int((train["candidate_count"] > 0).sum()),
            "blocks_determined": int(train["is_determined"].sum()),
            "blocks_ambiguous": int((train["candidate_count"] > 1).sum()),
            "blocks_unmatched": int(train["is_unmatched"].sum()),
            "fixed_layout_determined": bool(fixed["is_determined"].all()),
            "state_layout_determined": (
                bool(state["is_determined"].all())
                if state is not None
                else False
            ),
        }
    )
    return record


def scan_split(
    data_dir: str | Path,
    split: str,
    limit: int | None = None,
    background_color: int = 0,
) -> pd.DataFrame:
    """Describe every task in one ARC-AGI-2 split."""
    records = [
        scan_task(
            json.loads(path.read_text(encoding="utf-8")),
            task_id=path.stem,
            split=split,
            background_color=background_color,
        )
        for path in _task_paths(data_dir, split, limit)
    ]
    return pd.DataFrame.from_records(records, columns=_TASK_COLUMNS)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Roll task records up into one row per split."""
    rows = []
    for split, part in table.groupby("split", sort=True):
        in_frame = part[part["in_frame"].astype(bool)]
        non_trivial = in_frame[~in_frame["identity_layout"].astype(bool)]

        blocks = int(non_trivial["n_blocks"].sum()) if len(non_trivial) else 0
        rows.append(
            {
                "split": split,
                "tasks": len(part),
                "in_frame": len(in_frame),
                "identity_layout": len(in_frame) - len(non_trivial),
                "non_trivial_tiling": len(non_trivial),
                "blocks": blocks,
                "blocks_described": (
                    int(non_trivial["blocks_described"].sum()) if blocks else 0
                ),
                "blocks_unmatched": (
                    int(non_trivial["blocks_unmatched"].sum()) if blocks else 0
                ),
                "fixed_layout_determined": (
                    int(non_trivial["fixed_layout_determined"].astype(bool).sum())
                    if len(non_trivial)
                    else 0
                ),
                "state_layout_determined": (
                    int(non_trivial["state_layout_determined"].astype(bool).sum())
                    if len(non_trivial)
                    else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def _task_paths(
    data_dir: str | Path,
    split: str,
    limit: int | None,
) -> Iterator[Path]:
    split_dir = Path(data_dir) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"ARC-AGI-2 split directory not found: {split_dir}")

    paths = sorted(split_dir.glob("*.json"))
    return iter(paths[:limit] if limit else paths)


def _task_observations(
    task: dict[str, Any],
    task_id: str,
    split: str,
) -> pd.DataFrame:
    """Build a cell-level observation table for one in-memory task."""
    records = []
    for pair_type in ("train", "test"):
        for pair_index, pair in enumerate(task.get(pair_type, [])):
            for grid_role in ("input", "output"):
                grid = pair.get(grid_role)
                if grid is None:
                    continue
                records.extend(
                    {
                        "task_id": task_id,
                        "split": split,
                        "pair_type": pair_type,
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


def _dataset_revision(data_dir: str | Path) -> str | None:
    """Record the ARC-AGI-2 commit the scan read, when available."""
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(data_dir).parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        required=True,
        help="ARC-AGI-2 data directory containing training/ and evaluation/",
    )
    parser.add_argument(
        "--output",
        default="artifacts/arc/describability.csv",
        help="Path for the per-task CSV",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Path for the split-level summary JSON (defaults beside --output)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--background-color", type=int, default=0)
    arguments = parser.parse_args(argv)

    tables = [
        scan_split(
            arguments.data_dir,
            split,
            limit=arguments.limit,
            background_color=arguments.background_color,
        )
        for split in SPLITS
    ]
    table = pd.concat(tables, ignore_index=True)

    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    summary = summarize(table)
    summary_path = (
        Path(arguments.summary)
        if arguments.summary
        else output.with_name(f"{output.stem}_summary.json")
    )
    summary_path.write_text(
        json.dumps(
            {
                "arc_agi_version": "2",
                "dataset_revision": _dataset_revision(arguments.data_dir),
                "background_color": arguments.background_color,
                "operators": list(
                    BlockComposition(background_color=arguments.background_color)
                    .operator_names
                ),
                "splits": summary.to_dict("records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(f"\nwrote {output}\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
