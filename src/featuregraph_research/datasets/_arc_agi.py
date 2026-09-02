"""ARC-AGI-2 tasks represented as cell-level observation tables."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ARC_AGI_VERSION = "2"
ARC_AGI_SOURCE = "https://github.com/arcprize/ARC-AGI-2"
ARC_AGI_RAW_URL = (
    "https://raw.githubusercontent.com/arcprize/ARC-AGI-2/main/data"
)
ARC_AGI_SPLITS = frozenset({"training", "evaluation"})
_TASK_ID_PATTERN = re.compile(r"^[0-9a-f]{8}$")
_COLUMNS = [
    "task_id",
    "split",
    "pair_type",
    "pair_index",
    "grid_role",
    "row",
    "column",
    "color",
    "grid_height",
    "grid_width",
]


def arc_agi(
    task_id: str,
    *,
    split: str = "training",
    data_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load one ARC-AGI-2 task as a cell-level observation table.

    Parameters
    ----------
    task_id:
        Eight-character hexadecimal ARC task identifier.
    split:
        Public ARC-AGI-2 split: ``"training"`` or ``"evaluation"``.
    data_dir:
        Optional directory containing ``training/`` and ``evaluation/``
        subdirectories. When omitted, the task is downloaded from the
        official ARC-AGI-2 repository and stored in FeatureGraph's cache.
    refresh:
        Redownload an officially hosted task even when it is cached. This
        cannot be combined with ``data_dir``.

    Returns
    -------
    pandas.DataFrame
        One row per grid cell. ``pair_type`` distinguishes demonstrations
        (``"train"``) from test cases, and ``grid_role`` distinguishes input
        from output grids. Test outputs are included when present.
    """
    _validate_request(task_id=task_id, split=split, data_dir=data_dir, refresh=refresh)
    path = _task_path(task_id=task_id, split=split, data_dir=data_dir)

    if data_dir is None and (refresh or not path.exists()):
        _download_task(task_id=task_id, split=split, path=path)

    if not path.exists():
        raise FileNotFoundError(f"ARC-AGI-2 task file not found: {path}")

    with path.open(encoding="utf-8") as task_file:
        task = json.load(task_file)

    observations = _task_to_frame(task, task_id=task_id, split=split)
    observations.attrs.update(
        {
            "arc_agi_version": ARC_AGI_VERSION,
            "task_id": task_id,
            "split": split,
            "source_file": str(path),
            "source_url": f"{ARC_AGI_SOURCE}/blob/main/data/{split}/{task_id}.json",
        }
    )
    return observations


def get_arc_agi_cache_dir() -> Path:
    """Return the external cache used for official ARC-AGI-2 tasks."""
    cache_dir = Path.home() / ".cache" / "featuregraph" / "arc_agi" / ARC_AGI_VERSION
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def clear_arc_agi_cache() -> None:
    """Remove downloaded ARC-AGI-2 task files from the external cache."""
    cache_dir = get_arc_agi_cache_dir()
    for path in cache_dir.glob("*/*.json"):
        path.unlink()


def _task_path(
    *,
    task_id: str,
    split: str,
    data_dir: str | Path | None,
) -> Path:
    root = Path(data_dir) if data_dir is not None else get_arc_agi_cache_dir()
    return root / split / f"{task_id}.json"


def _download_task(*, task_id: str, split: str, path: Path) -> None:
    url = f"{ARC_AGI_RAW_URL}/{split}/{task_id}.json"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.part")
    temporary_path.write_bytes(response.content)
    temporary_path.replace(path)


def _task_to_frame(
    task: Any,
    *,
    task_id: str,
    split: str,
) -> pd.DataFrame:
    if not isinstance(task, dict):
        raise ValueError("ARC task must be a JSON object")

    records: list[dict[str, object]] = []
    for pair_type in ("train", "test"):
        pairs = task.get(pair_type)
        if not isinstance(pairs, list) or not pairs:
            raise ValueError(f"ARC task must contain a non-empty {pair_type!r} list")

        for pair_index, pair in enumerate(pairs):
            if not isinstance(pair, dict) or "input" not in pair:
                message = f"{pair_type}[{pair_index}] must contain an input grid"
                raise ValueError(message)

            records.extend(
                _grid_records(
                    pair["input"],
                    task_id=task_id,
                    split=split,
                    pair_type=pair_type,
                    pair_index=pair_index,
                    grid_role="input",
                )
            )
            if "output" in pair:
                records.extend(
                    _grid_records(
                        pair["output"],
                        task_id=task_id,
                        split=split,
                        pair_type=pair_type,
                        pair_index=pair_index,
                        grid_role="output",
                    )
                )

    return pd.DataFrame.from_records(records, columns=_COLUMNS)


def _grid_records(
    grid: Any,
    *,
    task_id: str,
    split: str,
    pair_type: str,
    pair_index: int,
    grid_role: str,
) -> list[dict[str, object]]:
    height, width = _validate_grid(grid)
    return [
        {
            "task_id": task_id,
            "split": split,
            "pair_type": pair_type,
            "pair_index": pair_index,
            "grid_role": grid_role,
            "row": row,
            "column": column,
            "color": color,
            "grid_height": height,
            "grid_width": width,
        }
        for row, values in enumerate(grid)
        for column, color in enumerate(values)
    ]


def _validate_grid(grid: Any) -> tuple[int, int]:
    if not isinstance(grid, list) or not grid:
        raise ValueError("ARC grid must be a non-empty list of rows")
    if not all(isinstance(row, list) and row for row in grid):
        raise ValueError("ARC grid rows must be non-empty lists")

    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("ARC grid must be rectangular")

    for row in grid:
        for color in row:
            if (
                not isinstance(color, int)
                or isinstance(color, bool)
                or not 0 <= color <= 9
            ):
                raise ValueError("ARC colors must be integers from 0 through 9")

    return len(grid), width


def _validate_request(
    *,
    task_id: str,
    split: str,
    data_dir: str | Path | None,
    refresh: bool,
) -> None:
    if not isinstance(task_id, str) or _TASK_ID_PATTERN.fullmatch(task_id) is None:
        message = "task_id must be an eight-character lowercase hexadecimal string"
        raise ValueError(message)
    if split not in ARC_AGI_SPLITS:
        choices = ", ".join(sorted(ARC_AGI_SPLITS))
        raise ValueError(f"split must be one of: {choices}")
    if data_dir is not None and refresh:
        raise ValueError("refresh cannot be used with data_dir")
