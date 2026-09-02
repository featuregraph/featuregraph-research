"""Measure what ARC-AGI-2 training pairs determine, without solving anything.

The solvers answer or refuse. This asks a different question of the same
evidence: across a corpus, how often does a declared operator vocabulary plus
a set of training pairs actually pin down a single answer, how often do they
rule everything out, and how often do they leave a choice open?

Nothing here is a solver score. A task counted as determined is one the
training pairs fully constrain under this vocabulary; whether that answer is
correct is reported separately, from the held-out test pair.

Usage:
    python scripts/arc_agi_determinacy.py --corpus /path/to/arc-agi-2/data
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from featuregraph.utils._arc_agi import (
    derive_state_instruction_layout,
    fixed_layout_evidence,
    state_evidence,
    test_cycle,
)

SPLITS = ("training", "evaluation")


def outcome(evidence) -> str:
    """One label per task, naming the most blocking condition first."""
    if evidence.contradicted:
        return "contradicted"
    if evidence.unobserved:
        return "unobserved"
    if evidence.underdetermined:
        return "underdetermined"
    return "determined"


def fixed_layout(evidence):
    shape = (
        max(index[0] for index in evidence.resolved) + 1,
        max(index[1] for index in evidence.resolved) + 1,
    )
    layout = np.empty(shape, dtype=object)
    for index, operator in evidence.resolved.items():
        layout[index] = operator
    return layout


def predicts_held_out(task, evidence, family) -> bool | None:
    """Whether the determined rules reproduce every held-out test output."""
    for case in task["test"]:
        if "output" not in case:
            return None
        grid = np.array(case["input"])
        try:
            if family == "fixed":
                predicted = test_cycle(grid, fixed_layout(evidence))
            else:
                layout = derive_state_instruction_layout(
                    grid, dict(evidence.resolved)
                )
                predicted = test_cycle(grid, layout)
        except ValueError:
            return False
        if not np.array_equal(predicted, np.array(case["output"])):
            return False
    return True


def analyse(corpus: Path) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    tally: dict = {
        split: {family: Counter() for family in ("fixed", "state")}
        for split in SPLITS
    }
    for split in SPLITS:
        for path in sorted((corpus / split).glob("*.json")):
            task = json.loads(path.read_text(encoding="utf-8"))
            pairs = [
                {"grid": pair["input"], "output": pair["output"]}
                for pair in task["train"]
            ]
            row = {"task": path.stem, "split": split}
            for family, evidence_of in (
                ("fixed", fixed_layout_evidence),
                ("state", state_evidence),
            ):
                try:
                    evidence = evidence_of(pairs)
                except ValueError as error:
                    row[family] = "inapplicable"
                    row[f"{family}_detail"] = str(error)
                    tally[split][family]["inapplicable"] += 1
                    continue
                label = outcome(evidence)
                row[family] = label
                row[f"{family}_detail"] = evidence.failure_message()
                tally[split][family][label] += 1
                if label == "determined":
                    correct = predicts_held_out(task, evidence, family)
                    row[f"{family}_held_out"] = (
                        "correct" if correct else "wrong" if correct is False else ""
                    )
                    tally[split][family][
                        "held_out_correct" if correct else "held_out_wrong"
                    ] += 1
            rows.append(row)
    return rows, tally


def corpus_revision(corpus: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(corpus), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/studies/arc_agi_2_determinacy"),
    )
    args = parser.parse_args()

    rows, tally = analyse(args.corpus)
    args.output.mkdir(parents=True, exist_ok=True)

    fields = [
        "task",
        "split",
        "fixed",
        "fixed_held_out",
        "fixed_detail",
        "state",
        "state_held_out",
        "state_detail",
    ]
    with (args.output / "task_determinacy.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"corpus revision: {corpus_revision(args.corpus)}")
    for split in SPLITS:
        total = sum(1 for row in rows if row["split"] == split)
        print(f"\n{split}: {total} tasks")
        for family in ("fixed", "state"):
            counts = tally[split][family]
            print(f"  {family}-layout")
            for key in (
                "determined",
                "underdetermined",
                "unobserved",
                "contradicted",
                "inapplicable",
            ):
                if counts[key]:
                    print(f"    {key:16s} {counts[key]:5d}  {counts[key] / total:6.1%}")
            if counts["determined"]:
                print(
                    f"    held-out correct {counts['held_out_correct']}"
                    f" / wrong {counts['held_out_wrong']}"
                )


if __name__ == "__main__":
    main()
