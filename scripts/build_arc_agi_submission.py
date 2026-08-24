"""Build a schema-validated ARC Prize submission from challenge JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from featuregraph.utils._arc_agi import run_harness  # noqa: E402


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build submission.json from an ARC Prize challenges file."
        )
    )
    parser.add_argument(
        "challenges_path",
        type=Path,
        help="Path to arc-agi_test_challenges.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.json"),
        help="Output path (default: submission.json).",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    report = run_harness(args.challenges_path, args.output)
    printable_report = {
        **report,
        "submission_path": str(report["submission_path"]),
    }
    print(json.dumps(printable_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
