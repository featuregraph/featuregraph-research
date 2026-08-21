"""Generate the disposition inventory for migrated foundations artifacts."""

import csv
from pathlib import Path

ROOT = Path("notebooks/foundations/archive/source_records")
OUTPUT = Path("notebooks/foundations/archive/INVENTORY.csv")

PROMOTE = {
    "01_linear_algebra/linear_algebra_plotting.ipynb": "array visualization",
    "01_linear_algebra/matrix_kronecker_product.ipynb": "block composition",
    "01_linear_algebra/reflection_matrices.ipynb": "spatial reflection",
    "01_linear_algebra/rotation_matrices.ipynb": "spatial rotation",
    "04_learning/numpy_auc.ipynb": "vectorized accumulation",
    "06_spatial_reasoning/neighbors_matrix.ipynb": "local spatial relations",
    "07_markov_processes/random walk mrp.ipynb": "state-value iteration",
    "08_rl_agents/CartPole visualizations 1.ipynb": "transition inspection",
    "08_rl_agents/random walk.ipynb": "explicit state transitions",
    "08_rl_agents/sample state space.ipynb": "state/action representation",
    "10_matrix_arithmetic/numpy workshop.ipynb": "boolean array operations",
    "11_feature_graph_origins/src/feature_graph/features.py": (
        "operator-parent provenance"
    ),
    "11_feature_graph_origins/src/feature_graph/operators.py": (
        "enter/exit/between operators"
    ),
    "12_eventql/README.md": "declarative operator model",
    "12_eventql/src/eventql/ast/nodes.py": "typed expression nodes",
    "12_eventql/src/eventql/compiler/compiler.py": "compile/evaluate separation",
    "12_eventql/src/eventql/compiler/expression.py": "compiled expression record",
    "12_eventql/src/eventql/parser/grammar.lark": "expression grammar",
    "12_eventql/src/eventql/parser/transformer.py": "syntax-to-AST mapping",
    "12_eventql/src/eventql/runtime/interpreter.py": "deterministic evaluation",
    "12_eventql/src/eventql/semantics/infer_type.py": "representation typing",
    "12_eventql/src/eventql/semantics/types.py": "representation types",
}

REMOVE = {
    "10_matrix_arithmetic/README.md": "empty or placeholder documentation",
    "11_feature_graph_origins/README.md": "empty or placeholder documentation",
    "11_feature_graph_origins/src/__init__.py": "empty historical package marker",
    "11_feature_graph_origins/src/feature_graph/__init__.py": (
        "empty historical package marker"
    ),
    "12_eventql/src/eventql/compiler/__init__.py": "empty historical package marker",
    "12_eventql/src/eventql/parser/__init__.py": "empty historical package marker",
    "12_eventql/src/eventql/runtime/__init__.py": "empty historical package marker",
    "12_eventql/src/eventql/semantics/__init__.py": "empty historical package marker",
}

REFERENCE_PREFIXES = (
    "01_linear_algebra/",
    "02_numerical_methods/",
    "03_probability/",
    "04_learning/",
    "07_markov_processes/absorbing_markov_chain",
    "07_markov_processes/multi-armed bandit",
    "07_markov_processes/q table",
    "07_markov_processes/random walk gridworld",
    "07_markov_processes/thompson-sampling/",
    "08_rl_agents/cartpole_solution.ipynb",
    "08_rl_agents/tutorial/README.md",
    "09_function_approximation/",
    "10_matrix_arithmetic/bit-transmission.py",
    "10_matrix_arithmetic/boolean-array.py",
    "10_matrix_arithmetic/linear functions workshop.ipynb",
    "11_feature_graph_origins/cp_example.py",
    "11_feature_graph_origins/example.py",
    "11_feature_graph_origins/src/feature_graph/project_constants.py",
    "12_eventql/pyproject.toml",
    "12_eventql/src/eventql/__init__.py",
    "12_eventql/src/eventql/parser/parser.py",
)


def classify(relative_path: str) -> tuple[str, str]:
    if relative_path in PROMOTE:
        return "promote", PROMOTE[relative_path]
    if relative_path in REMOVE:
        return "remove", REMOVE[relative_path]
    if relative_path.startswith(REFERENCE_PREFIXES):
        return "reference", "supporting mathematical or implementation record"
    return "archive", "superseded, duplicate, incomplete, or dependency-heavy"


def main() -> None:
    rows = []
    for path in sorted(item for item in ROOT.rglob("*") if item.is_file()):
        relative_path = path.relative_to(ROOT).as_posix()
        disposition, reason = classify(relative_path)
        rows.append(
            {
                "path": relative_path,
                "disposition": disposition,
                "reason": reason,
                "bytes": path.stat().st_size,
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("path", "disposition", "reason", "bytes"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = {
        disposition: sum(row["disposition"] == disposition for row in rows)
        for disposition in ("promote", "reference", "archive", "remove")
    }
    print(f"Wrote {len(rows)} classified artifacts to {OUTPUT}")
    print(counts)


if __name__ == "__main__":
    main()
