import json
from pathlib import Path


def test_kaggle_submission_notebook_executes_offline_fixture(monkeypatch):
    repository_root = Path(__file__).resolve().parents[1]
    notebook_path = (
        repository_root
        / "notebooks"
        / "arc-agi_kaggle_submission.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {}

    monkeypatch.chdir(repository_root)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(
            compile(source, f"notebook-cell-{index}", "exec"),
            namespace,
        )

    report = namespace["report"]
    submission = namespace["submission"]
    output_path = namespace["output_path"]

    assert output_path.exists()
    assert report["number_of_tasks"] == 2
    assert report["number_supported"] == 1
    assert report["number_fallback"] == 1
    assert set(submission) == {"supported", "unsupported"}
