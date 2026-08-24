import json

from scripts.build_arc_agi_submission import main


def test_build_arc_agi_submission_writes_complete_attempt_contract(
    tmp_path,
    capsys,
):
    challenges = {
        "supported": {
            "train": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[1, 2, 2, 1], [3, 4, 4, 3]],
                }
            ],
            "test": [{"input": [[5, 6], [7, 8]]}],
        },
        "unsupported": {
            "train": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[9, 9], [9, 9]],
                }
            ],
            "test": [{"input": [[5, 6], [7, 8]]}],
        },
    }
    challenges_path = tmp_path / "arc-agi_test_challenges.json"
    submission_path = tmp_path / "submission.json"
    challenges_path.write_text(json.dumps(challenges), encoding="utf-8")

    exit_code = main([
        str(challenges_path),
        "--output",
        str(submission_path),
    ])

    assert exit_code == 0
    assert submission_path.exists()

    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    assert set(submission) == set(challenges)
    for task_id, task_attempts in submission.items():
        assert len(task_attempts) == len(challenges[task_id]["test"])
        assert set(task_attempts[0]) == {"attempt_1", "attempt_2"}

    report = json.loads(capsys.readouterr().out)
    assert report["number_of_tasks"] == 2
    assert report["number_supported"] == 1
    assert report["number_fallback"] == 1
