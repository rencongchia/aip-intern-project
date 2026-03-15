from pathlib import Path
from aip_intern.core.task import DocumentTask


def test_document_task_fields():
    task = DocumentTask(
        feedback_dir=Path("workspace/data/feedback"),
        policy_path=Path("workspace/data/policy_snippets.md"),
        outputs_dir=Path("artifacts/test_run/outputs"),
    )
    assert task.feedback_dir == Path("workspace/data/feedback")


def test_document_task_defaults():
    task = DocumentTask(
        feedback_dir=Path("workspace/data/feedback"),
        policy_path=Path("workspace/data/policy_snippets.md"),
        outputs_dir=Path("artifacts/test_run/outputs"),
    )
    assert task.description == "Triage citizen feedback → action brief → response drafts"
