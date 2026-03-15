"""DocumentTask — the input/output contract for all phases.

All phases (baseline, mesh, failures) operate on the same task definition.
Changing the task means changing only this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocumentTask:
    """Describes what the agent pipeline should process and where to write outputs.

    Args:
        feedback_dir: Directory containing feedback .txt files (MCP-relative path).
        policy_path: Path to policy_snippets.md (MCP-relative path).
        outputs_dir: Where to write triage.csv, brief.md, response_templates.md.
        description: Human-readable task description passed to the agent as context.
    """

    feedback_dir: Path
    policy_path: Path
    outputs_dir: Path
    description: str = "Triage citizen feedback → action brief → response drafts"
