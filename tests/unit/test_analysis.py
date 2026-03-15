# tests/unit/test_analysis.py
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root for analysis/

from analysis.aggregate import load_runs


def test_load_runs_empty(tmp_path):
    df = load_runs(tmp_path, prefix="baseline")
    assert len(df) == 0


def test_load_runs_reads_metrics(tmp_path):
    run_dir = tmp_path / "baseline_abc"
    run_dir.mkdir()
    metrics = {
        "run_id": "baseline_abc",
        "total_latency_s": 3.5,
        "total_prompt_tokens": 500,
        "total_completion_tokens": 200,
        "step_trace": ["triage_node", "brief_node", "response_node"],
        "error": None,
        "nodes": [],
        "message_count": None,
        "state_size_bytes": None,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    df = load_runs(tmp_path, prefix="baseline")
    assert len(df) == 1
    assert df.iloc[0]["total_latency_s"] == 3.5
