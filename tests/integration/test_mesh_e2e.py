from __future__ import annotations

import os
from pathlib import Path

import pytest

from aip_intern.baseline.runner import RunConfig
from aip_intern.mesh.runner import run_once


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mesh_single_run_produces_outputs(live_llm, tmp_path):
    """Full mesh run: all 3 output files produced, mesh metrics recorded."""
    cfg = RunConfig(
        run_id_prefix="mesh_integration",
        n_runs=1,
        config_path=Path("config/mesh.yaml"),
        llm_model=os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-32B-Instruct"),
        llm_base_url=os.environ["OPENAI_BASE_URL"],
        llm_api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        workspace_root=Path("workspace/"),
        artifacts_dir=tmp_path,
    )
    result = await run_once(cfg)
    assert result.success, f"Run failed: {result.error}"
    assert result.metrics.get("state_size_bytes", 0) > 0
