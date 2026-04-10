"""Mesh run sweep — mirrors baseline/runner.py structure.

run(config) → list[RunResult]: execute n_runs mesh iterations.
run_once(config) → RunResult: single run.

Usage from notebooks:
    from aip_intern.mesh.runner import RunConfig, run
    results = asyncio.run(run(cfg))
"""

from __future__ import annotations

import dataclasses
import time
import uuid

from aip_intern.baseline.runner import RunConfig, RunResult
from aip_intern.core.exceptions import AIPInternError
from aip_intern.core.metrics import RunMetrics
from aip_intern.core.tracing import get_callback, get_langfuse
from aip_intern.mesh.graph import build_graph
from aip_intern.mesh.state import MeshState


async def run_once(cfg: RunConfig) -> RunResult:
    """Execute a single mesh run."""
    run_id = f"{cfg.run_id_prefix}_{uuid.uuid4().hex[:8]}"
    artifacts_run_dir = cfg.artifacts_dir / run_id
    outputs_dir = artifacts_run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    from aip_intern.core.config import LLMCfg
    llm_cfg = LLMCfg(
        model=cfg.llm_model,
        base_url=cfg.llm_base_url,
        api_key=cfg.llm_api_key,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
        request_timeout=cfg.llm_request_timeout,
    )

    metrics = RunMetrics(run_id=run_id)
    lf = get_langfuse()
    cb = get_callback(lf)
    invoke_config = {"callbacks": [cb]} if cb else {}

    initial_state: MeshState = {
        "run_id": run_id,
        "task_description": "Triage citizen feedback → action brief → response drafts",
        "triage_result": None,
        "brief_result": None,
        "response_result": None,
        "error": None,
        "step_trace": [],
        "message_count": 0,
        "state_size_bytes": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    graph = build_graph(llm_cfg, workspace_root=cfg.workspace_root, artifacts_outputs=outputs_dir)

    t0 = time.perf_counter()
    try:
        result_state = await graph.ainvoke(initial_state, config=invoke_config)
        metrics.total_latency_s = time.perf_counter() - t0
        metrics.step_trace = result_state.get("step_trace", [])
        metrics.message_count = result_state.get("message_count", 0)
        metrics.state_size_bytes = result_state.get("state_size_bytes", 0)
        metrics.total_prompt_tokens = result_state.get("prompt_tokens", 0)
        metrics.total_completion_tokens = result_state.get("completion_tokens", 0)
        success = result_state.get("error") is None
        error_msg = result_state.get("error")
    except AIPInternError as e:
        metrics.total_latency_s = time.perf_counter() - t0
        success = False
        error_msg = str(e)
        metrics.error = error_msg
    except Exception as e:
        metrics.total_latency_s = time.perf_counter() - t0
        success = False
        error_msg = f"Unexpected error: {e}"
        metrics.error = error_msg

    metrics.write(artifacts_run_dir / "metrics.json")

    return RunResult(
        run_id=run_id,
        success=success,
        error=error_msg if not success else None,
        metrics=dataclasses.asdict(metrics),
        outputs_path=outputs_dir,
    )


async def run(cfg: RunConfig) -> list[RunResult]:
    """Execute cfg.n_runs sequential mesh runs."""
    results = []
    for i in range(cfg.n_runs):
        print(f"  Mesh run {i + 1}/{cfg.n_runs}...", end=" ", flush=True)
        result = await run_once(cfg)
        print("OK" if result.success else f"ERR: {result.error}")
        results.append(result)
    return results
