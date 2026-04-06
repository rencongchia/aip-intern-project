"""Phase 3 fault injection sweep.

Extends the Phase 2 mesh runner with fault injection and recovery scoring.

run(cfg, fault_types) → list[RunResult]: execute n_runs × len(fault_types) iterations.
run_once(cfg, fault) → RunResult: single fault-injected run.

Each run_once():
  1. Builds the mesh graph with the fault injector applied to crew_node.
  2. Runs ainvoke() and captures t_fault + t_end.
  3. Calls score_recovery() and writes RecoveryScore fields to metrics.json.

metrics.json for Phase 3 runs has all standard mesh fields PLUS:
  - failure_type, recovery_time_s, output_quality, recovery_mode, notes

Usage from notebooks:
    from aip_intern.phase3.runner import FaultRunConfig, run
    results = asyncio.run(run(run_cfg, fault_types))
"""

from __future__ import annotations

import dataclasses
import time
import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from aip_intern.baseline.runner import RunConfig, RunResult
from aip_intern.core.config import LLMCfg
from aip_intern.core.exceptions import AIPInternError
from aip_intern.core.metrics import RunMetrics
from aip_intern.core.tracing import get_callback, get_langfuse
from aip_intern.failures.injectors import (
    inject_checkpoint_loss,
    inject_context_overflow,
    inject_malformed_json,
    inject_timeout,
)
from aip_intern.failures.scoring import RecoveryScore, score_recovery
from aip_intern.mesh.crew_node import crew_node
from aip_intern.mesh.graph import _build_crew
from aip_intern.mesh.nodes import supervisor_node
from aip_intern.mesh.state import MeshState

from langgraph.graph import END, START, StateGraph


@dataclass
class FaultRunConfig:
    """Parameters for a single fault type.

    Used by run() to parametrize run_once() across the 4 fault types.
    """
    fault_name: str           # "timeout" | "malformed_json" | "checkpoint_loss" | "context_overflow"
    after_seconds: float = 3.0    # inject_timeout: deadline in seconds
    fail_on_call: int = 1         # inject_malformed_json: which tool call to corrupt
    token_limit: int = 500        # inject_context_overflow: token threshold


def _build_fault_graph(llm_cfg: LLMCfg, workspace_root: Path, fault: FaultRunConfig):
    """Build a mesh StateGraph with the specified fault injector applied to crew_node."""
    crew = _build_crew(llm_cfg, workspace_root)

    if fault.fault_name == "timeout":
        injected = inject_timeout(crew_node, after_seconds=fault.after_seconds)
    elif fault.fault_name == "malformed_json":
        injected = inject_malformed_json(crew_node, fail_on_call=fault.fail_on_call)
    elif fault.fault_name == "checkpoint_loss":
        injected = inject_checkpoint_loss(crew_node)
    elif fault.fault_name == "context_overflow":
        injected = inject_context_overflow(
            crew_node,
            token_limit=fault.token_limit,
            workspace_root=workspace_root,
        )
    else:
        raise ValueError(f"Unknown fault_name: {fault.fault_name!r}")

    builder = StateGraph(MeshState)
    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node("crew_node", partial(injected, crew=crew))
    builder.add_edge(START, "supervisor_node")
    builder.add_edge("supervisor_node", "crew_node")
    builder.add_edge("crew_node", END)

    return builder.compile(), injected  # return injected so runner can read t_fault_holder


async def run_once(cfg: RunConfig, fault: FaultRunConfig) -> RunResult:
    """Execute a single fault-injected mesh run.

    Creates a fresh run_id, builds the fault graph, ainvokes it, scores recovery,
    and writes extended metrics.json (base mesh fields + RecoveryScore fields).
    """
    fault_prefix = f"failures_{fault.fault_name.replace('_', '')}"
    run_id = f"{fault_prefix}_{uuid.uuid4().hex[:8]}"
    artifacts_run_dir = cfg.artifacts_dir / run_id
    outputs_dir = artifacts_run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

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
    }

    graph, injected_node = _build_fault_graph(llm_cfg, cfg.workspace_root, fault)

    t_run_start = time.time()
    t_fault: float | None = None
    graph_reached_end = False
    success = False
    error_msg: str | None = None

    t0 = time.perf_counter()
    try:
        result_state = await graph.ainvoke(initial_state, config=invoke_config)
        metrics.total_latency_s = time.perf_counter() - t0
        metrics.step_trace = result_state.get("step_trace", [])
        metrics.message_count = result_state.get("message_count", 0)
        metrics.state_size_bytes = result_state.get("state_size_bytes", 0)
        error_msg = result_state.get("error")
        success = error_msg is None
        if not success:
            metrics.error = error_msg
        graph_reached_end = True  # graph reached END (even if error stored in state)
    except AIPInternError as e:
        metrics.total_latency_s = time.perf_counter() - t0
        success = False
        error_msg = str(e)
        metrics.error = error_msg
        graph_reached_end = False
        # t_fault is attached to the exception for injectors that raise directly
        t_fault = getattr(e, "t_fault", None)
    except Exception as e:
        metrics.total_latency_s = time.perf_counter() - t0
        success = False
        error_msg = f"Unexpected error: {e}"
        metrics.error = error_msg
        graph_reached_end = False

    t_end = time.time()

    # For malformed_json: fault fires inside crew (caught internally), read from holder
    if t_fault is None and hasattr(injected_node, "t_fault_holder"):
        t_fault = injected_node.t_fault_holder[0]

    # Fall back to run start if no injector timestamp was recorded
    if t_fault is None:
        t_fault = t_run_start

    recovery: RecoveryScore = score_recovery(
        failure_type=fault.fault_name,
        t_fault=t_fault,
        t_end=t_end,
        outputs_path=outputs_dir,
        graph_reached_end=graph_reached_end,
    )

    # Write extended metrics: base mesh fields + RecoveryScore fields
    metrics_dict = dataclasses.asdict(metrics)
    metrics_dict.update(dataclasses.asdict(recovery))
    metrics_dict["gcc_sim"] = False  # Phase 4 will set True

    import json
    (artifacts_run_dir / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))

    return RunResult(
        run_id=run_id,
        success=success,
        error=error_msg if not success else None,
        metrics=metrics_dict,
        outputs_path=outputs_dir,
    )


async def run(cfg: RunConfig, fault_types: list[FaultRunConfig]) -> list[RunResult]:
    """Execute cfg.n_runs × len(fault_types) sequential fault-injected runs.

    Runs each fault type cfg.n_runs times. Returns all RunResult objects ordered
    by fault_type then run number.
    """
    results = []
    total = cfg.n_runs * len(fault_types)
    completed = 0
    for fault in fault_types:
        for i in range(cfg.n_runs):
            completed += 1
            print(
                f"  Phase3 run {completed}/{total}"
                f" [{fault.fault_name}  {i + 1}/{cfg.n_runs}]...",
                end=" ",
                flush=True,
            )
            result = await run_once(cfg, fault)
            status = "OK" if result.success else f"ERR: {result.error[:60]}"
            print(status)
            results.append(result)
    return results
