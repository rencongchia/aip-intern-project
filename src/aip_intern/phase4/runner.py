"""Phase 4 GCC-constrained fault injection sweep.

Re-runs the same four Phase 3 fault types under simulated GCC constraints:
- 200ms added latency per LLM call
- Egress proxy jitter (50ms)
- 60k TPM rate cap

The key research question: which Phase 3 failures become unrecoverable
under GCC network constraints?

run(cfg, fault_types, gcc) → list[RunResult]
run_once(cfg, fault, gcc) → RunResult

metrics.json for Phase 4 adds:
  - gcc_sim: True
  - gcc_latency_ms: configured latency
  - gcc_tpm_limit: configured TPM cap
  - gcc_rate_wait_s: total seconds spent waiting for rate limit
"""

from __future__ import annotations

import dataclasses
import json
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
from aip_intern.phase4.gcc_constraints import GCCConstraints, apply_gcc_constraints

from langgraph.graph import END, START, StateGraph


@dataclass
class GCCRunConfig:
    """GCC constraint parameters from config YAML."""

    latency_ms: int = 200
    egress_jitter_ms: int = 50
    tpm_limit: int = 60_000


# Reuse FaultRunConfig from Phase 3
from aip_intern.phase3.runner import FaultRunConfig


def _build_gcc_fault_graph(
    llm_cfg: LLMCfg,
    workspace_root: Path,
    fault: FaultRunConfig,
    gcc: GCCConstraints,
    artifacts_outputs: Path | None = None,
):
    """Build a mesh StateGraph with fault injection + GCC constraints.

    Layering order: crew_node → fault injector → GCC constraints.
    GCC constraints wrap the outer layer so latency/rate-limit is applied
    even when the fault fires before the LLM call.
    """
    crew = _build_crew(llm_cfg, workspace_root)

    # Layer 1: fault injection
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

    # Layer 2: GCC constraints wrap the fault-injected node
    constrained = apply_gcc_constraints(injected, gcc)

    builder = StateGraph(MeshState)
    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node(
        "crew_node",
        partial(
            constrained,
            crew=crew,
            workspace_root=workspace_root,
            artifacts_outputs=artifacts_outputs,
        ),
    )
    builder.add_edge(START, "supervisor_node")
    builder.add_edge("supervisor_node", "crew_node")
    builder.add_edge("crew_node", END)

    return builder.compile(), constrained


async def run_once(
    cfg: RunConfig, fault: FaultRunConfig, gcc: GCCConstraints
) -> RunResult:
    """Execute a single GCC-constrained fault-injected mesh run."""
    fault_prefix = f"gcc_{fault.fault_name.replace('_', '')}"
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
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    graph, constrained_node = _build_gcc_fault_graph(
        llm_cfg, cfg.workspace_root, fault, gcc, artifacts_outputs=outputs_dir
    )

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
        metrics.total_prompt_tokens = result_state.get("prompt_tokens", 0)
        metrics.total_completion_tokens = result_state.get("completion_tokens", 0)
        error_msg = result_state.get("error")
        success = error_msg is None
        if not success:
            metrics.error = error_msg
        graph_reached_end = True
    except AIPInternError as e:
        metrics.total_latency_s = time.perf_counter() - t0
        success = False
        error_msg = str(e)
        metrics.error = error_msg
        graph_reached_end = False
        t_fault = getattr(e, "t_fault", None)
        # checkpoint_loss: crew ran (tokens spent) but state was discarded
        node_result = getattr(e, "node_result", None)
        if node_result:
            metrics.total_prompt_tokens = node_result.get("prompt_tokens", 0)
            metrics.total_completion_tokens = node_result.get("completion_tokens", 0)
    except Exception as e:
        metrics.total_latency_s = time.perf_counter() - t0
        success = False
        error_msg = f"Unexpected error: {e}"
        metrics.error = error_msg
        graph_reached_end = False

    t_end = time.time()

    # Read t_fault from malformed_json holder if available
    if t_fault is None and hasattr(constrained_node, "t_fault_holder"):
        t_fault = constrained_node.t_fault_holder[0]
    if t_fault is None:
        t_fault = t_run_start

    recovery: RecoveryScore = score_recovery(
        failure_type=fault.fault_name,
        t_fault=t_fault,
        t_end=t_end,
        outputs_path=outputs_dir,
        graph_reached_end=graph_reached_end,
    )

    # Determine if GCC constraints made this unrecoverable
    # Unrecoverable = manual recovery AND total latency exceeds 2× the non-GCC baseline
    # (heuristic: if GCC overhead pushed recovery_time beyond practical limits)
    gcc_rate_wait = initial_state.get("gcc_rate_wait_s", 0.0)
    if (
        recovery.recovery_mode == "manual"
        and metrics.total_latency_s > fault.after_seconds * 3
        and gcc_rate_wait > 0
    ):
        recovery.recovery_mode = "unrecoverable"
        recovery.notes += (
            " GCC rate-limit wait caused cascading failure — "
            "recovery not feasible within SLA."
        )

    # Write extended metrics
    metrics_dict = dataclasses.asdict(metrics)
    metrics_dict.update(dataclasses.asdict(recovery))
    metrics_dict["gcc_sim"] = True
    metrics_dict["gcc_latency_ms"] = gcc.latency_ms
    metrics_dict["gcc_tpm_limit"] = gcc.tpm_limit
    metrics_dict["gcc_rate_wait_s"] = gcc_rate_wait

    (artifacts_run_dir / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))

    return RunResult(
        run_id=run_id,
        success=success,
        error=error_msg if not success else None,
        metrics=metrics_dict,
        outputs_path=outputs_dir,
    )


async def run(
    cfg: RunConfig,
    fault_types: list[FaultRunConfig],
    gcc_cfg: GCCRunConfig,
) -> list[RunResult]:
    """Execute cfg.n_runs × len(fault_types) GCC-constrained fault runs."""
    import asyncio

    gcc = GCCConstraints(
        latency_ms=gcc_cfg.latency_ms,
        egress_jitter_ms=gcc_cfg.egress_jitter_ms,
        tpm_limit=gcc_cfg.tpm_limit,
    )

    results = []
    total = cfg.n_runs * len(fault_types)
    completed = 0
    inter_run_delay_s = getattr(cfg, "inter_run_delay_s", 0)

    for fault in fault_types:
        for i in range(cfg.n_runs):
            completed += 1
            print(
                f"  Phase4 run {completed}/{total}"
                f" [{fault.fault_name}  {i + 1}/{cfg.n_runs}]...",
                end=" ",
                flush=True,
            )
            result = await run_once(cfg, fault, gcc)
            status = "OK" if result.success else f"ERR: {result.error[:60]}"
            print(status)
            results.append(result)
            if inter_run_delay_s > 0 and completed < total:
                await asyncio.sleep(inter_run_delay_s)
    return results
