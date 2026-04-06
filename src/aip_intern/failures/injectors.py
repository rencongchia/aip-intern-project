"""Fault injection library — Phase 3 implementation.

Each injector wraps a node function and intercepts execution to simulate
a specific failure condition. The LangGraph graph does not need to change —
injectors are applied at graph construction time by replacing node functions.

Usage in the Phase 3 runner (build_fault_graph):

    from functools import partial
    from aip_intern.failures.injectors import inject_timeout
    from aip_intern.mesh.graph import _build_crew
    from aip_intern.mesh.crew_node import crew_node

    crew = _build_crew(llm_cfg, workspace_root)
    injected = inject_timeout(crew_node, after_seconds=3.0)
    builder.add_node("crew_node", partial(injected, crew=crew))

t_fault timestamps:
- inject_timeout, inject_checkpoint_loss, inject_context_overflow: attach t_fault as an
  attribute on the raised AIPInternError. Runner reads via getattr(e, 't_fault', t_run_start).
- inject_malformed_json: the JSONDecodeError is caught internally by crew_node (not an
  AIPInternError). t_fault is stored in `wrapped.t_fault_holder[0]` (a list of length 1).
  Runner reads via injected_fn.t_fault_holder[0].
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Callable

import tiktoken

from aip_intern.core.exceptions import (
    AgentTimeoutError,
    CheckpointLostError,
    ContextOverflowError,
    MalformedOutputError,  # noqa: F401 — re-exported for runner convenience
)


def inject_timeout(node_fn: Callable, after_seconds: float = 5.0) -> Callable:
    """Wrap a node function to raise AgentTimeoutError after after_seconds.

    Uses asyncio.wait_for to impose a deadline on the node coroutine.
    t_fault is attached to the raised AgentTimeoutError as err.t_fault (Unix time).

    Args:
        node_fn: The original async node function to wrap.
        after_seconds: Time budget before timeout is raised.

    Returns:
        Wrapped async function with the same signature as node_fn.
    """
    async def wrapped(state, **kwargs):
        # Pre-sleep guarantees the timeout fires before any LLM response (including
        # fast 429 rate-limit errors). crew_node catches all non-AIPInternError
        # exceptions internally, so a hung LLM call would return normally — the
        # timeout would never fire. A sleep(after_seconds * 2) always outlasts the
        # wait_for deadline, making timeout injection reliable and environment-agnostic.
        async def _hung():
            await asyncio.sleep(after_seconds * 2)
            return await node_fn(state, **kwargs)  # never reached

        try:
            return await asyncio.wait_for(_hung(), timeout=after_seconds)
        except asyncio.TimeoutError:
            err = AgentTimeoutError(
                f"Node timed out after {after_seconds}s (injected fault)"
            )
            err.t_fault = time.time()  # type: ignore[attr-defined]
            raise err

    return wrapped


def inject_malformed_json(node_fn: Callable, fail_on_call: int = 1) -> Callable:
    """Wrap a node to raise a JSONDecodeError on the Nth tool call inside the crew.

    Patches tool._run on all tools across all crew agents. The call counter is
    shared across all tools, counting total tool invocations in the crew run order.

    The JSONDecodeError is caught by crew_node's broad `except Exception` handler and
    stored as state['error']. The graph reaches END normally (automatic recovery).
    t_fault is NOT on an exception — read it via wrapped_fn.t_fault_holder[0] after
    the graph completes.

    Args:
        node_fn: The original async node function (crew_node).
        fail_on_call: Which tool call invocation to corrupt (1-indexed, across all tools).

    Returns:
        Wrapped async function. Has attribute t_fault_holder: list[float | None].
    """
    t_fault_holder: list[float | None] = [None]

    async def wrapped(state, **kwargs):
        crew = kwargs.get("crew")
        if crew is not None:
            call_counter = [0]
            for agent in crew.agents:
                for tool in agent.tools:
                    original_run = tool._run

                    def _make_patched(orig, counter, holder, n):
                        def patched(*args, **kw):
                            counter[0] += 1
                            if counter[0] == n:
                                holder[0] = time.time()
                                raise json.JSONDecodeError(
                                    "Injected malformed JSON from tool call", "", 0
                                )
                            return orig(*args, **kw)
                        return patched

                    tool._run = _make_patched(original_run, call_counter, t_fault_holder, fail_on_call)

        return await node_fn(state, **kwargs)

    wrapped.t_fault_holder = t_fault_holder  # type: ignore[attr-defined]
    return wrapped


def inject_checkpoint_loss(node_fn: Callable) -> Callable:
    """Wrap a node to raise CheckpointLostError after the node runs to completion.

    The node executes fully (crew runs, output files are written to disk), but
    the state update dict is discarded and CheckpointLostError is raised before
    LangGraph can record the state. The next node is never reached.

    This demonstrates checkpoint asymmetry: output files exist on disk, but
    LangGraph state has no record of them. Recovery requires human restart.
    t_fault is attached to the raised CheckpointLostError as err.t_fault.

    Args:
        node_fn: The original async node function.

    Returns:
        Wrapped async function with the same signature as node_fn.
    """
    async def wrapped(state, **kwargs):
        await node_fn(state, **kwargs)  # crew runs; output files written to disk
        err = CheckpointLostError(
            "State checkpoint written but lost on read-back (injected fault). "
            "Output files may exist on disk but LangGraph state has no record of them."
        )
        err.t_fault = time.time()  # type: ignore[attr-defined]
        raise err

    return wrapped


def inject_context_overflow(
    node_fn: Callable,
    token_limit: int = 1000,
    workspace_root: Path | None = None,
) -> Callable:
    """Wrap a node to raise ContextOverflowError when estimated context exceeds token_limit.

    Counts tokens using tiktoken cl100k_base (approximate — not model-specific).
    Counts task_description + policy_content from state, plus feedback files and
    policy snippets from workspace_root if provided. This approximates what the
    LLM would see as total prompt context.

    The check fires BEFORE the crew is invoked. t_fault is attached to the raised
    ContextOverflowError as err.t_fault.

    Args:
        node_fn: The original async node function.
        token_limit: Token threshold that triggers overflow.
        workspace_root: If provided, reads workspace files to build a realistic estimate.
            Defaults to Path("workspace/") if not provided.

    Returns:
        Wrapped async function with the same signature as node_fn.
    """
    _workspace = Path(workspace_root) if workspace_root is not None else Path("workspace/")

    async def wrapped(state, **kwargs):
        enc = tiktoken.get_encoding("cl100k_base")
        texts = [
            state.get("task_description", ""),
            state.get("policy_content", ""),
        ]
        # Include workspace files to simulate realistic prompt context
        feedback_dir = _workspace / "data" / "feedback"
        if feedback_dir.exists():
            for f in sorted(feedback_dir.iterdir()):
                if f.is_file():
                    texts.append(f.read_text(errors="replace"))
        policy_file = _workspace / "data" / "policy_snippets.md"
        if policy_file.exists():
            texts.append(policy_file.read_text(errors="replace"))

        n_tokens = len(enc.encode(" ".join(texts)))
        if n_tokens > token_limit:
            err = ContextOverflowError(
                f"Estimated context {n_tokens} tokens exceeds limit {token_limit} "
                f"(injected fault, tiktoken cl100k_base approximation)"
            )
            err.t_fault = time.time()  # type: ignore[attr-defined]
            raise err

        return await node_fn(state, **kwargs)

    return wrapped
