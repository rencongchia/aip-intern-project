"""GCC constraint simulation — Python-level network/rate constraints.

Simulates three GCC (Government Commercial Cloud) constraints without
requiring toxiproxy or external infrastructure:

1. Added latency (200ms per LLM call) — asyncio.sleep injected into crew_node
2. Egress proxy overhead — additional 50ms + jitter per request
3. TPM rate cap (60k tokens/min) — sliding-window token counter that sleeps
   when the budget is exhausted

These constraints wrap around the Phase 3 fault injectors so the same four
failure types can be re-run under GCC conditions.

Usage:
    from aip_intern.phase4.gcc_constraints import GCCConstraints, apply_gcc_constraints

    gcc = GCCConstraints(latency_ms=200, tpm_limit=60_000)
    constrained_node = apply_gcc_constraints(crew_node, gcc)
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class GCCConstraints:
    """Parameters for GCC simulation."""

    latency_ms: int = 200          # added latency per LLM call (ms)
    egress_jitter_ms: int = 50     # egress proxy jitter (ms)
    tpm_limit: int = 60_000        # tokens per minute rate cap
    _token_log: list[tuple[float, int]] = field(default_factory=list, repr=False)

    def record_tokens(self, n_tokens: int) -> None:
        """Record token usage for rate limiting."""
        self._token_log.append((time.time(), n_tokens))

    def tokens_used_last_minute(self) -> int:
        """Sum tokens consumed in the last 60 seconds."""
        cutoff = time.time() - 60.0
        self._token_log = [(t, n) for t, n in self._token_log if t > cutoff]
        return sum(n for _, n in self._token_log)

    async def enforce_rate_limit(self, estimated_tokens: int = 2000) -> float:
        """Sleep if adding estimated_tokens would exceed TPM limit.

        Returns the number of seconds slept (0.0 if no wait needed).
        """
        used = self.tokens_used_last_minute()
        if used + estimated_tokens > self.tpm_limit:
            # Sleep until oldest entry falls out of the window
            if self._token_log:
                oldest_t = self._token_log[0][0]
                wait = max(0.0, 60.0 - (time.time() - oldest_t) + 0.5)
            else:
                wait = 5.0
            await asyncio.sleep(wait)
            return wait
        return 0.0

    async def add_latency(self) -> None:
        """Simulate GCC network latency + egress jitter."""
        base_ms = self.latency_ms
        jitter_ms = random.randint(0, self.egress_jitter_ms)
        await asyncio.sleep((base_ms + jitter_ms) / 1000.0)


def apply_gcc_constraints(node_fn: Callable, gcc: GCCConstraints) -> Callable:
    """Wrap a node function with GCC latency and rate-limit constraints.

    The wrapper:
    1. Enforces TPM rate limit (sleeps if budget exhausted)
    2. Adds GCC latency (200ms + jitter) before node execution
    3. Records estimated token usage after completion

    Args:
        node_fn: The original async node function (may already be fault-injected).
        gcc: GCCConstraints configuration.

    Returns:
        Wrapped async function with the same signature.
    """
    async def wrapped(state, **kwargs):
        # 1. Rate limit check
        rate_wait = await gcc.enforce_rate_limit()
        if rate_wait > 0:
            # Store rate-limit wait in state for metrics
            state_gcc = state.get("gcc_rate_wait_s", 0.0)
            state["gcc_rate_wait_s"] = state_gcc + rate_wait

        # 2. GCC latency injection
        await gcc.add_latency()

        # 3. Run the actual node (may be fault-injected)
        result = await node_fn(state, **kwargs)

        # 4. Record approximate token usage (estimate from state)
        gcc.record_tokens(2000)  # conservative estimate per node call

        return result

    # Preserve t_fault_holder if present (from malformed_json injector)
    if hasattr(node_fn, "t_fault_holder"):
        wrapped.t_fault_holder = node_fn.t_fault_holder  # type: ignore[attr-defined]

    return wrapped
