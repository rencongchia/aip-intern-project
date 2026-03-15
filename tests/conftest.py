"""Shared test fixtures.

mock_llm: deterministic ChatModel for unit tests — no network required.
live_llm: real LLM from env — skipped if OPENAI_BASE_URL not set.
sample_baseline_state: minimal valid BaselineState for node tests.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture
def mock_llm():
    """A mock ChatModel that returns a deterministic AIMessage.

    Usage in tests:
        result = await triage_node(state, llm=mock_llm, tools=[])

    Override the return value for specific tests:
        mock_llm.ainvoke.return_value = AIMessage(content="custom response")
    """
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="mock response",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
    )
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def live_llm():
    """Real LLM from env. Skipped if OPENAI_BASE_URL is not set."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        pytest.skip("OPENAI_BASE_URL not set — skipping live LLM test")
    from aip_intern.core.config import LLMCfg
    from aip_intern.core.llm import create_llm
    cfg = LLMCfg(
        model=os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-32B-Instruct"),
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
    )
    return create_llm(cfg)


@pytest.fixture
def sample_baseline_state():
    """Minimal valid BaselineState for node unit tests."""
    return {
        "run_id": "test_run_001",
        "task_description": "Triage citizen feedback → action brief → response drafts",
        "error": None,
        "step_trace": [],
        "triage_result": None,
        "brief_result": None,
        "response_result": None,
    }
