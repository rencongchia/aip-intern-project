"""LLM client factory.

Returns a langchain_openai.ChatOpenAI configured for either vLLM or OpenAI.
Switching between them requires only a config change — no code change.

Usage:
    llm = create_llm(cfg.llm)
    response = await llm.ainvoke(messages)
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from aip_intern.core.config import LLMCfg


def create_llm(cfg: LLMCfg) -> ChatOpenAI:
    """Create a ChatOpenAI client from LLMCfg.

    Works with:
      - Local vLLM: base_url = "http://<gpu-ip>:8000/v1", api_key = "not-needed"
      - OpenAI API: base_url = "https://api.openai.com/v1", api_key = real key

    For unit tests, use the mock_llm fixture in tests/conftest.py instead.
    """
    return ChatOpenAI(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        request_timeout=cfg.request_timeout,
    )
