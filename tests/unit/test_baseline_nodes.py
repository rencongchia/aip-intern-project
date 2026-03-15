import pytest
from aip_intern.baseline.nodes import triage_node, brief_node, response_node
from aip_intern.baseline.state import BaselineState


def test_baseline_state_construction():
    state: BaselineState = {
        "run_id": "baseline_test",
        "task_description": "Triage feedback",
        "feedback_files": [],
        "policy_content": "",
        "triage_result": None,
        "brief_result": None,
        "response_result": None,
        "error": None,
        "step_trace": [],
    }
    assert state["run_id"] == "baseline_test"
    assert state["step_trace"] == []


def _make_state(**overrides) -> BaselineState:
    base: BaselineState = {
        "run_id": "test_run",
        "task_description": "test task",
        "feedback_files": ["data/feedback/msg_001.txt"],
        "policy_content": "SLA: 48h for HIGH",
        "triage_result": None,
        "brief_result": None,
        "response_result": None,
        "error": None,
        "step_trace": [],
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_triage_node_updates_step_trace(mock_llm):
    state = _make_state()
    result = await triage_node(state, llm=mock_llm, tools=[])
    assert "triage_node" in result["step_trace"]


@pytest.mark.asyncio
async def test_brief_node_requires_triage_result(mock_llm):
    """brief_node should set error if triage_result is None."""
    state = _make_state(triage_result=None)
    result = await brief_node(state, llm=mock_llm, tools=[])
    assert result.get("error") is not None


@pytest.mark.asyncio
async def test_response_node_requires_brief_result(mock_llm):
    """response_node should set error if brief_result is None."""
    state = _make_state(brief_result=None, triage_result="outputs/triage.csv")
    result = await response_node(state, llm=mock_llm, tools=[])
    assert result.get("error") is not None
