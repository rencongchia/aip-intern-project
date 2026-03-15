from __future__ import annotations

from aip_intern.mesh.state import MeshState


def test_mesh_state_has_mesh_metrics():
    state: MeshState = {
        "run_id": "mesh_test",
        "task_description": "test",
        "error": None,
        "step_trace": [],
        "triage_result": None,
        "brief_result": None,
        "response_result": None,
        "message_count": 0,
        "state_size_bytes": 0,
    }
    assert state["message_count"] == 0
    assert state["state_size_bytes"] == 0
