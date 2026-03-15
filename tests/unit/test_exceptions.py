import pytest
from aip_intern.core.exceptions import (
    AIPInternError,
    AgentTimeoutError,
    MalformedOutputError,
    CheckpointLostError,
    ContextOverflowError,
)

def test_hierarchy():
    assert issubclass(AgentTimeoutError, AIPInternError)
    assert issubclass(MalformedOutputError, AIPInternError)
    assert issubclass(CheckpointLostError, AIPInternError)
    assert issubclass(ContextOverflowError, AIPInternError)

def test_raise_and_catch_as_base():
    with pytest.raises(AIPInternError):
        raise AgentTimeoutError("node timed out after 30s")
