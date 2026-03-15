"""Typed exception hierarchy for aip-intern-project.

Phase 3 interns: these are the injectable fault types.
Nodes in baseline/ and mesh/ raise these; failures/injectors.py
intercepts them at graph boundaries to simulate failure conditions.
"""


class AIPInternError(Exception):
    """Base exception for all aip-intern errors."""


class AgentTimeoutError(AIPInternError):
    """Raised when an agent node exceeds its time budget."""


class MalformedOutputError(AIPInternError):
    """Raised when a tool call returns malformed or unparseable JSON."""


class CheckpointLostError(AIPInternError):
    """Raised when a LangGraph state checkpoint cannot be read or written."""


class ContextOverflowError(AIPInternError):
    """Raised when token count exceeds the model's context window."""
