"""OpenAI tool function definitions for CrewAI agents.

CrewAI agents use these as function-call tools (not MCP). They provide
read/write access to the workspace directory, analogous to phase3's MCP tools
but through the OpenAI tool-calling protocol.

Phase 3 interns: to inject a MalformedOutputError, replace read_file or
write_file with a version that raises MalformedOutputError on demand.
"""

from __future__ import annotations

from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel

_WORKSPACE_ROOT: Path = Path("workspace/")


def set_workspace_root(path: str | Path) -> None:
    """Set the workspace root. Call from runner before creating tools."""
    global _WORKSPACE_ROOT
    _WORKSPACE_ROOT = Path(path)


class _ReadFileInput(BaseModel):
    path: str  # relative to workspace root


class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = (
        "Read a file from the workspace. path must be relative to workspace root."
    )
    args_schema: type[BaseModel] = _ReadFileInput

    def _run(self, path: str) -> str:
        full_path = _WORKSPACE_ROOT / path
        if not full_path.exists():
            return f"Error: {path} not found"
        return full_path.read_text()


class _WriteFileInput(BaseModel):
    path: str
    content: str


class WriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = (
        "Write content to a file in the workspace."
        " path must be relative to workspace root."
    )
    args_schema: type[BaseModel] = _WriteFileInput

    def _run(self, path: str, content: str) -> str:
        full_path = _WORKSPACE_ROOT / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return f"Written: {path}"


class _ListDirInput(BaseModel):
    path: str


class ListDirectoryTool(BaseTool):
    name: str = "list_directory"
    description: str = (
        "List files in a workspace directory."
        " path must be relative to workspace root."
    )
    args_schema: type[BaseModel] = _ListDirInput

    def _run(self, path: str) -> str:
        full_path = _WORKSPACE_ROOT / path
        if not full_path.is_dir():
            return f"Error: {path} is not a directory"
        return "\n".join(p.name for p in sorted(full_path.iterdir()))


def get_tools() -> list[BaseTool]:
    """Return all workspace tools for CrewAI agents."""
    return [ReadFileTool(), WriteFileTool(), ListDirectoryTool()]
