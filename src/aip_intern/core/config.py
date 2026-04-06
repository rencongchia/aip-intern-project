"""Config loading for aip-intern-project.

Loads a YAML file and resolves ${ENV_VAR} placeholders from the environment.
Follows the same pattern as phase3's DemoConfig but with richer run/metrics fields.

Usage:
    cfg = load_config(Path("config/baseline.yaml"))
    llm = create_llm(cfg.llm)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


def _resolve_env(value: str) -> str:
    """Replace ${VAR} with os.environ.get(VAR, original_placeholder)."""
    return re.sub(
        r"\$\{(\w+)\}",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        value,
    )


def _resolve_dict(d: dict) -> dict:
    """Recursively resolve env var placeholders in all string values."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _resolve_dict(v)
        elif isinstance(v, str):
            result[k] = _resolve_env(v)
        else:
            result[k] = v
    return result


@dataclass(frozen=True)
class RunCfg:
    n_runs: int = 20
    run_id_prefix: str = "run"
    inter_run_delay_s: float = 0.0


@dataclass(frozen=True)
class LLMCfg:
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 4096
    request_timeout: int = 120

    @property
    def is_mock(self) -> bool:
        return self.base_url == "mock"


@dataclass(frozen=True)
class MCPCfg:
    workspace_root: str = "workspace/"


@dataclass(frozen=True)
class ArtifactsCfg:
    output_dir: str = "artifacts/"


@dataclass(frozen=True)
class AppConfig:
    run: RunCfg
    llm: LLMCfg
    mcp: MCPCfg
    artifacts: ArtifactsCfg
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"


def load_config(path: Path) -> AppConfig:
    """Load and validate config from a YAML file, resolving ${ENV_VAR} placeholders."""
    raw = yaml.safe_load(path.read_text())
    resolved = _resolve_dict(raw)
    return AppConfig(
        run=RunCfg(**resolved.get("run", {})),
        llm=LLMCfg(**resolved["llm"]),
        mcp=MCPCfg(**resolved.get("mcp", {})),
        artifacts=ArtifactsCfg(**resolved.get("artifacts", {})),
        langfuse_enabled=resolved.get("langfuse_enabled", False),
        langfuse_public_key=resolved.get("langfuse_public_key", ""),
        langfuse_secret_key=resolved.get("langfuse_secret_key", ""),
        langfuse_host=resolved.get("langfuse_host", "http://localhost:3000"),
    )


# ---------------------------------------------------------------------------
# Run-level data contracts (used by runner.py and returned to notebooks)
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Everything a single run needs — passed to run_once() and run().

    config_path preserves the YAML that produced this run, so notebooks
    can display exactly what config was used.
    """

    run_id: str               # e.g. "baseline_<uuid>" — assigned by runner if not provided
    n_runs: int               # number of sweep iterations
    config_path: Path         # path to the YAML that produced this run
    llm_model: str
    llm_base_url: str
    workspace_root: Path
    artifacts_dir: Path


@dataclass
class RunResult:
    """Result of a single run returned by run_once()."""

    run_id: str
    success: bool
    error: Optional[str]              # exception message if success=False
    metrics: dict                     # keys match Metrics Table in spec
    outputs_path: Path                # path to artifacts/{run_id}/outputs/
    langfuse_trace_url: Optional[str]
