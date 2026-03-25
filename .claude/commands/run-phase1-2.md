# /run-phase1-2 — Phase 1 & 2 Sweep Pipeline Orchestration

You are orchestrating the Phase 1 (LangGraph baseline) and Phase 2 (LangGraph + CrewAI mesh)
benchmark sweeps from a local machine. All commands run locally via Bash tool against a remote
vLLM endpoint defined in `.env`. No infrastructure deployment is required.

Working directory context: **all relative paths and commands below must be run from the repo root**
(`/Users/loremipsum/Code/aip-intern-project`). The analysis imports (`from analysis.aggregate
import load_runs`) resolve from that root.

---

## Pre-Flight Checks

Auto-fix everything you can. Only stop and ask the user for things that require human input.

### Step 1 — Auto-fix the environment

Run these in order. Each is idempotent (safe to re-run).

```bash
# Create .env from template if missing (never overwrites an existing .env)
test -f .env || cp .env.example .env

# Install package if missing
python3 -c "import aip_intern" 2>/dev/null || pip3 install -e ".[dev]"

# Create required directories
mkdir -p artifacts/ docs/
```

### Step 2 — Check what requires human input

Run both checks. If either fails, stop and tell the user exactly what to set:

```bash
# Fails if OPENAI_BASE_URL is still the placeholder value
grep -E "^OPENAI_BASE_URL=http" .env \
  && grep -v "your-vllm-endpoint" .env | grep -q "OPENAI_BASE_URL" \
  && echo "BASE_URL OK" \
  || echo "STOP: Set OPENAI_BASE_URL to your real vLLM endpoint in .env"

# Fails if key is empty or missing (vLLM accepts any non-empty string)
grep -E "^OPENAI_API_KEY=.+" .env && echo "API_KEY OK" || echo "STOP: Set OPENAI_API_KEY in .env (any non-empty string works for vLLM)"
```

If either prints `STOP:` — halt the pipeline and ask the user to fill in `.env`, then re-run
`/run-phase1-2`. Do not proceed.

### Step 3 — Verify endpoint reachability

```bash
OPENAI_BASE_URL=$(grep "^OPENAI_BASE_URL=" .env | cut -d= -f2-)
curl -s --max-time 5 "${OPENAI_BASE_URL}/models" | python3 -m json.tool | head -10
```

If curl fails (timeout, connection refused), stop and report to user — nothing will work without
a live endpoint.

### Step 4 — Confirm workspace data

```bash
ls workspace/data/feedback/
```

Must list at least one file. If empty or missing, stop and report.

---

## Idempotency Rules

Check artifact counts before running any sweep. These are the target counts from the configs:
- Baseline target: 20 runs (`config/baseline.yaml` → `run.n_runs: 20`)
- Mesh target: 20 runs (`config/mesh.yaml` → `run.n_runs: 20`)

```bash
BASELINE_COUNT=$(ls -d artifacts/baseline_* 2>/dev/null | wc -l | tr -d ' ')
MESH_COUNT=$(ls -d artifacts/mesh_* 2>/dev/null | wc -l | tr -d ' ')
echo "Existing baseline runs: $BASELINE_COUNT / 20"
echo "Existing mesh runs:     $MESH_COUNT / 20"
```

Apply these rules before each sweep stage:

| Condition | Action |
|-----------|--------|
| Count >= 20 | **SKIP** the dry-run and sweep for that phase; log as skipped |
| Count 1–19 | Run the full sweep — each run creates a new uuid so existing ones are safe; you will end up with more than 20 total, which is fine |
| Count == 0 | Run dry-run first, then full sweep |

The dry-run is always skipped when the sweep is skipped. Stages 1 and 3 (dry-runs) are
**never** skipped when their sweep is needed — run them first to validate connectivity.

---

## Pipeline

Run each stage in order. Track all outcomes in `docs/phase1-2-run-log.md`.

### Stage 1: Baseline Dry-Run (Connectivity Check)

Skip if `BASELINE_COUNT >= 20`. Otherwise:

```bash
python3 scripts/run_baseline.py --config config/baseline.yaml --dry-run
```

Success signal: `Complete: 1/1 runs succeeded.`

Fix any errors before proceeding to Stage 2.

### Stage 2: Full Baseline Sweep (20 Runs)

Skip if `BASELINE_COUNT >= 20` (print: `SKIP Stage 2: 20 baseline runs already exist`).
Otherwise:

```bash
python3 scripts/run_baseline.py --config config/baseline.yaml
```

Success signals:
```
Complete: 20/20 runs succeeded.
Artifacts: artifacts/
```

Each run writes to `artifacts/baseline_{uuid}/`:
- `outputs/triage.csv`
- `outputs/brief.md`
- `outputs/response_templates.md`
- `metrics.json`

Partial success (e.g. `18/20 runs succeeded`) is acceptable — log failures and continue to Stage 3.

### Stage 3: Mesh Dry-Run (Connectivity Check)

Skip if `MESH_COUNT >= 20`. Otherwise:

```bash
python3 scripts/run_mesh.py --config config/mesh.yaml --dry-run
```

Success signal: `Complete: 1/1 runs succeeded.`

Validates the LangGraph → CrewAI bridge and that `message_count` / `state_size_bytes`
fields are populated in `metrics.json`.

### Stage 4: Full Mesh Sweep (20 Runs)

Skip if `MESH_COUNT >= 20` (print: `SKIP Stage 4: 20 mesh runs already exist`).
Otherwise:

```bash
python3 scripts/run_mesh.py --config config/mesh.yaml
```

Success signals:
```
Complete: 20/20 runs succeeded.
Artifacts: artifacts/
```

Each run writes to `artifacts/mesh_{uuid}/` with the same outputs plus mesh-specific
metrics in `metrics.json`.

### Stage 5: Analysis & Comparison

Always run — it is read-only and cheap.

```bash
cd /Users/loremipsum/Code/aip-intern-project && python3 - <<'EOF'
from analysis.aggregate import load_runs
from analysis.compare import compare_phases

baseline_df = load_runs("artifacts/", prefix="baseline")
mesh_df = load_runs("artifacts/", prefix="mesh")

if baseline_df.empty and mesh_df.empty:
    print("WARNING: No artifacts found. Run Stages 2 and 4 first.")
else:
    comparison = compare_phases(baseline_df, mesh_df)
    print(comparison.to_string())
EOF
```

Success signal: printed comparison table with rows for both phases.

Sanity-check artifact counts:
```bash
ls -d artifacts/baseline_* 2>/dev/null | wc -l   # expect >= 20
ls -d artifacts/mesh_* 2>/dev/null | wc -l        # expect >= 20
```

---

## Error Recovery Protocol

When any stage fails, spawn diagnostic subagent first, then fix subagent with its findings:

**Diagnostic agent prompt template:**
```
Read the following error output and identify the root cause. Check relevant config
files and logs in the project. Return:
(1) Root cause in one sentence
(2) Specific file/config/command to fix
(3) Confidence level 1–10

Error output:
[paste full stderr/stdout]

Stage context: [which stage, e.g. "Stage 2: Full Baseline Sweep"]
Repo root: /Users/loremipsum/Code/aip-intern-project
```

**Fix agent prompt template:**
```
Apply the minimal fix to resolve this error. Do not refactor.
Do not modify workspace/data/, artifacts/, or .env.

Root cause: [diagnostic agent output]
Fix target: [file or command from diagnostic agent]
Repo root: /Users/loremipsum/Code/aip-intern-project
```

After both agents complete:
- Apply fix if confidence >= 7 and the change is safe.
- Retry the failed stage once.
- If retry fails: skip stage, append to run log, continue pipeline.

### Known Issues & Fixes

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| `OPENAI_BASE_URL` still `http://your-vllm-endpoint…` | `.env` not filled in | Edit `.env` and set real endpoint |
| `ModuleNotFoundError: aip_intern` | Package not installed | `pip3 install -e ".[dev]"` |
| `ModuleNotFoundError: dotenv` | python-dotenv missing | `pip3 install python-dotenv` |
| `ModuleNotFoundError: analysis` | Stage 5 not run from repo root | `cd /Users/loremipsum/Code/aip-intern-project` first |
| `FileNotFoundError: artifacts/` in Stage 5 | `artifacts/` dir doesn't exist yet | `mkdir -p artifacts/` then re-run Stage 5 |
| `CrewAI kickoff timeout` | Remote vLLM endpoint slow under load | Increase `llm.request_timeout` in `config/mesh.yaml` |
| `crew_node returned None for message_count` | `MeshState` field not populated | Check `src/aip_intern/mesh/crew_node.py` returns updated state with both fields |
| `load_runs() returns empty DataFrame` | `prefix` mismatch or no `metrics.json` | Verify `artifacts/` contains `baseline_*/metrics.json`; check prefix spelling |
| `KeyError: total_latency_s` in compare | Metrics schema mismatch | Verify both runner `metrics.json` files use same top-level keys |
| `command not found: python` | macOS ships `python3` only | Use `python3` (all commands in this skill already do) |
| `nbstripout strips 01_run or 02_analysis outputs` | gitattributes strips all notebooks | Check `.gitattributes` — `01_run.ipynb` and `02_analysis.ipynb` should commit outputs |
| `json.JSONDecodeError` in `load_runs()` | A `metrics.json` is corrupted | Delete or repair the corrupted `artifacts/<run_id>/metrics.json` and re-run Stage 5 |

---

## Run Log Format

Append to `docs/phase1-2-run-log.md` after each pipeline completes.

```markdown
## Run: YYYY-MM-DD HH:MM UTC

| Stage | Status | Notes |
|-------|--------|-------|
| 1. Baseline Dry-Run  | ✅ / ❌ / ⏭ skipped | |
| 2. Baseline Sweep    | ✅ / ❌ / ⏭ skipped | N/20 succeeded |
| 3. Mesh Dry-Run      | ✅ / ❌ / ⏭ skipped | |
| 4. Mesh Sweep        | ✅ / ❌ / ⏭ skipped | N/20 succeeded |
| 5. Analysis          | ✅ / ❌ | |

### Errors
<!-- For each ❌: paste root cause from diagnostic agent + what was attempted -->

### Artifacts
<!-- baseline: N runs, mesh: N runs, or "none" -->
```

---

## Key Paths

| What | Path |
|------|------|
| Baseline script | `scripts/run_baseline.py` |
| Mesh script | `scripts/run_mesh.py` |
| Baseline config | `config/baseline.yaml` |
| Mesh config | `config/mesh.yaml` |
| Baseline source | `src/aip_intern/baseline/` |
| Mesh source | `src/aip_intern/mesh/` (`crew_node.py`, `crew/agents.py`, `crew/tasks.py`) |
| Workspace input data | `workspace/data/` |
| Artifacts output | `artifacts/` (one dir per run) |
| Analysis module | `analysis/` (`aggregate.py`, `compare.py`, `plots.py`) |
| Phase 1 notebooks | `notebooks/phase1_baseline/` (`00_overview`, `01_run`, `02_analysis`) |
| Phase 2 notebooks | `notebooks/phase2_mesh/` (`00_overview`, `01_run`, `02_analysis`) |
| Run log | `docs/phase1-2-run-log.md` |
| Env template | `.env.example` → copy to `.env` |
| Config loader | `src/aip_intern/core/config.py` |

## Metrics Captured

| Metric | Phase 1 | Phase 2 | Notes |
|--------|---------|---------|-------|
| `total_latency_s` | Yes | Yes | Wall-clock time for full graph invocation |
| `total_prompt_tokens` | Yes | Yes | |
| `total_completion_tokens` | Yes | Yes | |
| `step_trace` | Yes | Yes | Ordered list of node names visited |
| `error` | Yes | Yes | Non-null string on failure, null on success |
| `message_count` | — | Yes | Inter-agent messages within the CrewAI crew; 0 if crew_node doesn't populate it |
| `state_size_bytes` | — | Yes | Serialised LangGraph state size at crew_node return; 0 if crew_node doesn't populate it |

## Langfuse Tracing (Optional)

Set `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_HOST` in `.env`.
Traces are submitted automatically when the keys are non-empty.

**Note on `LANGFUSE_HOST`:** If `LANGFUSE_HOST` is absent, the SDK defaults to
`https://cloud.langfuse.com` (Langfuse cloud), not `localhost:3000`. Set it explicitly
in `.env` if you're running a local Langfuse instance.

To run a sweep without tracing (without modifying `.env`):
```bash
LANGFUSE_SECRET_KEY= LANGFUSE_PUBLIC_KEY= python3 scripts/run_baseline.py --config config/baseline.yaml
```

---

## Extending to Phase 3+

To add a new phase to this pipeline, follow this checklist:

- [ ] Copy `config/mesh.yaml` → `config/phase3.yaml`, update `run_id_prefix` (e.g. `"failures"`)
- [ ] Copy `scripts/run_mesh.py` → `scripts/run_phase3.py`; change the runner import:
  ```python
  from aip_intern.phase3.runner import run   # update this line only
  ```
- [ ] Create `src/aip_intern/phase3/runner.py` with the same contract:
  ```python
  async def run(cfg: RunConfig) -> list[RunResult]: ...
  async def run_once(cfg: RunConfig) -> RunResult: ...
  ```
  `RunConfig` and `RunResult` are imported from `aip_intern.baseline.runner` (canonical location).
- [ ] If Phase 3 adds new metrics fields, add them to `analysis/aggregate.py`'s `load_runs()` row dict with a `.get("new_field")` default.
- [ ] Add a `phase3_dry_run` + `phase3_sweep` stage pair to a new `run-phase3.md` (or extend this skill if phases are tightly coupled).
- [ ] Update the run log format template to include the new stages.

**Convention:** run loop print prefix should be the phase name:
```python
print(f"  Phase3 run {i + 1}/{cfg.n_runs}...", end=" ", flush=True)
```
