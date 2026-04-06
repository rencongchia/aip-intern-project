# /run-phase3 — Phase 3 Fault Injection Sweep Orchestration

You are orchestrating the Phase 3 fault injection benchmark. This phase injects four failure
types into the Phase 2 LangGraph+CrewAI mesh and scores each on recovery time, output quality,
and whether recovery was automatic or required manual intervention.

Working directory: **all relative paths and commands below must be run from the repo root**
(`/home/rencongchia/projects/aip-intern-project`).

**Prerequisite**: Phase 1 and Phase 2 runs must exist in `artifacts/` before analysing
cross-phase comparisons. Phase 3 sweeps can run independently of Phase 1-2.

---

## Pre-Flight Checks

Auto-fix everything you can. Only stop and ask the user for things that require human input.

### Step 1 — Auto-fix the environment

```bash
# Create .env from template if missing
test -f .env || cp .env.example .env

# Install package (including tiktoken added for Phase 3)
python3 -c "import aip_intern, tiktoken" 2>/dev/null || pip3 install -e ".[dev]"

# Create required directories
mkdir -p artifacts/ docs/
```

### Step 2 — Check OpenRouter endpoint (required for Phase 3)

```bash
# Verify OPENAI_BASE_URL is set to OpenRouter (not the placeholder)
grep -E "^OPENAI_BASE_URL=https://openrouter\.ai" .env && echo "BASE_URL OK" \
  || echo "STOP: Set OPENAI_BASE_URL=https://openrouter.ai/api/v1 in .env"

# Verify API key is present
grep -E "^OPENAI_API_KEY=sk-or-" .env && echo "API_KEY OK" \
  || echo "STOP: Set OPENAI_API_KEY to your OpenRouter key (sk-or-v1-...) in .env"
```

If either prints `STOP:` — halt and ask the user to fill in `.env`.

**OpenRouter free model**: The config uses `meta-llama/llama-3.1-8b-instruct:free`.
Free models have a ~20 req/min rate limit. Expect ~3-6 minutes per 5-run fault sweep.

### Step 3 — Verify endpoint reachability

```bash
OPENAI_BASE_URL=$(grep "^OPENAI_BASE_URL=" .env | cut -d= -f2-)
curl -s --max-time 10 "${OPENAI_BASE_URL}/models" | python3 -m json.tool | head -10
```

### Step 4 — Confirm workspace data

```bash
ls workspace/data/feedback/
ls workspace/data/policy_snippets.md
```

Both must exist. If missing, Phase 3 fault runs will fail.

---

## Idempotency Rules

Check existing artifact counts before running. Each fault type produces its own prefix:
- `failures_timeout_*` (target: 5 runs)
- `failures_malformedjson_*` (target: 5 runs)
- `failures_checkpointloss_*` (target: 5 runs)
- `failures_contextoverflow_*` (target: 5 runs)

```bash
for prefix in failures_timeout failures_malformedjson failures_checkpointloss failures_contextoverflow; do
  count=$(ls -d artifacts/${prefix}_* 2>/dev/null | wc -l | tr -d ' ')
  echo "$prefix: $count / 5"
done
```

| Condition | Action |
|-----------|--------|
| Count >= 5 | **SKIP** the dry-run and sweep for that fault type; log as skipped |
| Count 1–4 | Run the full sweep (each run creates a new uuid, extras are fine) |
| Count == 0 | Run dry-run first, then full sweep |

---

## Pipeline

Run each stage in order. Track all outcomes in `docs/phase3-run-log.md`.

### Stage 1: Timeout — Dry-Run

Skip if `failures_timeout count >= 5`. Otherwise:

```bash
python3 scripts/run_phase3.py --config config/phase3.yaml --fault timeout --dry-run
```

Success signal: `Complete: 1/1 runs succeeded.` with `recovery_mode: manual` in summary.

### Stage 2: Timeout — Full Sweep (5 runs)

Skip if count >= 5. Otherwise:

```bash
python3 scripts/run_phase3.py --config config/phase3.yaml --fault timeout
```

Expected: all 5 runs succeed in recording (`success` may be False — that is expected for
fault injection; what matters is that `metrics.json` is written with recovery fields).

### Stage 3: Malformed JSON — Dry-Run

Skip if `failures_malformedjson count >= 5`.

```bash
python3 scripts/run_phase3.py --config config/phase3.yaml --fault malformed_json --dry-run
```

Success signal: summary shows `recovery_mode: automatic` (crew_node caught the error).

### Stage 4: Malformed JSON — Full Sweep

```bash
python3 scripts/run_phase3.py --config config/phase3.yaml --fault malformed_json
```

### Stage 5: Checkpoint Loss — Dry-Run

Skip if `failures_checkpointloss count >= 5`.

```bash
python3 scripts/run_phase3.py --config config/phase3.yaml --fault checkpoint_loss --dry-run
```

Success signal: `recovery_mode: manual`. Note: output files may exist on disk
(`output_quality > 0`) but LangGraph state has no record — this is expected behaviour.

### Stage 6: Checkpoint Loss — Full Sweep

```bash
python3 scripts/run_phase3.py --config config/phase3.yaml --fault checkpoint_loss
```

### Stage 7: Context Overflow — Dry-Run

Skip if `failures_contextoverflow count >= 5`.

```bash
python3 scripts/run_phase3.py --config config/phase3.yaml --fault context_overflow --dry-run
```

Success signal: `recovery_mode: manual`. The overflow fires before the crew runs
(token count exceeds 500 — expected given workspace files).

### Stage 8: Context Overflow — Full Sweep

```bash
python3 scripts/run_phase3.py --config config/phase3.yaml --fault context_overflow
```

### Stage 9: Analysis & Recovery Scoring

Always run — it is read-only and cheap.

```bash
python3 - <<'EOF'
import sys; sys.path.insert(0, ".")
from analysis.aggregate import load_runs
import pandas as pd

df = load_runs("artifacts/", prefix="failures")
if df.empty:
    print("WARNING: No Phase 3 artifacts found. Run Stages 2, 4, 6, 8 first.")
else:
    # Recovery summary table
    summary = df.groupby("failure_type").agg(
        runs=("run_id", "count"),
        automatic=("recovery_mode", lambda x: (x == "automatic").sum()),
        manual=("recovery_mode", lambda x: (x == "manual").sum()),
        avg_recovery_time_s=("recovery_time_s", "mean"),
        avg_output_quality=("output_quality", "mean"),
    ).round(3)
    print(summary.to_string())
    print(f"\nTotal: {len(df)} runs across {df['failure_type'].nunique()} fault types")
EOF
```

Success signal: printed table with 4 rows (one per fault type).

Verify artifact counts:
```bash
for prefix in failures_timeout failures_malformedjson failures_checkpointloss failures_contextoverflow; do
  echo "$prefix: $(ls -d artifacts/${prefix}_* 2>/dev/null | wc -l | tr -d ' ')"
done
```

---

## Error Recovery Protocol

When any stage fails, inspect root cause before fixing:

**Diagnostic agent prompt template:**
```
Read the following error output and identify the root cause. Check relevant source
files in src/aip_intern/phase3/ and src/aip_intern/failures/.

Error output:
[paste full stderr/stdout]

Stage context: [which stage, e.g. "Stage 1: Timeout Dry-Run"]
Repo root: /home/rencongchia/projects/aip-intern-project
```

### Known Issues & Fixes

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| `ModuleNotFoundError: tiktoken` | tiktoken not installed | `pip3 install tiktoken` or `pip3 install -e ".[dev]"` |
| `ModuleNotFoundError: aip_intern` | Package not installed | `pip3 install -e ".[dev]"` |
| `OPENAI_API_KEY` 401 error | Invalid or expired OpenRouter key | Get new key at openrouter.ai → Settings → API Keys |
| `429 Too Many Requests` | OpenRouter free tier rate limit | Wait 60s and retry; or reduce `n_runs` in config |
| `CrewAI kickoff timeout` | Request timeout exceeded | Increase `llm.request_timeout` in `config/phase3.yaml` |
| `context_overflow never fires` | token_limit too high for inputs | Lower `token_limit` in `config/phase3.yaml` (default 500 should fire) |
| `inject_malformed_json: t_fault_holder[0] is None` | Tool call never reached | crew failed before any tool call; check if LLM responded with tools |
| `checkpoint_loss output_quality 0.0` | Crew failed before writing files | Expected if crew errors before tool writes; check crew logs |
| `recovery_mode missing from metrics.json` | Phase 3 runner wrote old-style metrics | Ensure run used `phase3/runner.py`, not `mesh/runner.py` |
| `load_runs returns empty DataFrame` | prefix mismatch | Verify prefix: `failures_timeout`, not `failures-timeout` |

---

## Run Log Format

Append to `docs/phase3-run-log.md` after each pipeline completes.

```markdown
## Run: YYYY-MM-DD HH:MM UTC

| Stage | Fault Type | Status | Notes |
|-------|-----------|--------|-------|
| 1. Timeout Dry-Run         | timeout          | ✅ / ❌ / ⏭ | |
| 2. Timeout Sweep           | timeout          | ✅ / ❌ / ⏭ | N/5 recorded |
| 3. Malformed JSON Dry-Run  | malformed_json   | ✅ / ❌ / ⏭ | |
| 4. Malformed JSON Sweep    | malformed_json   | ✅ / ❌ / ⏭ | N/5 recorded |
| 5. Checkpoint Loss Dry-Run | checkpoint_loss  | ✅ / ❌ / ⏭ | |
| 6. Checkpoint Loss Sweep   | checkpoint_loss  | ✅ / ❌ / ⏭ | N/5 recorded |
| 7. Context Overflow Dry-Run| context_overflow | ✅ / ❌ / ⏭ | |
| 8. Context Overflow Sweep  | context_overflow | ✅ / ❌ / ⏭ | N/5 recorded |
| 9. Analysis                | all              | ✅ / ❌ | |

### Recovery Summary (from Stage 9)
<!-- Paste summary table output here -->

### Errors
<!-- For each ❌: root cause + fix attempted -->
```

---

## Expected Outcomes (Phase 3 baseline, no GCC)

| Fault Type | recovery_mode | output_quality | Notes |
|-----------|---------------|----------------|-------|
| timeout | manual | 0.0 | AgentTimeoutError propagates to runner; crew never completes |
| malformed_json | automatic | 0.0 | crew_node catches JSONDecodeError; stores in state['error']; graph reaches END |
| checkpoint_loss | manual | ~0.67–1.0 | Crew runs + writes files; CheckpointLostError raised; state lost |
| context_overflow | manual | 0.0 | ContextOverflowError raised before crew invoked |

These outcomes are compared against Phase 4 (GCC constraints) in `run-phase4.md`.

---

## Key Paths

| What | Path |
|------|------|
| Phase 3 script | `scripts/run_phase3.py` |
| Phase 3 config | `config/phase3.yaml` |
| Phase 3 runner | `src/aip_intern/phase3/runner.py` |
| Fault injectors | `src/aip_intern/failures/injectors.py` |
| Recovery scoring | `src/aip_intern/failures/scoring.py` |
| Mesh source (base) | `src/aip_intern/mesh/` |
| Workspace input data | `workspace/data/` |
| Artifacts output | `artifacts/failures_{type}_{uuid}/` |
| Analysis module | `analysis/aggregate.py` |
| Phase 3 notebooks | `notebooks/phase3_failures/` (`00_overview`, `01_run`, `02_analysis`) |
| Run log | `docs/phase3-run-log.md` |
| Technical log | `docs/phase3-technical-log.md` |

## Phase 3 Metrics Captured

| Metric | Description |
|--------|-------------|
| `total_latency_s` | Wall-clock time for full graph invocation |
| `step_trace` | Node names visited before fault fired |
| `error` | Error message (always non-null for fault runs) |
| `failure_type` | Injected fault: timeout / malformed_json / checkpoint_loss / context_overflow |
| `recovery_time_s` | Seconds from fault injection to graph END (or runner catch) |
| `output_quality` | Fraction of expected output files present and non-empty (0.0–1.0) |
| `recovery_mode` | "automatic" (graph handled it) / "manual" (runner caught AIPInternError) |
| `notes` | Diagnostic note (e.g. checkpoint asymmetry explanation) |
| `gcc_sim` | False for Phase 3; True in Phase 4 re-runs |

## Extending to Phase 4

Phase 4 reuses Phase 3's injectors under toxiproxy GCC constraints.
See `.claude/commands/run-phase4.md` for the full GCC simulation pipeline.
