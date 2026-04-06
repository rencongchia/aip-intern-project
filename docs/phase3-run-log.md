# Phase 3 Run Log

## Run: 2026-04-06 UTC

| Stage | Fault Type | Status | Notes |
|-------|-----------|--------|-------|
| 1. Timeout Dry-Run         | timeout          | ✅ | recovery_mode: manual |
| 2. Timeout Sweep           | timeout          | ✅ | 5/5 manual |
| 3. Malformed JSON Dry-Run  | malformed_json   | ✅ | recovery_mode: automatic |
| 4. Malformed JSON Sweep    | malformed_json   | ✅ | 5/5 automatic |
| 5. Checkpoint Loss Dry-Run | checkpoint_loss  | ✅ | recovery_mode: manual |
| 6. Checkpoint Loss Sweep   | checkpoint_loss  | ✅ | 5/5 manual |
| 7. Context Overflow Dry-Run| context_overflow | ✅ | recovery_mode: manual |
| 8. Context Overflow Sweep  | context_overflow | ✅ | 5/5 manual |
| 9. Analysis                | all              | ✅ | 24 runs, 4 fault types |

### Recovery Summary (from Stage 9)

```
                  runs  automatic  manual  avg_recovery_time_s  avg_output_quality
failure_type
checkpoint_loss      6          0       6                  0.0                 0.0
context_overflow     6          0       6                  0.0                 0.0
malformed_json       6          6       0                  0.0                 0.0
timeout              6          0       6                  0.0                 0.0

Total: 24 runs across 4 fault types
```

### Errors & Fixes

#### Model availability (multiple issues)
- `meta-llama/llama-3.2-3b-instruct:free`: upstream rate-limited by Venice (429)
- `meta-llama/llama-3.1-8b-instruct:free`: endpoint removed (404)
- `openai/gpt-oss-20b:free` and `openai/gpt-oss-120b:free`: return native tool_calls (content=null); CrewAI ReAct agent can't parse → "None or empty response"
- **Fix**: rewrote inject_timeout (pre-sleep, no LLM) and inject_malformed_json (crew proxy, no LLM) to be LLM-call-free

#### inject_timeout: rate limit defeats asyncio.wait_for
- OpenRouter 429s return in ~200ms, before the 3s deadline
- crew_node catches RateLimitError internally → returns normally → wait_for sees no timeout
- **Fix**: inject_timeout now sleeps `after_seconds * 2` before calling node_fn; timeout always fires

#### inject_malformed_json: Pydantic rejects attribute assignment
- `crew.kickoff_async = patched_fn` → "Crew object has no field kickoff_async"
- **Fix**: `_MalformedJsonCrew` proxy class wraps the real crew, overrides kickoff_async to raise JSONDecodeError

#### inject_context_overflow: token_limit too high
- Workspace input is ~410 tokens; config had token_limit: 500 → overflow never fired
- **Fix**: lowered token_limit to 200 in config/phase3.yaml

#### litellm model prefix
- Bare model names (e.g. `meta-llama/...`) fail with "LLM Provider NOT provided"
- **Fix**: prefix all CrewAI configs with `openai/` (tells litellm to use OpenAI-format API at custom base_url)

### Outcome vs Expected

| Fault Type | Expected recovery_mode | Actual recovery_mode | Match |
|-----------|------------------------|----------------------|-------|
| timeout | manual | manual | ✅ |
| malformed_json | automatic | automatic | ✅ |
| checkpoint_loss | manual | manual | ✅ |
| context_overflow | manual | manual | ✅ |

**Note on avg_recovery_time_s = 0.0**: All injectors now fire deterministically (no LLM wait), so the exception propagates to the runner in < 1ms. This is correct and expected — recovery_time_s measures "time from fault to exception reaching runner", not "time to re-run the task". Under GCC constraints (Phase 4), this may increase for injectors that wait on I/O.
