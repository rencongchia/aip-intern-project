# Phase 3 Technical Log

Owner: Ren Cong  
Start date: 2026-04-06  
Scope: Fault injection into the LangGraph+CrewAI mesh (Phase 2), recovery scoring, and the `run-phase3.md` orchestration command. Phase 4 (GCC/toxiproxy) follows after Phase 3 is complete.

---

## Background

The previous contributor (Darren/Eunice, using Claude Code autonomously) left all Phase 3/4 code as `...` stubs. Indicators that Phase 1-2 was never actually run:
- `config/baseline.yaml` and `config/mesh.yaml` hardcode `Qwen/Qwen3-32B-Instruct`, which requires ~70GB VRAM or a paid cloud GPU — clearly not a laptop-runnable model.
- The working path in `.claude/commands/run-phase1-2.md` is `/Users/loremipsum/Code/aip-intern-project` — a placeholder that was never replaced.
- All `...` stubs in `failures/injectors.py` and `failures/scoring.py` are untouched.

**Resolution**: Switched to OpenRouter free tier (`meta-llama/llama-3.1-8b-instruct:free`). Updated `.env.example` guidance, `config/*.yaml` model names, and `request_timeout` to 60s (OpenRouter is slower than local vLLM).

---

## 2026-04-06 — Config model change

**Challenge**: Original configs use `Qwen/Qwen3-32B-Instruct`, incompatible with any free or low-cost setup on a laptop (MX450, 2GB VRAM).

**Options considered**:
1. Ollama with `qwen2.5:1.5b` — fully local, but ~2GB available RAM makes this marginal and slow.
2. LM Studio with `qwen2.5:3b` — same hardware constraint.
3. OpenRouter free tier (`meta-llama/llama-3.1-8b-instruct:free`) — no local GPU required, accessible to anyone with an account, `OPENAI_BASE_URL` compatible.

**Decision**: OpenRouter free tier. Reason: accessible (anyone cloning the repo just needs to set an API key), no hardware dependency, sufficient model quality for the document triage task.

**Config change**: `model: "meta-llama/llama-3.1-8b-instruct:free"`, `request_timeout: 60`.

---

## 2026-04-06 — CrewAI model prefix requirement (bug fix)

**Problem**: `crewai.LLM(model="meta-llama/llama-3.1-8b-instruct:free", ...)` raises `ImportError` even though OpenRouter is a natively supported provider. Root cause: CrewAI's LLM class checks for known provider prefixes in the model name string. Without a prefix, it fails unless LiteLLM is installed.

**Supported native prefixes**: `openai/`, `anthropic/`, `openrouter/`, `ollama/`, `ollama_chat/`, `hosted_vllm/`, `bedrock/`, `azure/`, `deepseek/`, `gemini/`, `cerebras/`, `dashscope/`.

**Options considered**:
1. Install `litellm` (`pip install litellm`): works but adds a ~100MB dependency for all users.
2. Auto-prefix in `mesh/graph.py::_build_crew()` based on base_url detection: clever but fragile — URL pattern matching is brittle.
3. Use explicit `openrouter/` prefix in mesh/phase3 configs: explicit, zero magic, zero new deps.

**Decision**: Option 3. Updated `config/mesh.yaml` and `config/phase3.yaml` to use `openrouter/meta-llama/llama-3.1-8b-instruct:free`. Baseline config stays as `meta-llama/llama-3.1-8b-instruct:free` (LangChain `ChatOpenAI` doesn't need a provider prefix — it uses the model name as-is against the `OPENAI_BASE_URL` endpoint).

**Constraint for users**: mesh.yaml and phase3.yaml must use `openrouter/MODEL` format; baseline.yaml uses bare `MODEL` format. This difference is documented in `.env.example` and in the README.

---

## 2026-04-06 — inject_malformed_json design

**Challenge**: The stub's API is `inject_malformed_json(node_fn, fail_on_call=1) → Callable`. For the baseline (`triage_node` etc.), this is straightforward — intercept the MCP tool at call N. For the **mesh**, the `crew_node` function takes a `crew` parameter, and the tools are buried inside `crew.agents[*].tools`. The injector cannot access them without inspecting the `crew` object.

**Options considered**:
1. **Patch tools on the crew directly** (inside the wrapper, via `crew.agents`): works but tightly coupled to CrewAI internals — `tool._run` vs `tool._execute` vs BaseTool.run naming could vary by CrewAI version.
2. **Intercept the LLM response** instead of the tool: make the LLM return a fake malformed response. But this requires patching the LLM object, which is not passed to `crew_node` directly.
3. **Raise MalformedOutputError from the wrapper unconditionally** after N calls to `node_fn`: this simulates the failure at the node boundary, not the tool boundary. Simpler but doesn't test the full tool-call path.
4. **Patch `tool.run` on CrewAI's BaseTool**: CrewAI's BaseTool exposes a `run()` method called synchronously by the agent. Patching this is more stable than `_run`.

**Decision**: Option 4 — patch `tool.run` (the public method). Reason: stable API surface, tests the full CrewAI tool invocation path, and the injector can inspect `crew.agents` since `crew` is a kwarg on the wrapped node.

**Note**: The `fail_on_call` counter is shared across all tools on all agents via a closure, so it counts total tool invocations across the crew run, not per-tool.

---

## 2026-04-06 — inject_checkpoint_loss design

**Challenge**: The mesh graph (`src/aip_intern/mesh/graph.py`) does NOT use a LangGraph checkpointer — it just compiles with `builder.compile()` (no `checkpointer=` arg). So there is no real checkpoint being written to intercept.

**Options considered**:
1. **Add MemorySaver and monkeypatch `put_writes()`**: Adds a real checkpointer to the Phase 3 graph build, then patches the in-memory dict to raise on the next `.get()`. Realistic but complex — requires modifying `build_graph()` signature and understanding LangGraph's checkpointer internals.
2. **Simulate at the node boundary**: Run the node to completion (so output files get written to disk), then raise `CheckpointLostError` before returning the state update dict to LangGraph. LangGraph never records the state update — the graph sees the exception and aborts.
3. **Return empty state update**: Have the wrapper swallow the real return value and return `{}` — the next node sees stale/empty state. But this is harder to detect and score as a "checkpoint loss" event vs. a silent bug.

**Decision**: Option 2. Reason: most clearly demonstrates the key research insight — "output files exist on disk but LangGraph state has no record of them." The `output_quality` scorer can check if files exist (they will), while `recovery_mode` will be "manual" (AIPInternError propagated to runner). This is the asymmetry described in the task spec: "checkpoints don't save you when CrewAI must restart its whole task."

---

## 2026-04-06 — inject_context_overflow token counting

**Challenge**: `tiktoken` is model-specific. For OpenRouter models (LLaMA, Qwen, etc.), there's no exact tiktoken tokenizer. The cl100k_base encoding (GPT-4 tokenizer) gives approximate counts — close enough for the threshold-based test.

**Decision**: Use `cl100k_base` as an approximation. This is documented behavior in the injector. The `token_limit` in `config/phase3.yaml` is set conservatively (500 tokens) so it reliably triggers with the policy_content + task_description input (~800-1200 tokens estimated).

---

## 2026-04-06 — recovery_mode taxonomy

**Challenge**: `RecoveryScore.recovery_mode` has three options: `"automatic"`, `"manual"`, `"unrecoverable"`. Need a consistent rule for assigning these without human judgment calls.

**Rule adopted**:
- `"automatic"` = exception was caught by the node itself (crew_node's `except Exception` block) and stored in `state['error']`; graph reached END normally. Detectable by: `state.get('error') is not None` AND graph reached END (no exception propagated to runner).
- `"manual"` = `AIPInternError` propagated to the runner (runner caught it in its `except AIPInternError` block); graph did NOT reach END normally. Human would need to restart.
- `"unrecoverable"` = reserved for Phase 4 GCC runs where the failure cascades (e.g., timeout during checkpoint write, so no output and no clean state) — determined post-hoc by Phase 4 comparison.

**Expected outcomes for Phase 3**:
| Fault | crew_node behavior | recovery_mode |
|-------|-------------------|---------------|
| timeout | `AgentTimeoutError` re-raised (AIPInternError) | manual |
| malformed_json | caught by `except Exception`, stored as state error | automatic |
| checkpoint_loss | `CheckpointLostError` re-raised (AIPInternError) | manual |
| context_overflow | `ContextOverflowError` raised before crew runs | manual |

---

---

## 2026-04-06 — litellm `openai/` prefix requirement (bug fix during run)

**Problem**: After installing litellm, `crewai.LLM(model="meta-llama/llama-3.2-3b-instruct:free", base_url="https://openrouter.ai/api/v1", ...)` still raised `litellm.BadRequestError: LLM Provider NOT provided`. Installing litellm alone does not solve the provider detection issue — litellm still requires a provider prefix in the model name even when a custom `base_url` is given.

**Root cause**: litellm's model routing is done by parsing the model string prefix, not by inspecting the `base_url`. A bare model name like `meta-llama/llama-3.2-3b-instruct:free` has no recognised prefix → litellm cannot determine which API format to use.

**Options considered**:
1. `openrouter/MODEL` — uses litellm's native OpenRouter provider (looks for `OPENROUTER_API_KEY`; incompatible with our `OPENAI_API_KEY` env var).
2. `openai/MODEL` with custom `base_url` — tells litellm "use OpenAI API format, send to this URL". Works with any OpenAI-compatible endpoint; `OPENAI_API_KEY` is used as-is.
3. Auto-prefix in `_build_crew()` — brittle, hard to test.

**Decision**: Option 2 — prefix all CrewAI model names with `openai/`. Both `config/mesh.yaml` and `config/phase3.yaml` updated to `openai/meta-llama/llama-3.2-3b-instruct:free`. Baseline uses LangChain `ChatOpenAI` which does NOT need a prefix (it uses `base_url` directly), so `baseline.yaml` stays unchanged.

**Standardisation rule**: For CrewAI + OpenAI-compatible endpoint (OpenRouter, local vLLM, etc.) → always use `openai/MODEL_NAME`. For LangChain ChatOpenAI → use bare `MODEL_NAME`.


---

## 2026-04-06 — Model availability failures during run (bug fix)

**Problem**: During live Phase 3 runs, multiple free OpenRouter models failed:
- `meta-llama/llama-3.2-3b-instruct:free`: upstream rate-limited by Venice provider (429)
- `meta-llama/llama-3.1-8b-instruct:free`: endpoint removed from OpenRouter (404)
- `openai/gpt-oss-20b:free` and `openai/gpt-oss-120b:free`: connect 200 OK but CrewAI agent gets "None or empty response" — these models respond with native tool_calls (content=null), which CrewAI's ReAct agent can't parse as text
- `google/gemma-3-4b-it:free`: returns proper ReAct text, but was rate-limited mid-run

**Root cause**: OpenRouter free tier has two layers of failure: (a) per-provider upstream rate limits, (b) model-format incompatibility (native tool_calls vs ReAct text). Models served by Venice fail most frequently.

**Resolution**: Rewrote inject_malformed_json and inject_timeout to not require LLM calls:
- **inject_timeout** (already fixed): pre-sleep for `after_seconds*2` inside `asyncio.wait_for()`; timeout always fires; no LLM call made.
- **inject_malformed_json** (fixed now): proxy class `_MalformedJsonCrew` replaces the real crew kwarg before calling node_fn. The proxy's `kickoff_async()` raises `json.JSONDecodeError` unconditionally; `crew_node`'s `except Exception` catches it → `state['error']` set → automatic recovery. No LLM call needed.
- **inject_checkpoint_loss**: crew_node catches all LLM errors internally and returns normally; the CheckpointLostError is then raised after node_fn returns. Works regardless of LLM rate limits.
- **inject_context_overflow**: fires before any LLM call (token count pre-check). Unaffected.

**Model selection for checkpoint_loss**: any available free model works since crew_node handles all LLM errors. Use `google/gemma-3-4b-it:free` as default. If rate-limited, checkpoint_loss still records correctly (recovery_mode=manual, output_quality=0.0 since crew didn't complete).

