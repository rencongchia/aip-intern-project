# aip-intern-project

Research repo comparing a single-agent LangGraph baseline against a
LangGraph+CrewAI mesh on a citizen feedback triage task.

**For interns:** Start with `notebooks/phaseN_*/00_overview.ipynb` for your phase.

## Quick Start

```bash
git clone <repo> && cd aip-intern-project
cp .env.example .env          # fill in OPENAI_BASE_URL + OPENAI_API_KEY
pip install -e ".[dev]"
nbstripout --install --attributes .gitattributes

# Run Phase 1 baseline (requires live LLM endpoint)
python scripts/run_baseline.py --config config/baseline.yaml --dry-run

# Open the experiment record
jupyter lab notebooks/phase1_baseline/01_run.ipynb
```

## Phase overview

| Phase | Who | What |
|-------|-----|------|
| 1 | Owner | Single-agent LangGraph baseline, 20-run sweep |
| 2 | Owner | LangGraph + CrewAI mesh, same metrics |
| 3 | Intern | Fault injection — 4 failure types, recovery scoring |
| 4 | Intern | GCC simulation via toxiproxy |
| 5 | Intern | PinchBench / OpenClaw / KiloClaw safety scoring |
| 6 | Intern | Reference repo cleanup |

## Infrastructure

```bash
# Deploy GPU + CPU EC2
terraform -chdir=infra/terraform apply

# Install dependencies
cd infra/ansible && ansible-playbook -i inventory.sh playbooks/site.yml

# Run sweep + collect artifacts
ansible-playbook -i inventory.sh playbooks/run_sweep.yml -e sweep_type=baseline
```

## Testing

```bash
pytest                           # unit tests only (no LLM required)
pytest -m integration            # requires OPENAI_BASE_URL in env
```

## Notebook output policy
- `00_overview.ipynb` — outputs stripped (documentation)
- `01_run.ipynb` — outputs committed (experiment record)
- `02_analysis.ipynb` — outputs committed (analysis record)
