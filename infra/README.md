# Infrastructure

Terraform + Ansible for deploying the experiment environment.

## Setup

```bash
# Deploy GPU + CPU EC2 instances
terraform -chdir=infra/terraform apply

# Install dependencies on deployed instances
cd infra/ansible && ansible-playbook -i inventory.sh playbooks/site.yml

# Run a sweep
ansible-playbook -i inventory.sh playbooks/run_sweep.yml -e sweep_type=baseline
```

## Directory structure

```
infra/
├── terraform/     # EC2 provisioning (adapt from parent repo phase2/infra/)
└── ansible/
    ├── playbooks/ # site.yml, setup_gpu.yml, setup_cpu.yml, run_sweep.yml
    ├── templates/ # vllm.service.j2
    └── group_vars/all.yml
```

## Note for interns

The Terraform files need to be adapted from `phase2/infra/` in the parent `moe-optimization-example` repo (tag substitution: `phase2-*` → `aip-intern-*`). This was not automated to avoid dependencies on the parent repo structure.
