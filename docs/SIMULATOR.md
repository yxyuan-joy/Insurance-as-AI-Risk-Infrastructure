# Action-Risk Social Simulation

`run_simulation.py` is the primary entry point. The simulator combines firm profiles, AutoCLAW action risk, model-generated economic decisions, bounded vendor menus, contract lifecycle, insurance pricing and negotiation, claim settlement, experience memory, panic transmission, and accounting.

## Entry points

```bash
# Unit and mechanism smoke tests
bash scripts/run_smoke.sh

# One formal paired seed, with vLLM endpoints already running
bash scripts/run_formal_pair.sh 42

# Rule-policy comparison
bash scripts/run_rule_abm.sh
```

## Source modules

| Module | Responsibility |
|---|---|
| `data.py` | Input validation, panel loading, firm initialization, and prior-day industry snapshots |
| `decisions.py` | Heuristic and model decision policies, prompts, schemas, thresholds, and visible menus |
| `negotiation.py` | Vendor and insurer bargaining and model trace capture |
| `insurers.py` | Vendor profiles, insurer capacity, pricing, policy binding, and claims |
| `schema.py` | Firm, contract, quote, policy, and risk-state dataclasses |
| `simulator.py` | Daily ordering, cash flows, contracts, memory, panic, bankruptcy, logging, and checkpoints |

The frozen formal configuration is `configs/formal.yaml`. Relative paths are resolved from the repository root. Research data are downloaded separately; tests use a generated fixture defined in `tests/build_fixture.py`.
