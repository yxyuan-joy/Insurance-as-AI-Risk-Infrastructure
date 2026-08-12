# Configuration Reference

## Frozen formal configuration

`configs/formal.yaml` is the resolved configuration used by the formal experiment. It intentionally contains no inheritance and uses repository-relative data paths populated by `scripts/download_data.py`.

Principal blocks are:

| Block | Function |
|---|---|
| `paths` | Prepared risk panel and firm metadata |
| `simulation` | Horizon, seed, returns, AI value, and loss scale |
| `network` | Same-sector and cross-sector peer links |
| `decision_layer` | Model endpoint, decoding, timeout, and fallback policy |
| `decision_policy` | Adoption, renewal, insurance, and abandonment evidence weights |
| `negotiation` | Vendor and insurance bargaining limits |
| `contract_lifecycle` | Terms, expiry, cancellation, refunds, and re-entry |
| `claims` | Claim timing constraints |
| `experience_memory` | Decay, uninsured-loss reinforcement, and indemnity relief |
| `risk_mapping` | Risk-to-loss realization and event thresholds |
| `panic` | Persistence, event propagation, indemnity relief, and network transmission |
| `vendors` | Four vendor profiles |
| `insurers` | Four private insurer profiles and one residual backstop |
| `insurance_pricing` | Pure-risk, capital, coverage, deductible, limit, and premium-cap terms |

Selected formal values include `action_loss_scale=3.35`, `loss_realization_rate=0.70`, `global_multiplier=2.95`, `premium_cap_asset_share=0.18`, four vendors, and four private insurers plus a residual facility. The YAML file is authoritative for all values.

## Insurance counterfactual

The formal off arm is not a separate calibration. `run_simulation.py --disable-insurance-market` overrides only `simulation.enable_insurance_market`. This disables purchase and claim settlement while retaining the same firm population, operational-risk panel, model policy, AI value and loss mapping, vendor market, network, and random seed.

## Rule baseline

`configs/rule_abm.yaml` inherits the frozen formal market and replaces the decision layer with `rule_heuristic`. The insurance-purchase threshold is 0.30 with a minimum threshold of 0.22 because the deterministic score and model-generated score occupy different scales. No outcome variable is directly targeted by the rule policy.

## Sensitivity profiles

Each file in `configs/sensitivity/` changes one parameter family around the formal configuration:

| Family | Low / sparse / short | High / dense / long |
|---|---:|---:|
| AI loss scale | 3.00 | 3.70 |
| AI value scale | 0.070 | 0.086 |
| Insurance price multiplier | 2.65 | 3.25 |
| Coverage and deductible | weaker protection | stronger protection |
| Network | 8 same-sector, 3 cross-sector | 16 same-sector, 5 cross-sector |
| Uninsured panic weight | 0.48 | 0.68 |
| Loss and claimable-memory decay | 0.82 / 0.84 | 0.92 / 0.94 |

The launcher can generate a matched on/off pair for every profile.
