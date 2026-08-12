# Model Prompt Index

The executable source is the authoritative prompt disclosure. Prompts are assembled from structured state variables and request JSON-only responses. No hidden system prompt is loaded from an external file.

## Decision prompts

`src/action_risk_v2/decisions.py` defines prompts for:

| Decision | Main disclosed inputs | Main response fields |
|---|---|---|
| Initial AI adoption | Firm traits, visible vendor menu, local adoption, local insurance evidence, panic, and memory | `action`, `score`, `reason`, vendor choice, term, rounds |
| Vendor renewal | Existing vendor, visible alternatives, contract expiry, firm state, performance and risk memory | `action`, `score`, `reason`, selected vendor, term, rounds |
| Insurance purchase | Risk need, industry risk, memory, panic, recent claims, peer coverage, and available quotes | `action`, `score`, `reason`, quote choice, term, rounds |
| Insurance renewal | Existing policy, updated risk evidence, quote terms, cash, memory, and claim experience | `action`, `score`, `reason`, quote choice, term, rounds |
| AI abandonment | Contract state, realized loss, memory, panic, cash, and insurance status | `action`, `score`, `reason` |

The visible vendor set is bounded by `ads_bandwidth`; the model cannot select an identifier outside the supplied set. Vendor renewal prompts disclose both the incumbent and alternatives and do not instruct the model to retain the incumbent.

## Negotiation prompts

`src/action_risk_v2/negotiation.py` defines the firm-vendor and firm-insurer bargaining messages. Each round exposes current offer terms, firm budget or risk state, prior-round information, and a bounded action schema. The simulator validates parsed terms against contractual and financial limits before settlement.

## Trace-level audit support

Each new run writes `model_decisions.jsonl`, which records the rendered prompt payload, raw model response, parsed JSON, endpoint, and parse status. `interactions.jsonl` records negotiation rounds. These locally generated files permit verification that runtime prompts match the source templates. They are intentionally excluded from both release repositories.

## Parsing policy

Formal runs use temperature 0 and `fallback_to_rule=false` for general backend failures. Invalid JSON responses are retried. If all parse attempts fail with a schema `ValueError`, the code records the raw failure and applies a conservative no-change action so that exposure is not silently expanded or cancelled. The parser otherwise applies schema checks, allowed-identifier checks, numeric bounds, and explicit reason codes. `analysis/audit_model_traces.py` audits newly generated formal traces.
