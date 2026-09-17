# RecoMart benchmark results

This folder contains compact evidence files that are safe to commit. Large
raw data, processed datasets, trained models, prediction parquet files, and the
local prediction seal remain ignored.

## Final evaluation

| File | Purpose |
|---|---|
| `final_test_v1.json` | Full frozen final-test metrics, segment cuts, provenance, and sealed-artifact hash inventory. |
| `final_test_v1.csv` | Flat policy-level summary for quick review. |
| `final_test_v1_audit.json` | Independent metric recomputation for the six final policies. |

## Serving and demo checks

| File | Purpose |
|---|---|
| `api_smoke.json` | Snapshot-only FastAPI smoke check over all saved rankings plus sampled HTTP requests. |
| `api_database_http_smoke.json` | FastAPI smoke check with PostgreSQL-backed serving. |
| `backend_smoke.json` | PostgreSQL publication, Redis cache, and parity checks against the local snapshot. |
| `backend_outage_smoke.json` | Controlled Redis/PostgreSQL outage and recovery checks. |
| `demo_api_smoke.json` | Final demo-facing API smoke check, including all 206,209 saved rankings and 100 HTTP requests. |

## Development evidence

| File | Purpose |
|---|---|
| `diversity_architecture.json` | Architecture freeze comparison, category-cap experiments, and paired intervals. |
| `discovery_slot_diagnostic.json` | Forced discovery-slot accuracy and novelty trade-off. |
| `novel_ranker_v1*.json` / `.csv` | Dedicated novel-item ranker experiment and feature importance. |
| `history_availability.json` | Sparse-history routing and first-basket simulation. |
| `candidate_diagnostics*.json` | Candidate-pool coverage and incremental source diagnostics. |
| `ranking_segment_diagnostics.json` | Segment-level ranking diagnostics for repeat and novel behavior. |
| `two_tower_*` | Neural retrieval training, retrieval, and ranking experiments. |
| `xgb_ranker_*` | LambdaMART training/evaluation records and feature importance. |

These files are aggregate reports only. They are enough to inspect the modeling
decisions, but not enough to reproduce the full serving snapshot from a fresh
clone without the external Instacart data and local artifacts.
