# Frozen final next-basket evaluation

The experiment follows [the frozen contract](../configs/final_architecture.json).
The selected system is Repeat Purchase + ALS v2 + Content v1 retrieval followed
by the corrected 22-feature LambdaMART ranker. Architecture and hyperparameters
are fixed before opening test labels.

## Final result — 16 September 2026

Completed on **206,209 users** and **2,165,417 final-basket interactions**.
All predictions were saved and sealed before test labels were opened. The
selected model and policy were retained without test-driven tuning.

| Policy | Recall@10 | Precision@10 | NDCG@10 | Hit Rate@10 |
|---|---:|---:|---:|---:|
| Global popularity | 0.069942 | 0.072112 | 0.097616 | 0.457027 |
| Repeat purchase | 0.331622 | 0.275896 | 0.399060 | 0.856364 |
| Repeat + popularity fill | 0.332535 | 0.276296 | 0.399533 | 0.857625 |
| ALS v2 | 0.154914 | 0.123481 | 0.177601 | 0.648100 |
| Content v1 | 0.084644 | 0.074875 | 0.102998 | 0.475241 |
| **Selected LambdaMART** | **0.354376** | **0.296722** | **0.426033** | **0.872620** |

The selected ranker improves Recall@10 by **6.86% relative to repeat purchase**
and **6.57% relative to repeat with popularity filling**. NDCG improves by 6.76%
and 6.63%, respectively. These are observed offline differences; this experiment
does not measure online engagement, sales, or satisfaction.

### Retrieval, discovery, and coverage

- The three-source union contains **22,230,351 candidates**, averaging **107.805
  per user**, with macro candidate recall **0.623258**.
- Individual candidate Recall@50: repeat **0.549484**, ALS **0.338931**, content
  **0.202675**.
- The selected ranker returns ten unique products to every user and reaches
  **34,844 / 49,668 observed history products**, or **70.15% catalog coverage**.
- Conditional repeat recall is **0.585070** over **192,370 eligible users**.
  Conditional novel recall is **0.002737**, with novel hit rate **0.006367**, over
  **168,379 eligible users**. Novel-item discovery remains limited; the frozen
  accuracy-focused policy was not changed after seeing these test results.
- Recommendations span **7.294 aisles** on average, with aisle-pair diversity
  **0.906499**. Repeat with popularity filling has greater catalog coverage
  (**79.62%**) and aisle-pair diversity (**0.910301**), alongside lower accuracy.
- Pure repeat purchase produces short lists for **7.09%** of users. Those users
  remain in its evaluation, with precision still divided by ten.

### History segments

| Observed history orders | Users | Selected Recall@10 | Repeat + fill Recall@10 | Selected NDCG@10 |
|---|---:|---:|---:|---:|
| 1-2 | 8,686 | 0.314205 | 0.295034 | 0.364115 |
| 3-5 | 56,052 | 0.335472 | 0.322146 | 0.391304 |
| 6-10 | 50,092 | 0.353095 | 0.339455 | 0.420775 |
| Over 10 | 91,379 | 0.370494 | 0.338678 | 0.456104 |

The selected policy has higher observed recall than the filled repeat fallback
in each segment. Segment populations differ from development because prediction
history now includes the validation basket and evaluation includes all users.

### Verification and provenance

- **63 tests pass**, including a complete small-data evaluation and deliberate
  metric corruption that the independent audit rejects.
- An independent implementation recomputed Recall, Precision, NDCG, Hit Rate,
  and coverage for **all six policies**, matching the benchmark within `1e-12`.
- The ranker refit used **4,183,541 rows**, including **1,247,936 positives**,
  from **195,707 positive queries**. All **206,209 users** were considered before
  the prescribed zero-positive filter. Rebuilt features matched the development
  reference across **4,416 sampled rows / 207 users**.
- Prediction history contains **31,653,689 interactions**. The final model has
  exactly **486 rounds** and the frozen ordered **22 features**.
- **99 artifacts** were sealed at **07:04:16 UTC**. First test access was
  **13:14:50 UTC**; evaluation completed at **13:15:53 UTC**. Fitting and prediction
  stages are now locked, and the artifact seal still verifies.

Full [JSON results](../results/benchmarks/final_test_v1.json),
[CSV summary](../results/benchmarks/final_test_v1.csv), and
[independent audit](../results/benchmarks/final_test_v1_audit.json) are available.

The offline modeling step is complete. The [serving API](serving_api.md) now
serves the saved recommendations with the tested fallback routing.
[PostgreSQL and Redis integration](database_cache.md) is also complete.
The [Streamlit demo](demo.md) is also complete. Next are deployment and monitoring.

## Execution safeguards

1. Verify the contract's hashes for reference implementations, development
   checkpoint, and architecture-selection evidence. Record runtime versions and
   hashes of training/validation inputs and original candidate stores.
2. Rebuild refit features from **training history only**. Label using validation
   baskets. Keep all positives and at most 15 negatives per user, ordering only
   negative rows by Spark `xxhash64(user_id, product_id)`, then product ID.
3. Compare a deterministic cross-section of rebuilt features with the existing
   development feature tables and verify every sampled validation label.
   Remove zero-positive fitting queries and train exactly **486 rounds** without
   early stopping. Save a separate final model.
4. Validate the temporal separation and user populations of train and validation,
   then combine them into the prediction history. Rebuild item statistics,
   repeat recommendations, content profiles, ALS, and ranker features.
5. Save Top-10 predictions for all six frozen policies and all 206,209 users.
   Check uniqueness, catalog membership, user coverage, finite model scores,
   model feature order, model rounds, and history cutoff.
6. Seal all predictions, inputs, intermediate artifacts, and runner source with
   SHA-256 hashes. The evaluator verifies the seal before accessing test labels.
   Record first test access and lock fitting/prediction stages thereafter.
7. Calculate the fixed metrics without changing recommendations or tuning the
   model. Verify the seal again after metric calculation.

The immutable freeze manifest retains its historical `not_run` status. The
separate execution receipts, prediction seal, test-access record, and final
benchmark record the completed experiment without rewriting that decision.

## Metric definitions

- Recall, Precision, NDCG, and Hit Rate at 10 are macro averages over every test
  user, including users with empty or short baseline recommendation lists.
  Precision always divides by ten.
- Repeat and novel truth are determined by actual user/product membership in
  train plus validation history. Each conditional recall averages only users
  with at least one corresponding target product. Novel hit rate uses the same
  eligible-user denominator.
- Candidate recall measures the saved deduplicated three-source union. Each
  individual retrieval source also reports recall from up to 50 candidates.
- Catalog coverage divides distinct recommended products by the distinct
  products observed in prediction history; the exact denominator is reported.
- Aisle-pair diversity is the fraction of recommendation item pairs belonging
  to different aisles. It is a category proxy, not a satisfaction measure.
- History segments count distinct observed orders: 1-2, 3-5, 6-10, and over 10.
  The earlier excluded-user cold-start simulation remains separate.

## Artifacts and reproduction

Runner: [final_test_pipeline.py](../src/evaluation/final_test_pipeline.py).

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
# Fresh execution only; completed stages are verified and reused.
.\.venv\Scripts\python.exe -u -m src.evaluation.final_test_pipeline run
# After completion: independently recompute core metrics from saved predictions.
.\.venv\Scripts\python.exe -m src.evaluation.audit_final_test
```

Each stage runs in a separate process to release Spark and model memory.
Stages are also individually addressable through the runner's command-line
help. Completed artifacts are never silently replaced. After test access,
training and prediction commands are locked. A completed benchmark is not
re-evaluated by the runner.

- `models/final_v1/ranker.json`: final 486-round ranker.
- `models/final_v1/popularity.json`: popularity fallback from prediction history.
- `models/final_v1/stage_*.json`: stage receipts, statistics, and artifact hashes.
- `models/final_v1/prediction_seal.json`: pre-test artifact inventory and timestamp.
- `models/final_v1/test_access.json`: first access to final-test labels.
- `data/processed/final_v1/`: history, item features, source candidates, ranker
  feature partitions, saved recommendations, and per-user metrics.
- `results/benchmarks/final_test_v1.json` and `.csv`: final metrics and provenance.

Large data and model files remain local and ignored by Git. Development models
and candidate stores are preserved. Retrieval recommendations are precomputed
for this historical user population; live serving is a subsequent project step.
