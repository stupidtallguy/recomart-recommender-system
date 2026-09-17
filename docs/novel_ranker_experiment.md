# Dedicated unseen-item ranker v1

## Result and decision

**Do not promote this dedicated ranker into the serving architecture.** It did
not demonstrate an improvement over the existing one-slot policy. Keep the
experiment and artifacts for reproducibility; retain zero forced slots as the
accuracy baseline and the existing one-slot policy as a discovery comparator.

The completed run evaluated all 41,114 held-out ranker users:

| Policy | Recall@10 | Precision@10 | NDCG@10 | Novel Recall@10 | Novel hit rate | Catalog coverage |
|---|---:|---:|---:|---:|---:|---:|
| Original, zero forced slots | 0.354390 | 0.292920 | 0.423534 | 0.004735 | 0.013521 | 45.29% |
| Original, minimum one slot | 0.341581 | 0.278944 | 0.413684 | 0.013980 | 0.056847 | 43.40% |
| Dedicated ranker, minimum one slot | 0.341422 | 0.278854 | 0.413607 | 0.013785 | 0.055747 | 43.40% |

Treatment minus the existing one-slot policy:

| Metric | Mean difference | Paired 95% interval |
|---|---:|---:|
| Recall@10 | -0.000159 | [-0.000328, +0.000010] |
| NDCG@10 | -0.000076 | [-0.000174, +0.000012] |
| Novel Recall@10 | -0.000196 | [-0.000666, +0.000286] |
| Novel hit rate | -0.001099 | [-0.002674, +0.000416] |

These intervals include zero. The point estimates are slightly worse, but this
run does not establish a statistically clear degradation or improvement. It
does not justify adding another model to the selected system.

Of the 36,470 users originally lacking a novel item, the dedicated ranker
changed the inserted product for 11,950. Repeat recall and the per-user novelty
counts stayed identical to the one-slot control, as intended. Both one-slot
policies recommend 21,545 unique products across the cohort. The small positive
recall difference in the >20-order segment is exploratory and does not justify
segment routing without a separate test.

Compared with zero forced slots, the dedicated policy still loses 0.012968
overall recall while gaining 0.009050 novel recall. The experiment therefore
does not resolve the accuracy/discovery trade-off.

### Interpretation and next work

**Follow-up completed:** sparse-history and zero-history evaluation now support
keeping the ranker for positive-history users and using explicit availability
fallbacks. See the [history-availability report](history_availability_experiment.md).

Separating unseen-product training alone was insufficient under this feature
set, existing negative sample, and insertion policy. This result does not prove
that all dedicated discovery rankers fail. In particular, internal training uses
sampled negatives, while outer evaluation ranks the full frozen candidate pool.

No additional tuning is justified by this run. Move to the planned
**sparse-history and cold-start evaluation**: define eligible history segments,
compare existing fallback strategies on matched users, and evaluate a true
zero-history scenario separately before deciding on routing. Existing history
segments do not themselves establish performance for brand-new users. Keep the
final test reserved until architecture and evaluation choices are frozen.

## Hypothesis

A ranker trained only on unseen products may select a more relevant discovery
item than the ordinary five-source LambdaMART model, at the same slot budget.
The controls are the existing zero-slot and minimum-one-slot policies.

This experiment does not add retrieval sources, change the candidate union, or
change the outer training/validation user split. It does not use the final test.

## Training protocol

The input is the existing corrected v3 ranker-training dataset. Filtering to
`seen_before == 0` produces 2,109,261 rows: 145,740 positive and 1,963,521 negative
examples. These are the unseen subset of the original 15 sampled negatives per
user, so the unseen negative count is not necessarily 15. No new negatives or
products are added.

Only users with both positive and negative unseen examples provide ranking
pairs. This leaves 1,054,745 rows across 76,072 users, retaining all 145,740 unseen
positives in this dataset. Users with no retrieved novel positive do not train
the ranker, but remain included in outer evaluation.

An internal user split (NumPy seed 42) reserves 7,608 of these training users for
early stopping and uses 68,464 for fitting. There is no user overlap with the
41,114 outer ranker-validation users. The internal criterion is NDCG@1 because
the intervention selects one product. All tree and pair-generation parameters
remain those of LambdaMART v3; no hyperparameter sweep is performed.

The selected checkpoint is iteration 23, equivalent to 24 boosting rounds. A
final model is then fitted on all 76,072 informative training users for those
24 rounds. Internal NDCG uses sampled negative pools and must not be compared
with full-candidate outer metrics.

All 30 v3 features are retained. Six repeat/history features are constant within
the unseen subset and have no splitting value. User/product IDs identify rows
and ranking queries; they are not model features.

## Controlled intervention

The saved ordinary ranker scores the unchanged 6,895,062 validation candidates.
The dedicated model scores the 5,310,138 unseen candidates within that pool.

For a user whose ordinary Top-10 already contains an unseen item, every policy
with a one-slot minimum preserves that list. For a user with no unseen item:

1. Remove the lowest-ranked seen product.
2. The one-slot control selects the unseen candidate with the highest ordinary
   model score; the treatment selects the highest dedicated-model score.
3. Retain ordinary model-score ordering for the final list, using product ID to
   break ties. Scores from the two models are not combined or compared directly.

This keeps exactly the same number of novel items per user in control and
treatment. It tests selection for missing discovery slots, not replacement of
novel items already preferred by the ordinary ranker.

## Evaluation and uncertainty

Overall Recall/Precision/NDCG, repeat recall, novel recall and hit rate, catalog
coverage, and history segments use the same diagnostic as the preceding quota
experiment. Precision divides by 10. Novel metrics condition on users with
novel truth; repeat recall conditions on users with repeat truth. Catalog
coverage uses 49,641 training-observed products.

Both controls must reproduce the saved quota experiment before comparisons are
accepted. The runner saves per-user metrics locally and calculates 2,000 paired
user-bootstrap samples, seed 42, for treatment differences against both
controls. The reported 95% percentile intervals are marginal, conditional on
the fitted models and the development cohort. They do not account for earlier
validation-driven model selection, multiple comparisons, or training variance.
Catalog coverage is reported as an aggregate without a bootstrap interval.

## Reproduction and artifacts

From the repository root, with the v3 feature data, ordinary model, and prior
discovery benchmark available:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m src.models.run_novel_ranker_v1
```

To evaluate the saved novel model without retraining:

```powershell
.\.venv\Scripts\python.exe -m src.models.run_novel_ranker_v1 --evaluate-only
```

- [Runner](../src/models/run_novel_ranker_v1.py)
- [Training report](../results/benchmarks/novel_ranker_v1_training.json)
- [Evaluation and paired intervals](../results/benchmarks/novel_ranker_v1.json)
- [Feature importance](../results/benchmarks/novel_ranker_v1_feature_importance.csv)
- Model: `models/xgb_novel_ranker_v1.json` (local, ignored by Git).
- Per-user metrics: `data/processed/novel_ranker_v1/evaluation_user_metrics.parquet`
  (local, ignored by Git).

The training report records model and user-cohort hashes, parameters, sample
counts, and internal checkpoint history. Tests cover unseen-query filtering,
user separation, slot preservation, score-scale separation, deterministic ties,
and paired uncertainty calculations. All 19 repository tests passed, and the
full train-and-evaluate command exited successfully. Model training and internal
checkpoint selection took 8.15 seconds; full evaluation took 226.93 seconds on
the local environment. These are run timings, not serving-latency benchmarks.
