# Sparse-history and cold-start evaluation

**Follow-up completed:** diversity assessment and architecture selection are now
complete. The [frozen three-source architecture](final_architecture.md) preserves
the routing policy below and was rechecked against the fallback in every segment.

## Decision

**Keep LambdaMART for every tested positive-history segment.** None of the
tested alternatives improves overall Recall@10 or NDCG@10 within these segments.
Use a fallback according to available information, without introducing a new
trained model or a history-count threshold that overrides an available ranker.

| Available information | Recommendation policy |
|---|---|
| Zero history | Global purchase-count popularity |
| Positive history and ranker list available | Preserve the ranker list |
| Positive history, ranker unavailable, repeat list available | Repeat purchase, filling short lists with unique popular products |
| Positive history, neither personalized list available | Global popularity |

The policy is implemented in [routing.py](../src/serving/routing.py). It consumes
precomputed lists and has no Spark, pandas, or model-training dependency. It
deduplicates products, preserves personalized order, and reports `complete=False`
when the available inputs cannot supply K unique products. Zero-history requests
ignore supplied personalized lists. A partial ranker list is preserved and filled
with popularity; this is tested recovery behavior, not a separately measured
accuracy result. All evaluated ranker lists already contained ten products.

This completes the history-availability step. API, cache, and database integration
remain part of the later production phase.

## Known-user experiment

The cohort is the same 41,114 held-out ranker-validation users. Targets are their
official validation baskets. History size is the number of distinct historical
orders in `train_interactions.parquet`, not the maximum order number. Each model
is evaluated on identical users within each segment.

| Historical orders | Users | LambdaMART Recall@10 | Repeat Recall@10 | Repeat + popularity Recall@10 |
|---|---:|---:|---:|---:|
| 1-2 | 6,175 | 0.298455 | 0.270318 | 0.279128 |
| 3-5 | 9,285 | 0.346031 | 0.332094 | 0.333990 |
| 6-10 | 8,788 | 0.361609 | 0.348690 | 0.349201 |
| >10 | 16,866 | 0.375709 | 0.344651 | 0.344726 |

The runner also compares the saved ALS, content, two-tower v1, two-tower v2, and
global-popularity lists. All have lower overall recall and NDCG than LambdaMART
within every segment. This conclusion concerns next-basket accuracy; retrievers
may have greater novel-item recall and still be valuable as retrieval sources.

The strongest tested alternative in all four segments is repeat purchase with
popularity filling. Its paired Recall@10 differences from LambdaMART are:

| Historical orders | Alternative minus ranker | Paired 95% interval |
|---|---:|---:|
| 1-2 | -0.019328 | [-0.021991, -0.016656] |
| 3-5 | -0.012041 | [-0.014088, -0.009923] |
| 6-10 | -0.012408 | [-0.014513, -0.010336] |
| >10 | -0.030983 | [-0.032933, -0.029079] |

Intervals use 2,000 paired user-bootstrap samples with seed 42. They are
exploratory development intervals, unadjusted for multiple comparisons or earlier
model selection. They do not motivate replacing the ranker for sparse users.

### Ranker-unavailable fallback

Across the full known-user cohort, repeat purchase with popularity filling
achieves Recall@10 **0.333406**, compared with **0.331514** for repeat alone and
**0.354390** for LambdaMART. It eliminates short lists in this evaluation.

For users with 1-2 orders, repeat alone returns fewer than ten products for
**38.20%** of users. Filling preserves those repeat products and raises recall
from 0.270318 to 0.279128. This supports its use when the ranker is unavailable,
while the ranker remains the primary recommendation source.

## Separate zero-history scenario

There are **no naturally zero-history users** in the known-user cohort. Its
metrics cannot establish performance for a brand-new user, so this experiment
creates a separate held-user simulation:

1. Exclude all 41,114 evaluation users and all their historical interactions
   from popularity fitting.
2. Fit popularity using 23,630,268 interactions from the other 165,095 users.
3. Recommend the same ten products to every held user, without using their
   history, user embeddings, inferred categories, or onboarding preferences.
4. Evaluate against each held user's first recorded basket: 416,988 target
   interactions in total.

| Metric | Result |
|---|---:|
| Recall@10 | 0.069244 |
| Precision@10 | 0.067916 |
| NDCG@10 | 0.093305 |
| Users with at least one hit | 44.35% |
| Mean returned products | 10 |

The fitting catalog contains 49,546 products; 22 target user-product interactions
involve products absent from that catalog. The fallback recommends only ten
distinct products globally, so its catalog coverage is just **0.02018%**. This
is a simple initial fallback, not evidence of personalization or diversity.

These first-basket metrics are **not directly comparable** with the known-user
next-basket metrics. The scenario simulates history exclusion for dataset users;
it is not an online experiment with new customers. Instacart does not supply a
shared calendar timeline here, so a global-time cutoff across users cannot be
enforced. The simulation does not establish that popularity is the best possible
cold-start approach. Category onboarding would require additional user input
that is not available in this dataset.

## Reproduction, artifacts, and checks

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m src.evaluation.history_availability
```

- [Evaluation runner](../src/evaluation/history_availability.py)
- [Benchmark, segments, and intervals](../results/benchmarks/history_availability.json)
- [Serving routing policy](../src/serving/routing.py)
- Per-user metrics: `data/processed/history_availability/user_metrics.parquet`
  (local, ignored by Git).
- Future unknown-user fallback: `models/popularity_fallback_v1.json`
  (local, ignored by Git), fitted on all training users. This is distinct from
  the held-user-excluded popularity fit used for cold-start evaluation.

The verification run exercises the router across the full cohort: every available
ranker list is preserved, the missing-ranker branch produces the evaluated
repeat-plus-popularity lists, and the zero-history branch generates the cold
predictions. The original ranker control reproduces the prior benchmark within
floating-point tolerance. All 43 tests pass, including excluded-user isolation,
segment boundaries, fixed-K precision, and fallback routing behavior.

The final test remains untouched. The next bounded step is to check whether a
diversity constraint is justified, then choose the simplest supported candidate
and ranking architecture and freeze it before final-test evaluation.
