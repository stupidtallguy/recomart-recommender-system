# Discovery-slot experiment

## Question and setup

Can a minimum novel-item quota improve discovery enough to justify displacing
repeat purchases in the final Top-10?

The experiment uses the saved five-source LambdaMART v3 model, checkpoint
iteration 362, and its full candidate pool for the same 41,114 held-out ranker
validation users (6,895,062 candidate rows). The only intervention is a minimum
quota of 0, 1, 2, or 3 previously unseen products. No models are retrained.

If the original list already satisfies the quota, it stays unchanged. Otherwise,
the highest-scoring unseen candidates outside the Top-10 replace the
lowest-ranked seen items. The selected products retain model-score order.
Quotas are best-effort when too few unseen candidates exist.

## Validation results

| Minimum novel slots | Recall@10 | Precision@10 | NDCG@10 | Novel Recall@10 | Novel hit rate | Catalog coverage |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.354390 | 0.292920 | 0.423534 | 0.004735 | 0.013521 | 45.29% |
| 1 | 0.341581 | 0.278944 | 0.413684 | 0.013980 | 0.056847 | 43.40% |
| 2 | 0.326666 | 0.263107 | 0.402009 | 0.021125 | 0.088732 | 41.15% |
| 3 | 0.309785 | 0.245398 | 0.388330 | 0.027485 | 0.114911 | 38.65% |

Full results, including history segments, are in
[the benchmark JSON](../results/benchmarks/discovery_slot_diagnostic.json).

Metrics are averaged per user. Precision always divides by 10, including short
recommendation lists. Novel recall and novel hit rate use only the 33,652 users
with at least one novel product in their validation basket; repeat recall uses
the 37,940 users with repeat-purchase truth. Catalog coverage uses the 49,641
training-observed products as its denominator. Novel truth comes from the raw
`reordered` flag; recommendation novelty comes from the training-history
`seen_before` feature.

## Interpretation and decision

One novel slot increases novel recall by approximately **2.95 times**, but costs
**0.012809 absolute Recall@10**, a **3.61% relative loss**. NDCG@10 falls by
0.009850, and catalog coverage falls by 1.88 percentage points. Requiring two or
three slots costs 7.82% or 12.59% of baseline recall, respectively.

The one-slot policy also reduces overall recall in every measured history
segment. Therefore, these results do not support a free accuracy improvement
through a uniform quota. Higher per-user novelty also does not imply broader
catalog coverage.

**Keep zero forced slots as the current accuracy baseline. Retain quota policies
as experimental comparators; do not promote a uniform quota to the default.**
This is a conservative engineering decision in the absence of a business target
for the value of discovery or an agreed accuracy-loss budget. An online retailer
could make a different choice based on measured customer outcomes.

The next justified model experiment is a **dedicated unseen-item ranker**. Its
hypothesis is that ranking novel products separately can produce better
discovery than the existing ranker's novel-item ordering at the same slot count.
The present experiment does not establish that this hypothesis is true.

### Next experiment contract

**Follow-up completed:** the dedicated unseen-item ranker did not demonstrate
an improvement over the existing one-slot policy. See the
[experiment report](novel_ranker_experiment.md) for results and the decision.
The protocol below records the planned comparison.

1. Freeze the existing five candidate sources and deterministic user split.
2. Train only on unseen candidates from ranker-training users, preserving the
   corrected negative-sampling strategy and keeping held-out users out of fitting.
3. At one fixed novel slot, compare the dedicated novel ordering with the
   existing one-slot policy. Also report the zero-slot accuracy control.
4. Report overall Recall/Precision/NDCG, novel recall and hit rate, coverage, and
   history segments on the same users. Include paired uncertainty estimates
   before interpreting small improvements.
5. Keep the added model only if measured improvements justify its complexity.
   These remain development results because validation informs model selection.

Cold-start and diversity work follow that decision. The final test remains
reserved until the recommendation architecture and evaluation protocol freeze.

## Reproduction and checks

Run from the repository root with the prepared data and saved model available:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m src.evaluation.discovery_slot_diagnostic
```

The reported `recommendations`/`recommended` typo was already corrected in the
working tree. The fresh run reproduced the saved metrics, then exposed a Windows
console encoding error in the printed delta labels. Those labels now use ASCII.
Regression tests cover minimum-quota semantics, preservation of
existing novel products, replacement priority, insufficient candidates, short
lists, ranker-validation population selection, and fixed-K precision.

Runtime: Python 3.14.0, pandas 3.0.5, NumPy 2.5.2, XGBoost 3.4.1.
Saved model SHA-256:
`0295A31F2FDCDFFA70981CD611CA35E615B13C84D26D895C9991B6753A336199`.

The diagnostic reads validation artifacts and the saved model. It does not read
the final test interactions. The separate v3 training runner's precision was also
corrected to use K instead of returned-list length. Re-evaluating its existing
repeat candidates changed control Precision@10 from 0.286421 to 0.271788; recall
and NDCG stayed unchanged. The corrected v3 precision improvement over repeat is
0.021131. Its saved benchmark now reflects that correction, without retraining.
