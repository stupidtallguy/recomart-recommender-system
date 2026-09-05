# RecoMart

**Production-Oriented Personalized E-Commerce Recommendation System**

RecoMart is an end-to-end recommender-system project built on the Instacart Market Basket Analysis dataset.

The objective is to predict a personalized ranked **Top-K list of products a customer is likely to purchase in a future order**, while progressively evolving from simple baselines into collaborative filtering, hybrid retrieval, neural recommendation, ranking, cold-start handling, and production serving.

The project is intentionally structured around one principle:

> **Every new model must earn its complexity by being evaluated under the same temporal recommendation framework.**

---

## Current Results

### Offline Validation Benchmark

All current models are evaluated against the same future validation order for **206,209 users**.

| Model | Recall@10 | Precision@10 | NDCG@10 |
|---|---:|---:|---:|
| Global Popularity | 0.0701 | 0.0706 | 0.0964 |
| **Repeat Purchase** | **0.3316** | **0.2714** | **0.3951** |
| Item Similarity — Discovery | 0.0209 | 0.0207 | 0.0270 |
| Spark ALS — Purchase Count | 0.1111 | 0.0818 | 0.1207 |
### Interpretation

The personalized repeat-purchase baseline strongly outperforms global popularity, demonstrating the importance of habitual purchase behavior in grocery recommendation.

The current item-item model intentionally excludes products the user has already purchased. Its lower overall next-basket Recall@10 therefore should not be interpreted as a direct failure against the repeat-purchase model: it is currently optimized as a **discovery candidate generator** for unseen products.

Future evaluation will separately measure overall next-basket performance and novel-item discovery.

---

# Dataset Scale

RecoMart currently processes the full known interaction portion of the Instacart Market Basket Analysis dataset.

| Entity                           |      Count |
| -------------------------------- | ---------: |
| Users                            |    206,209 |
| Orders                           |  3,421,083 |
| Products                         |     49,688 |
| Aisles                           |        134 |
| Departments                      |         21 |
| Known order-product interactions | 33,819,106 |

### Interaction Sources

| Source    |   Interactions |
| --------- | -------------: |
| Prior     |     32,434,489 |
| Train     |      1,384,617 |
| **Total** | **33,819,106** |

---

# Data Engineering Pipeline

Raw CSV files are treated as immutable inputs.

```text
RAW INSTACART DATA
        │
        ▼
Schema Validation
        │
        ▼
Data Quality Checks
        │
        ▼
Chunked ETL
        │
        ▼
Typed Parquet Storage
        │
        ├───────────────┐
        ▼               ▼
 orders.parquet    products.parquet
        │
        └───────┬───────┘
                ▼
       interactions.parquet
                │
                ▼
          Temporal Split
```

The preprocessing pipeline validates:

* Required files
* Required schemas
* Primary-key uniqueness
* Product foreign keys
* Order foreign keys
* Valid reorder indicators
* Valid order metadata
* Unknown products
* Unknown orders
* Source/evaluation-set consistency
* Duplicate order-product pairs

All current data-quality checks pass with zero detected violations.

---

# Processed Data Layer

```text
data/processed/
│
├── orders.parquet
├── products.parquet
├── interactions.parquet
│
├── train_interactions.parquet
├── validation_interactions.parquet
├── test_interactions.parquet
│
├── data_quality_report.json
└── split_report.json
```

Parquet is used for typed, compressed analytical storage and interoperability with Pandas, PyArrow, and Spark.

---

# Temporal Evaluation

RecoMart does **not** use a random train/test split.

For every eligible user:

```text
Historical Orders
      │
      ├── Order 1
      ├── Order 2
      ├── ...
      ├── Order N-2   → TRAIN
      │
      ├── Order N-1   → VALIDATION
      │
      └── Order N     → TEST
```

This ensures the model learns only from past behavior when predicting future purchases.

## Split Statistics

| Split      | Interactions |    Orders |   Users |
| ---------- | -----------: | --------: | ------: |
| Train      |   29,524,435 | 2,933,665 | 206,209 |
| Validation |    2,129,254 |   206,209 | 206,209 |
| Test       |    2,165,417 |   206,209 | 206,209 |

The test split remains reserved for later final model comparison.

---

# Evaluation Framework

Current ranking metrics:

* Recall@K
* Precision@K
* NDCG@K

Metric implementations are covered by automated unit tests.

Planned additions:

* HitRate@K
* Catalog Coverage@K
* Novel Recall@K
* Recommendation diversity
* Cold-start metrics
* Candidate retrieval recall
* Serving latency

---

# Implemented Models

## 1. Global Popularity

Non-personalized baseline that recommends globally frequent products.

```text
All Users
    │
    ▼
Same Globally Popular Products
```

### Result

```text
Recall@10    0.0701
Precision@10 0.0706
NDCG@10      0.0964
```

This establishes the minimum benchmark more sophisticated models must beat.

---

## 2. Repeat Purchase Recommender

Personalized baseline using historical user-product behavior.

Signals currently include:

* Purchase frequency
* Reorder behavior
* Purchase recency

```text
User Purchase History
        │
        ▼
User-Product Aggregation
        │
        ▼
Historical Preference Score
        │
        ▼
Personalized Top-K
```

### Result

```text
Recall@10    0.3316
Precision@10 0.2714
NDCG@10      0.3951
```

Repeat-purchase behavior currently represents the strongest baseline.

---

## 3. Sparse Item-Item Collaborative Filtering

Item similarity is derived from collaborative purchase behavior.

The training data contains approximately:

```text
29.5M training interactions
        │
        ▼
12.1M unique user-product relationships
```

A binary sparse user-item matrix is constructed:

```text
206,209 users
     ×
49,641 training products
```

Instead of materializing a complete item-item similarity matrix, RecoMart performs sparse Top-N multiplication and retains only useful item relationships.

Current model:

```text
Sparse User × Item Matrix
        │
        ▼
Normalized Item Vectors
        │
        ▼
Top-50 Similar Items / Product
        │
        ▼
2.48M Stored Similarities
        │
        ▼
Top-100 User Candidates
        │
        ▼
Remove Previously Seen Products
        │
        ▼
Precomputed Top-K Recommendations
```

The original per-user nearest-neighbor implementation was replaced because repeated nearest-neighbor searches during evaluation did not scale to the full 206K-user validation population.

Recommendation generation is now performed offline and evaluation uses precomputed user recommendations.

### Discovery Result

```text
Recall@10    0.0209
Precision@10 0.0207
NDCG@10      0.0270
```

This variant excludes previously purchased products and will later contribute novel candidates to the hybrid recommender.

---

# Current System Architecture

```mermaid
flowchart TD

    A[Raw Instacart Data] --> B[Schema Validation]
    B --> C[Chunked ETL]
    C --> D[Parquet Data Layer]

    D --> E[Temporal Split]

    E --> F[Train]
    E --> G[Validation]
    E --> H[Test]

    F --> I[Global Popularity]
    F --> J[Repeat Purchase]
    F --> K[Item Similarity]
    F --> L[Spark ALS - Next]

    I --> M[Common Evaluation Framework]
    J --> M
    K --> M
    L --> M

    G --> M

    M --> N[Recall@K]
    M --> O[Precision@K]
    M --> P[NDCG@K]
```

---

# Project Structure

```text
recomart-recommender-system/
│
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_baseline_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── validate_raw.py
│   │   ├── preprocess.py
│   │   └── split.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate.py
│   │
│   └── models/
│       ├── popularity.py
│       ├── repeat_purchase.py
│       └── item_similarity.py
│
├── tests/
│   └── test_metrics.py
│
├── results/
│   └── benchmarks/
│
├── models/
└── docs/
```

---

# Roadmap

## Foundation

* [x] Repository architecture
* [x] Isolated Python environment
* [x] Raw schema validation
* [x] Data-quality validation
* [x] Chunk-based preprocessing
* [x] Parquet analytical data layer
* [x] Temporal train/validation/test split
* [x] Ranking metric implementation
* [x] Unit tests

## Baseline Recommendation

* [x] Global popularity
* [x] Personalized repeat purchase
* [x] Sparse item-item collaborative filtering
* [ ] Coverage and novel-item evaluation

## Collaborative Filtering

- [x] PySpark implicit ALS
* [ ] Interaction-confidence engineering
* [ ] ALS hyperparameter validation
* [ ] Latent-factor recommendation analysis

## Content & Hybrid Retrieval

* [ ] Product metadata features
* [ ] Content-based recommendation
* [ ] Multi-source candidate generation
* [ ] Collaborative + content hybrid

## Deep Recommendation

* [ ] PyTorch Two-Tower model
* [ ] User embeddings
* [ ] Product embeddings
* [ ] Negative sampling
* [ ] Embedding retrieval

## Ranking

* [ ] Candidate feature table
* [ ] Learning-to-rank model
* [ ] Purchase-frequency features
* [ ] Recency features
* [ ] Collaborative scores
* [ ] Content scores
* [ ] User-category affinity

## Production

* [ ] Cold-start strategy
* [ ] Diversity reranking
* [ ] FastAPI serving
* [ ] PostgreSQL
* [ ] Redis recommendation cache
* [ ] Docker
* [ ] Model versioning
* [ ] Monitoring
* [ ] API latency benchmarks

## Demo

* [ ] Streamlit application
* [ ] User history explorer
* [ ] Personalized recommendations
* [ ] Similar-product explorer
* [ ] Model benchmark dashboard

---

# Current Focus

## Spark ALS — Implicit Collaborative Filtering

RecoMart uses PySpark ALS with implicit feedback rather than treating
purchases as explicit ratings.

The first experiment aggregates historical purchases into user-product
purchase counts and uses those counts as implicit confidence signals.

### Dataset

- 206,209 users
- 49,641 products observed during training
- 12,084,910 unique user-product relationships
- 29,524,435 training interactions

### Configuration

- Rank: 32
- Iterations: 10
- Regularization: 0.05
- Alpha: 20
- Interaction signal: purchase count

### Validation Result

- Recall@10: 0.1111
- Precision@10: 0.0818
- NDCG@10: 0.1207

ALS substantially outperforms global popularity but does not outperform the
strong personalized repeat-purchase baseline. This indicates that latent
collaborative preference alone does not fully capture the highly repetitive
nature of grocery purchasing.

The ALS signal will later be combined with repeat behavior, recency, content,
and ranking features rather than treated as a standalone final recommender.
```text
User × Product Interaction Strength
                │
                ▼
          PySpark ALS
                │
       ┌────────┴────────┐
       ▼                 ▼
 User Factors       Product Factors
       │                 │
       └────────┬────────┘
                ▼
        Collaborative Score
                │
                ▼
       Personalized Top-K
```

The first ALS experiment will use historical purchase count as implicit interaction strength.

Later experiments will incorporate:

* Reorder confidence
* Recency
* Frequency weighting
* Tuned latent dimension
* Regularization
* Implicit-feedback alpha

ALS must be evaluated using the same validation users and ranking metrics as all existing models.

---

# Core Engineering Principle

RecoMart is not intended to be a collection of unrelated recommendation algorithms.

Its progression is:

```text
DATA
 ↓
TEMPORAL EVALUATION
 ↓
SIMPLE BASELINES
 ↓
COLLABORATIVE FILTERING
 ↓
CONTENT RETRIEVAL
 ↓
HYBRID CANDIDATES
 ↓
DEEP RETRIEVAL
 ↓
RANKING
 ↓
COLD START / DIVERSITY
 ↓
API SERVING
 ↓
MONITORING
```

Every additional layer must justify its complexity through measurable improvement or a clearly defined system capability.
