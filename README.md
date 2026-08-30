# RecoMart

Production-oriented personalized e-commerce recommendation system built on the
Instacart Market Basket Analysis dataset.

## Problem

Given a user's historical grocery purchases, generate a personalized ranked
Top-K list of products that the user is likely to purchase in a future order.

RecoMart is designed as a multi-stage recommender system rather than a
collection of isolated recommendation algorithms.

## Objectives

The system will optimize and evaluate:

- Recommendation relevance
- Ranking quality
- Catalog coverage
- Recommendation diversity
- Cold-start behavior
- Serving latency

## Planned Architecture
```
Raw Instacart Data
        ↓
Data Validation
        ↓
SQL / PySpark ETL
        ↓
User / Item / Interaction Features
        ↓
Temporal Train / Validation / Test Split
        ↓
Candidate Generation
        ↓
Ranking
        ↓
Diversity / Cold-Start Reranking
        ↓
Top-K Recommendations
        ↓
FastAPI / Redis / PostgreSQL
```
## Dataset

Instacart Market Basket Analysis

Core tables:

- orders
- order_products_prior
- order_products_train
- products
- aisles
- departments

Raw data is not stored in this repository.

## Evaluation Strategy

RecoMart uses temporal evaluation rather than random train/test splitting.

For each eligible user:

- Historical orders → training
- Second-to-last order → validation
- Last order → test

Models are evaluated using the same evaluation pipeline.

Initial metrics:

- Recall@K
- Precision@K
- NDCG@K
- HitRate@K
- Catalog Coverage

## Initial Models

The first benchmark will compare:

1. Global popularity
2. Repeat-purchase recommendation
3. Item-item recommendation
4. Collaborative filtering
5. Content-based recommendation
6. Hybrid recommendation

More complex models will only be added when they improve the common offline
evaluation benchmark.

## Current Status

Milestone 1 — Data engineering and baseline recommender evaluation.
