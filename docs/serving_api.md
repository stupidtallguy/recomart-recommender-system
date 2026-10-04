# RecoMart serving API

The FastAPI service serves the frozen recommendations and product metadata from
one verified, read-only snapshot. Loading happens once at startup; requests use
the unchanged evaluated fallback policy. Optional PostgreSQL storage and Redis
caching are implemented; see the [backend setup guide](database_cache.md).
The [deployment guide](deployment.md) documents Docker and Compose startup.
The command below starts snapshot-only mode when no backend URLs are set.

## Run locally

From the repository root, using the existing project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-api-lock.txt
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open [interactive API documentation](http://127.0.0.1:8000/docs),
[health](http://127.0.0.1:8000/health), or
[user 1's recommendations](http://127.0.0.1:8000/recommendations/1).
Stop a foreground server with Ctrl+C. Use one worker initially; each additional
worker loads its own snapshot into memory.

The small serving runtime is listed in `requirements-api.txt`; exact resolved
versions are in `requirements-api-lock.txt`. Spark, XGBoost, PyTorch, training
features, and final-test labels are not needed by the API.

An optional environment variable selects an artifact root with the same layout:

```powershell
$env:RECOMART_ARTIFACT_ROOT = 'E:/Programming/recomart-recommender-system'
```

The service reads the environment directly. `.env.example` documents this
setting; `.env` files are not loaded automatically.

## HTTP contract

| Endpoint | Behavior |
|---|---|
| `GET /recommendations/{user_id}?k=10` | Ranked products with metadata; `k` must be 1-10. |
| `GET /products/{product_id}` | Product name, aisle, and department; unknown products return 404. |
| `GET /health` | Snapshot identity, availability, user/product counts, and source coverage. |
| `GET /docs` | Interactive OpenAPI documentation. |
| `GET /openapi.json` | Machine-readable API schema. |

User and product IDs must be positive integers within the signed 64-bit range.
Invalid IDs or `k` values return 422. A valid unknown user returns 200 with
popularity recommendations. A snapshot that is not ready returns 503.

Example request:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/recommendations/1?k=3'
```

Responses contain `user_id`, requested `k`, `model_version`, `snapshot_id`,
`strategy`, `complete`, and `recommendations`. Each recommendation includes
`product_id`, `product_name`, `aisle_id`, `aisle`, `department_id`, and
`department`. No relevance score is exposed because the serving artifacts
contain ranked product lists. For `k` below ten, the service preserves the
corresponding prefix of the evaluated ranking.

### Fallback behavior

| Available data | Response strategy |
|---|---|
| Known user with enough ranked products | `ranker` |
| Partial ranked list | `ranker_with_popularity_fill` |
| Missing ranked list, available repeat history | `repeat_with_popularity_fill` |
| Unknown user or no personalized list | `global_popularity` |

Filling preserves order and removes duplicates. If the catalog cannot provide
the requested number of unique products, the response contains available
products and sets `complete=false`.

## Artifact loading and integrity

Required files, relative to the artifact root:

```text
configs/final_architecture.json
models/final_v1/prediction_seal.json
models/final_v1/popularity.json
data/processed/final_v1/item_features.parquet
data/processed/final_v1/history_summary.parquet
```

Optional personalized sources:

```text
data/processed/final_v1/predictions/selected_lambdamart.parquet
data/processed/final_v1/predictions/repeat_purchase.parquet
```

Every artifact that is loaded must match its SHA-256 entry in the prediction
seal. The loader parses the exact bytes it verified and checks the frozen
contract, routing implementation, IDs, product membership, uniqueness, and user
population. The running routing source must match the frozen implementation.
These checks establish consistency with a trusted local seal; they are not a
digital signature or substitute for deployment access controls.

A missing optional source permits startup with fallback service. `/health`
returns HTTP 200 with `status="degraded"` and `ready=true`, because the fallback
can serve requests. Missing mandatory files or a corrupt present artifact abort
startup. Corruption is never treated as ordinary source unavailability.

The snapshot is immutable while the process runs. Replacing files on disk does
not refresh an existing worker. Snapshot updates require a coordinated restart.
The health response exposes the snapshot creation time and full seal hash.

## Verification

```powershell
# Full project environment required for the complete test suite.
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
# Run while the server is listening.
.\.venv\Scripts\python.exe -m src.api.smoke_test
```

**135 tests pass** with the live PostgreSQL test configuration in the
[backend guide](database_cache.md), including 30 API/store tests, 20 backend
tests, six live database tests, and sixteen demo tests. Without that configuration, six tests skip.
The API/store tests exercise known and unknown
users, missing and partial sources, missing mandatory data, checksum failures,
invalid artifacts and parameters, short catalogs, immutability, and readiness.

The [full-snapshot smoke report](../results/benchmarks/api_smoke.json) confirms:

- **206,209 users** and **49,688 products** loaded successfully.
- Routing preserves the saved ranking for **all 206,209 users**.
- **100 real HTTP requests** match the saved recommendation prefixes.
- Unknown-user fallback, product lookup, invalid input, missing products, and
  OpenAPI checks pass.
- In the initial snapshot-only run, loading took approximately **1.8 seconds** locally. Sequential
  loopback requests had **15.3 ms median / 21.2 ms p95** latency in this smoke run.
  These timings include the local client and are not a capacity benchmark.
- The serving smoke process imported no XGBoost, Spark, or PyTorch modules.

The implementation follows FastAPI's documented
[lifespan loading](https://fastapi.tiangolo.com/advanced/events/) and
[testing](https://fastapi.tiangolo.com/tutorial/testing/) patterns.

## Current scope and next step

This is a local, unauthenticated API over a static historical snapshot. It does
not yet ingest new purchases or run online ranking. Public deployment and its
authentication, rate limits, and TLS remain future work.

PostgreSQL storage and Redis caching are complete and verified; see the
[integration guide](database_cache.md). The [Streamlit demo](demo.md) is also
complete, including new read-only explorer endpoints. Reproducible Docker
Compose startup is documented in the [deployment guide](deployment.md). Next:
authentication, rate limits, metrics, and monitoring.
