# Deployment packaging

RecoMart now includes a Docker image and Compose stack for reproducible local
startup of the API, demo, PostgreSQL, Redis, and snapshot import step.

The image contains source code and the lightweight serving/demo runtime from
`requirements-api-lock.txt` and `requirements-demo-lock.txt`. It does not
contain raw Instacart data, processed parquet files, trained models, local
credentials, or database state. Mount a local artifact root at runtime.

## Required artifact layout

`RECOMART_ARTIFACT_ROOT` must point to a directory with the serving files used
by the API:

```text
configs/final_architecture.json
models/final_v1/prediction_seal.json
models/final_v1/popularity.json
data/processed/final_v1/item_features.parquet
data/processed/final_v1/history_summary.parquet
data/processed/final_v1/predictions/selected_lambdamart.parquet
data/processed/final_v1/predictions/repeat_purchase.parquet
```

The API verifies every loaded artifact against the prediction seal. If the
mandatory files are missing or corrupt, startup fails.

## Compose startup

From the repository root:

```powershell
Copy-Item deploy/compose.env.example .env
notepad .env
docker compose up --build
```

Set `RECOMART_ARTIFACT_ROOT` in `.env` to the host path containing the sealed
artifacts. On Windows, use forward slashes, for example:

```text
RECOMART_ARTIFACT_ROOT=E:/Programming/recomart-recommender-system
```

Compose starts PostgreSQL and Redis, imports the snapshot, starts FastAPI, then
starts the Streamlit demo. The import is idempotent for an already published
snapshot.

Open:

- API: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/health>
- Demo: <http://127.0.0.1:8501>

Stop the stack:

```powershell
docker compose down
```

Remove the local PostgreSQL volume:

```powershell
docker compose down -v
```

## Image-only startup

Build the image:

```powershell
docker build -t recomart-app:local .
```

Snapshot-only API mode:

```powershell
docker run --rm -p 8000:8000 `
  -e RECOMART_ARTIFACT_ROOT=/artifacts `
  -v E:/Programming/recomart-recommender-system:/artifacts:ro `
  recomart-app:local
```

Demo against a running API:

```powershell
docker run --rm -p 8501:8501 `
  -e RECOMART_API_URL=http://host.docker.internal:8000 `
  recomart-app:local `
  python -m streamlit run demo/app.py --server.address 0.0.0.0 --server.port 8501
```

## Configuration

| Variable | Meaning |
|---|---|
| `RECOMART_ARTIFACT_ROOT` | Host path mounted into the containers as `/artifacts`. |
| `POSTGRES_PASSWORD` | Local Compose PostgreSQL password. |
| `REDIS_PASSWORD` | Local Compose Redis password. |
| `POSTGRES_PORT` | Host PostgreSQL port; default `55432`. |
| `REDIS_PORT` | Host Redis port; default `56379`. |
| `API_PORT` | Host FastAPI port; default `8000`. |
| `DEMO_PORT` | Host Streamlit port; default `8501`. |
| `RECOMART_CACHE_TTL_SECONDS` | Redis recommendation cache lifetime. |

For public exposure, put the API behind TLS and add authentication, rate limits,
request metrics, error metrics, and health alerts. The current stack is a
reproducible local deployment over a static historical snapshot.
