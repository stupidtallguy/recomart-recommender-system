"""Run with: python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Path as PathParameter, Query, Request, Response
from pydantic import BaseModel

from src.serving.store import ROOT, RecommendationStore, load_snapshot
from src.serving.service import RecommendationService
from src.api.explorer import CatalogIndex, router as explorer_router

logger = logging.getLogger(__name__)
PositiveID = Annotated[int, PathParameter(gt=0, le=9223372036854775807)]


class ProductResponse(BaseModel):
    product_id: int
    product_name: str
    aisle_id: int
    aisle: str
    department_id: int
    department: str


class RecommendationResponse(BaseModel):
    user_id: int
    k: int
    model_version: str
    snapshot_id: str
    strategy: Literal["ranker", "ranker_with_popularity_fill", "repeat_with_popularity_fill", "global_popularity"]
    complete: bool
    recommendations: list[ProductResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ready: bool
    model_version: str
    snapshot_id: str
    snapshot_created_at: str
    known_users: int
    catalog_products: int
    users_with_ranked_lists: int
    users_with_repeat_lists: int
    database: Literal["disabled", "ok", "unavailable"]
    cache: Literal["disabled", "ok", "unavailable"]
    cache_counters: dict[str, int]


def get_store(request: Request) -> RecommendationStore:
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Recommendation snapshot is not ready")
    return store


def create_app(*, artifact_root: Path | None = None, store: RecommendationStore | None = None,
               service_factory=RecommendationService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load once before accepting requests. Failure aborts startup instead of
        # exposing inconsistent recommendations. Each worker has its own copy.
        root = artifact_root or Path(os.environ.get("RECOMART_ARTIFACT_ROOT", str(ROOT)))
        snapshot = store if store is not None else load_snapshot(root)
        service = service_factory(snapshot, database_url=os.environ.get("RECOMART_DATABASE_URL"),
            redis_url=os.environ.get("RECOMART_REDIS_URL"),
            ttl_seconds=int(os.environ.get("RECOMART_CACHE_TTL_SECONDS", "3600")))
        app.state.store = snapshot
        app.state.service = service
        app.state.catalog = CatalogIndex(snapshot)
        logger.info("Loaded snapshot %s (%d users, degraded=%s)",
                    snapshot.snapshot_id, len(snapshot.histories), snapshot.degraded)
        try:
            yield
        finally:
            app.state.store = None
            app.state.service = None
            service.close()

    app = FastAPI(
        title="RecoMart Recommendation API", version="1.0.0", lifespan=lifespan,
        description="Precomputed next-basket recommendations from the frozen RecoMart model. "
                    "Unknown users receive the popularity fallback. Rankings are refreshed offline.",
    )
    app.include_router(explorer_router)

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    def health(request: Request):
        snapshot = get_store(request)
        dependencies = request.app.state.service.health()
        return HealthResponse(
            status="degraded" if snapshot.degraded or "unavailable" in dependencies.values() else "ok", ready=True,
            model_version=snapshot.model_version, snapshot_id=snapshot.snapshot_id,
            snapshot_created_at=snapshot.snapshot_created_at,
            known_users=len(snapshot.histories), catalog_products=len(snapshot.products),
            users_with_ranked_lists=len(snapshot.ranked), users_with_repeat_lists=len(snapshot.repeat),
            **dependencies,
        )

    @app.get("/recommendations/{user_id}", response_model=RecommendationResponse, tags=["recommendations"])
    def recommendations(user_id: PositiveID, request: Request, response: Response, k: Annotated[int, Query(ge=1, le=10)] = 10):
        snapshot = get_store(request)
        route, cache_status, source = request.app.state.service.recommend(user_id, k)
        response.headers["X-RecoMart-Cache"] = cache_status
        response.headers["X-RecoMart-Source"] = source
        return RecommendationResponse(
            user_id=user_id, k=k, model_version=snapshot.model_version, snapshot_id=snapshot.snapshot_id,
            strategy=route.strategy, complete=route.complete,
            recommendations=[ProductResponse(**{
                field: getattr(snapshot.products[product], field)
                for field in ProductResponse.model_fields
            }) for product in route.recommendations],
        )

    @app.get("/products/{product_id}", response_model=ProductResponse, tags=["products"])
    def product(product_id: PositiveID, request: Request):
        get_store(request)
        item = request.app.state.service.product(product_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Product not found")
        return ProductResponse(**{field: getattr(item, field) for field in ProductResponse.model_fields})

    return app


app = create_app()
