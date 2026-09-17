"""Read-only catalog and shopper context for the demo; no new ranking policy."""
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Path, Query, Request

router = APIRouter(tags=["explorer"])


class CatalogIndex:
    def __init__(self, store):
        self.products = tuple(sorted(store.products.values(), key=lambda p: (p.product_name.casefold(), p.product_id)))
        self.search = tuple((p, f"{p.product_name} {p.aisle} {p.department}".casefold()) for p in self.products)
        self.departments = sorted({p.department for p in self.products})

    def find(self, query, department, offset, limit):
        terms = query.casefold().split()
        matches = [p for p, text in self.search if (not department or p.department == department)
                   and all(term in text for term in terms)]
        return {"total": len(matches), "offset": offset, "limit": limit,
                "products": [asdict(p) for p in matches[offset:offset + limit]]}


@router.get("/catalog")
def catalog(request: Request):
    index = request.app.state.catalog
    return {"total": len(index.products), "departments": index.departments}


@router.get("/catalog/products")
def search_products(request: Request, q: str = Query(default="", max_length=120),
                    department: str = Query(default="", max_length=80),
                    offset: int = Query(default=0, ge=0, le=1000000),
                    limit: int = Query(default=12, ge=1, le=48)):
    return request.app.state.catalog.find(q, department, offset, limit)


@router.get("/catalog/products/{product_id}/neighbors")
def neighbors(request: Request, product_id: int = Path(gt=0, le=9223372036854775807)):
    store = request.app.state.store
    product = store.products.get(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    items = [asdict(p) for p in request.app.state.catalog.products
             if p.aisle_id == product.aisle_id and p.product_id != product_id][:6]
    return {"basis": "same_aisle", "products": items}


@router.get("/users/{user_id}/profile")
def profile(request: Request, user_id: int = Path(gt=0, le=9223372036854775807)):
    store = request.app.state.store
    return {"user_id": user_id, "known": user_id in store.histories,
            "history_orders": store.histories.get(user_id, 0),
            "usual_picks": [asdict(store.products[p]) for p in store.repeat.get(user_id, ())],
            "basis": "repeat_purchase_baseline", "snapshot_id": store.snapshot_id}
