# RecoMart demo

The Streamlit demo connects to the real FastAPI service at
`http://127.0.0.1:8000`. Open the running interface at
[localhost:8501](http://127.0.0.1:8501).

## Run

From the repository root, install the pinned demo runtime:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-demo-lock.txt
```

Start the API using the [PostgreSQL/Redis guide](database_cache.md), then run the
demo in a second terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run demo/app.py
```

The API can also run in snapshot-only mode. The demo does not connect directly
to PostgreSQL or Redis. To point it at another API, set `RECOMART_API_URL` before
starting Streamlit. Run from the repository root so `.streamlit/config.toml`
loads the theme and loopback server settings. Restart the demo after changing
shared Python modules or configuration. Stop a foreground server with Ctrl+C.

The minimal demo environment does not need Spark, PyTorch, XGBoost, the training
data, or test labels. The Model report view reads only the saved aggregate
benchmark in `results/benchmarks/final_test_v1.json`; it does not reevaluate or
change the model.

## Views

- **For you:** enter a shopper ID, switch to an example shopper, or try the new
  shopper case. Recommendations preserve the API's ranked order, with 3, 6, or
  10 displayed products. Usual picks show the saved repeat-purchase baseline
  and the number of historical orders. They are not a complete order history.
- **Explore catalog:** case-insensitive search across names, aisles, and
  departments; department filtering; pagination; and product detail dialogs.
  More in this aisle is a category browse feature, not learned item similarity.
- **Saved list:** save/remove products, keep separate lists for each selected
  shopper, and download a spreadsheet-safe CSV. Lists belong to the live
  Streamlit session; refreshing or disconnecting can clear them. Download a
  list to retain it. Saving does not create a purchase or retrain the model.
- **Model report:** compare all six final-test policies using Recall,
  Precision, or NDCG, with metric definitions and the downloadable report.

## Visual system and accessibility

| Role | Color |
|---|---|
| Navigation and primary text | `#091540` |
| Primary actions and selected states | `#1B2CC1` |
| Accents and illustrations | `#7692FF` |

The third and fourth requested colors were identical, so both use the same
accent. White and pale neutral surfaces provide readable contrast. Product
artwork is a set of local SVG category illustrations, not product photography.
The app makes no claims about live prices, inventory, or checkout.

Native controls retain keyboard operation and accessible names. Focus outlines
are visible, selected products have a text marker, and reduced-motion preferences
are respected. Responsive cards wrap to a single column at phone widths; the
sidebar becomes a drawer. Product details expose full names where card titles
are visually shortened. No externally hosted fonts or images are required.

## API additions

All explorer data comes from the already verified snapshot. The ranking policy
and sealed artifacts are unchanged.

| Route | Purpose |
|---|---|
| `GET /catalog` | Catalog count and department options. |
| `GET /catalog/products?q=&department=&offset=0&limit=12` | Alphabetical, paginated search; maximum page size 48. |
| `GET /catalog/products/{product_id}/neighbors` | Up to six other products from the same aisle. |
| `GET /users/{user_id}/profile` | Known/unknown state, historical order count, usual picks, snapshot ID. |

Unknown shoppers receive a clear popularity message. Missing or short rankings
show the actual fallback state. API outages offer retry actions while saved
lists remain usable. Health uses the same live API as recommendations. Snapshot
identity mismatches are surfaced instead of silently combining profile data.

## Verification

**135 tests pass** with the live PostgreSQL test configuration from the backend
guide. Six tests skip when that configuration is absent. Sixteen new demo tests
cover catalog filtering/pagination, profiles, rank preservation, saved-list
isolation, dialogs, invalid IDs, cold start, API outages, report controls, and
safe rendering/export of catalog strings.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_demo.py -q -p no:cacheprovider
```

Browser checks exercised product dialogs, saving and dismissal, mobile
navigation, catalog search, and responsive layouts at desktop and 390 px phone
width. This is functional and visual verification, not a formal accessibility
audit or a production load test.

The [live API smoke report](../results/benchmarks/demo_api_smoke.json) also
confirms ranking parity for all 206,209 shoppers and 100 real HTTP requests.
All 99 sealed artifacts still match their original hashes.

Implementation references: [Streamlit themes](https://docs.streamlit.io/develop/concepts/configuration/theming-customize-colors-and-borders),
[containers](https://docs.streamlit.io/develop/api-reference/layout/st.container),
and [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).

## Next step

Prepare deployment and monitoring: reproducible service startup, authentication
and rate limits before public access, request/error metrics, and health alerts.
This demo remains a local interface over a static historical snapshot.
