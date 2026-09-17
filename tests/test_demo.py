from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType
from urllib.error import URLError

from fastapi.testclient import TestClient
import pytest
from streamlit.testing.v1 import AppTest

from demo import client as demo_client
from demo.design import card_html
from src.api.main import create_app
from src.serving.service import RecommendationService
from src.serving.store import Product, RecommendationStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def api():
    products = {i: Product(i, f"Apple {i:02}", 1 if i < 10 else 2,
                          "fresh fruit" if i < 10 else "juice", 1 if i < 10 else 2,
                          "produce" if i < 10 else "beverages") for i in range(1, 16)}
    store = RecommendationStore("test", "a" * 64, "2026-09-16T00:00:00Z",
        MappingProxyType(products), MappingProxyType({1: 7, 42: 2}),
        MappingProxyType({1: tuple(range(10, 0, -1)), 42: tuple(range(1, 11))}),
        MappingProxyType({1: (2, 3), 42: (4, 5)}), tuple(range(1, 16)), False)
    with TestClient(create_app(store=store, service_factory=lambda s, **kw: RecommendationService(s))) as api:
        yield api


@pytest.fixture
def app(api, monkeypatch):
    def fetch(path, **params):
        response = api.get(path, params=params)
        response.raise_for_status()
        return demo_client.Result(response.json(), response.headers.get("X-RecoMart-Source", ""),
                                  response.headers.get("X-RecoMart-Cache", ""))
    monkeypatch.setattr(demo_client, "fetch", fetch)
    return AppTest.from_file(str(ROOT / "demo/app.py"), default_timeout=10).run()


def test_catalog_search_filters_and_paginates_without_overlap(api):
    first = api.get("/catalog/products", params={"q": "APPLE", "limit": 5}).json()
    second = api.get("/catalog/products", params={"q": "apple", "limit": 5, "offset": 5}).json()
    assert first["total"] == 15
    assert not ({p["product_id"] for p in first["products"]} & {p["product_id"] for p in second["products"]})
    result = api.get("/catalog/products", params={"q": "apple juice", "department": "beverages"}).json()
    assert result["total"] == 6
    assert all(p["department"] == "beverages" for p in result["products"])
    assert api.get("/catalog/products", params={"q": "not-a-real-product"}).json()["total"] == 0
    assert api.get("/catalog/products", params={"limit": 1000}).status_code == 422
    assert api.get("/catalog/products", params={"offset": -1}).status_code == 422


def test_profile_and_same_aisle_explorer_are_explicit(api):
    profile = api.get("/users/1/profile").json()
    assert profile["known"] and profile["history_orders"] == 7
    assert [p["product_id"] for p in profile["usual_picks"]] == [2, 3]
    unknown = api.get("/users/999/profile").json()
    assert not unknown["known"] and unknown["usual_picks"] == []
    result = api.get("/catalog/products/1/neighbors").json()
    assert result["basis"] == "same_aisle"
    assert all(p["aisle_id"] == 1 and p["product_id"] != 1 for p in result["products"])
    assert api.get("/catalog/products/999/neighbors").status_code == 404
    assert api.get("/users/0/profile").status_code == 422


def test_ui_preserves_rank_and_saved_lists_are_per_shopper(app):
    assert not app.exception
    assert [b.key for b in app.button if b.key and b.key.startswith("details_recommendations")] == [f"details_recommendations_{i}" for i in range(10, 0, -1)]
    app.button(key="save_recommendations_10").click().run()
    assert not app.exception
    assert 10 in app.session_state["lists"][1]
    app.radio(key="nav").set_value("Saved list").run()
    assert app.button(key="save_saved_10").label == "Saved ✓"
    next(b for b in app.sidebar.button if b.label == "Try #42").click().run()
    assert app.session_state["shopper"] == 42
    assert not app.exception
    assert app.session_state["lists"].get(42, {}) == {}
    assert 10 in app.session_state["lists"][1]
    assert app.session_state["nav"] == "For you"


def test_ui_invalid_id_keeps_previous_profile_and_unknown_falls_back(app):
    app.text_input(key="shopper_input").set_value("0")
    next(b for b in app.button if b.label == "Load recommendations →").click().run()
    assert app.session_state["shopper"] == 1
    assert app.error
    next(b for b in app.button if b.label == "New shopper").click().run()
    assert not app.exception
    assert any("no recorded history" in info.value for info in app.info)
    app.radio(key="recommendation_view").set_value("Usual picks").run()
    assert not app.exception


def test_detail_dialog_save_and_close_refreshes_the_main_card(app):
    app.button(key="details_recommendations_10").click().run()
    assert not app.exception
    app.button(key="dialog_save_10").click().run()
    assert 10 in app.session_state["lists"][1]
    # AppTest performs a full rerun for the dialog callback. Reopen to exercise
    # its explicit Done action; real fragment/dismiss behavior is browser-tested.
    app.button(key="details_recommendations_10").click().run()
    app.button(key="close_details").click().run()
    assert not app.exception
    assert app.button(key="save_recommendations_10").label == "Saved ✓"


def test_ui_catalog_search_empty_clear_and_pagination(app):
    app.radio(key="nav").set_value("Explore catalog").run()
    assert not app.exception
    next(b for b in app.button if b.label == "Next →").click().run()
    assert app.session_state["catalog_page"] == 1
    app.text_input(key="catalog_query").set_value("doesnotexist").run()
    assert app.session_state["catalog_page"] == 0
    next(b for b in app.button if b.label == "Clear filters").click().run()
    assert app.text_input(key="catalog_query").value == ""
    assert not app.exception


def test_ui_api_failure_has_retry_and_saved_list_stays_available(app, monkeypatch):
    app.button(key="save_recommendations_10").click().run()
    def offline(*args, **kwargs):
        raise demo_client.APIError("Offline; retry shortly.")
    monkeypatch.setattr(demo_client, "fetch", offline)
    app.run()
    assert not app.exception
    assert app.button(key="retry_recommendations")
    app.radio(key="nav").set_value("Saved list").run()
    assert app.button(key="save_saved_10")
    assert not app.exception


def test_ui_report_metric_switch(app):
    app.radio(key="nav").set_value("Model report").run()
    assert not app.exception
    app.selectbox[0].set_value("precision_at_10").run()
    assert not app.exception


@pytest.mark.parametrize("value", ["0", "-1", "abc", "1.5", "9223372036854775808", "١"])
def test_user_ids_reject_invalid_input(value):
    with pytest.raises(ValueError):
        demo_client.parse_user_id(value)


def test_network_errors_are_actionable_and_do_not_expose_connection_details(monkeypatch):
    def fail(*args, **kwargs):
        raise URLError("secret connection details")
    monkeypatch.setattr(demo_client, "urlopen", fail)
    with pytest.raises(demo_client.APIError, match="retry") as error:
        demo_client.fetch("/health")
    assert "secret" not in str(error.value)


def test_untrusted_product_text_is_escaped_in_html_and_csv():
    product = asdict(Product(1, '<script>alert(1)</script>', 1, '=1+1', 1, '<img>'))
    html = card_html(product)
    assert "<script>" not in html and "<img>" not in html
    assert "&lt;script&gt;" in html
    assert "'=1+1" in demo_client.shopping_list_csv([product])
