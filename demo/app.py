"""Run from the repository root: python -m streamlit run demo/app.py."""
from html import escape
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from demo import client
from demo.design import CSS, art, card_html

st.set_page_config(page_title="RecoMart · Your next basket", page_icon="✦", layout="wide")
st.html(CSS)

DEFAULTS = {"shopper": 1, "shopper_input": "1", "nav": "For you", "lists": {},
            "catalog_page": 0, "catalog_query": "", "catalog_department": "All departments"}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def saved_list():
    return st.session_state.lists.setdefault(st.session_state.shopper, {})


def toggle_saved(product):
    saved = saved_list()
    pid = product["product_id"]
    if pid in saved:
        del saved[pid]
    else:
        saved[pid] = product


def set_shopper(user):
    st.session_state.shopper = user
    st.session_state.shopper_input = str(user)
    st.session_state.nav = "For you"
    st.session_state.pop("shopper_error", None)


def submit_shopper():
    try:
        user = client.parse_user_id(st.session_state.shopper_input)
        set_shopper(user)
    except ValueError as exc:
        st.session_state.shopper_error = str(exc)


def navigate(page):
    st.session_state.nav = page


def reset_catalog():
    st.session_state.catalog_page = 0


def empty(title, message):
    st.html(f'<div class="empty"><h2>{escape(title)}</h2><p>{escape(message)}</p></div>')


def section(title, subtitle):
    st.html(f'<h2 class="section-title">{escape(title)}</h2><p class="section-sub">{escape(subtitle)}</p>')


def save_button(product, key):
    saved = product["product_id"] in saved_list()
    st.button("Saved ✓" if saved else "+ Save", key=key, type="primary" if saved else "secondary",
              width="stretch", on_click=toggle_saved, args=(product,),
              help=("Remove " if saved else "Save ") + product["product_name"])


@st.dialog("A closer look", width="large", on_dismiss="rerun")
def product_details(product):
    left, right = st.columns([1, 1.5])
    with left:
        st.html(f'<div class="product-art" style="height:220px">{art(product["department"], product["product_name"])}</div>')
        st.caption("Category illustration")
    with right:
        st.subheader(product["product_name"])
        st.caption(f'{product["department"].title()} / {product["aisle"].title()}')
        st.write(f'Product #{product["product_id"]}')
        st.caption("Save it to your list to keep it in mind for your next basket.")
        save_button(product, f'dialog_save_{product["product_id"]}')
    st.divider()
    st.markdown("**More in this aisle**")
    st.caption("Browse the same category. These are not personalized similarity rankings.")
    try:
        neighbors = client.fetch(f'/catalog/products/{product["product_id"]}/neighbors').data["products"]
        if not neighbors:
            st.info("There are no other products in this aisle.")
        for item in neighbors:
            label, action = st.columns([4, 1])
            label.write(item["product_name"])
            with action:
                save_button(item, f'neighbor_{item["product_id"]}')
    except client.APIError as exc:
        st.warning(str(exc))
    if st.button("Done", key="close_details", width="stretch"):
        st.rerun()


def product_grid(products, prefix, ranked=False):
    # Horizontal containers wrap cards naturally on narrow screens.
    with st.container(horizontal=True, gap="medium", key=f"grid_{prefix}"):
        for i, product in enumerate(products, 1):
            pid = product["product_id"]
            with st.container(width=228, key=f"card_{prefix}_{pid}"):
                st.html(card_html(product, i if ranked else None))
                left, right = st.columns(2, gap="small")
                with left:
                    if st.button("Details", key=f"details_{prefix}_{pid}", width="stretch"):
                        product_details(product)
                with right:
                    save_button(product, f"save_{prefix}_{pid}")


with st.sidebar:
    st.html('<div class="brand"><span class="brand-mark">✦</span>RecoMart<span style="color:#7692FF">.</span></div><div class="brand-sub">A better next basket</div>')
    st.html('<div class="side-label">YOUR SPACE</div>')
    st.radio("Navigation", ["For you", "Explore catalog", "Saved list", "Model report"],
             key="nav", label_visibility="collapsed", width="stretch")
    st.divider()
    st.html('<div class="side-label">SHOPPER PROFILE</div>')
    with st.form("shopper_form"):
        st.text_input("Shopper ID", key="shopper_input", placeholder="e.g. 1", max_chars=19)
        st.form_submit_button("Load recommendations →", width="stretch", on_click=submit_shopper)
        if st.session_state.get("shopper_error"):
            st.error(st.session_state.shopper_error)
    st.caption(f'Viewing shopper #{st.session_state.shopper:,}')
    a, b = st.columns(2)
    a.button("Try #42", on_click=set_shopper, args=(42,), width="stretch")
    b.button("New shopper", on_click=set_shopper, args=(999999,), width="stretch",
             help="See popularity recommendations for a shopper with no recorded history.")
    st.html('<div class="sidebar-note"><strong>Your taste. A head start.</strong>Explore real grocery recommendations, shaped by past shopping patterns.</div>')
    st.caption("Instacart sample profiles · historical data")

nav = st.session_state.nav
user = st.session_state.shopper
try:
    health = client.fetch("/health").data
except client.APIError:
    health = None

status = "Service connected" if health and health["status"] == "ok" else "Using backup service" if health else "Service offline"
st.html(f'<div class="topline"><span class="eyebrow">Your market / {escape(nav)}</span><span class="status"><span class="status-dot"></span>{status}</span></div>')
if health and health["status"] == "degraded":
    st.info("Some services are unavailable. Recommendations are being served from the available snapshot.")


def for_you():
    st.html(f'''<section class="hero"><div class="hero-copy"><div class="eyebrow">A little less searching. A little more you.</div>
        <h1>Your next basket,<br><span>a little smarter.</span></h1>
        <p>Rediscover your usual favorites and find your next pick.<br>A thoughtful starting point for your grocery list.</p></div>
        <div class="hero-art">{art('', hero=True)}</div></section>''')
    try:
        with st.spinner("Finding your next picks…"):
            profile = client.fetch(f"/users/{user}/profile").data
            result = client.fetch(f"/recommendations/{user}", k=10)
    except client.APIError as exc:
        empty("Your picks are taking a pause", str(exc))
        st.button("Try again", key="retry_recommendations")
        return
    data = result.data
    if profile["snapshot_id"] != data["snapshot_id"]:
        st.warning("The recommendation collection is being updated. Please refresh to see a consistent profile.")
        st.button("Refresh profile")
        return
    orders = profile["history_orders"]
    st.html(f'<div class="context-strip"><span><strong>Shopper #{user:,}</strong></span><span><strong>{orders:,}</strong> past orders</span><span><strong>{len(data["recommendations"])}</strong> selected picks</span><span>Saved list · <strong>{len(saved_list())}</strong></span></div>')
    if not profile["known"]:
        st.info("A fresh start. This shopper has no recorded history, so these picks are popular across the market.")
    elif data["strategy"] != "ranker":
        st.info({"ranker_with_popularity_fill": "Your ranked picks are topped up with market favorites.",
                 "repeat_with_popularity_fill": "Showing repeat-purchase favorites, topped up with popular picks.",
                 "global_popularity": "Personalized picks aren't available, so we're showing market favorites."}.get(data["strategy"], "Showing available picks."))
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    with left:
        mode = st.radio("Recommendation view", ["Recommended", "Usual picks"], horizontal=True,
                        label_visibility="collapsed", key="recommendation_view")
    with right:
        count = st.selectbox("Show", [10, 6, 3], format_func=lambda n: f"{n} products", key="show_count")
    if mode == "Recommended":
        section("Picked for your next basket", "In recommendation order. Save the ones you love.")
        products = data["recommendations"][:count]
        if not data["complete"]:
            st.info("This collection has fewer products than requested. Here are all the available picks.")
    else:
        section("Your usual picks", "Favorites from the repeat-purchase baseline, based on historical shopping patterns.")
        products = profile["usual_picks"][:count]
        st.caption("This is a selection of recurring favorites, not your complete order history.")
    if products:
        product_grid(products, "recommendations" if mode == "Recommended" else "usual", ranked=mode == "Recommended")
    else:
        empty("A blank slate", "No picks are available in this view. Explore the catalog to start your list.")
        st.button("Explore the catalog", on_click=navigate, args=("Explore catalog",))
    with st.expander("How are these picks chosen?"):
        st.write("RecoMart combines repeat-purchase patterns, collaborative preferences, and product content. A learned ranker orders the final picks. New shoppers receive popular products.")
        st.caption("These are saved recommendations from historical data. Saving a product does not retrain the model. No individual score or product-level explanation is inferred.")
        st.caption(f'Model: {data["model_version"]} · Source: {result.source or "API"} · Cache: {result.cache or "not reported"}')


def explore():
    section("Find your next favorite", "Search the grocery catalog, explore an aisle, and make a list of your own.")
    try:
        catalog = client.fetch("/catalog").data
        a, b = st.columns([2, 1])
        a.text_input("Search products", placeholder="Try bananas, yogurt, or coffee…", key="catalog_query", on_change=reset_catalog, max_chars=120)
        b.selectbox("Department", ["All departments", *catalog["departments"]], key="catalog_department", on_change=reset_catalog)
        page = st.session_state.catalog_page
        with st.spinner("Searching the market…"):
            result = client.fetch("/catalog/products", q=st.session_state.catalog_query,
                    department="" if st.session_state.catalog_department == "All departments" else st.session_state.catalog_department,
                    offset=page * 12, limit=12).data
        total = result["total"]
        if not total:
            empty("Nothing in this aisle yet", "Try a shorter search or choose another department.")
            if st.button("Clear filters"):
                # Reset via callback on the next rerun to respect widget state.
                st.session_state.clear_catalog = True
                st.rerun()
            return
        st.caption(f'{page * 12 + 1}–{min((page + 1) * 12, total)} of {total:,} products · Alphabetical order')
        product_grid(result["products"], "catalog")
        previous, label, next_page = st.columns([1, 3, 1])
        if previous.button("← Previous", disabled=page == 0, width="stretch"):
            st.session_state.catalog_page -= 1
            st.rerun()
        label.markdown(f'<p style="text-align:center;color:#5c698a;font-size:13px;padding:8px">Page {page + 1} of {math.ceil(total / 12)}</p>', unsafe_allow_html=True)
        if next_page.button("Next →", disabled=(page + 1) * 12 >= total, width="stretch"):
            st.session_state.catalog_page += 1
            st.rerun()
    except client.APIError as exc:
        st.warning(str(exc))
        st.button("Retry catalog")


def saved():
    section("Good picks, kept together", f"Your saved list for shopper #{user:,}. A little planning for your next basket.")
    products = list(saved_list().values())
    if not products:
        empty("Your next basket starts here", "Tap + Save on any product to collect your favorites in one place.")
        st.button("Find your picks →", type="primary", on_click=navigate, args=("For you",))
        return
    a, b = st.columns([3, 1])
    a.caption(f'{len(products)} saved products · This session only · Download to keep your list')
    b.download_button("Download list ↓", data=client.shopping_list_csv(products), file_name=f"recomart-shopper-{user}.csv", mime="text/csv", width="stretch")
    product_grid(products, "saved")


def report():
    section("Good recommendations. Measured.", "A transparent look at how RecoMart performed on the held-out next basket.")
    try:
        benchmark = json.loads((ROOT / "results/benchmarks/final_test_v1.json").read_text(encoding="utf-8"))
        selected = benchmark["policies"]["selected_lambdamart"]["overall"]
        repeat = benchmark["policies"]["repeat_purchase"]["overall"]
        if health and benchmark["prediction_seal_sha256"] != health["snapshot_id"]:
            st.warning("This report describes a different snapshot from the running API.")
        uplift = (selected["recall_at_10"] / repeat["recall_at_10"] - 1) * 100
        st.html(f'<div class="metric-grid"><div class="metric"><span>Recall @ 10</span><strong>{selected["recall_at_10"]:.1%}</strong><small>{uplift:.2f}% relative gain over repeat purchase</small></div><div class="metric"><span>Precision @ 10</span><strong>{selected["precision_at_10"]:.1%}</strong><small>Relevant products among the ten picks</small></div><div class="metric"><span>NDCG @ 10</span><strong>{selected["ndcg_at_10"]:.3f}</strong><small>Rewards relevant products ranked higher</small></div></div>')
        section("The comparison that counts", f'{benchmark["test_users"]:,} shoppers · Same held-out basket · Ten recommendations each')
        metric = st.selectbox("Compare models by", ["recall_at_10", "precision_at_10", "ndcg_at_10"], format_func=lambda v: {"recall_at_10": "Recall @ 10", "precision_at_10": "Precision @ 10", "ndcg_at_10": "NDCG @ 10"}[v])
        names = {"global_popularity": "Global popularity", "als_v2": "Collaborative (ALS)", "content_v1": "Content-based", "repeat_purchase": "Repeat purchase", "repeat_with_popularity_fill": "Repeat + popularity", "selected_lambdamart": "RecoMart · selected"}
        scale = max(p["overall"][metric] for p in benchmark["policies"].values())
        rows = []
        for key, label in names.items():
            value = benchmark["policies"][key]["overall"][metric]
            rows.append(f'<div class="bar-row"><span>{label}</span><div class="bar-track"><div class="bar-fill {"selected" if key == "selected_lambdamart" else ""}" style="width:{value / scale * 100:.2f}%"></div></div><strong>{value:.4f}</strong></div>')
        st.html("".join(rows))
        with st.expander("What do these metrics mean?", expanded=True):
            st.write("**Recall** measures how much of the actual next basket appears in the picks. **Precision** measures how many of the ten picks belong in that basket. **NDCG** also rewards putting relevant products near the top.")
            st.caption("Evaluation uses a temporal split: earlier purchases inform recommendations for a later basket. Predictions were frozen before final-test access. These offline metrics are not live conversion rates.")
        st.download_button("Download benchmark JSON", json.dumps(benchmark, indent=2), file_name="recomart-final-benchmark.json", mime="application/json")
    except (OSError, ValueError, KeyError):
        empty("The report isn't available", "The saved benchmark could not be loaded. Your shopping list is still available.")


if st.session_state.pop("clear_catalog", False):
    st.session_state.catalog_query = ""
    st.session_state.catalog_department = "All departments"
    st.session_state.catalog_page = 0

{"For you": for_you, "Explore catalog": explore, "Saved list": saved, "Model report": report}[nav]()
st.html('<footer class="footer"><span>RecoMart · A better next basket.</span><span>Historical Instacart catalog · Category illustrations · No live prices or checkout</span></footer>')
