"""Local vector category illustrations and escaped, reusable presentation."""
from html import escape
from base64 import b64encode


def art(department, name="", hero=False):
    """Decorative category art, deliberately not a photograph of the product."""
    text = f"{department} {name}".lower()
    if hero:
        drawing = '''<circle cx="125" cy="115" r="98" fill="#dce4ff"/><circle cx="208" cy="40" r="8" fill="#7692FF"/>
        <path d="M62 91h128l-15 108H79z" fill="#1B2CC1"/><path d="M95 96V77a29 29 0 0 1 58 0v19" fill="none" stroke="#091540" stroke-width="9" stroke-linecap="round"/>
        <path d="M85 84c-22-19-25-44-13-55 27 9 38 30 35 58" fill="#7692FF"/><path d="M167 78c-1-31 18-47 34-44 3 26-10 45-34 52" fill="#7692FF"/>
        <path d="M110 138l13 13 28-29" fill="none" stroke="white" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
        <rect x="175" y="155" width="65" height="38" rx="19" fill="white"/><path d="M194 174h26m-13-13v26" stroke="#1B2CC1" stroke-width="3"/>
        <path d="M30 123h18m-9-9v18" stroke="#7692FF" stroke-width="3"/>'''
    elif any(w in text for w in ("milk", "dairy", "beverage", "water", "juice", "soda", "alcohol")):
        drawing = '''<ellipse cx="125" cy="195" rx="49" ry="7" fill="#dce4ff"/><path d="M104 47h42v26l17 27v86H87v-86l17-27z" fill="white" stroke="#091540" stroke-width="3"/><path d="M104 73h42l17 27H87z" fill="#7692FF"/><rect x="87" y="116" width="76" height="46" fill="#1B2CC1"/><path d="M117 132h16m-8-8v24" stroke="white" stroke-width="4"/><rect x="103" y="40" width="44" height="12" rx="3" fill="#091540"/>'''
    elif any(w in text for w in ("produce", "fruit", "vegetable", "banana")):
        drawing = '''<ellipse cx="125" cy="194" rx="54" ry="8" fill="#dce4ff"/><path d="M124 88c-65-40-78 38-40 85 23 27 35 11 42 11s24 16 44-14c34-48 17-117-44-80z" fill="#7692FF" stroke="#091540" stroke-width="3"/><path d="M125 93c-2-19 0-29 12-42" fill="none" stroke="#091540" stroke-width="6" stroke-linecap="round"/><path d="M129 70c3-28 27-35 42-30-3 23-18 34-42 30" fill="#1B2CC1"/><path d="M90 114c-7 8-8 23-2 33" fill="none" stroke="white" stroke-width="6" stroke-linecap="round"/>'''
    elif any(w in text for w in ("bakery", "bread", "snack", "breakfast", "cereal")):
        drawing = '''<ellipse cx="125" cy="195" rx="59" ry="7" fill="#dce4ff"/><path d="M76 111c-13-2-20-13-16-26 12-41 115-41 130 0 4 14-4 24-17 26v72H76z" fill="#7692FF" stroke="#091540" stroke-width="3"/><path d="M90 83c18-10 54-10 73 0" fill="none" stroke="white" stroke-width="5" stroke-linecap="round"/><path d="M96 125v31m29-31v31m29-31v31" stroke="#1B2CC1" stroke-width="7" stroke-linecap="round"/>'''
    elif any(w in text for w in ("frozen", "meat", "deli", "seafood")):
        drawing = '''<ellipse cx="125" cy="192" rx="60" ry="7" fill="#dce4ff"/><rect x="64" y="63" width="122" height="122" rx="24" fill="white" stroke="#091540" stroke-width="3"/><rect x="74" y="73" width="102" height="102" rx="17" fill="#7692FF"/><path d="M125 90v68m-30-51 60 34m-60 0 60-34m-41-10 11 10 11-10m-22 51 11-10 11 10" fill="none" stroke="white" stroke-width="5" stroke-linecap="round"/>'''
    else:
        drawing = '''<ellipse cx="125" cy="195" rx="54" ry="7" fill="#dce4ff"/><path d="M83 57h84l10 129H73z" fill="#7692FF" stroke="#091540" stroke-width="3"/><path d="M83 57h84v20H83z" fill="#1B2CC1"/><rect x="89" y="100" width="72" height="57" rx="10" fill="white"/><path d="M109 128l11 11 22-24" fill="none" stroke="#1B2CC1" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>'''
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 250 230">{drawing}</svg>'
    encoded = b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{encoded}" alt="" aria-hidden="true" />'


def card_html(product, rank=None):
    badge = f'<span class="rank">{rank:02d}</span>' if rank else ''
    return f'''<div class="product-art">{badge}{art(product['department'], product['product_name'])}</div>
    <div class="product-category">{escape(product['department'])}</div>
    <h3 class="product-name" title="{escape(product['product_name'], quote=True)}">{escape(product['product_name'])}</h3>
    <div class="product-aisle">{escape(product['aisle'])}</div>'''


CSS = """
<style>
:root { --navy:#091540; --blue:#1B2CC1; --accent:#7692FF; }
.stApp { font-family: 'Segoe UI', sans-serif; }
.stMainBlockContainer { max-width:1400px; padding:2rem 3rem 3rem; }
header[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { min-width:258px; max-width:258px; }
[data-testid="stSidebar"] > div:first-child { padding-top:1.2rem; }
[data-testid="stSidebar"] hr { border-color:#283562; margin:1.4rem 0; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#bfcaf0; }
[data-testid="stSidebar"] [role="radiogroup"] { gap:.45rem; width:100% !important; }
[data-testid="stSidebar"] [role="radiogroup"] > div { width:100%; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding:.7rem .85rem; border-radius:10px; width:100% !important; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:#1B2CC1; }
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child { display:none; }
[data-testid="stSidebar"] [data-testid="stRadioOption"]:focus-within { outline:2px solid #7692FF; outline-offset:2px; }
[data-testid="stAppDeployButton"] { display:none; }
[data-testid="stSidebar"] [data-testid="stForm"] { border:0; padding:0; }
[data-testid="stSidebar"] .stButton button[kind="primary"], [data-testid="stSidebar"] .stFormSubmitButton button { background:#7692FF; color:#091540; border:0; font-weight:650; }
.brand { display:flex; align-items:center; gap:10px; margin:0 0 8px; font-size:27px; font-weight:750; letter-spacing:-1px; color:white; }
.brand-mark { display:grid; place-items:center; border-radius:11px; background:#7692FF; color:#091540; width:38px; height:38px; font-size:25px; }
.brand-sub { color:#aab9ea; font-size:11px; letter-spacing:2px; text-transform:uppercase; margin-bottom:30px; }
.side-label { color:#aab9ea; font-size:10px; letter-spacing:1.8px; font-weight:650; margin-bottom:12px; }
.sidebar-note { margin-top:25px; padding:16px; background:#13204b; border:1px solid #253461; border-radius:12px; font-size:12px; color:#bdc9f1; line-height:1.7; }
.sidebar-note strong { display:block; color:white; margin-bottom:4px; }
.topline { display:flex; justify-content:space-between; align-items:center; margin-bottom:23px; gap:12px; }
.eyebrow { font-size:11px; font-weight:700; letter-spacing:1.8px; text-transform:uppercase; color:#536188; }
.status { font-size:11px; background:#eef1fb; border:1px solid #dce3f7; color:#233366; padding:7px 11px; border-radius:30px; white-space:nowrap; }
.status-dot { display:inline-block; width:6px; height:6px; border-radius:50%; background:#1B2CC1; margin-right:6px; }
.hero { display:flex; align-items:center; justify-content:space-between; background:#eaf0ff; border:1px solid #dbe3fc; border-radius:20px; padding:32px 36px; margin-bottom:4px; overflow:hidden; }
.hero-copy { max-width:560px; }
.hero .eyebrow { color:#1B2CC1; margin-bottom:14px; }
.hero h1 { font-size:clamp(30px,3.1vw,46px); letter-spacing:-1.7px; line-height:1.12; font-weight:750; padding:0; margin:0 0 17px; color:#091540; }
.hero h1 span { color:#1B2CC1; }
.hero p { font-size:14px; line-height:1.75; color:#4b5b84; max-width:420px; margin:0; }
.hero-art { width:230px; flex:0 0 230px; margin:-15px -10px -15px 15px; }
.hero-art img { display:block; width:100%; }
.context-strip { display:flex; gap:25px; align-items:center; flex-wrap:wrap; padding:16px 3px 21px; border-bottom:1px solid #e0e6f3; font-size:12px; color:#5a6789; margin-bottom:8px; }
.context-strip strong { color:#091540; font-weight:650; }
.section-title { font-size:23px; font-weight:700; letter-spacing:-.5px; margin:7px 0 4px; }
.section-sub { color:#65718f; font-size:13px; margin:0 0 15px; }
[class*="st-key-card_"] { background:white; border:1px solid #e1e6f2; padding:12px; border-radius:15px; transition:border-color .15s,box-shadow .15s; min-width:0; }
[class*="st-key-grid_"] { display:grid !important; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:16px !important; }
[class*="st-key-grid_"] > [data-testid="stLayoutWrapper"] { width:100% !important; min-width:0; }
[class*="st-key-card_"]:hover { border-color:#a6b8fb; box-shadow:0 8px 25px #09154009; }
.product-art { height:148px; background:#f0f3fc; border-radius:10px; position:relative; display:flex; justify-content:center; }
.product-art img { width:155px; height:100%; }
.rank { position:absolute; top:10px; left:10px; color:#1B2CC1; font-size:11px; background:white; border-radius:6px; padding:4px 7px; font-weight:700; }
.product-category { color:#1B2CC1; text-transform:uppercase; letter-spacing:1.15px; font-size:9px; font-weight:700; margin:13px 0 7px; }
h3.product-name { font-size:15px; line-height:1.4; font-weight:650; height:42px; overflow:hidden; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; padding:0; margin:0 0 6px; letter-spacing:-.15px; }
.product-aisle { font-size:11px; color:#6a7795; margin-bottom:2px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
[class*="st-key-card_"] button { min-height:35px; font-size:12px; border-radius:8px; }
[class*="st-key-card_"] button p { font-size:12px; }
.footer { font-size:11px; color:#73809d; display:flex; justify-content:space-between; gap:15px; border-top:1px solid #e0e6f3; padding-top:19px; margin-top:28px; line-height:1.7; }
.empty { text-align:center; padding:60px 24px; border:1px dashed #bbc8e8; border-radius:18px; background:#f0f3fc; }
.empty h2 { font-size:24px; }.empty p { color:#5c698a; font-size:14px; }
.metric-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:20px 0; }
.metric { background:white; border:1px solid #e0e6f3; border-radius:14px; padding:22px; }
.metric span { font-size:12px; color:#5c698a; }.metric strong { display:block; font-size:32px; letter-spacing:-1px; margin:6px 0; }.metric small { color:#1B2CC1; font-size:11px; }
.bar-row { display:grid; grid-template-columns:170px 1fr 65px; gap:16px; align-items:center; font-size:12px; margin:19px 0; }
.bar-track { background:#e8ecf7; border-radius:8px; height:14px; overflow:hidden; }.bar-fill { background:#7692FF; height:100%; border-radius:8px; }.bar-fill.selected { background:#1B2CC1; }
button:focus-visible, a:focus-visible { outline:3px solid #7692FF !important; outline-offset:3px; }
@media (max-width:1100px) { .stMainBlockContainer { padding:1.5rem; }.hero { padding:26px; }.hero-art { width:180px; flex-basis:180px; } }
@media (max-width:700px) { .stMainBlockContainer { padding:1rem; }.hero { padding:25px; }.hero-art { display:none; }.hero h1 { font-size:34px; }.hero p { font-size:13px; }.context-strip { gap:12px; }.metric-grid { grid-template-columns:1fr; }.topline { margin-top:20px; }.bar-row { grid-template-columns:110px 1fr 50px; gap:8px; }.footer { flex-direction:column; } }
@media (prefers-reduced-motion:reduce) { * { transition:none !important; } }
</style>
"""
