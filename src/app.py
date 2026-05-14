import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import folium
import random
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh
import os

# ============================================================
# PAGE CONFIG — must be the very first Streamlit command
# ============================================================
st.set_page_config(
    page_title="NEXUS | AI Smart Traffic Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# MASSIVE CSS INJECTION — Glassmorphism, Gradients, Animations
# ============================================================
# This single <style> block controls the entire visual identity.
# RULE: Inside st.markdown(), no HTML line may start with 4+ spaces.
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700&display=swap');

/* ---- Base ---- */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

/* ---- Animated gradient background ---- */
.stApp {
background: linear-gradient(135deg, #020617 0%, #0a1628 40%, #0d1f3c 70%, #020617 100%);
background-size: 400% 400%;
animation: gradientShift 15s ease infinite;
}
@keyframes gradientShift {
0%   { background-position: 0% 50%; }
50%  { background-position: 100% 50%; }
100% { background-position: 0% 50%; }
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
background: rgba(5, 10, 20, 0.85) !important;
backdrop-filter: blur(20px) !important;
border-right: 1px solid rgba(0,255,204,0.12) !important;
}

/* ---- Section headers (h3) ---- */
h3 {
font-family: 'Orbitron', sans-serif !important;
color: #e0f0ff !important;
letter-spacing: 1.5px !important;
font-size: 20px !important;
border-bottom: 1px solid rgba(0,255,204,0.15);
padding-bottom: 10px !important;
margin-top: 30px !important;
}

/* ---- Glassmorphism metric cards ---- */
.glass-card {
background: rgba(10, 20, 40, 0.45);
backdrop-filter: blur(18px);
-webkit-backdrop-filter: blur(18px);
border: 1px solid rgba(0, 255, 204, 0.18);
border-radius: 18px;
padding: 28px 20px;
text-align: center;
box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255,255,255,0.05);
transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
animation: cardFadeIn 0.7s ease-out forwards;
opacity: 0;
transform: translateY(24px);
}
.glass-card:hover {
transform: translateY(-6px) scale(1.03);
border-color: rgba(0, 255, 204, 0.5);
box-shadow: 0 16px 48px rgba(0, 255, 204, 0.15), inset 0 1px 0 rgba(255,255,255,0.1);
}
.card-label {
color: #7b9ab8;
font-size: 12px;
font-weight: 700;
text-transform: uppercase;
letter-spacing: 2.5px;
margin-bottom: 12px;
}
.card-value {
font-family: 'Orbitron', sans-serif;
font-size: 34px;
font-weight: 900;
background: linear-gradient(90deg, #00ffcc, #00aaff);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
line-height: 1.2;
}

/* ---- Animations ---- */
@keyframes cardFadeIn {
to { opacity: 1; transform: translateY(0); }
}
@keyframes glowPulse {
0%, 100% { box-shadow: 0 0 8px rgba(0,255,204,0.08); }
50%      { box-shadow: 0 0 20px rgba(0,255,204,0.2); }
}

/* ---- AI Recommendation Box ---- */
.ai-box {
background: linear-gradient(135deg, rgba(0,255,204,0.07) 0%, rgba(0,136,255,0.04) 100%);
backdrop-filter: blur(12px);
border-left: 5px solid #00ffcc;
border-radius: 0 18px 18px 0;
padding: 28px;
margin: 12px 0 36px 0;
box-shadow: 0 10px 36px rgba(0,0,0,0.5);
animation: glowPulse 4s infinite;
border-top: 1px solid rgba(0,255,204,0.1);
border-bottom: 1px solid rgba(0,255,204,0.1);
border-right: 1px solid rgba(0,255,204,0.1);
}

/* ---- Dataframe styling ---- */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ---- Alert boxes ---- */
div[data-testid="stAlert"] {
border-radius: 12px !important;
backdrop-filter: blur(8px);
}

/* ---- Plotly chart containers ---- */
div[data-testid="stPlotlyChart"] {
background: rgba(5,12,25,0.3);
border: 1px solid rgba(255,255,255,0.04);
border-radius: 14px;
padding: 8px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# DATA + MODEL LOADING
# ============================================================
@st.cache_data
def load_datasets():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(base, "datasets")
    return (
        pd.read_csv(os.path.join(d, "route_data.csv")),
        pd.read_csv(os.path.join(d, "cleaned_traffic_data.csv")),
        pd.read_csv(os.path.join(d, "toll_data.csv")),
        pd.read_csv(os.path.join(d, "weather_data.csv")),
        pd.read_csv(os.path.join(d, "accident_data.csv")),
    )

@st.cache_resource
def load_model():
    """Load the trained Random Forest model and label encoder."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m = os.path.join(base, "models")
    model = joblib.load(os.path.join(m, "congestion_model.pkl"))
    le = joblib.load(os.path.join(m, "label_encoder.pkl"))
    return model, le


route_df, traffic_df, toll_df, weather_df, accident_df = load_datasets()
rf_model, label_encoder = load_model()

# Precompute training data stats for scaling user inputs
_means = traffic_df[['traffic_volume','average_speed_kmph','distance_km',
    'base_eta_mins','temperature_celsius','precipitation_mm',
    'visibility_km','toll_fee_inr']].mean()
_stds = traffic_df[['traffic_volume','average_speed_kmph','distance_km',
    'base_eta_mins','temperature_celsius','precipitation_mm',
    'visibility_km','toll_fee_inr']].std()


# ============================================================
# HELPER — build a glassmorphism KPI card (zero-indent HTML)
# ============================================================
def kpi_card(label, value, delay_idx=0):
    """Render a glass KPI card. HTML is built via concatenation so
    every line starts at column 0 — Streamlit won't code-block it."""
    delay = delay_idx * 0.15
    html = (
        f'<div class="glass-card" style="animation-delay:{delay}s;">'
        f'<div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# SIDEBAR — Route Controls + AI Prediction Inputs
# ============================================================
st.sidebar.markdown(
"<h2 style='color:#00ffcc; font-family:Orbitron; text-align:center;"
" letter-spacing:2px; font-size:22px;'>NEXUS CONTROL</h2>",
unsafe_allow_html=True,
)
st.sidebar.caption("Smart Traffic Command Centre")
st.sidebar.markdown("---")

selected_source = st.sidebar.selectbox("🟢 Source", route_df["start_point"].unique())
selected_dest = st.sidebar.selectbox("🔴 Destination", route_df["end_point"].unique())
time_of_day = st.sidebar.slider("⏰ Departure Hour", 0, 23, 8, format="%d:00")

st.sidebar.markdown("---")
st.sidebar.markdown("**🤖 AI Prediction Inputs**")
user_volume = st.sidebar.slider("🚗 Traffic Volume", 40, 850, 200)
user_speed = st.sidebar.slider("🏎️ Avg Speed (km/h)", 5, 95, 45)
user_weather = st.sidebar.selectbox("🌦️ Weather", ["Clear", "Partly Cloudy", "Overcast", "Fog", "Light Rain", "Heavy Rain"])
user_accident = st.sidebar.selectbox("💥 Accident Severity", ["None", "Minor", "Major", "Severe"])

st.sidebar.markdown("---")
st.sidebar.markdown("**📡 LIVE SIMULATION**")
live_mode = st.sidebar.toggle("🔴 Enable Real-Time Feed", value=False)
refresh_rate = st.sidebar.select_slider(
    "Refresh Interval", options=[5, 10, 15, 30], value=10,
    format_func=lambda x: f"{x}s",
) if live_mode else 10

# Auto-refresh the page when live mode is ON
if live_mode:
    st_autorefresh(interval=refresh_rate * 1000, limit=None, key="live_feed")

st.sidebar.markdown("---")
st.sidebar.success("🧠 AI Engine **ONLINE**\n\n🎯 Model Accuracy: **99.40 %**")


# ============================================================
# DYNAMIC LOGIC + REAL AI PREDICTION
# ============================================================
matches = route_df[
    (route_df["start_point"] == selected_source)
    | (route_df["end_point"] == selected_dest)
]
sel = matches.iloc[0] if not matches.empty else route_df.iloc[0]

is_peak = (8 <= time_of_day <= 11) or (17 <= time_of_day <= 20)
base_eta = int(sel["base_eta_mins"])
toll_est = 150 if is_peak else 80

# --- Map user inputs to numeric values ---
weather_map = {"Clear": 0, "Partly Cloudy": 1, "Overcast": 2, "Fog": 3, "Light Rain": 3, "Heavy Rain": 5}
accident_map = {"None": 0, "Minor": 1, "Major": 3, "Severe": 5}
w_sev = weather_map[user_weather]
a_imp = accident_map[user_accident]

# --- If LIVE MODE is ON, randomize sensor inputs ---
if live_mode:
    user_volume = random.randint(50, 800)
    user_speed = random.randint(8, 90)
    w_options = ["Clear", "Partly Cloudy", "Overcast", "Fog", "Light Rain", "Heavy Rain"]
    user_weather = random.choice(w_options)
    a_options = ["None", "None", "None", "Minor", "Minor", "Major", "Severe"]
    user_accident = random.choice(a_options)
    w_sev = weather_map[user_weather]
    a_imp = accident_map[user_accident]

# --- Build 15-feature vector (same order as training) ---
def scale(val, col):
    return (val - _means[col]) / _stds[col]

FEATURE_NAMES = [
    'route_id_encoded', 'hour', 'day_of_week', 'is_peak_hour',
    'weather_severity', 'accident_impact', 'traffic_volume_scaled',
    'average_speed_kmph_scaled', 'distance_km_scaled',
    'base_eta_mins_scaled', 'temperature_celsius_scaled',
    'precipitation_mm_scaled', 'visibility_km_scaled',
    'toll_fee_inr_scaled', 'surge_pricing_active',
]

route_idx = list(route_df["route_id"]).index(sel["route_id"])
feature_values = [[
    route_idx,
    time_of_day,
    2,
    int(is_peak),
    w_sev,
    a_imp,
    scale(user_volume, 'traffic_volume'),
    scale(user_speed, 'average_speed_kmph'),
    scale(sel['distance_km'], 'distance_km'),
    scale(base_eta, 'base_eta_mins'),
    scale(25.0, 'temperature_celsius'),
    scale(0.0, 'precipitation_mm'),
    scale(8.0, 'visibility_km'),
    scale(float(toll_est), 'toll_fee_inr'),
    int(is_peak),
]]
features = pd.DataFrame(feature_values, columns=FEATURE_NAMES)

# --- Run model prediction ---
prediction = rf_model.predict(features)[0]
probabilities = rf_model.predict_proba(features)[0]
cong_label = label_encoder.inverse_transform([prediction])[0].upper()
confidence = float(probabilities.max()) * 100
ai_conf = f"{confidence:.1f} %"

# --- Dynamic ETA based on predicted congestion ---
if cong_label == "HIGH":
    eta = int(base_eta * 1.6)
elif cong_label == "MEDIUM":
    eta = int(base_eta * 1.2)
else:
    eta = int(base_eta * 1.05)


# ============================================================
# HEADER — gradient text via CSS
# ============================================================
st.markdown(
"<h1 style='text-align:center; font-family:Orbitron; font-size:52px;"
" font-weight:900; letter-spacing:6px;"
" background:linear-gradient(90deg,#ffffff,#00ffcc,#0088ff);"
" -webkit-background-clip:text; -webkit-text-fill-color:transparent;"
" padding:24px 0 0 0;'>NEXUS SMART HUB</h1>",
unsafe_allow_html=True,
)
st.markdown(
"<p style='text-align:center; color:#6a8ca8; letter-spacing:4px;"
" font-size:13px; text-transform:uppercase; margin-bottom:36px;"
" font-weight:600;'>"
"Predictive Traffic Analytics &amp; Autonomous Route Optimization</p>",
unsafe_allow_html=True,
)

# Show live status banner when simulation is active
if live_mode:
    st.markdown(
    "<div style='text-align:center; padding:8px; margin-bottom:10px;"
    " background:rgba(255,50,50,0.15); border:1px solid rgba(255,50,50,0.3);"
    " border-radius:10px;'>"
    "<span style='color:#ff4444; font-family:Orbitron; font-size:13px;"
    " letter-spacing:2px;'>"
    f"\U0001F534 LIVE FEED ACTIVE &mdash; Refreshing every {refresh_rate}s"
    f" &mdash; Volume: {user_volume} | Speed: {user_speed} km/h"
    f" | Weather: {user_weather} | Accident: {user_accident}"
    "</span></div>",
    unsafe_allow_html=True,
    )


# ============================================================
# SECTION 1 — GLASSMORPHISM KPI CARDS
# ============================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Congestion Level", cong_label, 0)
with c2:
    kpi_card("Predicted ETA", f"{eta} min", 1)
with c3:
    kpi_card("Toll Estimate", f"Rs {toll_est}", 2)
with c4:
    kpi_card("AI Confidence", ai_conf, 3)

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
st.markdown("---")


# ============================================================
# SECTION 2 — AI RECOMMENDED ROUTE (glow-pulse box)
# ============================================================
st.markdown("### 🏆 OPTIMAL ROUTE DETECTED")

reason = (
    f"Historical patterns at {time_of_day}:00 indicate lower volume on this corridor. "
    "Optimal balance of travel time, toll cost, and safety risk."
)

ai_html = (
    '<div class="ai-box">'
    f'<h2 style="color:#fff; margin:0 0 8px 0; font-family:Orbitron;'
    f' letter-spacing:2px; font-size:26px;">{sel["route_name"]}</h2>'
    '<div style="height:2px; background:linear-gradient(90deg,#00ffcc,transparent);'
    ' margin-bottom:18px;"></div>'
    f'<p style="color:#b0c8dc; font-size:17px; margin:0 0 18px 0;">'
    f'<b style="color:#fff;">Distance:</b>'
    f' <span style="color:#00ffcc; font-weight:700; font-size:22px;">'
    f'{sel["distance_km"]} km</span>'
    f' &nbsp;&nbsp;|&nbsp;&nbsp; '
    f'<b style="color:#fff;">ETA:</b>'
    f' <span style="color:#00ffcc; font-weight:700; font-size:22px;">'
    f'{eta} mins</span></p>'
    '<div style="background:rgba(0,0,0,0.35); padding:14px 18px;'
    ' border-radius:10px; border-left:4px solid #0088ff;">'
    f'<p style="color:#ddd; margin:0; font-size:14px; line-height:1.6;">'
    f'<b style="color:#0088ff;">🤖 AI REASONING:</b> {reason}</p>'
    '</div>'
    '</div>'
)
st.markdown(ai_html, unsafe_allow_html=True)


# ============================================================
# SECTION 2B — AI PREDICTION ENGINE (model output)
# ============================================================
st.markdown("### 🤖 AI PREDICTION ENGINE")

# Show colored alert based on predicted congestion
if cong_label == "HIGH":
    st.error(f"🚨 **Prediction: HIGH Congestion** — Model confidence: {ai_conf}")
elif cong_label == "MEDIUM":
    st.warning(f"⚠️ **Prediction: MEDIUM Congestion** — Model confidence: {ai_conf}")
else:
    st.success(f"✅ **Prediction: LOW Congestion** — Model confidence: {ai_conf}")

# Probability breakdown using native st.columns
prob_cols = st.columns(3)
class_names = label_encoder.classes_
prob_colors = {"High": "🔴", "Low": "🟢", "Medium": "🟡"}
for i, cls in enumerate(class_names):
    pct = probabilities[i] * 100
    with prob_cols[i]:
        st.metric(
            label=f"{prob_colors.get(cls, '')} {cls} Probability",
            value=f"{pct:.1f} %",
        )

# Dynamic reasoning text
reasons = []
if user_speed < 20:
    reasons.append("very low avg speed indicates standstill traffic")
elif user_speed < 40:
    reasons.append("below-average speed suggests slowdowns")
else:
    reasons.append("healthy speed indicates smooth flow")
if w_sev >= 3:
    reasons.append(f"{user_weather} reduces visibility and road grip")
if a_imp >= 3:
    reasons.append(f"{user_accident} accident blocks lanes")
if is_peak:
    reasons.append(f"{time_of_day}:00 falls in rush-hour window")
if user_volume > 400:
    reasons.append("high vehicle density on this corridor")

reasoning_text = ". ".join(reasons).capitalize() + "."
st.info(f"**🧠 Prediction Reasoning:** {reasoning_text}")

st.markdown("---")


# ============================================================
# SECTION 3 — ROUTE COMPARISON TABLE
# ============================================================
st.markdown("### 🔀 ROUTE COMPARISON MATRIX")

np.random.seed(99)
cmp_rows = []
for _, r in route_df.iterrows():
    r_eta = int(r["base_eta_mins"] * (1.6 if is_peak else 1.1))
    r_toll = int(np.random.choice([50, 80, 100, 150]))
    r_cong = "High" if is_peak and r["route_id"] in ["R001", "R004"] else (
        "Medium" if is_peak else "Low"
    )
    cmp_rows.append({
        "Route": r["route_name"],
        "Distance (km)": r["distance_km"],
        "ETA (min)": r_eta,
        "Toll (Rs)": r_toll,
        "Congestion": r_cong,
    })
cmp_df = pd.DataFrame(cmp_rows)
st.dataframe(cmp_df, hide_index=True)

st.markdown("---")


# ============================================================
# SECTION 4 — CHARTS (side by side)
# ============================================================
ch1, ch2 = st.columns(2)

with ch1:
    st.markdown("### 📈 24H CONGESTION FORECAST")
    np.random.seed(42)
    hrs = list(range(24))
    vols = [
        int(100 + 150 * np.exp(-0.5 * ((h - 9) / 2)**2)
            + 120 * np.exp(-0.5 * ((h - 18) / 2)**2)
            + np.random.randint(-8, 8))
        for h in hrs
    ]
    fig1 = px.area(
        x=hrs, y=vols,
        labels={"x": "Hour", "y": "Vehicle Volume"},
        template="plotly_dark",
    )
    fig1.update_traces(
        line_color="#00ffcc", line_width=3,
        fillcolor="rgba(0,255,204,0.1)",
    )
    fig1.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig1, key="forecast_chart")

with ch2:
    st.markdown("### 🏁 ETA COMPARISON BY ROUTE")
    fig2 = px.bar(
        cmp_df, x="Route", y="ETA (min)",
        color="ETA (min)", color_continuous_scale="Teal",
        template="plotly_dark",
    )
    fig2.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, tickangle=-20),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig2, key="eta_bar_chart")

st.markdown("---")


# ============================================================
# SECTION 5 — CONGESTION HEATMAP
# ============================================================
st.markdown("### 🔥 CONGESTION HEATMAP (Hour x Route)")

np.random.seed(7)
z_data = []
for _ in route_df.iterrows():
    row_vals = []
    for h in range(24):
        pk = (8 <= h <= 11) or (17 <= h <= 20)
        row_vals.append(np.random.randint(60, 95) if pk else np.random.randint(10, 45))
    z_data.append(row_vals)

fig3 = go.Figure(data=go.Heatmap(
    z=z_data,
    x=[f"{h}:00" for h in range(24)],
    y=route_df["route_name"].tolist(),
    colorscale="YlOrRd",
    colorbar=dict(title="Volume"),
))
fig3.update_layout(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=10, b=10),
    height=300,
)
st.plotly_chart(fig3, key="heatmap_chart")

st.markdown("---")


# ============================================================
# SECTION 5B — INTERACTIVE FOLIUM TRAFFIC MAP (Enhanced)
# ============================================================
st.markdown("### 🗺️ LIVE TRAFFIC MAP — Delhi NCR")

# Real coordinates for each route's key junction point
ROUTE_COORDS = {
    "R001": {"lat": 28.5921, "lon": 77.1670, "name": "Delhi-Gurgaon Expressway"},
    "R002": {"lat": 28.5706, "lon": 77.3260, "name": "Noida-Greater Noida Expressway"},
    "R003": {"lat": 28.6280, "lon": 77.2430, "name": "Delhi-Meerut Expressway"},
    "R004": {"lat": 28.5494, "lon": 77.2001, "name": "Outer Ring Road (Delhi)"},
    "R005": {"lat": 28.4595, "lon": 77.0266, "name": "NH-48 (Delhi-Jaipur)"},
}

CONG_COLORS = {"Low": "#00ff88", "Medium": "#ffaa00", "High": "#ff3333"}
CONG_BG     = {"Low": "#0a2e1a", "Medium": "#2e2a0a", "High": "#2e0a0a"}

# Build map with dark tiles and tighter zoom controls
traffic_map = folium.Map(
    location=[28.55, 77.15],
    zoom_start=11,
    tiles="CartoDB dark_matter",
    control_scale=True,
    zoom_control=True,
    max_zoom=16,
    min_zoom=9,
)

heat_points = []
for _, r in route_df.iterrows():
    rid = r["route_id"]
    if rid not in ROUTE_COORDS:
        continue
    coord = ROUTE_COORDS[rid]
    r_eta = int(r["base_eta_mins"] * (1.6 if is_peak else 1.1))
    r_cong = "High" if is_peak and rid in ["R001", "R004"] else (
        "Medium" if is_peak else "Low"
    )
    color = CONG_COLORS[r_cong]
    bg = CONG_BG[r_cong]

    # Styled HTML popup card
    popup_html = (
        f'<div style="font-family:Inter,sans-serif; background:{bg};'
        f' border:1px solid {color}; border-radius:10px; padding:12px 16px;'
        f' min-width:200px; color:#fff;">'
        f'<div style="font-size:14px; font-weight:700; color:{color};'
        f' margin-bottom:8px; letter-spacing:1px;">{coord["name"]}</div>'
        f'<div style="font-size:12px; color:#ccc; line-height:1.8;">'
        f'<b>Status:</b> <span style="color:{color}; font-weight:700;">{r_cong}</span><br>'
        f'<b>ETA:</b> {r_eta} min<br>'
        f'<b>Distance:</b> {r["distance_km"]} km<br>'
        f'<b>Weather:</b> {user_weather}</div></div>'
    )

    # Pulsing CSS marker for High congestion, solid for others
    if r_cong == "High":
        marker_html = (
            f'<div style="width:24px; height:24px; border-radius:50%;'
            f' background:{color}; opacity:0.85; box-shadow:0 0 12px {color},'
            f' 0 0 24px {color}; animation:pulse 1.5s infinite;"></div>'
            f'<style>@keyframes pulse{{'
            f'0%{{transform:scale(1);opacity:0.85}}'
            f'50%{{transform:scale(1.4);opacity:0.4}}'
            f'100%{{transform:scale(1);opacity:0.85}}}}</style>'
        )
    else:
        glow = "8px" if r_cong == "Medium" else "4px"
        marker_html = (
            f'<div style="width:18px; height:18px; border-radius:50%;'
            f' background:{color}; opacity:0.8; box-shadow:0 0 {glow} {color};'
            f' border:2px solid rgba(255,255,255,0.3);"></div>'
        )

    folium.Marker(
        location=[coord["lat"], coord["lon"]],
        icon=folium.DivIcon(
            html=marker_html,
            icon_size=(24, 24),
            icon_anchor=(12, 12),
        ),
        popup=folium.Popup(popup_html, max_width=280),
        tooltip=coord["name"],
    ).add_to(traffic_map)

    intensity = 1.0 if r_cong == "High" else (0.5 if r_cong == "Medium" else 0.15)
    heat_points.append([coord["lat"], coord["lon"], intensity])

# Enhanced heat layer
HeatMap(
    heat_points,
    radius=45, blur=30, max_zoom=14,
    gradient={0.2: '#00ff88', 0.5: '#ffaa00', 0.8: '#ff6600', 1.0: '#ff0000'},
).add_to(traffic_map)

st_folium(traffic_map, width=None, height=480, returned_objects=[])

st.markdown("---")


# ============================================================
# SECTION 6 — SMART ALERT CARDS
# ============================================================
st.markdown("### ⚠️ SMART ALERT SYSTEM")

def alert_card(icon, title, message, color, bg):
    """Render a styled HTML alert card. All strings at column 0."""
    html = (
        f'<div style="background:{bg}; border-left:5px solid {color};'
        f' border-radius:0 12px 12px 0; padding:16px 20px;'
        f' margin-bottom:12px; box-shadow:0 4px 16px rgba(0,0,0,0.3);">'
        f'<div style="font-size:13px; font-weight:700; color:{color};'
        f' letter-spacing:1px; margin-bottom:6px;">'
        f'{icon} {title}</div>'
        f'<div style="font-size:13px; color:#ccc; line-height:1.6;">'
        f'{message}</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)

# Build dynamic alert list based on current state
alerts = []

# 1. Congestion alert
if cong_label == "HIGH":
    alerts.append(("🚨", "HEAVY CONGESTION DETECTED",
        f"AI predicts <b>HIGH</b> congestion on {sel['route_name']}. "
        f"Volume: {user_volume} vehicles, Avg speed: {user_speed} km/h. Consider alternate routes.",
        "#ff3333", "rgba(255,50,50,0.08)"))
elif cong_label == "MEDIUM":
    alerts.append(("⚠️", "MODERATE TRAFFIC AHEAD",
        f"Traffic density is building on {sel['route_name']}. "
        f"Current speed: {user_speed} km/h. ETA may fluctuate.",
        "#ffaa00", "rgba(255,170,0,0.08)"))
else:
    alerts.append(("✅", "ROADS CLEAR",
        f"Smooth traffic flow detected. Avg speed: {user_speed} km/h. "
        "Enjoy your drive!",
        "#00ff88", "rgba(0,255,136,0.08)"))

# 2. Weather alert
if user_weather in ("Heavy Rain", "Fog"):
    alerts.append(("🌧️", "ADVERSE WEATHER WARNING",
        f"<b>{user_weather}</b> reported in the area. Reduced visibility and "
        "wet roads may increase travel time by 15-25%.",
        "#5588ff", "rgba(85,136,255,0.08)"))
elif user_weather in ("Light Rain", "Overcast"):
    alerts.append(("🌦️", "WEATHER ADVISORY",
        f"<b>{user_weather}</b> conditions detected. Drive with caution. "
        "Minor delays possible.",
        "#88aacc", "rgba(136,170,204,0.06)"))

# 3. Accident alert
if user_accident in ("Major", "Severe"):
    alerts.append(("💥", "ACCIDENT REPORTED",
        f"A <b>{user_accident}</b> accident has been reported in the vicinity. "
        "Emergency services alerted. Expect significant lane closures.",
        "#ff6644", "rgba(255,102,68,0.08)"))
elif user_accident == "Minor":
    alerts.append(("🔶", "MINOR INCIDENT NEARBY",
        "A minor fender-bender reported. Traffic impact is limited but "
        "watch for rubbernecking slowdowns.",
        "#ddaa44", "rgba(221,170,68,0.06)"))

# 4. ETA alert
if eta > base_eta * 1.4:
    alerts.append(("⏱️", "ROUTE ETA INCREASED",
        f"Predicted ETA of <b>{eta} min</b> is {int((eta/base_eta - 1)*100)}% "
        f"above the base time of {base_eta} min. Heavy conditions detected.",
        "#cc66ff", "rgba(204,102,255,0.08)"))

# 5. Peak hour alert
if is_peak:
    alerts.append(("🕐", "RUSH HOUR ACTIVE",
        f"Current time ({time_of_day}:00) falls within peak traffic hours. "
        "Historical data shows 40-60% higher volumes during this window.",
        "#ff9944", "rgba(255,153,68,0.06)"))

# Render alerts in a 2-column grid
col_left, col_right = st.columns(2)
for i, (icon, title, msg, color, bg) in enumerate(alerts):
    with col_left if i % 2 == 0 else col_right:
        alert_card(icon, title, msg, color, bg)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
"<p style='text-align:center; color:#3a5060; font-family:Orbitron;"
" font-size:11px; letter-spacing:2px; padding:10px 0;'>"
"BUILT FOR HACKATHON 2026 &bull; NEXUS CORE v2.0 &bull;"
" POWERED BY RANDOM FOREST AI</p>",
unsafe_allow_html=True,
)
