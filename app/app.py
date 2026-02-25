"""
app.py
------
Streamlit web application for the NHL Attendance Predictor.
Loads the trained model and provides an interactive prediction UI.

Run:
    streamlit run app/app.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_engineering import FEATURE_COLS, ORIGINAL_SIX
from src.data_collection import ARENA_CAPACITY

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NHL Attendance Predictor",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #003f8a;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #003f8a 0%, #0066cc 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,63,138,0.3);
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.85;
        margin-top: 0.3rem;
    }
    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #003f8a;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.92rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #003f8a, #0066cc);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #002a5e, #004fa3);
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))   # .../app
REPO_ROOT = os.path.dirname(APP_DIR)                   # repo root
MODEL_DIR = os.path.join(REPO_ROOT, "models")

@st.cache_resource
def load_model():
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    enc_path   = os.path.join(MODEL_DIR, "encoders.joblib")
    feat_path  = os.path.join(MODEL_DIR, "feature_cols.joblib")
    meta_path  = os.path.join(MODEL_DIR, "model_meta.json")

    model = joblib.load(model_path)
    encoders = joblib.load(enc_path)
    feature_cols = joblib.load(feat_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, encoders, feature_cols, meta

# IMPORTANT: always define these variables
try:
    model, encoders, feature_cols, model_meta = load_model()
except Exception as e:
    model, encoders, feature_cols, model_meta = None, None, None, None
    st.error(f"Model load failed: {e}")
    st.stop()
# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🏒 NHL Attendance Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML-powered game attendance forecasting for arena operations and revenue planning</p>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model not found. Please run `python src/train.py` first to train the model.")
    st.code("cd nhl-attendance-predictor\npython src/train.py", language="bash")
    st.stop()

# ── Sidebar — Model Info ────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/3/3a/05_NHL_Shield.svg", width=80)
    st.markdown("### Model Performance")

    metrics = model_meta.get("metrics", {})
    st.metric("Best Model", model_meta.get("best_model", "N/A"))
    st.metric("RMSE", f"{metrics.get('rmse', 0):,.0f} fans")
    st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
    st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This model predicts NHL game attendance using:
    - **Elo ratings** — dynamic team quality
    - **Rolling win %** — recent form
    - **Temporal features** — day, month, holidays
    - **Matchup factors** — rivalries, back-to-backs
    - **Arena capacity**

    Trained on {n_train:,} regular-season games.
    """.format(n_train=model_meta.get("n_train", 0)))

    st.markdown("---")
    st.caption("Built by Khalid Wasim Mushir · GitHub")

# ── Main Input Panel ────────────────────────────────────────────────────────────
teams = sorted(ARENA_CAPACITY.keys())

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 🏟️ Game Details")

    home_team = st.selectbox("Home Team", teams, index=teams.index("TOR"))
    away_team = st.selectbox("Away Team", [t for t in teams if t != home_team],
                              index=0)

    game_date = st.date_input("Game Date", value=pd.Timestamp("2025-01-15"))

    col1a, col1b = st.columns(2)
    with col1a:
        home_back_to_back = st.checkbox("Home B2B", help="Home team played yesterday")
    with col1b:
        away_back_to_back = st.checkbox("Away B2B", help="Away team played yesterday")

with col2:
    st.markdown("### 📊 Team Form")

    home_win_pct = st.slider(
        f"{home_team} Win %", 0.20, 0.80, 0.50, 0.01,
        format="%.2f"
    )
    away_win_pct = st.slider(
        f"{away_team} Win %", 0.20, 0.80, 0.50, 0.01,
        format="%.2f"
    )
    home_elo = st.slider(f"{home_team} Elo Rating", 1300, 1700, 1500, 10)
    away_elo = st.slider(f"{away_team} Elo Rating", 1300, 1700, 1500, 10)

# ── Prediction ─────────────────────────────────────────────────────────────────
st.markdown("---")
predict_col, _ = st.columns([1, 2])

with predict_col:
    predict_btn = st.button("🔮 Predict Attendance")

if predict_btn:
    game_date_dt = pd.Timestamp(game_date)
    dow = game_date_dt.dayofweek
    month = game_date_dt.month
    is_weekend = int(dow >= 4)
    is_saturday = int(dow == 5)
    near_holiday = int(game_date_dt.strftime("%m-%d") in [
        "12-24","12-25","12-26","12-31","01-01","02-14","03-17","11-11"
    ])

    is_rivalry = int(
        home_team in ORIGINAL_SIX and away_team in ORIGINAL_SIX
    )
    from src.feature_engineering import HISTORIC_RIVALRIES
    is_historic = int(frozenset({home_team, away_team}) in HISTORIC_RIVALRIES)

    arena_cap = ARENA_CAPACITY.get(home_team, 18000)
    combined_win_pct = (home_win_pct + away_win_pct) / 2
    win_pct_diff = home_win_pct - away_win_pct
    elo_diff = home_elo - away_elo
    elo_sum = home_elo + away_elo

    # Encode teams
    def safe_encode(encoder, value):
        try:
            return encoder.transform([value])[0]
        except ValueError:
            return 0

    home_enc = safe_encode(encoders["home_team"], home_team)
    away_enc = safe_encode(encoders["away_team"], away_team)

    features = {
        "day_of_week": dow,
        "month": month,
        "is_weekend": is_weekend,
        "is_saturday": is_saturday,
        "near_holiday": near_holiday,
        "is_early_season": int(month in [10, 11]),
        "is_late_season": int(month in [2, 3]),
        "is_playoff_push": int(month == 4),
        "home_win_pct": home_win_pct,
        "away_win_pct": away_win_pct,
        "combined_win_pct": combined_win_pct,
        "win_pct_diff": win_pct_diff,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": elo_diff,
        "elo_sum": elo_sum,
        "is_original_six": is_rivalry,
        "is_historic_rivalry": is_historic,
        "home_back_to_back": int(home_back_to_back),
        "away_back_to_back": int(away_back_to_back),
        "arena_capacity": arena_cap,
        "home_team_enc": home_enc,
        "away_team_enc": away_enc,
    }

    X_input = pd.DataFrame([features])[feature_cols]
    prediction = int(model.predict(X_input)[0])
    prediction = max(1000, min(prediction, int(arena_cap * 1.02)))
    fill_pct = prediction / arena_cap * 100

    # ── Results ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Prediction Results")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{prediction:,}</p>
            <p class="metric-label">Predicted Attendance</p>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{fill_pct:.1f}%</p>
            <p class="metric-label">Arena Fill Rate</p>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{arena_cap:,}</p>
            <p class="metric-label">Arena Capacity ({home_team})</p>
        </div>""", unsafe_allow_html=True)

    # Fill gauge
    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.barh([""], [100], color="#e8eaf0", height=0.5, edgecolor="none")
    color = "#003f8a" if fill_pct >= 85 else "#f4a100" if fill_pct >= 70 else "#cc2200"
    ax.barh([""], [fill_pct], color=color, height=0.5, edgecolor="none")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Arena Fill %")
    ax.axvline(85, color="gray", linestyle="--", lw=1, alpha=0.5)
    ax.text(85.5, 0, "85%\ntarget", va="center", fontsize=8, color="gray")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Insights
    st.markdown("### 💡 Key Factors")
    insights = []
    if is_weekend:
        insights.append("✅ Weekend game — historically **+5–8%** attendance boost")
    else:
        insights.append("⚠️ Weekday game — typically lower attendance than weekends")
    if is_rivalry or is_historic:
        insights.append(f"🔥 **Rivalry matchup** ({home_team} vs {away_team}) — strong fan draw")
    if home_back_to_back:
        insights.append(f"😓 {home_team} on a back-to-back — fatigue may affect performance buzz")
    if away_back_to_back:
        insights.append(f"😓 {away_team} on a back-to-back — away team fatigue")
    if home_win_pct >= 0.60:
        insights.append(f"⭐ {home_team} is a strong home team ({home_win_pct:.0%} win rate) — fan confidence high")
    elif home_win_pct <= 0.40:
        insights.append(f"📉 {home_team} struggling ({home_win_pct:.0%} win rate) — may suppress attendance")
    if near_holiday:
        insights.append("🎄 Near a major holiday — attendance patterns may vary")
    if month == 4:
        insights.append("🏆 Playoff push month — elevated fan urgency and ticket demand")

    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    # Scenario comparison
    st.markdown("### 🔄 Scenario Comparison")
    scenarios = {
        "Current Input": prediction,
        "If Weekend (Sat)": None,
        "If Rivalry Game": None,
        "If Both B2B": None,
        "If Top Teams (65% win)": None,
    }

    def predict_scenario(override: dict) -> int:
        f = features.copy()
        f.update(override)
        X = pd.DataFrame([f])[feature_cols]
        p = int(model.predict(X)[0])
        return max(1000, min(p, int(arena_cap * 1.02)))

    scenarios["If Weekend (Sat)"] = predict_scenario({"is_weekend": 1, "is_saturday": 1, "day_of_week": 5})
    scenarios["If Rivalry Game"] = predict_scenario({"is_original_six": 1, "is_historic_rivalry": 1})
    scenarios["If Both B2B"] = predict_scenario({"home_back_to_back": 1, "away_back_to_back": 1})
    scenarios["If Top Teams (65% win)"] = predict_scenario({
        "home_win_pct": 0.65, "away_win_pct": 0.65,
        "combined_win_pct": 0.65, "home_elo": 1600, "away_elo": 1600,
        "elo_sum": 3200
    })

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    names = list(scenarios.keys())
    values = list(scenarios.values())
    bar_colors = ["#003f8a"] + ["#6699cc"] * (len(names) - 1)
    bars = ax2.bar(names, values, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax2.axhline(arena_cap, color="red", linestyle="--", lw=1.2, label=f"Capacity ({arena_cap:,})")
    ax2.set_ylabel("Predicted Attendance")
    ax2.set_title("What-If Scenario Analysis", fontweight="bold")
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax2.legend(fontsize=9)
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                 f"{v:,}", ha="center", fontsize=9, fontweight="bold")
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()
