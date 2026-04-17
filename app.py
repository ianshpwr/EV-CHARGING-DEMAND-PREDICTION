"""
app.py — EV Charging Intelligence Hub
======================================
Clean, thin Streamlit UI.

Flow:
    1. User selects station + date
    2. Clicks "Run Prediction"  →  ML model runs, per-station result shown
    3. Clicks "Generate AI Recommendation"  →  Groq LLM produces structured output
    4. Dashboard expander shows system-wide charts (full predictions dict)
"""

import streamlit as st

from src.models.demand_model import predict_station_demand
from src.agent.agent        import generate_recommendation
from src.utils.helpers      import (
    load_daily_data,
    load_raw_csv,
    resolve_station_key,
    get_station_stats,
    classify_demand,
)
from src.charts import (
    plot_historical_trend,
    plot_monthly_trend,
    plot_weekday_heatmap,
    plot_top_stations,
    plot_demand_distribution,
    plot_system_trend,
    plot_system_map,
)

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="⚡ EV Intelligence Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: rgba(30, 30, 46, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(126, 184, 247, 0.2);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px 0 rgba(126, 184, 247, 0.15);
    border-color: rgba(126, 184, 247, 0.5);
}
.metric-card .label  { font-size: 14px; font-weight: 500; color: #a1a1aa; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-card .value  { font-size: 32px; font-weight: 800; color: #ffffff; text-shadow: 0 0 10px rgba(255,255,255,0.2); }
.metric-card .delta  { font-size: 13px; font-weight: 600; color: #7EB8F7; margin-top: 8px; }

.status-card {
    background: rgba(30, 30, 46, 0.6);
    backdrop-filter: blur(8px);
    padding: 20px 24px;
    border-radius: 12px;
    border-left: 5px solid;
    margin: 16px 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}
.status-card h3 { margin: 0 0 6px; font-size: 20px; font-weight: 700; letter-spacing: -0.5px; }
.status-card p  { margin: 0; font-size: 15px; color: #d1d5db; }

.rec-item {
    background: rgba(30, 30, 46, 0.5);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    border-left: 3px solid #7EB8F7;
    font-size: 15px;
    color: #e5e7eb;
    transition: background 0.2s;
}
.rec-item:hover {
    background: rgba(42, 42, 62, 0.8);
}

.section-divider {
    border: none;
    border-top: 1px solid rgba(126, 184, 247, 0.1);
    margin: 36px 0;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════
for key, default in {
    "predictions_dict":    None,   # full {station: kwh} dict from model
    "selected_prediction": None,   # float — kwh for selected station only
    "groq_result":         None,   # dict from generate_recommendation()
    "station_stats":       None,   # dict from get_station_stats()
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════════
# DATA LOAD  (cached — runs once per session)
# ═══════════════════════════════════════════════════════════════════
daily_data = load_daily_data()
stations   = sorted(daily_data["stationID"].unique())


# ═══════════════════════════════════════════════════════════════════
# ── SECTION 1: HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown("## ⚡ EV Charging Intelligence Hub")
st.markdown(
    "<p style='color:#aaa; margin-top:-10px; margin-bottom:24px;'>"
    "ML-powered demand forecasting · Groq LLM recommendations · Real-time insights"
    "</p>",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════
# ── SECTION 2: INPUT (SIDEBAR)
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔌 Prediction Controls", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top:0; border-color: rgba(126, 184, 247, 0.2);'>", unsafe_allow_html=True)

    selected_station = st.selectbox(
        "⚡ Charging Station",
        stations,
        key="station_selector",
        help="Select the EV charging station to forecast.",
    )

    selected_date = st.date_input(
        "📅 Forecast Date",
        key="date_picker",
        help="Target date for the demand forecast.",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button(
        "🚀 Run Prediction",
        width="stretch",
        type="primary",
        key="run_prediction_btn",
    )

# Station historical slice (used for charts + stats)
station_df = (
    daily_data[daily_data["stationID"] == selected_station]
    .sort_values("date")
)


# ═══════════════════════════════════════════════════════════════════
# ── PREDICTION LOGIC (on button click)
# ═══════════════════════════════════════════════════════════════════
if run_clicked:
    # Reset downstream results when re-running
    st.session_state.groq_result = None

    with st.spinner("⚙️ Running ML model across all stations…"):
        try:
            raw_df      = load_raw_csv()
            predictions = predict_station_demand(raw_df, target_date=selected_date)
            st.session_state.predictions_dict = predictions

            matched_key = resolve_station_key(selected_station, predictions)

            if matched_key is not None:
                st.session_state.selected_prediction = float(predictions[matched_key])
                st.session_state.station_stats       = get_station_stats(station_df)
                st.success(f"✅ Prediction ready for **{selected_station}**", icon="⚡")
            else:
                st.session_state.selected_prediction = None
                sample_keys = list(predictions.keys())[:5]
                st.warning(
                    f"⚠️ Station **{selected_station}** not found in predictions. "
                    f"Sample available keys: `{sample_keys}`"
                )

        except Exception as exc:
            st.error(f"❌ Prediction failed: {exc}")


# ═══════════════════════════════════════════════════════════════════
# ── SECTION 3: TABS DIRECTORY  (shown only after prediction runs)
# ═══════════════════════════════════════════════════════════════════
if st.session_state.selected_prediction is not None:
    pred  = st.session_state.selected_prediction
    stats = st.session_state.station_stats or {}
    meta  = classify_demand(pred)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    t1, t2, t3 = st.tabs(["📊 Station Overview", "🌍 System Network", "🤖 AI Diagnostics"])

    # ── TAB 1: STATION OVERVIEW ───────────────────────────────────
    with t1:
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">🔮 Predicted Demand</div>
                <div class="value">{pred:.1f} <span style="font-size:16px;color:#aaa;">kWh</span></div>
                <div class="delta">Next-day forecast</div>
            </div>""", unsafe_allow_html=True)

        with m2:
            last = stats.get("last_demand")
            last_str = f"{last:.1f} kWh" if last is not None else "—"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">📅 Last Day Demand</div>
                <div class="value">{last_str}</div>
                <div class="delta">Most recent recorded</div>
            </div>""", unsafe_allow_html=True)

        with m3:
            avg7 = stats.get("avg_7")
            avg7_str = f"{avg7:.1f} kWh" if avg7 is not None else "—"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">📈 7-Day Average</div>
                <div class="value">{avg7_str}</div>
                <div class="delta">Rolling mean</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="status-card" style="
            background:{meta['bg']};
            border-color:{meta['color']};
        ">
            <h3 style="color:{meta['color']};">{meta['emoji']} {meta['label']}</h3>
            <p>Station <strong>{selected_station}</strong> — predicted
               <strong>{pred:.1f} kWh</strong> for {selected_date}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>##### 📈 Historical Demand — Last 60 Days", unsafe_allow_html=True)
        st.plotly_chart(
            plot_historical_trend(station_df, selected_date, pred),
            width="stretch",
        )

    # ── TAB 2: SYSTEM NETWORK ─────────────────────────────────────
    with t2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.predictions_dict:
            st.markdown("##### 🌍 System-Wide Predicted Load Map")
            st.plotly_chart(plot_system_map(st.session_state.predictions_dict), width="stretch")
            st.markdown("<br>", unsafe_allow_html=True)

        d1, d2 = st.columns(2)

        with d1:
            st.markdown("##### 🏆 Top 10 High-Demand Stations")
            st.plotly_chart(plot_top_stations(daily_data), width="stretch")

        with d2:
            st.markdown("##### 📊 Demand Distribution (Selected Station)")
            st.plotly_chart(plot_demand_distribution(station_df), width="stretch")

        st.markdown("##### 🔥 Weekly Pattern — Selected Station")
        st.plotly_chart(plot_weekday_heatmap(station_df), width="stretch")

        st.markdown("##### 📅 Monthly Average Demand (System-Wide)")
        st.plotly_chart(plot_monthly_trend(daily_data), width="stretch")

        st.markdown("##### 📈 System-Wide Daily Demand Trend")
        st.plotly_chart(plot_system_trend(daily_data), width="stretch")


    # ── TAB 3: AI DIAGNOSTICS ──────────────────────────────────────
    with t3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🤖 Agentic Demand Diagnostics")
        st.markdown("Get an automated analysis report using Groq LLM detailing grid health and actions.")

        ai_col, _ = st.columns([1, 2])
        with ai_col:
            gen_clicked = st.button(
                "⚡ Generate AI Report",
                width="stretch",
                key="run_groq_btn",
            )

        if gen_clicked:
            with st.spinner("🧠 Groq LLM analyzing system..."):
                result = generate_recommendation(selected_station, pred)
                st.session_state.groq_result = result

        if st.session_state.groq_result is not None:
            result = st.session_state.groq_result
            status      = result.get("status", "Unknown")
            recs        = result.get("recommendations", [])
            reasoning   = result.get("reasoning", "")

            badge_style = {
                "normal":     ("#1a9e5c", "#e6faf1", "🟢"),
                "high load":  ("#c28500", "#fffbe6", "🟡"),
                "overloaded": ("#c0392b", "#fdecea", "🔴"),
            }
            key_lower = status.lower()
            bkey  = next((k for k in badge_style if k in key_lower), "normal")
            color, bg, emoji = badge_style[bkey]

            st.markdown(f"""
            <div class="status-card" style="background:{bg}; border-color:{color}; margin-top: 1rem;">
                <h3 style="color:{color};">{emoji} Agent Status: {status}</h3>
                <p>Station <strong>{selected_station}</strong> — {pred:.1f} kWh forecast</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### ⚡ Recommendations")
            for rec in recs:
                st.markdown(
                    f'<div class="rec-item">• {rec}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📊 Agent Reasoning")
            st.info(reasoning)