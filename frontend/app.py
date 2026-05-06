"""
app.py — Streamlit Dashboard for ChurnLens.

Gym operators use this dashboard to:
  1. Analyze individual reviews (sentiment + churn risk)
  2. View aggregate sentiment & churn trends
  3. Explore complaint theme clusters
  4. Identify at-risk locations
"""

import streamlit as st
import requests
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import pandas as pd

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnLens Dashboard",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────
API_URL = "http://localhost:9000"

SENTIMENT_COLORS = {
    "positive": "#10B981",
    "neutral": "#F59E0B",
    "negative": "#EF4444",
}

CHURN_COLORS = {
    True: "#EF4444",
    False: "#10B981",
}

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        background: linear-gradient(90deg, #e2e8f0, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        font-size: 1rem;
        color: #94a3b8;
        margin: 0;
    }

    .metric-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }

    .metric-card .label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #e2e8f0;
    }

    .result-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .sentiment-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .sentiment-positive { background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid #10B981; }
    .sentiment-neutral  { background: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid #F59E0B; }
    .sentiment-negative { background: rgba(239, 68, 68, 0.15);  color: #EF4444; border: 1px solid #EF4444; }

    .churn-high { background: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid #EF4444; }
    .churn-low  { background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid #10B981; }

    .theme-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid #818CF8;
        padding: 0.25rem 0.75rem;
        border-radius: 16px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }

    .stTextArea textarea {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e293b);
    }

    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: #334155;
        overflow: hidden;
        margin-top: 0.3rem;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────
def call_api(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    """Call the ChurnLens API."""
    try:
        url = f"{API_URL}{endpoint}"
        if method == "POST":
            resp = requests.post(url, json=data, timeout=300)
        else:
            resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def render_sentiment_badge(sentiment: str) -> str:
    return f'<span class="sentiment-badge sentiment-{sentiment}">{sentiment}</span>'


def render_churn_badge(churn_risk: bool) -> str:
    label = "HIGH RISK" if churn_risk else "LOW RISK"
    cls = "churn-high" if churn_risk else "churn-low"
    return f'<span class="sentiment-badge {cls}">⚠ {label}</span>' if churn_risk else f'<span class="sentiment-badge {cls}">✓ {label}</span>'


def render_themes(themes: list) -> str:
    if not themes:
        return '<span style="color: #64748b;">No specific themes detected</span>'
    return " ".join([f'<span class="theme-tag">{t}</span>' for t in themes])


def render_confidence_bar(confidence: float, color: str) -> str:
    pct = int(confidence * 100)
    return f"""
    <div class="confidence-bar">
        <div class="confidence-fill" style="width: {pct}%; background: {color};"></div>
    </div>
    <span style="font-size: 0.8rem; color: #94a3b8;">{pct}% confidence</span>
    """


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏋️ ChurnLens")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🔍 Review Analyzer", "📊 Batch Analysis"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # API status
    health = call_api("/health")
    if health:
        status_color = "#10B981"
        status_text = "Connected"
        model_text = "BERT Model Loaded" if health.get("model_loaded") else "Demo Mode (rule-based)"
    else:
        status_color = "#EF4444"
        status_text = "Disconnected"
        model_text = "Start the API server"

    st.markdown(f"""
    <div style="background: #0f172a; padding: 1rem; border-radius: 8px; border: 1px solid #334155;">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background: {status_color};"></div>
            <span style="color: #e2e8f0; font-weight: 500;">API: {status_text}</span>
        </div>
        <div style="font-size: 0.8rem; color: #64748b;">{model_text}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div style="color: #475569; font-size: 0.75rem; text-align: center;">ChurnLens v1.0 · BERT Fine-tuned</div>',
        unsafe_allow_html=True,
    )


# ── Page: Review Analyzer ───────────────────────────────────────
if page == "🔍 Review Analyzer":
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Review Analyzer</h1>
        <p>Analyze individual gym reviews for sentiment, churn risk, and complaint themes</p>
    </div>
    """, unsafe_allow_html=True)

    # Sample reviews
    sample_reviews = {
        "Select a sample...": "",
        "😡 Angry member (churn risk)": "This gym is absolutely terrible. The equipment is broken and dirty. Staff is rude and unhelpful. I'm cancelling my membership and switching to Planet Fitness. Waste of money.",
        "😊 Happy member": "Amazing gym! The trainers are incredibly knowledgeable and friendly. Equipment is always clean and well-maintained. Best gym I've ever been to. Highly recommend!",
        "😐 Mixed feelings": "The gym itself is decent and has good equipment, but it gets really crowded during peak hours. Wish they had better hours. The price is fair though.",
        "🏃 Leaving soon": "I've been a member for 2 years but I'm seriously thinking about leaving. The quality has gone downhill - dirty showers, broken machines, and the new management doesn't seem to care.",
        "⭐ Class lover": "The yoga and spin classes are fantastic! The instructors really know what they're doing. Only complaint is parking can be tough on weekends.",
    }

    selected = st.selectbox("Try a sample review:", list(sample_reviews.keys()))
    default_text = sample_reviews.get(selected, "")

    review_text = st.text_area(
        "Enter a gym review to analyze:",
        value=default_text,
        height=150,
        placeholder="Type or paste a gym review here...",
    )

    col_btn, col_space = st.columns([1, 3])
    with col_btn:
        analyze_btn = st.button("🔬 Analyze Review", type="primary", use_container_width=True)

    if analyze_btn and review_text.strip():
        with st.spinner("Analyzing..."):
            result = call_api("/predict", method="POST", data={"text": review_text})

        if result:
            st.markdown("---")

            # Top metrics row
            c1, c2, c3 = st.columns(3)

            with c1:
                s_color = SENTIMENT_COLORS.get(result["sentiment"], "#94a3b8")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Sentiment</div>
                    <div class="value" style="color: {s_color};">{result["sentiment"].upper()}</div>
                    {render_confidence_bar(result["sentiment_confidence"], s_color)}
                </div>
                """, unsafe_allow_html=True)

            with c2:
                c_color = CHURN_COLORS.get(result["churn_risk"], "#94a3b8")
                churn_label = "HIGH RISK" if result["churn_risk"] else "LOW RISK"
                churn_icon = "⚠️" if result["churn_risk"] else "✅"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Churn Risk</div>
                    <div class="value" style="color: {c_color};">{churn_icon} {churn_label}</div>
                    {render_confidence_bar(result["churn_confidence"], c_color)}
                </div>
                """, unsafe_allow_html=True)

            with c3:
                theme_count = len(result.get("themes", []))
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Themes Detected</div>
                    <div class="value" style="color: #818CF8;">{theme_count}</div>
                    <div style="margin-top: 0.5rem;">{render_themes(result.get("themes", []))}</div>
                </div>
                """, unsafe_allow_html=True)

            # Sentiment scores chart
            st.markdown("<br>", unsafe_allow_html=True)
            scores = result.get("sentiment_scores", {})
            if scores:
                fig = go.Figure(go.Bar(
                    x=list(scores.keys()),
                    y=list(scores.values()),
                    marker_color=[SENTIMENT_COLORS.get(k, "#64748b") for k in scores.keys()],
                    text=[f"{v:.1%}" for v in scores.values()],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Sentiment Score Distribution",
                    yaxis_title="Probability",
                    yaxis_range=[0, 1.1],
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Could not connect to the API. Make sure the server is running: `uvicorn backend.main:app --port 8000`")

    elif analyze_btn:
        st.warning("Please enter a review to analyze.")


# ── Page: Batch Analysis ─────────────────────────────────────────
elif page == "📊 Batch Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Batch Analysis</h1>
        <p>Upload a CSV of reviews to analyze sentiment, churn risk, and themes across locations</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="result-card">
        <strong>CSV Format:</strong> Your file needs a <code>text</code> column (required) and optionally a <code>location</code> column to group results by gym.
        <br><br>
        <code>location,text</code><br>
        <code>Downtown Fitness,"Great gym, love the trainers!"</code><br>
        <code>Westside Gym,"Dirty equipment, thinking about cancelling"</code>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    # Also keep text area as fallback
    batch_input = st.text_area(
        "Or paste reviews (one per line):",
        height=150,
        placeholder="Review 1: This gym is great...\nReview 2: Terrible experience...",
    )

    if st.button("🚀 Analyze Batch", type="primary"):
        texts = []
        locations = []
        has_locations = False

        # Parse CSV upload
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            if "text" not in df_upload.columns:
                st.error("CSV must have a `text` column.")
                st.stop()
            if "location" in df_upload.columns:
                df_upload = df_upload.dropna(subset=["text", "location"])
                texts = df_upload["text"].astype(str).tolist()
                locations = df_upload["location"].astype(str).tolist()
                has_locations = True
            else:
                df_upload = df_upload.dropna(subset=["text"])
                texts = df_upload["text"].astype(str).tolist()
        elif batch_input.strip():
            lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
            # Auto-detect CSV format pasted into text area
            if lines and ("," in lines[0]) and ("text" in lines[0].lower()):
                import io
                df_paste = pd.read_csv(io.StringIO(batch_input.strip()))
                if "text" in df_paste.columns:
                    if "location" in df_paste.columns:
                        df_paste = df_paste.dropna(subset=["text", "location"])
                        texts = df_paste["text"].astype(str).tolist()
                        locations = df_paste["location"].astype(str).tolist()
                        has_locations = True
                    else:
                        df_paste = df_paste.dropna(subset=["text"])
                        texts = df_paste["text"].astype(str).tolist()
                else:
                    texts = lines[1:]  # skip header, treat as plain text
            else:
                texts = lines
        
        if not texts:
            st.warning("Please upload a CSV or enter reviews.")
        else:
            if has_locations:
                unique_locs = len(set(locations))
                st.success(f"✅ Detected **{unique_locs} locations** across {len(texts)} reviews")
            with st.spinner(f"Analyzing {len(texts)} reviews with BERT..."):
                result = call_api("/predict/batch", method="POST", data={"texts": texts})

            if result and "predictions" in result:
                preds = result["predictions"]

                # Attach location info
                if has_locations and len(locations) == len(preds):
                    for i, p in enumerate(preds):
                        p["location"] = locations[i]

                # ── KPI Cards ──
                sentiments = [p["sentiment"] for p in preds]
                churn_flags = [p["churn_risk"] for p in preds]
                all_themes = []
                for p in preds:
                    all_themes.extend(p.get("themes", []))

                neg_pct = sentiments.count("negative") / len(sentiments) * 100
                churn_pct = sum(churn_flags) / len(churn_flags) * 100
                pos_pct = sentiments.count("positive") / len(sentiments) * 100

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f'<div class="metric-card"><div class="label">Total Reviews</div><div class="value">{len(preds):,}</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card"><div class="label">Positive %</div><div class="value" style="color:#10B981;">{pos_pct:.0f}%</div></div>', unsafe_allow_html=True)
                with c3:
                    color = "#EF4444" if neg_pct > 30 else "#F59E0B"
                    st.markdown(f'<div class="metric-card"><div class="label">Negative %</div><div class="value" style="color:{color};">{neg_pct:.0f}%</div></div>', unsafe_allow_html=True)
                with c4:
                    color = "#EF4444" if churn_pct > 20 else "#F59E0B" if churn_pct > 10 else "#10B981"
                    st.markdown(f'<div class="metric-card"><div class="label">Churn Risk %</div><div class="value" style="color:{color};">{churn_pct:.0f}%</div></div>', unsafe_allow_html=True)

                st.markdown("---")

                # ── Location Breakdown (if locations exist) ──
                if has_locations:
                    from collections import defaultdict
                    loc_groups = defaultdict(list)
                    for p in preds:
                        loc_groups[p.get("location", "Unknown")].append(p)

                    locations_data = []
                    for name, reviews in loc_groups.items():
                        sents = [r["sentiment"] for r in reviews]
                        churns = [r["churn_risk"] for r in reviews]
                        themes_list = []
                        for r in reviews:
                            themes_list.extend(r.get("themes", []))
                        top_theme = Counter(themes_list).most_common(1)[0][0] if themes_list else "none"
                        total = len(sents)
                        locations_data.append({
                            "name": name,
                            "positive": round(sents.count("positive") / total * 100),
                            "neutral": round(sents.count("neutral") / total * 100),
                            "negative": round(sents.count("negative") / total * 100),
                            "churn_rate": round(sum(churns) / len(churns) * 100),
                            "top_theme": top_theme,
                            "reviews": total,
                            "themes_counter": Counter(themes_list),
                        })

                    high_risk_count = sum(1 for l in locations_data if l["churn_rate"] > 20)
                    if high_risk_count > 0:
                        st.error(f"⚠️ **{high_risk_count} location(s) above 20% churn threshold**")

                    # Charts: Churn by Location + Sentiment by Location
                    col_left, col_right = st.columns(2)
                    with col_left:
                        sorted_locs = sorted(locations_data, key=lambda x: x["churn_rate"], reverse=True)
                        colors = ["#EF4444" if l["churn_rate"] > 20 else "#F59E0B" if l["churn_rate"] > 10 else "#10B981" for l in sorted_locs]
                        fig_churn = go.Figure()
                        fig_churn.add_trace(go.Bar(x=[l["name"] for l in sorted_locs], y=[l["churn_rate"] for l in sorted_locs], marker_color=colors, text=[f"{l['churn_rate']}%" for l in sorted_locs], textposition="outside"))
                        fig_churn.add_hline(y=20, line_dash="dash", line_color="#EF4444", annotation_text="Risk Threshold")
                        fig_churn.update_layout(title="Churn Risk by Location", yaxis_title="Churn Rate %", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400, font=dict(family="Inter"))
                        st.plotly_chart(fig_churn, use_container_width=True)

                    with col_right:
                        fig_sent = go.Figure()
                        names = [l["name"] for l in locations_data]
                        fig_sent.add_trace(go.Bar(name="Positive", x=names, y=[l["positive"] for l in locations_data], marker_color="#10B981"))
                        fig_sent.add_trace(go.Bar(name="Neutral", x=names, y=[l["neutral"] for l in locations_data], marker_color="#F59E0B"))
                        fig_sent.add_trace(go.Bar(name="Negative", x=names, y=[l["negative"] for l in locations_data], marker_color="#EF4444"))
                        fig_sent.update_layout(title="Sentiment by Location", barmode="stack", yaxis_title="%", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400, font=dict(family="Inter"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(fig_sent, use_container_width=True)

                    # Theme Heatmap
                    st.markdown("### 🎯 Theme Analysis by Location")
                    theme_names = ["cleanliness", "equipment", "staff", "pricing", "overcrowding", "classes", "facilities", "hours"]
                    heatmap_data = [[loc["themes_counter"].get(t, 0) for t in theme_names] for loc in locations_data]
                    fig_heat = go.Figure(data=go.Heatmap(z=heatmap_data, x=theme_names, y=[l["name"] for l in locations_data], colorscale=[[0, "#0f172a"], [0.5, "#6366F1"], [1, "#EF4444"]], text=heatmap_data, texttemplate="%{text}", textfont={"size": 12}))
                    fig_heat.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, font=dict(family="Inter"), xaxis_title="Theme", yaxis_title="Location")
                    st.plotly_chart(fig_heat, use_container_width=True)

                    # Location Cards
                    st.markdown("### 📍 Location Summary")
                    for loc in sorted(locations_data, key=lambda x: x["churn_rate"], reverse=True):
                        risk_color = "#EF4444" if loc["churn_rate"] > 20 else "#F59E0B" if loc["churn_rate"] > 10 else "#10B981"
                        risk_label = "HIGH" if loc["churn_rate"] > 20 else "MEDIUM" if loc["churn_rate"] > 10 else "LOW"
                        st.markdown(f"""
                        <div class="result-card" style="border-left: 4px solid {risk_color};">
                            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                                <div>
                                    <span style="font-size: 1.2rem; font-weight: 600; color: #e2e8f0;">{loc["name"]}</span>
                                    <span class="sentiment-badge" style="margin-left: 1rem; background: {risk_color}22; color: {risk_color}; border: 1px solid {risk_color};">{risk_label} RISK</span>
                                </div>
                                <div style="color: #94a3b8;">{loc["reviews"]} reviews</div>
                            </div>
                            <div style="display: flex; gap: 2rem; margin-top: 1rem; flex-wrap: wrap;">
                                <div><span style="color: #10B981; font-weight: 600;">{loc["positive"]}%</span> <span style="color: #64748b;">positive</span></div>
                                <div><span style="color: #F59E0B; font-weight: 600;">{loc["neutral"]}%</span> <span style="color: #64748b;">neutral</span></div>
                                <div><span style="color: #EF4444; font-weight: 600;">{loc["negative"]}%</span> <span style="color: #64748b;">negative</span></div>
                                <div><span style="color: {risk_color}; font-weight: 600;">{loc["churn_rate"]}%</span> <span style="color: #64748b;">churn rate</span></div>
                                <div><span class="theme-tag">{loc["top_theme"]}</span> <span style="color: #64748b;">top complaint</span></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                else:
                    # No locations — show aggregate charts
                    col_left, col_right = st.columns(2)
                    with col_left:
                        s_counts = Counter(sentiments)
                        fig_s = px.pie(names=list(s_counts.keys()), values=list(s_counts.values()), color=list(s_counts.keys()), color_discrete_map=SENTIMENT_COLORS, title="Sentiment Distribution", hole=0.45)
                        fig_s.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=350, font=dict(family="Inter"))
                        st.plotly_chart(fig_s, use_container_width=True)
                    with col_right:
                        if all_themes:
                            t_counts = Counter(all_themes)
                            fig_t = px.bar(x=list(t_counts.keys()), y=list(t_counts.values()), title="Theme Frequency", color_discrete_sequence=["#818CF8"])
                            fig_t.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Theme", yaxis_title="Count", height=350, font=dict(family="Inter"))
                            st.plotly_chart(fig_t, use_container_width=True)

                st.markdown("---")

                # Download
                results_df = pd.DataFrame(preds)
                csv_data = results_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Results CSV", csv_data, "churnlens_results.csv", "text/csv")

                # Detailed Results (collapsible)
                with st.expander(f"📋 View All {len(preds)} Detailed Results", expanded=False):
                    for i, p in enumerate(preds):
                        sentiment_html = render_sentiment_badge(p["sentiment"])
                        churn_html = render_churn_badge(p["churn_risk"])
                        themes_html = render_themes(p.get("themes", []))
                        loc_label = f"<span style='color:#818CF8;font-weight:600;'>[{p['location']}]</span> " if "location" in p else ""
                        st.markdown(f"""
                        <div class="result-card">
                            <div style="display: flex; gap: 0.75rem; margin-bottom: 0.75rem; flex-wrap: wrap;">
                                {sentiment_html} {churn_html}
                            </div>
                            <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">{loc_label}{p.get("text", texts[i] if i < len(texts) else "")}</div>
                            <div style="margin-top: 0.75rem;">{themes_html}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("API error. Ensure the server is running.")

