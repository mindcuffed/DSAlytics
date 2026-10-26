import os
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DSAlytics | Developer Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CYBER DEVELOPER THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #070A0F;
        --panel: #0D131D;
        --panel-2: #101A27;
        --border: #202D3D;
        --text: #E7EEF8;
        --muted: #8EA0B7;
        --green: #35F2A0;
        --cyan: #4DE1F2;
        --purple: #A78BFA;
        --red: #FF647C;
        --amber: #FBBF55;
    }

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 15% 0%, rgba(25, 90, 78, 0.13), transparent 34%),
            radial-gradient(ellipse at 90% 12%, rgba(32, 92, 132, 0.10), transparent 30%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(7, 10, 15, 0.82);
    }

    .block-container {
        max-width: 1600px;
        padding: 2.0rem 2.5rem 3rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.035em;
    }

    h1 {
        font-size: clamp(2rem, 3vw, 3rem) !important;
        font-weight: 700 !important;
    }

    h2 { font-size: 1.55rem !important; font-weight: 650 !important; }
    h3 { font-size: 1.15rem !important; font-weight: 600 !important; }

    p, li, label, .stMarkdown {
        color: #B7C4D6;
    }

    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0C121B 0%, #080C13 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.6rem;
    }

    [data-testid="stSidebar"] h1 {
        color: var(--green) !important;
        font-size: 1.65rem !important;
        letter-spacing: -0.06em;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }

    [data-testid="stRadio"] > label {
        color: #AFC0D4 !important;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 0.35rem;
    }

    [data-testid="stRadio"] div[role="radiogroup"] label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 10px;
        padding: 0.55rem 0.65rem;
        transition: all 160ms ease;
    }

    [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background: #111B28;
        border-color: #26394D;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(17, 28, 42, 0.96), rgba(10, 17, 27, 0.96));
        border: 1px solid #24364A;
        border-radius: 15px;
        padding: 1rem 1.15rem;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.13);
        min-height: 112px;
    }

    [data-testid="stMetricLabel"] {
        color: #8FA3BB !important;
        font-size: 0.84rem !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--green) !important;
        font-family: 'DM Mono', monospace !important;
        font-size: clamp(1.5rem, 2vw, 2.1rem) !important;
        font-weight: 500 !important;
    }

    [data-testid="stMetricDelta"] {
        color: var(--cyan) !important;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.1rem;
        border: 1px solid #24473F;
        border-radius: 22px;
        background:
            radial-gradient(circle at 85% 20%, rgba(53, 242, 160, 0.12), transparent 28%),
            linear-gradient(120deg, #101B27 0%, #0B151D 58%, #0A1718 100%);
        margin-bottom: 1.25rem;
    }

    .eyebrow {
        color: var(--green);
        font-family: 'DM Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.72rem;
        margin-bottom: 0.75rem;
    }

    .hero-title {
        color: #F3F7FC;
        font-size: clamp(2rem, 4vw, 3.35rem);
        line-height: 1.06;
        letter-spacing: -0.06em;
        font-weight: 700;
        margin: 0 0 0.8rem 0;
    }

    .hero-title span { color: var(--green); }

    .hero-copy {
        max-width: 740px;
        color: #A9B8CA;
        font-size: 1rem;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .section-kicker {
        color: var(--green);
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 0.35rem;
    }

    .panel {
        background: linear-gradient(145deg, #0F1824, #0B111A);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 1.1rem 1.2rem;
        height: 100%;
    }

    .panel-title {
        color: #E7EEF8;
        font-size: 1rem;
        font-weight: 650;
        margin-bottom: 0.35rem;
    }

    .panel-copy {
        color: #8FA1B8;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    .step-number {
        color: var(--green);
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        margin-bottom: 0.75rem;
    }

    .status-pill {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        background: rgba(53, 242, 160, 0.09);
        border: 1px solid rgba(53, 242, 160, 0.27);
        color: var(--green);
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
    }

    .stButton > button {
        background: #10352D;
        color: var(--green);
        border: 1px solid #26765C;
        border-radius: 10px;
        padding: 0.55rem 1rem;
        font-weight: 600;
        transition: all 160ms ease;
    }

    .stButton > button:hover {
        background: #17483B;
        color: #FFFFFF;
        border-color: var(--green);
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        background: #101B29;
        border: 1px solid #284158;
        border-radius: 12px;
        color: #C9D8E8;
    }

    [data-testid="stExpander"] {
        background: #0D141E;
        border: 1px solid var(--border);
        border-radius: 12px;
    }

    hr {
        border-color: #1F2B3A !important;
        margin: 1.4rem 0 !important;
    }

    a { color: var(--cyan) !important; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #070A0F; }
    ::-webkit-scrollbar-thumb { background: #29394D; border-radius: 8px; }

    @media (max-width: 768px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .hero { padding: 1.35rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ML_DATASET = os.path.join(BASE_DIR, "data", "processed", "ml_dataset.csv")
WEAKNESS_FILE = os.path.join(BASE_DIR, "results", "topic_weakness_analysis.csv")
RECOMMENDATION_FILE = os.path.join(BASE_DIR, "results", "final_recommendations.csv")
FEATURE_IMPORTANCE_FILE = os.path.join(BASE_DIR, "results", "feature_importance.csv")
VALIDATION_FILE = os.path.join(BASE_DIR, "results", "walk_forward_validation.csv")


# ============================================================
# PLOTLY THEME
# ============================================================

PLOT_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Space Grotesk, sans-serif", "color": "#B7C4D6"},
    "title": {"text": "", "font": {"size": 17, "color": "#E7EEF8"}, "x": 0.02, "xanchor": "left"},
    "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#AFC0D4"}},
    "margin": {"l": 20, "r": 20, "t": 65, "b": 25},
    "xaxis": {
        "gridcolor": "#1D2A39",
        "linecolor": "#2A3B4F",
        "zerolinecolor": "#2A3B4F",
        "tickfont": {"color": "#8FA1B8"},
        "title_font": {"color": "#AFC0D4"},
    },
    "yaxis": {
        "gridcolor": "#1D2A39",
        "linecolor": "#2A3B4F",
        "zerolinecolor": "#2A3B4F",
        "tickfont": {"color": "#8FA1B8"},
        "title_font": {"color": "#AFC0D4"},
    },
}


def style_chart(fig, height=390):
    """Apply the shared cyber theme to a Plotly figure."""
    fig.update_layout(**PLOT_LAYOUT, height=height, title_text="")
    return fig


def section_header(kicker, title, description=None):
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.subheader(title)
    if description:
        st.caption(description)


def metric_row(items):
    """Render metric cards from (label, value) pairs."""
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.metric(label, value)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    required_files = {
        "ML dataset": ML_DATASET,
        "Weakness analysis": WEAKNESS_FILE,
        "Recommendations": RECOMMENDATION_FILE,
        "Feature importance": FEATURE_IMPORTANCE_FILE,
        "Model validation": VALIDATION_FILE,
    }

    missing = [f"{name}: {path}" for name, path in required_files.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required data file(s):\n" + "\n".join(missing))

    ml_data = pd.read_csv(ML_DATASET)
    weakness_data = pd.read_csv(WEAKNESS_FILE)
    recommendation_data = pd.read_csv(RECOMMENDATION_FILE)
    feature_data = pd.read_csv(FEATURE_IMPORTANCE_FILE)
    validation_data = pd.read_csv(VALIDATION_FILE)

    return ml_data, weakness_data, recommendation_data, feature_data, validation_data


try:
    (
        ml_data,
        weakness_data,
        recommendation_data,
        feature_data,
        validation_data,
    ) = load_data()
except Exception as error:
    st.error(f"Unable to load DSAlytics data: {error}")
    st.info("Run `py update_dsalytics.py` from your project folder, then refresh this page.")
    st.stop()


# ============================================================
# COMMON METRICS
# ============================================================

total_submissions = len(ml_data)
accepted = int(ml_data["target"].fillna(0).sum()) if "target" in ml_data.columns else 0
failed = max(total_submissions - accepted, 0)
acceptance_rate = (accepted / total_submissions * 100) if total_submissions else 0.0
failure_rate = (failed / total_submissions * 100) if total_submissions else 0.0


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## ⚡ DSAlytics")
    st.caption("DEVELOPER INTELLIGENCE")
    st.markdown('<span class="status-pill">● ANALYTICS ONLINE</span>', unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Home",
            "📈 Performance",
            "🔥 Weakness Analysis",
            "🤖 ML Insights",
            "🎯 Recommendations",
        ],
        label_visibility="visible",
    )

    st.divider()
    st.caption("Track → Analyze → Predict → Improve")
    st.caption("DSA practice intelligence, built around your submission history.")


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">// PERSONAL DSA INTELLIGENCE</div>
            <div class="hero-title">Turn every submission<br>into <span>signal.</span></div>
            <p class="hero-copy">
                Your coding history, translated into actionable insight.
                Track performance, surface weak topics, explore model signals,
                and decide what to practice next.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">01 / YOUR ACTIVITY</div>', unsafe_allow_html=True)
    metric_row(
        [
            ("TOTAL SUBMISSIONS", f"{total_submissions:,}"),
            ("ACCEPTED", f"{accepted:,}"),
            ("ACCEPTANCE RATE", f"{acceptance_rate:.1f}%"),
            ("FAILURE RATE", f"{failure_rate:.1f}%"),
        ]
    )

    st.divider()
    section_header("02 / THE WORKFLOW", "From raw history to next steps", "Four stages turn your practice history into a focused plan.")

    flow_columns = st.columns(4)
    flow_items = [
        ("01", "TRACK", "Collect submission history and build a consistent activity dataset."),
        ("02", "ANALYZE", "Measure topic-level accuracy, repeated failures, and recency."),
        ("03", "PREDICT", "Use historical patterns to estimate future failure risk."),
        ("04", "RECOMMEND", "Prioritize the topics that deserve your next practice session."),
    ]
    for column, (number, title, copy) in zip(flow_columns, flow_items):
        with column:
            st.markdown(
                f"""
                <div class="panel">
                    <div class="step-number">[{number}]</div>
                    <div class="panel-title">{title}</div>
                    <div class="panel-copy">{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    section_header("03 / EXPLORE", "Your workspace", "Choose a view to dig deeper into your practice data.")

    feature_columns = st.columns(2)
    feature_items = [
        ("📈", "Performance analytics", "Review acceptance patterns, difficulty-level accuracy, and recent submissions."),
        ("🔥", "Weakness detection", "Find topics where historical or recent performance suggests a learning gap."),
        ("🤖", "Machine-learning insights", "Inspect feature importance and chronological model validation."),
        ("🎯", "Personalized practice", "Use ranked recommendations to choose a topic for your next session."),
    ]
    for index, (icon, title, copy) in enumerate(feature_items):
        with feature_columns[index % 2]:
            st.markdown(
                f"""
                <div class="panel" style="margin-bottom: 0.85rem;">
                    <div class="panel-title">{icon} &nbsp; {title}</div>
                    <div class="panel-copy">{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption("Tip: refresh the data pipeline after new LeetCode activity to update these insights.")


# ============================================================
# PERFORMANCE
# ============================================================

elif page == "📈 Performance":
    st.markdown('<div class="section-kicker">01 / ACTIVITY MONITOR</div>', unsafe_allow_html=True)
    st.title("Performance analytics")
    st.caption("Understand the outcomes behind your coding practice.")

    metric_row(
        [
            ("SUBMISSIONS", f"{total_submissions:,}"),
            ("ACCEPTED", f"{accepted:,}"),
            ("ACCEPTANCE RATE", f"{acceptance_rate:.1f}%"),
            ("FAILURE RATE", f"{failure_rate:.1f}%"),
        ]
    )

    st.divider()
    left, right = st.columns(2)

    with left:
        section_header("OUTCOMES", "Accepted vs failed")
        performance_data = pd.DataFrame(
            {"Status": ["Accepted", "Failed"], "Count": [accepted, failed]}
        )
        if performance_data["Count"].sum() > 0:
            fig = px.pie(
                performance_data,
                names="Status",
                values="Count",
                hole=0.68,
                color="Status",
                color_discrete_map={"Accepted": "#35F2A0", "Failed": "#FF647C"},
            )
            fig.update_traces(textposition="inside", textinfo="percent+label", marker_line_color="#0B111A", marker_line_width=3)
            style_chart(fig, 360)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No submission outcomes are available yet.")

    with right:

        section_header("DIFFICULTY", "Accuracy by difficulty")

        if (
            "difficulty_encoded" in ml_data.columns
            and "target" in ml_data.columns
            and not ml_data.empty
        ):
            difficulty_map = {1: "Easy", 2: "Medium", 3: "Hard"}

            # Keep only valid difficulty codes and binary outcomes.
            difficulty_data = ml_data[
                ml_data["difficulty_encoded"].isin([1, 2, 3])
                & ml_data["target"].isin([0, 1])
            ].copy()

            if not difficulty_data.empty:
                difficulty_data["Difficulty"] = (
                    difficulty_data["difficulty_encoded"].map(difficulty_map)
                )

                difficulty_data = (
                    difficulty_data.groupby("Difficulty", as_index=False)
                    .agg(
                        Attempts=("target", "count"),
                        Accuracy=("target", "mean"),
                    )
                )

                difficulty_data["Accuracy"] *= 100

                difficulty_order = ["Easy", "Medium", "Hard"]
                difficulty_data["Difficulty"] = pd.Categorical(
                    difficulty_data["Difficulty"],
                    categories=difficulty_order,
                    ordered=True,
                )
                difficulty_data = difficulty_data.sort_values("Difficulty")

                fig = px.bar(
                    difficulty_data,
                    x="Difficulty",
                    y="Accuracy",
                    text="Accuracy",
                    color="Difficulty",
                    category_orders={"Difficulty": difficulty_order},
                    color_discrete_map={
                        "Easy": "#35F2A0",
                        "Medium": "#4DE1F2",
                        "Hard": "#A78BFA",
                    },
                    custom_data=["Attempts"],
                )

                fig.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=(
                        "Difficulty: %{x}<br>"
                        "Accuracy: %{y:.1f}%<br>"
                        "Attempts: %{customdata[0]}"
                        "<extra></extra>"
                    ),
                )

                max_accuracy = float(difficulty_data["Accuracy"].max())
                fig.update_yaxes(
                    range=[0, min(110, max(100, max_accuracy + 10))],
                    title="Acceptance accuracy (%)",
                )
                fig.update_xaxes(title="Difficulty")

                style_chart(fig, 360)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "No valid Easy, Medium, or Hard difficulty records were found."
                )
        else:
            st.info(
                "Difficulty information is not available in the current dataset."
            )


    st.divider()
    section_header("RECENT ACTIVITY", "Latest submissions", "Most recent rows available in the processed dataset.")

    if "submission_date" in ml_data.columns and "target" in ml_data.columns and not ml_data.empty:
        recent = ml_data.tail(10).copy()
        recent["submission_date"] = pd.to_datetime(recent["submission_date"], errors="coerce")
        recent = recent.dropna(subset=["submission_date"])
        recent["Status"] = recent["target"].map({1: "Accepted", 0: "Failed"}).fillna("Unknown")
        if not recent.empty:
            hover_fields = [column for column in ["difficulty_encoded"] if column in recent.columns]
            fig = px.scatter(
                recent,
                x="submission_date",
                y="target",
                color="Status",
                hover_data=hover_fields,
                color_discrete_map={"Accepted": "#35F2A0", "Failed": "#FF647C", "Unknown": "#8EA0B7"},
            )
            fig.update_traces(marker={"size": 12, "line": {"width": 1, "color": "#070A0F"}})
            fig.update_yaxes(tickvals=[0, 1], ticktext=["Failed", "Accepted"], title="")
            fig.update_xaxes(title="")
            style_chart(fig, 320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid submission dates were found.")
    else:
        st.info("The dataset does not contain the date and outcome columns needed for this chart.")


# ============================================================
# WEAKNESS ANALYSIS
# ============================================================

elif page == "🔥 Weakness Analysis":
    st.markdown('<div class="section-kicker">02 / GAP DETECTION</div>', unsafe_allow_html=True)
    st.title("Weakness analysis")
    st.caption("Topics are ranked using historical performance, recent performance, and practice recency.")

    if weakness_data.empty:
        st.info("No weakness-analysis data is available yet.")
    else:
        if "adjusted_weakness_score" in weakness_data.columns and "topic" in weakness_data.columns:
            top_10 = weakness_data.head(10).copy()
            section_header("RISK MAP", "Highest-risk topics", "A higher score indicates a higher-priority area according to the current scoring pipeline.")
            fig = px.bar(
                top_10,
                x="adjusted_weakness_score",
                y="topic",
                orientation="h",
                text="adjusted_weakness_score",
                color="adjusted_weakness_score",
                color_continuous_scale=["#173C3A", "#35F2A0"],
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            fig.update_yaxes(categoryorder="total ascending", title="")
            fig.update_xaxes(title="Weakness score")
            style_chart(fig, max(360, 35 * len(top_10) + 100))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        section_header("PRIORITY LIST", "Topics to review first")

        priority_columns = ["topic", "attempts", "accuracy", "recent_accuracy", "adjusted_weakness_score"]
        available_columns = [column for column in priority_columns if column in weakness_data.columns]
        if available_columns:
            st.dataframe(weakness_data[available_columns].head(5), use_container_width=True, hide_index=True)
        else:
            st.info("Expected topic-level columns are not present in the weakness file.")

        with st.expander("View complete topic-level analysis"):
            st.dataframe(weakness_data, use_container_width=True, hide_index=True)


# ============================================================
# ML INSIGHTS
# ============================================================

elif page == "🤖 ML Insights":
    st.markdown('<div class="section-kicker">03 / MODEL EXPLORER</div>', unsafe_allow_html=True)
    st.title("Machine-learning insights")
    st.caption("Explore model signals and chronological validation results.")

    if feature_data.empty:
        st.info("No feature-importance data is available yet.")
    elif "importance" in feature_data.columns and "feature" in feature_data.columns:
        top_features = feature_data.head(15).copy()
        section_header("FEATURE IMPORTANCE", "Signals influencing prediction", "These are the feature-importance values produced by your current training pipeline.")
        fig = px.bar(
            top_features,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=["#183B43", "#4DE1F2"],
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_yaxes(categoryorder="total ascending", title="")
        fig.update_xaxes(title="Importance")
        style_chart(fig, max(400, 28 * len(top_features) + 120))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("The feature-importance file does not contain the expected columns.")

    st.divider()
    section_header("MODEL VALIDATION", "Walk-forward validation", "Each test window contains later submissions than the training data.")

    if validation_data.empty:
        st.info("No walk-forward validation results are available yet.")
    else:
        st.dataframe(validation_data, use_container_width=True, hide_index=True)
        if {"roc_auc", "fold"}.issubset(validation_data.columns):
            fig = px.line(
                validation_data,
                x="fold",
                y="roc_auc",
                markers=True,
                color_discrete_sequence=["#35F2A0"],
            )
            fig.update_traces(line={"width": 3}, marker={"size": 9})
            fig.update_yaxes(range=[0, 1], title="ROC-AUC")
            fig.update_xaxes(title="Validation fold")
            style_chart(fig, 330)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "🎯 Recommendations":
    st.markdown('<div class="section-kicker">04 / NEXT BEST ACTION</div>', unsafe_allow_html=True)
    st.title("Personalized recommendations")
    st.caption("Use the current ranking to decide what to practice next.")

    if recommendation_data.empty:
        st.info("No recommendations are available yet. Run the update pipeline to regenerate them.")
    else:
        top = recommendation_data.iloc[0]

        st.markdown(
            f"""
            <div class="hero">
                <div class="eyebrow">// TOP PRIORITY TOPIC</div>
                <div class="hero-title" style="font-size: clamp(1.8rem, 3vw, 2.6rem);">
                    Focus on <span>{str(top.get('topic', 'your next topic'))}</span>
                </div>
                <p class="hero-copy">
                    Start here based on the current recommendation ranking.
                    Use the metrics below to understand why this topic is prioritized.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        recommendation_metrics = []
        if "final_risk_score" in top.index:
            recommendation_metrics.append(("RISK SCORE", f"{float(top['final_risk_score']):.1f}/100"))
        if "accuracy" in top.index:
            recommendation_metrics.append(("HISTORICAL ACCURACY", f"{float(top['accuracy']) * 100:.1f}%"))
        if "ml_failure_probability" in top.index:
            recommendation_metrics.append(("ML FAILURE RISK", f"{float(top['ml_failure_probability']) * 100:.1f}%"))

        if recommendation_metrics:
            metric_row(recommendation_metrics)

        st.divider()
        section_header("PRACTICE QUEUE", "Recommended practice order", "Work from the top down, then regenerate recommendations after new practice.")
        st.dataframe(recommendation_data, use_container_width=True, hide_index=True)

        if "topic" in recommendation_data.columns:
            st.divider()
            st.markdown("#### Topic shortlist")
            shortlist = recommendation_data.head(5)
            for index, (_, row) in enumerate(shortlist.iterrows(), start=1):
                topic = row.get("topic", "Unknown topic")
                score = row.get("final_risk_score", None)
                score_text = f" · Risk {float(score):.1f}/100" if pd.notna(score) else ""
                st.markdown(
                    f'<div class="panel" style="margin-bottom:0.6rem;">'
                    f'<span class="step-number">#{index:02d}</span> &nbsp; '
                    f'<span class="panel-title">{topic}</span>'
                    f'<span class="panel-copy">{score_text}</span></div>',
                    unsafe_allow_html=True,
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("DSAlytics • Developer Intelligence • Insights depend on the latest processed dataset.")
