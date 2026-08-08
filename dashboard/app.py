import streamlit as st
import pandas as pd
import plotly.express as px
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DSAlytics",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ML_DATASET = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ml_dataset.csv"
)

WEAKNESS_FILE = os.path.join(
    BASE_DIR,
    "results",
    "topic_weakness_analysis.csv"
)

RECOMMENDATION_FILE = os.path.join(
    BASE_DIR,
    "results",
    "final_recommendations.csv"
)

FEATURE_IMPORTANCE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "feature_importance.csv"
)

VALIDATION_FILE = os.path.join(
    BASE_DIR,
    "results",
    "walk_forward_validation.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    ml_data = pd.read_csv(
        ML_DATASET
    )

    weakness_data = pd.read_csv(
        WEAKNESS_FILE
    )

    recommendation_data = pd.read_csv(
        RECOMMENDATION_FILE
    )

    feature_data = pd.read_csv(
        FEATURE_IMPORTANCE_FILE
    )

    validation_data = pd.read_csv(
        VALIDATION_FILE
    )

    return (
        ml_data,
        weakness_data,
        recommendation_data,
        feature_data,
        validation_data
    )


try:

    (
        ml_data,
        weakness_data,
        recommendation_data,
        feature_data,
        validation_data
    ) = load_data()

except Exception as e:

    st.error(
        f"Unable to load DSAlytics data: {e}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 DSAlytics")

st.sidebar.caption(
    "DSA Performance Analytics"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📈 Performance",
        "🔥 Weakness Analysis",
        "🤖 ML Insights",
        "🎯 Recommendations"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Personalized DSA learning analytics"
)


# ============================================================
# COMMON METRICS
# ============================================================

total_submissions = len(
    ml_data
)

accepted = int(
    ml_data["target"].sum()
)

failed = (
    total_submissions - accepted
)

acceptance_rate = (
    accepted /
    total_submissions *
    100
)

failure_rate = (
    failed /
    total_submissions *
    100
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title(
        "📊 DSAlytics"
    )

    st.subheader(
        "AI-Powered DSA Performance Analytics & Personalized Practice"
    )

    st.write(
        "DSAlytics analyzes coding practice history to identify "
        "weak areas, estimate future failure risk and recommend "
        "what to practice next."
    )

    st.divider()

    # --------------------------------------------------------
    # HOW DSALYTICS WORKS
    # --------------------------------------------------------

    st.header(
        "How DSAlytics Works"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            "### 📈 Track"
        )

        st.write(
            "Collect and analyze coding submission history."
        )

    with col2:

        st.markdown(
            "### 🔎 Analyze"
        )

        st.write(
            "Measure topic accuracy, failures and practice recency."
        )

    with col3:

        st.markdown(
            "### 🤖 Predict"
        )

        st.write(
            "Use machine learning to estimate future failure risk."
        )

    with col4:

        st.markdown(
            "### 🎯 Recommend"
        )

        st.write(
            "Prioritize topics that need the most attention."
        )

    st.divider()

    # --------------------------------------------------------
    # CORE FEATURES
    # --------------------------------------------------------

    st.header(
        "Core Features"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 📈 Performance Analytics"
        )

        st.write(
            "Understand acceptance rate, failure rate and "
            "performance across different difficulty levels."
        )

        st.markdown(
            "### 🔥 Weakness Detection"
        )

        st.write(
            "Identify topics where repeated failures or poor "
            "recent performance indicate learning gaps."
        )

    with col2:

        st.markdown(
            "### 🤖 ML-Based Prediction"
        )

        st.write(
            "Machine learning models use historical and recent "
            "submission patterns to estimate failure probability."
        )

        st.markdown(
            "### 🎯 Personalized Practice"
        )

        st.write(
            "Combine weakness scores with ML predictions to "
            "prioritize the next topics to practice."
        )

    st.divider()

    st.info(
        "Use the navigation panel to explore your performance, "
        "weaknesses, machine-learning insights and personalized recommendations."
    )


# ============================================================
# PERFORMANCE
# ============================================================

elif page == "📈 Performance":

    st.title(
        "📈 Performance Analytics"
    )

    st.write(
        "Analyze submission outcomes and historical performance."
    )

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Submissions",
        total_submissions
    )

    col2.metric(
        "Acceptance Rate",
        f"{acceptance_rate:.1f}%"
    )

    col3.metric(
        "Failure Rate",
        f"{failure_rate:.1f}%"
    )

    st.divider()

    # --------------------------------------------------------
    # ACCEPTED VS FAILED
    # --------------------------------------------------------

    st.header(
        "Submission Outcomes"
    )

    performance_data = pd.DataFrame(
        {
            "Status": [
                "Accepted",
                "Failed"
            ],
            "Count": [
                accepted,
                failed
            ]
        }
    )

    fig = px.pie(
        performance_data,
        names="Status",
        values="Count",
        title="Accepted vs Failed Submissions"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # DIFFICULTY ANALYSIS
    # --------------------------------------------------------

    st.header(
        "Accuracy by Difficulty"
    )

    if "difficulty_encoded" in ml_data.columns:

        difficulty_map = {
            1: "Easy",
            2: "Medium",
            3: "Hard"
        }

        difficulty_data = (
            ml_data
            .groupby("difficulty_encoded")["target"]
            .agg(["count", "mean"])
            .reset_index()
        )

        difficulty_data["Difficulty"] = (
            difficulty_data["difficulty_encoded"]
            .map(difficulty_map)
        )

        difficulty_data["Accuracy"] = (
            difficulty_data["mean"] * 100
        )

        fig = px.bar(
            difficulty_data,
            x="Difficulty",
            y="Accuracy",
            text="Accuracy",
            title="Acceptance Accuracy by Difficulty"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # RECENT SUBMISSIONS
    # --------------------------------------------------------

    st.header(
        "Recent Submissions"
    )

    recent = ml_data.tail(10).copy()

    if "submission_date" in recent.columns:

        recent["submission_date"] = pd.to_datetime(
            recent["submission_date"]
        )

        recent["Status"] = (
            recent["target"]
            .map(
                {
                    1: "Accepted",
                    0: "Failed"
                }
            )
        )

        fig = px.scatter(
            recent,
            x="submission_date",
            y="target",
            color="Status",
            hover_data=[
                "difficulty_encoded"
            ],
            title="Last 10 Submissions"
        )

        fig.update_yaxes(
            tickvals=[0, 1],
            ticktext=["Failed", "Accepted"]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# WEAKNESS ANALYSIS
# ============================================================

elif page == "🔥 Weakness Analysis":

    st.title(
        "🔥 Weakness Analysis"
    )

    st.write(
        "Topics are ranked using historical performance, "
        "recent performance and practice recency."
    )

    st.divider()

    # --------------------------------------------------------
    # TOP WEAKNESSES
    # --------------------------------------------------------

    st.header(
        "Highest-Risk Topics"
    )

    top_10 = (
        weakness_data
        .head(10)
        .copy()
    )

    fig = px.bar(
        top_10,
        x="adjusted_weakness_score",
        y="topic",
        orientation="h",
        title="Top 10 Weaknesses",
        text="adjusted_weakness_score"
    )

    fig.update_layout(
        yaxis=dict(
            categoryorder="total ascending"
        )
    )

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # TOP 5
    # --------------------------------------------------------

    st.header(
        "Priority Weaknesses"
    )

    priority_columns = [
        "topic",
        "attempts",
        "accuracy",
        "recent_accuracy",
        "adjusted_weakness_score"
    ]

    available_columns = [
        column
        for column in priority_columns
        if column in weakness_data.columns
    ]

    st.dataframe(
        weakness_data[
            available_columns
        ].head(5),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # FULL ANALYSIS
    # --------------------------------------------------------

    st.header(
        "Topic-Level Analysis"
    )

    st.dataframe(
        weakness_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ML INSIGHTS
# ============================================================

elif page == "🤖 ML Insights":

    st.title(
        "🤖 Machine Learning Insights"
    )

    st.write(
        "DSAlytics uses machine learning to estimate "
        "the probability of failure on future submissions."
    )

    st.divider()

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.header(
        "Most Important ML Features"
    )

    top_features = (
        feature_data
        .head(15)
        .copy()
    )

    fig = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 15 Features Influencing Prediction"
    )

    fig.update_layout(
        yaxis=dict(
            categoryorder="total ascending"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # WALK-FORWARD VALIDATION
    # --------------------------------------------------------

    st.header(
        "Walk-Forward Validation"
    )

    st.write(
        "The model is evaluated chronologically: each test window "
        "contains later submissions than the training data."
    )

    st.dataframe(
        validation_data,
        use_container_width=True,
        hide_index=True
    )

    if "roc_auc" in validation_data.columns:

        fig = px.line(
            validation_data,
            x="fold",
            y="roc_auc",
            markers=True,
            title="ROC-AUC Across Validation Folds"
        )

        fig.update_yaxes(
            range=[0, 1]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "🎯 Recommendations":

    st.title(
        "🎯 Personalized Recommendations"
    )

    st.write(
        "Topics are prioritized according to their calculated "
        "risk of future failure."
    )

    st.divider()

    # --------------------------------------------------------
    # TOP RECOMMENDATION
    # --------------------------------------------------------

    if len(recommendation_data) > 0:

        top = (
            recommendation_data
            .iloc[0]
        )

        st.subheader(
            f"🚨 Priority Topic: {top['topic']}"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Risk Score",
                f"{top['final_risk_score']:.1f}/100"
            )

        with col2:

            st.metric(
                "Historical Accuracy",
                f"{top['accuracy'] * 100:.1f}%"
            )

        with col3:

            st.metric(
                "ML Failure Risk",
                f"{top['ml_failure_probability'] * 100:.1f}%"
            )

        st.divider()

    # --------------------------------------------------------
    # RECOMMENDATION TABLE
    # --------------------------------------------------------

    st.header(
        "Recommended Practice Order"
    )

    st.dataframe(
        recommendation_data,
        use_container_width=True,
        hide_index=True
    )