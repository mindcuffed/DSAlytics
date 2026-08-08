import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# CONFIGURATION
# ============================================================

WEAKNESS_FILE = (
    "results/topic_weakness_analysis.csv"
)

ML_DATASET_FILE = (
    "data/processed/ml_dataset.csv"
)

MODEL_FILE = (
    "models/best_model.joblib"
)

FEATURE_FILE = (
    "models/feature_names.json"
)

OUTPUT_DIR = "results"

OUTPUT_FILE = (
    "results/final_recommendations.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("DSAlytics - Personalized Practice Recommender")
print("=" * 60)


print("\nLoading weakness analysis...")

weakness_df = pd.read_csv(
    WEAKNESS_FILE
)

print(
    f"Topics loaded: {len(weakness_df)}"
)


print("\nLoading ML dataset...")

ml_df = pd.read_csv(
    ML_DATASET_FILE
)

print(
    f"ML records loaded: {len(ml_df)}"
)


print("\nLoading trained model...")

model = joblib.load(
    MODEL_FILE
)

print(
    "Model loaded successfully."
)


# ============================================================
# LOAD FEATURE NAMES
# ============================================================

import json


with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    feature_names = json.load(f)


# ============================================================
# PREPARE ML DATA
# ============================================================

columns_to_remove = [
    "target",
    "is_accepted",
    "is_failure",

    "id",
    "title",
    "titleSlug",
    "topics",

    "statusDisplay",
    "lang",
    "runtime",
    "memory",

    "timestamp",
    "submission_date"
]


X = ml_df.drop(
    columns=[
        column
        for column in columns_to_remove
        if column in ml_df.columns
    ]
)


X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)

X = X.select_dtypes(
    include=["number"]
)


# ============================================================
# ALIGN FEATURES WITH TRAINED MODEL
# ============================================================

for feature in feature_names:

    if feature not in X.columns:

        X[feature] = 0


X = X[
    feature_names
]


# ============================================================
# CALCULATE ML FAILURE PROBABILITY
# ============================================================

ml_df["success_probability"] = (
    model.predict_proba(X)[:, 1]
)


ml_df["failure_probability"] = (
    1
    -
    ml_df["success_probability"]
)


# ============================================================
# CALCULATE TOPIC-LEVEL ML RISK
# ============================================================

ml_df["topics"] = (
    ml_df["topics"]
    .fillna("")
    .astype(str)
)


topic_ml_records = []


for _, row in ml_df.iterrows():

    topics = [

        topic.strip()

        for topic in row["topics"].split("|")

        if topic.strip()

    ]


    for topic in topics:

        topic_ml_records.append({

            "topic": topic,

            "failure_probability":
                row["failure_probability"],

            "submission_date":
                row["submission_date"]
        })


topic_ml_df = pd.DataFrame(
    topic_ml_records
)


# ============================================================
# AGGREGATE ML RISK BY TOPIC
# ============================================================

topic_ml_risk = (

    topic_ml_df

    .groupby("topic")

    .agg(

        ml_failure_probability=(
            "failure_probability",
            "mean"
        )

    )

    .reset_index()

)


# ============================================================
# MERGE ANALYTICAL + ML RISK
# ============================================================

final_df = weakness_df.merge(

    topic_ml_risk,

    on="topic",

    how="left"

)


final_df[
    "ml_failure_probability"
] = (

    final_df[
        "ml_failure_probability"
    ]

    .fillna(0.5)

)


# ============================================================
# NORMALIZE ANALYTICAL WEAKNESS SCORE
# ============================================================

max_score = (

    final_df[
        "adjusted_weakness_score"
    ]

    .max()
)


if max_score > 0:

    final_df[
        "normalized_weakness"
    ] = (

        final_df[
            "adjusted_weakness_score"
        ]

        / max_score

    )

else:

    final_df[
        "normalized_weakness"
    ] = 0


# ============================================================
# FINAL RISK SCORE
# ============================================================

# 60% historical / analytical weakness
# 40% ML predicted failure probability

final_df[
    "final_risk_score"
] = (

    0.60
    *
    final_df[
        "normalized_weakness"
    ]

    +

    0.40
    *
    final_df[
        "ml_failure_probability"
    ]

)


# Convert to 0-100

final_df[
    "final_risk_score"
] = (

    final_df[
        "final_risk_score"
    ]

    * 100

)


# ============================================================
# RISK CATEGORY
# ============================================================

def get_risk_category(score):

    if score >= 70:

        return "HIGH"

    elif score >= 45:

        return "MEDIUM"

    else:

        return "LOW"


final_df[
    "risk_level"
] = (

    final_df[
        "final_risk_score"
    ]

    .apply(
        get_risk_category
    )

)


# ============================================================
# RECOMMENDATION
# ============================================================

def generate_recommendation(row):

    score = row[
        "final_risk_score"
    ]

    accuracy = row[
        "accuracy"
    ]

    attempts = row[
        "attempts"
    ]

    recent_accuracy = row[
        "recent_accuracy"
    ]

    days = row[
        "days_since_practice"
    ]


    reasons = []


    # Historical performance

    if accuracy < 0.50:

        reasons.append(
            "very low historical accuracy"
        )

    elif accuracy < 0.70:

        reasons.append(
            "below-average historical accuracy"
        )


    # Recent performance

    if recent_accuracy < 0.50:

        reasons.append(
            "recent performance is weak"
        )


    # Repeated attempts

    if attempts >= 10:

        reasons.append(
            "repeated attempts show persistent difficulty"
        )


    # Recency

    if days >= 14:

        reasons.append(
            "topic has not been practiced recently"
        )


    # ML

    if row[
        "ml_failure_probability"
    ] >= 0.60:

        reasons.append(
            "ML model predicts high failure probability"
        )


    if not reasons:

        reasons.append(
            "moderate historical difficulty"
        )


    reason_text = "; ".join(
        reasons
    )


    if score >= 70:

        priority = "URGENT"

    elif score >= 45:

        priority = "HIGH"

    else:

        priority = "NORMAL"


    return (
        priority
        + ": "
        + reason_text
    )


final_df[
    "recommendation_reason"
] = (

    final_df

    .apply(
        generate_recommendation,
        axis=1
    )

)


# ============================================================
# SORT
# ============================================================

final_df = (

    final_df

    .sort_values(
        "final_risk_score",
        ascending=False
    )

    .reset_index(
        drop=True
    )

)


# ============================================================
# ROUND VALUES
# ============================================================

numeric_columns = [

    "accuracy",

    "failure_rate",

    "recent_accuracy",

    "days_since_practice",

    "adjusted_weakness_score",

    "ml_failure_probability",

    "normalized_weakness",

    "final_risk_score"

]


for column in numeric_columns:

    if column in final_df.columns:

        final_df[column] = (

            final_df[column]

            .round(3)

        )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY TOP WEAKNESSES
# ============================================================

print("\n" + "=" * 60)

print("FINAL PERSONALIZED WEAKNESS RANKING")

print("=" * 60)


display_columns = [

    "topic",

    "attempts",

    "accuracy",

    "recent_accuracy",

    "ml_failure_probability",

    "adjusted_weakness_score",

    "final_risk_score",

    "risk_level"

]


print(

    final_df[
        display_columns
    ]

    .head(15)

    .to_string(
        index=False
    )

)


# ============================================================
# TOP RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 60)

print("WHAT SHOULD YOU PRACTICE NEXT?")

print("=" * 60)


top_topics = final_df.head(5)


for rank, (_, row) in enumerate(

    top_topics.iterrows(),

    start=1

):

    print(
        f"\n{rank}. {row['topic']}"
    )

    print(
        f"   Risk Score       : "
        f"{row['final_risk_score']:.1f}/100"
    )

    print(
        f"   Risk Level       : "
        f"{row['risk_level']}"
    )

    print(
        f"   Accuracy         : "
        f"{row['accuracy']:.1%}"
    )

    print(
        f"   Recent Accuracy  : "
        f"{row['recent_accuracy']:.1%}"
    )

    print(
        f"   ML Failure Risk  : "
        f"{row['ml_failure_probability']:.1%}"
    )

    print(
        f"   Reason           : "
        f"{row['recommendation_reason']}"
    )


# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 60)

print("RECOMMENDER COMPLETE")

print("=" * 60)

print(
    f"\nRecommendations saved to:"
)

print(
    OUTPUT_FILE
)