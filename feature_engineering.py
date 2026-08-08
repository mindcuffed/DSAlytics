import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/master_dataset.csv"

OUTPUT_DIR = "data/processed"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "ml_dataset.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("DSAlytics - Leakage-Free Feature Engineering")
print("=" * 60)

print("\nLoading master dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} submission records.")

# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df["submission_date"] = pd.to_datetime(
    df["submission_date"]
)

df = (
    df
    .sort_values("submission_date")
    .reset_index(drop=True)
)

# ============================================================
# BASIC FEATURES
# ============================================================

print("\nCreating basic features...")

# Difficulty encoding

difficulty_map = {
    "Easy": 1,
    "Medium": 2,
    "Hard": 3
}

df["difficulty_encoded"] = (
    df["difficulty"]
    .map(difficulty_map)
    .fillna(0)
)

# ============================================================
# HISTORICAL GLOBAL PERFORMANCE
# ============================================================

# IMPORTANT:
#
# Every feature below describes the user's history BEFORE
# the current submission.
#
# The current submission must never influence its own features.
# ============================================================

df["previous_total_submissions"] = (
    df.index
)

df["previous_accepted"] = (
    df["is_accepted"]
    .shift(1)
    .fillna(0)
    .cumsum()
)

df["previous_failures"] = (
    df["is_failure"]
    .shift(1)
    .fillna(0)
    .cumsum()
)

df["historical_accuracy"] = np.where(
    df["previous_total_submissions"] > 0,
    df["previous_accepted"]
    / df["previous_total_submissions"],
    0.5
)

# ============================================================
# PREVIOUS ATTEMPTS ON SAME PROBLEM
# ============================================================

df["previous_problem_attempts"] = (
    df
    .groupby("titleSlug")
    .cumcount()
)

df["previous_problem_failures"] = (
    df
    .groupby("titleSlug")["is_failure"]
    .transform(
        lambda x:
        x.shift(1)
        .fillna(0)
        .cumsum()
    )
)

# ============================================================
# RECENT GLOBAL PERFORMANCE
# ============================================================

# Previous 10 submissions only.
#
# shift(1) is critical because the current submission
# must not be included.

df["recent_10_accuracy"] = (
    df["is_accepted"]
    .shift(1)
    .rolling(
        window=10,
        min_periods=1
    )
    .mean()
    .fillna(0.5)
)

df["recent_10_failures"] = (
    df["is_failure"]
    .shift(1)
    .rolling(
        window=10,
        min_periods=1
    )
    .sum()
    .fillna(0)
)

# ============================================================
# TOPIC PROCESSING
# ============================================================

df["topics"] = (
    df["topics"]
    .fillna("")
    .astype(str)
)

df["topic_list"] = (
    df["topics"]
    .apply(
        lambda x: [
            topic.strip()
            for topic in x.split("|")
            if topic.strip()
        ]
    )
)

# ============================================================
# HISTORICAL TOPIC FEATURES
# ============================================================

print("\nCalculating historical topic features...")

topic_history = {}

topic_accuracy_features = []
topic_attempt_features = []
topic_failure_features = []
topic_recency_features = []

for _, row in df.iterrows():

    topics = row["topic_list"]

    current_date = row["submission_date"]

    accuracy_values = []
    attempt_values = []
    failure_values = []
    recency_values = []

    # --------------------------------------------------------
    # READ ONLY PREVIOUS TOPIC HISTORY
    # --------------------------------------------------------

    for topic in topics:

        if topic not in topic_history:

            accuracy_values.append(0.5)

            attempt_values.append(0)

            failure_values.append(0)

            recency_values.append(-1)

            continue

        history = topic_history[topic]

        attempts = history["attempts"]

        accepted = history["accepted"]

        failures = history["failures"]

        last_date = history["last_date"]

        # Historical accuracy

        if attempts > 0:

            accuracy = (
                accepted / attempts
            )

        else:

            accuracy = 0.5

        accuracy_values.append(
            accuracy
        )

        # Historical attempts

        attempt_values.append(
            attempts
        )

        # Historical failures

        failure_values.append(
            failures
        )

        # Days since topic practice

        if last_date is not None:

            days = (
                current_date - last_date
            ).total_seconds() / 86400

            recency_values.append(
                days
            )

        else:

            recency_values.append(-1)

    # --------------------------------------------------------
    # AGGREGATE TOPIC HISTORY
    # --------------------------------------------------------

    if accuracy_values:

        topic_accuracy_features.append(
            np.mean(accuracy_values)
        )

        topic_attempt_features.append(
            np.sum(attempt_values)
        )

        topic_failure_features.append(
            np.sum(failure_values)
        )

        valid_recency = [
            value
            for value in recency_values
            if value >= 0
        ]

        if valid_recency:

            topic_recency_features.append(
                min(valid_recency)
            )

        else:

            topic_recency_features.append(
                -1
            )

    else:

        topic_accuracy_features.append(
            0.5
        )

        topic_attempt_features.append(
            0
        )

        topic_failure_features.append(
            0
        )

        topic_recency_features.append(
            -1
        )

    # --------------------------------------------------------
    # UPDATE HISTORY AFTER FEATURE CALCULATION
    # --------------------------------------------------------
    #
    # This is essential.
    #
    # The current submission is added to history ONLY AFTER
    # its features have been calculated.
    #
    # Therefore there is no target leakage here.
    # --------------------------------------------------------

    for topic in topics:

        if topic not in topic_history:

            topic_history[topic] = {
                "attempts": 0,
                "accepted": 0,
                "failures": 0,
                "last_date": None
            }

        topic_history[topic]["attempts"] += 1

        if row["is_accepted"] == 1:

            topic_history[topic]["accepted"] += 1

        else:

            topic_history[topic]["failures"] += 1

        topic_history[topic]["last_date"] = (
            current_date
        )

# ============================================================
# ADD HISTORICAL TOPIC FEATURES
# ============================================================

df["historical_topic_accuracy"] = (
    topic_accuracy_features
)

df["historical_topic_attempts"] = (
    topic_attempt_features
)

df["historical_topic_failures"] = (
    topic_failure_features
)

df["days_since_topic_practice"] = (
    topic_recency_features
)

# ============================================================
# ONE-HOT ENCODE TOPICS
# ============================================================

print("\nCreating topic features...")

all_topics = sorted(
    set(
        topic
        for topics in df["topic_list"]
        for topic in topics
    )
)

for topic in all_topics:

    column_name = (
        "topic_"
        + topic.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )

    df[column_name] = (
        df["topic_list"]
        .apply(
            lambda topics:
            int(topic in topics)
        )
    )

print(
    f"Topics found: {len(all_topics)}"
)

# ============================================================
# ONE-HOT ENCODE DIFFICULTY
# ============================================================

difficulty_dummies = pd.get_dummies(
    df["difficulty"],
    prefix="difficulty",
    dtype=int
)

df = pd.concat(
    [
        df,
        difficulty_dummies
    ],
    axis=1
)

# ============================================================
# TARGET
# ============================================================

# Target:
#
# 1 = current submission accepted
# 0 = current submission failed
#
# This is what the model is supposed to predict.
#
# IMPORTANT:
# is_accepted / is_failure are retained in the dataset
# for analysis, BUT train_model.py MUST remove them from X.
# ============================================================

df["target"] = (
    df["is_accepted"]
)

# ============================================================
# CLEANUP
# ============================================================

# Convert boolean columns to integers

for column in df.columns:

    if df[column].dtype == "bool":

        df[column] = (
            df[column]
            .astype(int)
        )

# Remove temporary topic list

df = df.drop(
    columns=[
        "topic_list"
    ]
)

# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 60)
print("LEAKAGE CHECK")
print("=" * 60)

dangerous_columns = [
    "is_accepted",
    "is_failure",
    "target",
    "solved"
]

print(
    "\nThese columns must NEVER be used as ML features:"
)

for column in dangerous_columns:

    if column in df.columns:

        print(f"  - {column}")

# ============================================================
# SAVE DATASET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)

print(
    f"\nRows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)

print(
    f"Target = Accepted: "
    f"{df['target'].sum()}"
)

print(
    f"Target = Failed: "
    f"{(df['target'] == 0).sum()}"
)

print(
    "\nML dataset saved to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nFeature engineering complete."
)