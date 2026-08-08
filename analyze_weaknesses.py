import pandas as pd
import numpy as np
import os


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/master_dataset.csv"

OUTPUT_DIR = "results"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "topic_weakness_analysis.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("DSAlytics - Weakness Analysis")
print("=" * 60)

print("\nLoading master dataset...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Submission records: {len(df)}"
)


# ============================================================
# BASIC CLEANING
# ============================================================

df["submission_date"] = pd.to_datetime(
    df["submission_date"]
)

df["topics"] = (
    df["topics"]
    .fillna("")
    .astype(str)
)


# ============================================================
# EXPAND TOPICS
# ============================================================

# One submission can have multiple topics.
#
# Example:
#
# Array|Hash Table
#
# becomes:
#
# submission 1 -> Array
# submission 1 -> Hash Table

topic_rows = []


for _, row in df.iterrows():

    topics = [
        topic.strip()
        for topic in row["topics"].split("|")
        if topic.strip()
    ]

    for topic in topics:

        topic_rows.append({

            "id": row["id"],

            "title": row["title"],

            "titleSlug": row["titleSlug"],

            "topic": topic,

            "difficulty": row["difficulty"],

            "is_accepted": row["is_accepted"],

            "is_failure": row["is_failure"],

            "submission_date": row["submission_date"],

            "runtime": row["runtime"],

            "memory": row["memory"]
        })


topic_df = pd.DataFrame(
    topic_rows
)


print(
    f"Topic-level records: "
    f"{len(topic_df)}"
)


# ============================================================
# TOPIC STATISTICS
# ============================================================

topic_stats = (
    topic_df
    .groupby("topic")
    .agg(

        attempts=(
            "id",
            "count"
        ),

        accepted=(
            "is_accepted",
            "sum"
        ),

        failures=(
            "is_failure",
            "sum"
        ),

        unique_problems=(
            "titleSlug",
            "nunique"
        ),

        avg_difficulty=(
            "difficulty",
            lambda x:
                x.map({
                    "Easy": 1,
                    "Medium": 2,
                    "Hard": 3
                }).mean()
        ),

        last_practiced=(
            "submission_date",
            "max"
        )
    )
    .reset_index()
)


# ============================================================
# ACCURACY
# ============================================================

topic_stats["accuracy"] = (
    topic_stats["accepted"]
    /
    topic_stats["attempts"]
)


topic_stats["failure_rate"] = (
    topic_stats["failures"]
    /
    topic_stats["attempts"]
)


# ============================================================
# RECENCY
# ============================================================

latest_date = (
    topic_df["submission_date"]
    .max()
)


topic_stats["days_since_practice"] = (
    latest_date
    -
    topic_stats["last_practiced"]
).dt.total_seconds() / 86400


# ============================================================
# RECENT PERFORMANCE
# ============================================================

# We calculate the performance from the most recent
# 5 attempts for each topic.

recent_topic_records = []


for topic, group in topic_df.groupby("topic"):

    group = group.sort_values(
        "submission_date"
    )

    recent = group.tail(5)

    recent_attempts = len(
        recent
    )

    recent_accepted = (
        recent["is_accepted"]
        .sum()
    )

    recent_accuracy = (
        recent_accepted
        /
        recent_attempts
        if recent_attempts > 0
        else 0
    )

    recent_topic_records.append({

        "topic": topic,

        "recent_attempts": recent_attempts,

        "recent_accuracy": recent_accuracy,

        "recent_failures": (
            recent["is_failure"]
            .sum()
        )
    })


recent_df = pd.DataFrame(
    recent_topic_records
)


topic_stats = topic_stats.merge(
    recent_df,
    on="topic",
    how="left"
)


# ============================================================
# WEAKNESS SCORE
# ============================================================

# We combine several signals:
#
# 1. Overall failure rate
# 2. Recent failure rate
# 3. Lack of recent practice
# 4. Number of attempts
#
# Higher score = greater weakness.
#
# This is NOT an ML prediction yet.
# It is our analytical baseline.


# Failure component

failure_component = (
    topic_stats["failure_rate"]
    * 100
)


# Recent failure component

recent_failure_component = (
    (1 - topic_stats["recent_accuracy"])
    * 100
)


# Recency component
#
# Cap at 30 days so extremely old topics
# don't completely dominate the score.

recency_component = (
    np.minimum(
        topic_stats["days_since_practice"],
        30
    )
    / 30
    * 100
)


# Difficulty component

difficulty_component = (
    (topic_stats["avg_difficulty"] - 1)
    / 2
    * 100
)


# ============================================================
# COMBINED SCORE
# ============================================================

topic_stats["weakness_score"] = (

    0.40 * failure_component

    +

    0.35 * recent_failure_component

    +

    0.15 * recency_component

    +

    0.10 * difficulty_component
)


# ============================================================
# CONFIDENCE
# ============================================================

# A topic with 2 attempts should not be treated
# as equally reliable as a topic with 30 attempts.

topic_stats["confidence"] = np.minimum(
    topic_stats["attempts"] / 10,
    1
)


# ============================================================
# ADJUSTED WEAKNESS SCORE
# ============================================================

# Shrink scores for topics with very little evidence.

topic_stats["adjusted_weakness_score"] = (

    topic_stats["weakness_score"]
    *
    (
        0.5
        +
        0.5 * topic_stats["confidence"]
    )
)


# ============================================================
# SORT
# ============================================================

topic_stats = topic_stats.sort_values(
    "adjusted_weakness_score",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# ROUND VALUES
# ============================================================

numeric_columns = [
    "accuracy",
    "failure_rate",
    "avg_difficulty",
    "days_since_practice",
    "recent_accuracy",
    "weakness_score",
    "confidence",
    "adjusted_weakness_score"
]


for column in numeric_columns:

    topic_stats[column] = (
        topic_stats[column]
        .round(3)
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

topic_stats.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 60)

print("TOPIC WEAKNESS RANKING")

print("=" * 60)


display_columns = [

    "topic",

    "attempts",

    "failures",

    "accuracy",

    "recent_accuracy",

    "days_since_practice",

    "unique_problems",

    "adjusted_weakness_score"
]


print(
    topic_stats[
        display_columns
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# TOP 5 WEAKNESSES
# ============================================================

print("\n" + "=" * 60)

print("TOP 5 CURRENT WEAKNESSES")

print("=" * 60)


top_5 = topic_stats.head(5)


for rank, (_, row) in enumerate(
    top_5.iterrows(),
    start=1
):

    print(
        f"{rank}. "
        f"{row['topic']} "
        f"→ "
        f"score={row['adjusted_weakness_score']:.1f}, "
        f"accuracy={row['accuracy']:.1%}, "
        f"attempts={int(row['attempts'])}"
    )


# ============================================================
# OUTPUT
# ============================================================

print(
    f"\nAnalysis saved to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nWeakness analysis complete."
)