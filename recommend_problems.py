import pandas as pd
import numpy as np
import json
import os


# ============================================================
# DSAlytics - Personalized Problem Recommender
# ============================================================

MASTER_DATASET = "data/processed/master_dataset.csv"
PROBLEM_CATALOG = "data/raw/leetcode_problems.json"

OUTPUT_FILE = "results/problem_recommendations.csv"

TOP_N = 10

# Maximum number of recommendations whose PRIMARY weakness
# is the same topic.
MAX_PER_WEAKNESS = 3


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("DSAlytics - Personalized Problem Recommender")
    print("=" * 60)

    print("\nLoading submission dataset...")

    df = pd.read_csv(MASTER_DATASET)

    print(f"Submission records: {len(df)}")

    print("\nLoading LeetCode problem catalog...")

    with open(
        PROBLEM_CATALOG,
        "r",
        encoding="utf-8"
    ) as file:

        problems = json.load(file)

    problems_df = pd.DataFrame(problems)

    print(
        f"Catalog problems: {len(problems_df)}"
    )

    return df, problems_df


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(df, candidates):

    for column in candidates:

        if column in df.columns:

            return column

    return None


# ============================================================
# SPLIT TOPICS
# ============================================================

def split_topics(value):

    if pd.isna(value):

        return []

    value = str(value).strip()

    if not value:

        return []

    # Dataset uses |
    if "|" in value:

        return [
            x.strip()
            for x in value.split("|")
            if x.strip()
        ]

    # Also support comma-separated topics
    if "," in value:

        return [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]

    return [value]


# ============================================================
# NORMALIZE TOPIC
# ============================================================

def normalize_topic(topic):

    return (
        str(topic)
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
        .replace("–", " ")
        .replace("—", " ")
    )


# ============================================================
# BUILD WEAKNESS PROFILE
# ============================================================

def build_weakness_profile(df):

    print("\nAnalyzing your weaknesses...")

    topic_column = find_column(
        df,
        [
            "topic",
            "topics",
            "topic_name",
            "Topic",
            "Topics"
        ]
    )

    status_column = find_column(
        df,
        [
            "statusDisplay",
            "status",
            "Status",
            "result"
        ]
    )

    if topic_column is None:

        raise ValueError(
            "Could not find topic column."
        )

    if status_column is None:

        raise ValueError(
            "Could not find status column."
        )

    df = df.copy()

    # --------------------------------------------------------
    # SUCCESS FLAG
    # --------------------------------------------------------

    df["success"] = (
        df[status_column]
        .astype(str)
        .str.lower()
        .eq("accepted")
        .astype(int)
    )

    # --------------------------------------------------------
    # SORT CHRONOLOGICALLY
    # --------------------------------------------------------

    if "submission_date" in df.columns:

        df["submission_date"] = pd.to_datetime(
            df["submission_date"],
            errors="coerce"
        )

        df = df.sort_values(
            "submission_date"
        ).reset_index(
            drop=True
        )

    # --------------------------------------------------------
    # EXPAND TOPICS
    # --------------------------------------------------------

    topic_records = []

    for _, row in df.iterrows():

        topics = split_topics(
            row[topic_column]
        )

        for topic in topics:

            topic_records.append({

                "topic": topic,

                "topic_normalized":
                    normalize_topic(topic),

                "success":
                    int(row["success"])

            })

    topic_df = pd.DataFrame(
        topic_records
    )

    if topic_df.empty:

        raise ValueError(
            "No topic information found."
        )

    # --------------------------------------------------------
    # HISTORICAL PERFORMANCE
    # --------------------------------------------------------

    profile = (
        topic_df
        .groupby(
            [
                "topic",
                "topic_normalized"
            ]
        )
        .agg(
            attempts=("success", "count"),
            accuracy=("success", "mean")
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # RECENT PERFORMANCE
    # --------------------------------------------------------

    recent_count = max(
        20,
        int(len(df) * 0.20)
    )

    recent_submissions = df.tail(
        recent_count
    )

    recent_records = []

    for _, row in recent_submissions.iterrows():

        topics = split_topics(
            row[topic_column]
        )

        for topic in topics:

            recent_records.append({

                "topic_normalized":
                    normalize_topic(topic),

                "success":
                    int(row["success"])

            })

    recent_topics = pd.DataFrame(
        recent_records
    )

    if not recent_topics.empty:

        recent_profile = (
            recent_topics
            .groupby("topic_normalized")
            .agg(
                recent_attempts=("success", "count"),
                recent_accuracy=("success", "mean")
            )
            .reset_index()
        )

        profile = profile.merge(
            recent_profile,
            on="topic_normalized",
            how="left"
        )

    else:

        profile["recent_attempts"] = 0
        profile["recent_accuracy"] = (
            profile["accuracy"]
        )

    profile["recent_accuracy"] = (
        profile["recent_accuracy"]
        .fillna(profile["accuracy"])
    )

    # --------------------------------------------------------
    # WEAKNESS SCORE
    # --------------------------------------------------------
    #
    # Historical accuracy       -> 60%
    # Recent accuracy           -> 30%
    # Repeated exposure         -> 10%
    #
    # Lower accuracy = higher weakness.
    #

    profile["weakness_score"] = (

        (1 - profile["accuracy"]) * 60

        +

        (1 - profile["recent_accuracy"]) * 30

        +

        np.minimum(
            profile["attempts"],
            20
        ) / 20 * 10

    )

    profile = profile.sort_values(
        "weakness_score",
        ascending=False
    )

    print("\nTop individual weaknesses:")

    print(
        profile[
            [
                "topic",
                "attempts",
                "accuracy",
                "recent_accuracy",
                "weakness_score"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    return profile


# ============================================================
# GET ATTEMPTED PROBLEMS
# ============================================================

def get_attempted_problems(df):

    slug_column = find_column(
        df,
        [
            "titleSlug",
            "title_slug",
            "problem_slug",
            "slug"
        ]
    )

    if slug_column:

        return set(
            df[slug_column]
            .dropna()
            .astype(str)
            .str.lower()
        )

    return set()


# ============================================================
# CREATE WEAKNESS MAP
# ============================================================

def create_weakness_map(profile):

    weakness_map = {}

    for _, row in profile.iterrows():

        weakness_map[
            normalize_topic(
                row["topic"]
            )
        ] = {

            "topic":
                row["topic"],

            "weakness_score":
                float(
                    row["weakness_score"]
                ),

            "accuracy":
                float(
                    row["accuracy"]
                ),

            "recent_accuracy":
                float(
                    row["recent_accuracy"]
                ),

            "attempts":
                int(
                    row["attempts"]
                )
        }

    return weakness_map


# ============================================================
# SCORE DIFFICULTY
# ============================================================

def difficulty_score(difficulty):

    difficulty = str(
        difficulty
    ).upper()

    if difficulty == "EASY":

        return 8

    if difficulty == "MEDIUM":

        return 15

    if difficulty == "HARD":

        return 5

    return 0


# ============================================================
# DIFFICULTY ADJUSTMENT
# ============================================================

def difficulty_adjustment(
    problem,
    strongest
):

    difficulty = str(
        problem.get(
            "difficulty",
            ""
        )
    ).upper()

    accuracy = strongest[
        "accuracy"
    ]

    # If accuracy is extremely low,
    # do not push Hard problems.
    if (
        difficulty == "HARD"
        and accuracy < 0.30
    ):

        return -20

    # If user is struggling,
    # Easy problems are useful for rebuilding
    # the underlying pattern.
    if (
        difficulty == "EASY"
        and accuracy < 0.30
    ):

        return 5

    return 0


# ============================================================
# SCORE PROBLEM
# ============================================================

def score_problem(
    problem,
    weakness_map
):

    topics = problem.get(
        "topics",
        []
    )

    if not isinstance(
        topics,
        list
    ):

        topics = []

    matched = []

    for topic in topics:

        key = normalize_topic(
            topic
        )

        if key in weakness_map:

            matched.append(
                weakness_map[key]
            )

    if not matched:

        return None

    # --------------------------------------------------------
    # SORT BY WEAKNESS
    # --------------------------------------------------------

    matched.sort(
        key=lambda x:
            x["weakness_score"],
        reverse=True
    )

    strongest = matched[0]

    # --------------------------------------------------------
    # BASE SCORE
    # --------------------------------------------------------

    score = (
        strongest["weakness_score"]
    )

    # --------------------------------------------------------
    # MULTIPLE WEAK TOPICS
    # --------------------------------------------------------

    if len(matched) >= 2:

        score += 8

    if len(matched) >= 3:

        score += 5

    # --------------------------------------------------------
    # DIFFICULTY
    # --------------------------------------------------------

    score += difficulty_score(
        problem.get(
            "difficulty",
            ""
        )
    )

    score += difficulty_adjustment(
        problem,
        strongest
    )

    # --------------------------------------------------------
    # RECENT PERFORMANCE
    # --------------------------------------------------------

    # If recent accuracy is substantially worse
    # than historical accuracy, increase priority.

    recent_accuracy = strongest[
        "recent_accuracy"
    ]

    historical_accuracy = strongest[
        "accuracy"
    ]

    if recent_accuracy < historical_accuracy:

        score += 5

    # --------------------------------------------------------
    # REPEATED ATTEMPTS
    # --------------------------------------------------------

    if strongest["attempts"] >= 20:

        score += 3

    elif strongest["attempts"] >= 10:

        score += 2

    # --------------------------------------------------------
    # CAP
    # --------------------------------------------------------

    score = min(
        max(score, 0),
        100
    )

    return {

        "score":
            round(
                score,
                2
            ),

        "matched_topics":
            [
                x["topic"]
                for x in matched
            ],

        "strongest_topic":
            strongest["topic"],

        "strongest_accuracy":
            strongest["accuracy"],

        "recent_accuracy":
            strongest["recent_accuracy"],

        "reason":
            (
                "Targets your weakness in "
                +
                strongest["topic"]
                +
                " "
                +
                f"(historical accuracy "
                +
                f"{strongest['accuracy']:.1%})"
            )
    }


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    df,
    problems_df,
    weakness_profile
):

    print(
        "\nGenerating recommendations..."
    )

    # --------------------------------------------------------
    # ATTEMPTED PROBLEMS
    # --------------------------------------------------------

    attempted = get_attempted_problems(
        df
    )

    print(
        f"Problems already attempted: "
        f"{len(attempted)}"
    )

    # --------------------------------------------------------
    # WEAKNESS MAP
    # --------------------------------------------------------

    weakness_map = create_weakness_map(
        weakness_profile
    )

    recommendations = []

    # --------------------------------------------------------
    # SCORE ALL CATALOG PROBLEMS
    # --------------------------------------------------------

    for _, problem in problems_df.iterrows():

        slug = str(
            problem.get(
                "titleSlug",
                ""
            )
        ).lower()

        # ----------------------------------------------------
        # Skip invalid problems
        # ----------------------------------------------------

        if not slug:

            continue

        # ----------------------------------------------------
        # Skip already attempted problems
        # ----------------------------------------------------

        if slug in attempted:

            continue

        # ----------------------------------------------------
        # Skip Premium problems
        # ----------------------------------------------------

        if problem.get(
            "isPaidOnly",
            False
        ):

            continue

        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        result = score_problem(
            problem,
            weakness_map
        )

        if result is None:

            continue

        recommendations.append({

            "title":
                problem.get(
                    "title",
                    ""
                ),

            "titleSlug":
                problem.get(
                    "titleSlug",
                    ""
                ),

            "difficulty":
                problem.get(
                    "difficulty",
                    ""
                ),

            "topics":
                ", ".join(
                    problem.get(
                        "topics",
                        []
                    )
                ),

            "recommendation_score":
                result["score"],

            "matched_topics":
                ", ".join(
                    result[
                        "matched_topics"
                    ]
                ),

            "primary_weakness":
                result[
                    "strongest_topic"
                ],

            "historical_accuracy":
                round(
                    result[
                        "strongest_accuracy"
                    ],
                    3
                ),

            "recent_accuracy":
                round(
                    result[
                        "recent_accuracy"
                    ],
                    3
                ),

            "reason":
                result["reason"]
        })

    recommendations_df = pd.DataFrame(
        recommendations
    )

    if recommendations_df.empty:

        return recommendations_df

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    recommendations_df = (
        recommendations_df
        .sort_values(
            [
                "recommendation_score",
                "historical_accuracy"
            ],
            ascending=[
                False,
                True
            ]
        )
        .drop_duplicates(
            subset=[
                "titleSlug"
            ]
        )
    )

    # ========================================================
    # DIVERSITY SELECTION
    # ========================================================
    #
    # Instead of simply taking the top 10,
    # limit each primary weakness.
    #
    # This prevents:
    #
    # Knapsack
    # Knapsack
    # Knapsack
    # Knapsack
    #
    # from dominating the entire recommendation list.
    #

    selected = []

    topic_counts = {}

    for _, row in recommendations_df.iterrows():

        primary = row[
            "primary_weakness"
        ]

        count = topic_counts.get(
            primary,
            0
        )

        if count >= MAX_PER_WEAKNESS:

            continue

        selected.append(
            row
        )

        topic_counts[
            primary
        ] = count + 1

        if len(selected) >= TOP_N:

            break

    recommendations_df = pd.DataFrame(
        selected
    )

    return recommendations_df


# ============================================================
# DISPLAY RECOMMENDATIONS
# ============================================================

def display_recommendations(
    recommendations
):

    print("\n")

    print("=" * 60)

    print(
        "YOUR PERSONALIZED NEXT PROBLEMS"
    )

    print("=" * 60)

    if recommendations.empty:

        print(
            "\nNo recommendations found."
        )

        return

    for i, row in recommendations.iterrows():

        print(
            f"\n{i + 1}. "
            f"{row['title']}"
        )

        print(
            f"   Difficulty       : "
            f"{row['difficulty']}"
        )

        print(
            f"   Topics           : "
            f"{row['topics']}"
        )

        print(
            f"   Primary Weakness : "
            f"{row['primary_weakness']}"
        )

        print(
            f"   Your Accuracy    : "
            f"{row['historical_accuracy']:.1%}"
        )

        print(
            f"   Recent Accuracy  : "
            f"{row['recent_accuracy']:.1%}"
        )

        print(
            f"   Score            : "
            f"{row['recommendation_score']}/100"
        )

        print(
            f"   Reason           : "
            f"{row['reason']}"
        )

        print(
            f"   LeetCode         : "
            f"https://leetcode.com/problems/"
            f"{row['titleSlug']}/"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    recommendations
):

    os.makedirs(
        "results",
        exist_ok=True
    )

    recommendations.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nRecommendations saved to:"
    )

    print(
        f"✓ {OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df, problems_df = load_data()

    weakness_profile = (
        build_weakness_profile(
            df
        )
    )

    recommendations = (
        generate_recommendations(
            df,
            problems_df,
            weakness_profile
        )
    )

    display_recommendations(
        recommendations
    )

    save_results(
        recommendations
    )

    print(
        "\nRecommendation generation complete."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()