import pandas as pd
import os


# ============================================================
# FILE PATHS
# ============================================================

SUBMISSIONS_FILE = "data/raw/leetcode_submissions.csv"
METADATA_FILE = "data/raw/problem_metadata.csv"

OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "master_dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("DSAlytics - Dataset Preparation")
print("=" * 60)

print("\nLoading submission data...")

submissions = pd.read_csv(
    SUBMISSIONS_FILE
)

print(
    f"Submissions loaded: {len(submissions)}"
)


print("\nLoading problem metadata...")

metadata = pd.read_csv(
    METADATA_FILE
)

print(
    f"Problems loaded: {len(metadata)}"
)


# ============================================================
# SELECT RELEVANT METADATA COLUMNS
# ============================================================

metadata = metadata[
    [
        "titleSlug",
        "difficulty",
        "topics"
    ]
]


# ============================================================
# MERGE SUBMISSIONS + METADATA
# ============================================================

print("\nMerging datasets...")

master = submissions.merge(
    metadata,
    on="titleSlug",
    how="left"
)


# ============================================================
# CONVERT TIMESTAMP
# ============================================================

master["timestamp"] = pd.to_numeric(
    master["timestamp"],
    errors="coerce"
)

master["submission_date"] = pd.to_datetime(
    master["timestamp"],
    unit="s"
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

master = master.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


# ============================================================
# CREATE SUCCESS / FAILURE FEATURES
# ============================================================

master["is_accepted"] = (
    master["statusDisplay"] == "Accepted"
).astype(int)


master["is_failure"] = (
    master["statusDisplay"] != "Accepted"
).astype(int)


# ============================================================
# ATTEMPT NUMBER
# ============================================================

master["attempt_number"] = (
    master
    .groupby("titleSlug")
    .cumcount()
    + 1
)


# ============================================================
# PROBLEM SOLVED EVENT
# ============================================================

# True only for the first Accepted submission
# for a particular problem.

master["solved"] = (
    master
    .groupby("titleSlug")["is_accepted"]
    .cumsum()
    .eq(1)
    &
    (master["is_accepted"] == 1)
).astype(int)


# ============================================================
# CLEAN TOPICS
# ============================================================

master["topics"] = (
    master["topics"]
    .fillna("")
)


# ============================================================
# DAYS SINCE PREVIOUS SUBMISSION OF SAME PROBLEM
# ============================================================

master["previous_submission_date"] = (
    master
    .groupby("titleSlug")["submission_date"]
    .shift(1)
)


master["days_since_previous_attempt"] = (
    master["submission_date"]
    - master["previous_submission_date"]
).dt.total_seconds() / 86400


master["days_since_previous_attempt"] = (
    master["days_since_previous_attempt"]
    .fillna(-1)
)


# ============================================================
# REMOVE TEMPORARY COLUMN
# ============================================================

master = master.drop(
    columns=["previous_submission_date"]
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SAVE MASTER DATASET
# ============================================================

master.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DATASET CREATED")
print("=" * 60)

print(
    f"\nTotal submission records: {len(master)}"
)

print(
    f"Unique problems: "
    f"{master['titleSlug'].nunique()}"
)

print(
    f"Accepted submissions: "
    f"{master['is_accepted'].sum()}"
)

print(
    f"Failed submissions: "
    f"{master['is_failure'].sum()}"
)

print(
    f"\nDifficulty distribution:"
)

print(
    master["difficulty"]
    .value_counts()
)

print(
    f"\nOutput file:"
)

print(
    OUTPUT_FILE
)

print("\nDataset preparation complete.")