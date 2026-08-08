import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/ml_dataset.csv"

N_FOLDS = 5

# Columns that must NEVER be used as ML features.
FORBIDDEN_FEATURES = [
    "target",
    "is_accepted",
    "is_failure",
    "solved",

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

# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("DSAlytics - Random Forest Walk-Forward Validation")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading ML dataset...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

if "target" not in df.columns:

    raise ValueError(
        "ERROR: target column is missing."
    )

if "submission_date" not in df.columns:

    raise ValueError(
        "ERROR: submission_date column is missing."
    )

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
# TARGET
# ============================================================

y = df["target"].astype(int)

# ============================================================
# PREPARE FEATURES
# ============================================================

print("\nRemoving forbidden features...")

X = df.drop(
    columns=[
        column
        for column in FORBIDDEN_FEATURES
        if column in df.columns
    ]
)

# Handle invalid values.

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)

# Keep numeric features only.

X = X.select_dtypes(
    include=["number"]
)

# ============================================================
# SAFETY CHECK
# ============================================================

leakage_columns = [
    column
    for column in X.columns
    if column in FORBIDDEN_FEATURES
]

if leakage_columns:

    raise ValueError(
        "LEAKAGE DETECTED: "
        + str(leakage_columns)
    )

print(
    f"\nNumber of ML features: {X.shape[1]}"
)

# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

print("\n" + "=" * 60)

print(
    "WALK-FORWARD VALIDATION"
)

print("=" * 60)

print(
    f"\nTotal samples: {len(X)}"
)

print(
    f"Number of folds: {N_FOLDS}"
)

print(
    "\nEach fold trains ONLY on earlier submissions "
    "and tests on later submissions."
)

# ------------------------------------------------------------
# Sequential test windows
#
# 250 samples / 6 ≈ 41 samples per test window
#
# Fold 1:
# train = first 41
# test  = next 41
#
# Fold 2:
# train = first 82
# test  = next 41
#
# etc.
# ------------------------------------------------------------

test_size = len(X) // (N_FOLDS + 1)

results = []

# ============================================================
# FOLDS
# ============================================================

for fold in range(1, N_FOLDS + 1):

    train_end = test_size * fold

    test_end = test_size * (fold + 1)

    X_train = X.iloc[
        :train_end
    ]

    y_train = y.iloc[
        :train_end
    ]

    X_test = X.iloc[
        train_end:test_end
    ]

    y_test = y.iloc[
        train_end:test_end
    ]

    print("\n" + "=" * 60)

    print(
        f"FOLD {fold}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Sample information
    # --------------------------------------------------------

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    print(
        f"Training period : "
        f"{df.iloc[0]['submission_date']} "
        f"-> "
        f"{df.iloc[train_end - 1]['submission_date']}"
    )

    print(
        f"Testing period  : "
        f"{df.iloc[train_end]['submission_date']} "
        f"-> "
        f"{df.iloc[test_end - 1]['submission_date']}"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nTraining target distribution:")

    print(
        y_train.value_counts()
    )

    print("\nTesting target distribution:")

    print(
        y_test.value_counts()
    )

    # --------------------------------------------------------
    # Check both classes
    # --------------------------------------------------------

    if y_train.nunique() < 2:

        print(
            "\nSkipping fold because training data "
            "contains only one class."
        )

        continue

    if y_test.nunique() < 2:

        print(
            "\nWarning: test set contains only one class. "
            "ROC-AUC may be unavailable."
        )

    # --------------------------------------------------------
    # RANDOM FOREST
    #
    # EXACT SAME CONFIGURATION AS train_model.py
    # --------------------------------------------------------

    model = RandomForestClassifier(

        n_estimators=300,

        max_depth=8,

        min_samples_leaf=3,

        class_weight="balanced",

        random_state=42
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

    except ValueError:

        roc_auc = np.nan

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    if not np.isnan(roc_auc):

        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )

    else:

        print(
            "ROC-AUC  : N/A"
        )

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append({

        "fold": fold,

        "train_samples": len(X_train),

        "test_samples": len(X_test),

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": roc_auc
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 60)

print(
    "WALK-FORWARD RESULTS"
)

print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

# ============================================================
# AVERAGE PERFORMANCE
# ============================================================

print("\n" + "=" * 60)

print(
    "AVERAGE PERFORMANCE"
)

print("=" * 60)

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]

for metric in metrics:

    mean_value = (
        results_df[metric]
        .mean()
    )

    std_value = (
        results_df[metric]
        .std()
    )

    print(
        f"{metric.upper():<10}: "
        f"{mean_value:.4f} "
        f"+/- {std_value:.4f}"
    )

# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

output_file = (
    "results/walk_forward_validation.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print(
    f"\nResults saved to: {output_file}"
)

print(
    "\nValidation complete."
)