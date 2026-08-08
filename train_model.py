import pandas as pd
import numpy as np
import os
import json
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/ml_dataset.csv"

MODEL_DIR = "models"
RESULTS_DIR = "results"

# Columns that are NEVER allowed to become ML features.
# They either directly reveal the target or are identifiers/
# raw submission information that should not be used.
FORBIDDEN_FEATURES = [
    # Target leakage
    "target",
    "is_accepted",
    "is_failure",
    "solved",

    # Identifiers / text
    "id",
    "title",
    "titleSlug",
    "topics",

    # Raw submission information
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
print("DSAlytics - Leakage-Free ML Training")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading ML dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

if "target" not in df.columns:
    raise ValueError(
        "ERROR: 'target' column is missing from the ML dataset."
    )

if "submission_date" not in df.columns:
    raise ValueError(
        "ERROR: 'submission_date' column is required "
        "for chronological splitting."
    )

# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")

print(
    df["target"].value_counts()
)

print("\nTarget proportions:")

print(
    df["target"].value_counts(normalize=True)
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
# CREATE TARGET
# ============================================================

y = df["target"].astype(int)

# ============================================================
# REMOVE FORBIDDEN FEATURES
# ============================================================

print("\nRemoving forbidden features...")

existing_forbidden = [
    column
    for column in FORBIDDEN_FEATURES
    if column in df.columns
]

print("\nThe following columns will NOT be used as ML features:")

for column in existing_forbidden:
    print(f"- {column}")

X = df.drop(
    columns=existing_forbidden
)

# ============================================================
# KEEP ONLY NUMERIC FEATURES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)

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
        "LEAKAGE DETECTED! Forbidden columns found in X: "
        + str(leakage_columns)
    )

# Also check for suspicious target-derived columns.
suspicious_columns = [
    column
    for column in X.columns
    if column.lower() in [
        "accepted",
        "failure",
        "success",
        "target"
    ]
]

if suspicious_columns:

    print(
        "\nWARNING: Suspicious feature names detected:"
    )

    for column in suspicious_columns:
        print(f"- {column}")

# ============================================================
# FINAL FEATURE INFORMATION
# ============================================================

print(
    f"\nNumber of ML features: {X.shape[1]}"
)

print("\nML features:")

for feature in X.columns:
    print(f"- {feature}")

# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

# IMPORTANT:
# We do NOT use random splitting.
#
# Earlier submissions -> training
# Later submissions  -> testing
#
# This better represents how DSAlytics will behave
# when predicting future submissions.

split_index = int(
    len(X) * 0.80
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n" + "=" * 60)
print("CHRONOLOGICAL TRAIN / TEST SPLIT")
print("=" * 60)

print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)

print(
    f"Training period: "
    f"{df.iloc[0]['submission_date']} "
    f"-> "
    f"{df.iloc[split_index - 1]['submission_date']}"
)

print(
    f"Testing period : "
    f"{df.iloc[split_index]['submission_date']} "
    f"-> "
    f"{df.iloc[-1]['submission_date']}"
)

print("\nTraining target distribution:")

print(
    y_train.value_counts()
)

print("\nTesting target distribution:")

print(
    y_test.value_counts()
)

# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

# ============================================================
# TRAIN + EVALUATE
# ============================================================

results = []

trained_models = {}

for model_name, model in models.items():

    print("\n" + "=" * 60)

    print(
        f"TRAINING: {model_name}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # PREDICTIONS
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
    # PRINT METRICS
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

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({

        "model": model_name,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": roc_auc
    })

    trained_models[
        model_name
    ] = model

# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

# ============================================================
# SELECT BEST MODEL
# ============================================================
#
# ROC-AUC is used as the primary selection metric because
# this is a probability-based failure prediction system.
#
# F1 is retained as an additional metric.
# ============================================================

valid_results = results_df.dropna(
    subset=["roc_auc"]
)

if len(valid_results) > 0:

    best_model_name = (
        valid_results
        .sort_values(
            "roc_auc",
            ascending=False
        )
        .iloc[0]["model"]
    )

else:

    best_model_name = (
        results_df
        .sort_values(
            "f1",
            ascending=False
        )
        .iloc[0]["model"]
    )

best_model = trained_models[
    best_model_name
]

print(
    f"\nBest model based on ROC-AUC: "
    f"{best_model_name}"
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

if hasattr(best_model, "feature_importances_"):

    importance_df = pd.DataFrame({

        "feature": X.columns,

        "importance":
            best_model.feature_importances_

    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        importance_df
        .head(20)
        .to_string(index=False)
    )

else:

    print(
        "Feature importance is not available "
        "for this model."
    )

# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# ============================================================
# SAVE ALL MODELS
# ============================================================

print("\nSaving trained models...")

for model_name, model in trained_models.items():

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + ".joblib"
    )

    filepath = os.path.join(
        MODEL_DIR,
        filename
    )

    joblib.dump(
        model,
        filepath
    )

    print(
        f"Saved: {filepath}"
    )

# ============================================================
# SAVE BEST MODEL
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "best_model.joblib"
)

joblib.dump(
    best_model,
    best_model_path
)

# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    RESULTS_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

# ============================================================
# SAVE FEATURE NAMES
# ============================================================

feature_file = os.path.join(
    MODEL_DIR,
    "feature_names.json"
)

with open(
    feature_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        list(X.columns),
        f,
        indent=2
    )

# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

if hasattr(best_model, "feature_importances_"):

    importance_file = os.path.join(
        RESULTS_DIR,
        "feature_importance.csv"
    )

    importance_df.to_csv(
        importance_file,
        index=False
    )

    print(
        f"Saved feature importance: "
        f"{importance_file}"
    )

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"\nBest model: {best_model_name}"
)

print(
    f"Saved model: {best_model_path}"
)

print(
    f"Saved comparison: {results_file}"
)

print(
    f"Saved features: {feature_file}"
)

print(
    f"\nFinal feature count: {X.shape[1]}"
)

print("\nNo target/leakage columns were used as features.")

print("\nDone.")