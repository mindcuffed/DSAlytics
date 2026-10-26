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
# 
# ============================================================
# CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# Earlier submissions -> training
# Next submissions    -> validation and model selection
# Latest submissions -> final testing
#
# The final test set must NOT be used to select the best model.

train_end = int(len(X) * 0.70)
validation_end = int(len(X) * 0.85)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:validation_end]
X_test = X.iloc[validation_end:]

y_train = y.iloc[:train_end]
y_val = y.iloc[train_end:validation_end]
y_test = y.iloc[validation_end:]

print("\n" + "=" * 60)
print("CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)

print(f"\nTraining samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")
print(f"Final test samples : {len(X_test)}")

print(
    f"\nTraining period: "
    f"{df.iloc[0]['submission_date']} -> "
    f"{df.iloc[train_end - 1]['submission_date']}"
)

print(
    f"Validation period: "
    f"{df.iloc[train_end]['submission_date']} -> "
    f"{df.iloc[validation_end - 1]['submission_date']}"
)

print(
    f"Final test period: "
    f"{df.iloc[validation_end]['submission_date']} -> "
    f"{df.iloc[-1]['submission_date']}"
)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nValidation target distribution:")
print(y_val.value_counts())

print("\nFinal test target distribution:")
print(y_test.value_counts())


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
#
# ============================================================
# TRAIN + VALIDATION EVALUATION
# ============================================================

results = []
trained_models = {}

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)

    # Train only on the training period
    model.fit(X_train, y_train)

    # Evaluate candidates on validation data
    val_predictions = model.predict(X_val)
    val_probabilities = model.predict_proba(X_val)[:, 1]

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision = precision_score(
        y_val, val_predictions, zero_division=0
    )
    val_recall = recall_score(
        y_val, val_predictions, zero_division=0
    )
    val_f1 = f1_score(
        y_val, val_predictions, zero_division=0
    )

    try:
        val_roc_auc = roc_auc_score(y_val, val_probabilities)
    except ValueError:
        val_roc_auc = np.nan

    print(f"\nValidation accuracy : {val_accuracy:.4f}")
    print(f"Validation precision: {val_precision:.4f}")
    print(f"Validation recall   : {val_recall:.4f}")
    print(f"Validation F1       : {val_f1:.4f}")
    print(f"Validation ROC-AUC  : {val_roc_auc:.4f}")

    print("\nValidation classification report:")
    print(
        classification_report(
            y_val,
            val_predictions,
            zero_division=0
        )
    )

    # Store validation metrics for model comparison
    results.append({
        "model": model_name,
        "accuracy": val_accuracy,
        "precision": val_precision,
        "recall": val_recall,
        "f1": val_f1,
        "roc_auc": val_roc_auc
    })

    trained_models[model_name] = model


# ============================================================
# MODEL COMPARISON USING VALIDATION DATA ONLY
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("VALIDATION MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# ============================================================
# SELECT BEST MODEL
# ============================================================
# Select by validation F1; use validation ROC-AUC as a
# secondary criterion. The final test set is not used here.
# ============================================================

valid_results = results_df.copy()

valid_results["_roc_auc_sort"] = (
    valid_results["roc_auc"].fillna(-1)
)

valid_results = valid_results.sort_values(
    by=["f1", "_roc_auc_sort"],
    ascending=[False, False]
)

best_model_name = valid_results.iloc[0]["model"]
best_model = trained_models[best_model_name]

print(f"\nSelected model: {best_model_name}")
print("Selection basis: validation F1, then validation ROC-AUC")


# ============================================================
# FINAL EVALUATION ON UNSEEN TEST DATA
# ============================================================

test_predictions = best_model.predict(X_test)
test_probabilities = best_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(
    y_test, test_predictions, zero_division=0
)
test_recall = recall_score(
    y_test, test_predictions, zero_division=0
)
test_f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

try:
    test_roc_auc = roc_auc_score(y_test, test_probabilities)
except ValueError:
    test_roc_auc = np.nan

print("\n" + "=" * 60)
print("FINAL HELD-OUT TEST RESULTS")
print("=" * 60)

print(f"Selected model : {best_model_name}")
print(f"Test accuracy  : {test_accuracy:.4f}")
print(f"Test precision : {test_precision:.4f}")
print(f"Test recall    : {test_recall:.4f}")
print(f"Test F1        : {test_f1:.4f}")

if not np.isnan(test_roc_auc):
    print(f"Test ROC-AUC   : {test_roc_auc:.4f}")

print("\nFinal test classification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)

# Preserve the existing comparison columns for dashboard compatibility.
# Add the selected model's held-out test metrics separately.
for column, value in {
    "test_accuracy": test_accuracy,
    "test_precision": test_precision,
    "test_recall": test_recall,
    "test_f1": test_f1,
    "test_roc_auc": test_roc_auc
}.items():
    results_df[column] = np.nan

    results_df.loc[
        results_df["model"] == best_model_name,
        column
    ] = value

# Keep the existing variable expected by the save section.
results_df = results_df.drop(
    columns=["_roc_auc_sort"],
    errors="ignore"
)

print("\n" + "=" * 60)
print("MODEL COMPARISON WITH FINAL TEST RESULTS")
print("=" * 60)
print(results_df.to_string(index=False))

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