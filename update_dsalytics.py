import subprocess
import sys
import time


def run_step(name, command):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    start = time.time()

    result = subprocess.run(
        [sys.executable] + command,
        check=False
    )

    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\nERROR: {name} failed.")
        print(f"Time: {elapsed:.2f} seconds")
        sys.exit(result.returncode)

    print(f"\n✓ {name} completed")
    print(f"Time: {elapsed:.2f} seconds")


def main():

    print("=" * 70)
    print("DSAlytics - Automatic ML Pipeline")
    print("=" * 70)

    # ========================================================
    # STEP 0: FETCH LATEST LEETCODE SUBMISSIONS
    # ========================================================

    run_step(
        "STEP 0: FETCHING LATEST LEETCODE DATA",
        ["fetch_leetcode_data.py"]
    )

    # ========================================================
    # STEP 1: PREPARE MASTER DATASET
    # ========================================================

    run_step(
        "STEP 1: PREPARING MASTER DATASET",
        ["prepare_dataset.py"]
    )

    # ========================================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================================

    run_step(
        "STEP 2: FEATURE ENGINEERING",
        ["feature_engineering.py"]
    )

    # ========================================================
    # STEP 3: WEAKNESS ANALYSIS
    # ========================================================

    run_step(
        "STEP 3: WEAKNESS ANALYSIS",
        ["analyze_weaknesses.py"]
    )

    # ========================================================
    # STEP 4: MODEL TRAINING
    # ========================================================

    run_step(
        "STEP 4: MODEL TRAINING",
        ["train_model.py"]
    )

    # ========================================================
    # STEP 5: WALK-FORWARD VALIDATION
    # ========================================================

    run_step(
        "STEP 5: WALK-FORWARD VALIDATION",
        ["validate_model.py"]
    )

    # ========================================================
    # STEP 6: PERSONALIZED RECOMMENDATIONS
    # ========================================================

    run_step(
        "STEP 6: PERSONALIZED RECOMMENDATIONS",
        ["recommend_next.py"]
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("DSAlytics pipeline completed successfully.")
    print("=" * 70)

    print("\nUpdated:")
    print("✓ LeetCode submission data")
    print("✓ Master dataset")
    print("✓ ML dataset")
    print("✓ Weakness analysis")
    print("✓ Trained models")
    print("✓ Walk-forward validation")
    print("✓ Personalized recommendations")


if __name__ == "__main__":
    main()