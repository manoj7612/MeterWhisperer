from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from features import (
    build_dataset,
    FEATURE_COLUMNS
)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "raw"

MODEL_DIR = ROOT / "models"

REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# FIVE CLASSES
# ---------------------------------------------------------

CLASSES = [
    "normal",
    "tap_water_leak",
    "washing_machine_leak",
    "toilet_leak",
    "pipe_fitting",
]


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("       METER WHISPERER - MODEL TRAINING")
    print("=" * 70)

    print("\nReading recordings...")

    dataset = build_dataset(DATA_DIR)

    if dataset.empty:
        print("\nERROR: No usable CSV data found.")
        return

    print("\nFeature dataset created.")

    print(
        f"Total feature windows: {len(dataset)}"
    )

    print("\nWindows per class:")

    print(
        dataset["label"].value_counts()
    )

    # -----------------------------------------------------
    # INPUTS / OUTPUT
    # -----------------------------------------------------

    X = dataset[FEATURE_COLUMNS]

    y = dataset["label"]

    groups = dataset["source_file"]

    # -----------------------------------------------------
    # RECORDING-LEVEL SPLIT
    # -----------------------------------------------------
    #
    # This prevents windows from the same CSV appearing
    # in both training and testing.
    #

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_indices, test_indices = next(
        splitter.split(
            X,
            y,
            groups=groups
        )
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    print("\nTraining windows:", len(X_train))
    print("Testing windows:", len(X_test))

    # -----------------------------------------------------
    # RANDOM FOREST
    # -----------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1
    )

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # TEST
    # -----------------------------------------------------

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 70)
    print("MODEL RESULTS")
    print("=" * 70)

    print(
        f"\nTest accuracy: {accuracy * 100:.2f}%"
    )

    print("\nClassification report:")

    report = classification_report(
        y_test,
        predictions,
        labels=CLASSES,
        zero_division=0
    )

    print(report)

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=CLASSES
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=CLASSES,
        columns=CLASSES
    )

    confusion_path = (
        REPORT_DIR /
        "confusion_matrix.csv"
    )

    confusion_df.to_csv(
        confusion_path
    )

    # -----------------------------------------------------
    # SAVE REPORT
    # -----------------------------------------------------

    report_path = (
        REPORT_DIR /
        "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "METER WHISPERER MODEL RESULTS\n"
        )

        file.write(
            "==============================\n\n"
        )

        file.write(
            f"Test accuracy: "
            f"{accuracy * 100:.2f}%\n\n"
        )

        file.write(report)

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    model_path = (
        MODEL_DIR /
        "meter_whisperer_model.joblib"
    )

    model_package = {

        "model": model,

        "features": FEATURE_COLUMNS,

        "classes": CLASSES,

        "version": "5-class-v2"
    }

    joblib.dump(
        model_package,
        model_path
    )

    print("\nModel saved to:")

    print(model_path)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()