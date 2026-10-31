"""Evaluate the ML classifier with a fresh, leakage-free held-out split."""

import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.app.core.utils.processing import preprocess_data


DATASET_PATH = REPOSITORY_ROOT / "backend" / "data" / "requirements_dataset.csv"
MODELS_PATH = REPOSITORY_ROOT / "backend" / "models"

LABEL_MAP = {
    "ambiguous": "ambiguous",
    "unambiguous": "clear",
    "clear": "clear",
}
CLASS_LABELS = ["ambiguous", "clear"]


def normalize_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return the dataset with canonical requirement and label columns."""
    dataset = dataset.rename(columns={
        "Context": "requirement",
        "DetectedAs": "label",
    }).copy()

    missing_columns = {"requirement", "label"} - set(dataset.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    dataset["requirement"] = dataset["requirement"].astype("string").str.strip()
    dataset["label"] = dataset["label"].astype("string").str.strip().str.lower()
    dataset = dataset.dropna(subset=["requirement", "label"])
    dataset = dataset[dataset["requirement"] != ""].copy()

    unknown_labels = sorted(set(dataset["label"]) - set(LABEL_MAP))
    if unknown_labels:
        raise ValueError(f"Unsupported dataset labels: {unknown_labels}")

    dataset["label"] = dataset["label"].map(LABEL_MAP)
    return dataset


def prepare_model_input(dataset: pd.DataFrame) -> pd.DataFrame:
    """Apply the same requirement preprocessing used by MLService.predict."""
    model_input = pd.DataFrame({
        "Requirement": dataset["requirement"].tolist(),
        "Comment": "NA",
        "isAmbiguous": dataset["label"].map({
            "ambiguous": "YES",
            "clear": "NO",
        }).tolist(),
    })
    return preprocess_data(model_input)


def evaluate(
    dataset_path: Path,
    output_path: Path,
    sample_size: int | None,
) -> None:
    dataset = normalize_dataset(pd.read_csv(dataset_path))
    train_df, eval_df = train_test_split(
        dataset,
        test_size=0.75,
        stratify=dataset["label"],
        random_state=42,
    )

    if sample_size is not None:
        if sample_size < 1:
            raise ValueError("--sample-size must be at least 1")
        if sample_size > len(eval_df):
            raise ValueError(
                f"--sample-size ({sample_size}) exceeds evaluation set size "
                f"({len(eval_df)})"
            )
        eval_df = eval_df.sample(n=sample_size, random_state=42)

    vectorizer_template = joblib.load(MODELS_PATH / "vectorizer.joblib")
    classifier_template = joblib.load(MODELS_PATH / "svc_model.joblib")
    vectorizer = clone(vectorizer_template)
    classifier = clone(classifier_template)

    train_input = prepare_model_input(train_df)
    eval_input = prepare_model_input(eval_df)
    y_train = train_df["label"].map({
        "ambiguous": 1,
        "clear": 0,
    }).to_numpy()
    vectorizer.fit(train_input["Requirement"].tolist())
    classifier.fit(vectorizer.transform(train_input["Requirement"].tolist()), y_train)

    results = []
    for row, clean_requirement in zip(
        eval_df.itertuples(index=False),
        eval_input["Requirement"],
    ):
        try:
            features = vectorizer.transform([clean_requirement])
            predicted_class = int(classifier.predict(features)[0])
            predicted = "ambiguous" if predicted_class == 1 else "clear"
            class_index = list(classifier.classes_).index(predicted_class)
            confidence = float(classifier.predict_proba(features)[0][class_index])
            error = ""
        except Exception as exc:
            predicted = "ERROR"
            confidence = 0.0
            error = str(exc)

        results.append({
            "requirement": row.requirement,
            "ground_truth": row.label,
            "predicted": predicted,
            "confidence": confidence,
            "error": error,
        })

    results_df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    errors = results_df["predicted"] == "ERROR"
    valid_results = results_df.loc[~errors]
    print(f"Saved predictions to: {output_path}")
    print(f"Evaluation rows: {len(results_df)}")
    print(f"Errors: {errors.sum()}")
    print(
        "Evaluation uses a fresh model trained on the 25% training split; "
        "the saved artifacts are used only for their hyperparameters."
    )

    if valid_results.empty:
        print("No valid predictions; metrics were not computed.")
        return

    y_true = valid_results["ground_truth"]
    y_pred = valid_results["predicted"]
    print(f"Accuracy (valid predictions): {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=["AMBIGUOUS", "UNAMBIGUOUS"],
        zero_division=0,
    ))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_true, y_pred, labels=CLASS_LABELS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help=f"CSV dataset path (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dev_results_ml_service.csv"),
        help="Where to save per-requirement predictions",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Evaluate a reproducible sample from the held-out split",
    )
    args = parser.parse_args()
    evaluate(args.dataset, args.output, args.sample_size)


if __name__ == "__main__":
    main()
