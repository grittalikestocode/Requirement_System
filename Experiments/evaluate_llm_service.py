"""Evaluate the backend LLM service on a held-out dataset split."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from backend.app.core.llm_service import DATASET_PATH, LLMService


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


def evaluate(dataset_path: Path, output_path: Path, sample_size: int | None) -> None:
    dataset = normalize_dataset(pd.read_csv(dataset_path))
    example_df, eval_df = train_test_split(
        dataset,
        test_size=0.85,
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
        # eval_df = eval_df.sample(n=sample_size, random_state=42)
        eval_df = eval_df.reset_index(drop=True).iloc[:sample_size]

    service = LLMService()
    service.initialize(examples_df=example_df)

    results = []
    for row in eval_df.itertuples(index=False):
        try:
            prediction = service.analyze_ambiguity(row.requirement)
            predicted = prediction["verdict"]
            if predicted not in CLASS_LABELS:
                raise ValueError(f"Unexpected service verdict: {predicted!r}")
            error = ""
            confidence = prediction["confidence"]
            ambiguity_type = prediction["ambiguity_type"]
            explanation = " ".join(prediction["reasons"])
        except Exception as exc:
            predicted = "ERROR"
            error = str(exc)
            confidence = 0.0
            ambiguity_type = ""
            explanation = ""

        results.append({
            "requirement": row.requirement,
            "ground_truth": row.label,
            "predicted": predicted,
            "confidence": confidence,
            "ambiguity_type": ambiguity_type,
            "explanation": explanation,
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
        default=Path("dev_results_llm_service.csv"),
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
