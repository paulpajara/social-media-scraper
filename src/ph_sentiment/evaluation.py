"""Human-label evaluation of a fixed sentiment model without data leakage."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .csvio import write_csv
from .errors import ConfigurationError, DataValidationError
from .preprocessing import clean_text, tokenize
from .schema import LABELS, Record
from .sentiment import SentimentClassifier

_EVALUATION_NOTE = (
    "Human labels required. Fixed lexicon model: no training or tuning; "
    "CV measures variation across development partitions. Holdout excluded from CV."
)


def _validate_records(records: Sequence[Record]) -> list[str]:
    if not records:
        raise DataValidationError("Evaluation requires human-labeled records.")
    identifiers: set[tuple[str, str]] = set()
    seen_text: dict[str, str] = {}
    labels: list[str] = []
    for record in records:
        record.validate()
        identity = (record.platform, record.record_id)
        if identity in identifiers:
            raise DataValidationError(
                f"Duplicate record_id in {record.platform}: {record.record_id}"
            )
        identifiers.add(identity)
        if record.gold_label not in LABELS:
            raise DataValidationError(
                f"Record {record.record_id} needs a human gold_label: "
                "negative, neutral, or positive. Automatic labels are not ground truth."
            )
        normalized = " ".join(tokenize(clean_text(record.text)))
        if not normalized:
            raise DataValidationError(f"Record {record.record_id} is empty after preprocessing.")
        if normalized in seen_text:
            raise DataValidationError(
                f"Duplicate preprocessed text in records {seen_text[normalized]} and "
                f"{record.record_id}; remove duplicates before splitting to prevent leakage."
            )
        seen_text[normalized] = f"{record.platform}/{record.record_id}"
        labels.append(record.gold_label)
    missing = set(LABELS) - set(labels)
    if missing:
        raise DataValidationError(
            f"Human labels must include every class; missing: {', '.join(sorted(missing))}."
        )
    return labels


def _metrics(actual: Sequence[str], predicted: Sequence[str]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
    }


def _append_diagnostics(
    actual: Sequence[str],
    predicted: Sequence[str],
    split: str,
    fold: int | str,
    metric_rows: list[dict[str, Any]],
    report_rows: list[dict[str, Any]],
    confusion_rows: list[dict[str, Any]],
) -> dict[str, float]:
    values = _metrics(actual, predicted)
    for name, value in values.items():
        metric_rows.append(
            {
                "split": split,
                "fold": fold,
                "n_records": len(actual),
                "metric": name,
                "value": value,
                "model_version": SentimentClassifier.version,
                "notes": _EVALUATION_NOTE,
            }
        )
    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=LABELS,
        average=None,
        zero_division=0,
    )
    for index, label in enumerate(LABELS):
        report_rows.append(
            {
                "split": split,
                "fold": fold,
                "label": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
        )
    matrix = confusion_matrix(actual, predicted, labels=LABELS)
    for left, actual_label in enumerate(LABELS):
        for right, predicted_label in enumerate(LABELS):
            confusion_rows.append(
                {
                    "split": split,
                    "fold": fold,
                    "actual_label": actual_label,
                    "predicted_label": predicted_label,
                    "count": int(matrix[left, right]),
                }
            )
    return values


def evaluate(
    records: list[Record],
    output_dir: Path,
    seed: int = 42,
    holdout_fraction: float = 0.1,
    folds: int = 10,
) -> dict[str, Path]:
    """Reserve a stratified holdout before CV, then write auditable CSV metrics.

    This baseline is fixed. No model is fitted on either development or holdout
    text; CV is diagnostic rather than an estimate of retraining variability.
    Gold labels must be independently assigned by humans. This function cannot
    verify provenance, so callers must maintain their annotation audit trail.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ConfigurationError("holdout_fraction must be strictly between 0 and 1.")
    if folds < 2:
        raise ConfigurationError("Evaluation needs at least two cross-validation folds.")
    if not 0 <= seed < 2**32:
        raise ConfigurationError("seed must be an integer between 0 and 2**32 - 1.")
    labels = _validate_records(records)
    indices = np.arange(len(records))
    try:
        development, holdout = train_test_split(
            indices,
            test_size=holdout_fraction,
            random_state=seed,
            stratify=labels,
            shuffle=True,
        )
    except ValueError as exc:
        raise DataValidationError(
            "Insufficient human labels for a stratified holdout with all three classes. "
            "Add independently labeled examples or increase the holdout size."
        ) from exc
    development_labels = [labels[int(index)] for index in development]
    class_counts = Counter(development_labels)
    insufficient = {label: class_counts[label] for label in LABELS if class_counts[label] < folds}
    if insufficient:
        detail = ", ".join(f"{label}={count}" for label, count in insufficient.items())
        raise DataValidationError(
            f"Each development class needs at least {folds} records for {folds}-fold "
            f"stratified CV after reserving the holdout; found {detail}."
        )
    if set(labels[int(index)] for index in holdout) != set(LABELS):
        raise DataValidationError(
            "Holdout must include every class; add labels or increase its size."
        )

    classifier = SentimentClassifier()
    metric_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    prediction_rows: dict[int, dict[str, Any]] = {}
    fold_metrics: list[dict[str, float]] = []
    validation_folds: dict[int, int] = {}

    def predict_partition(partition: Sequence[int], split: str, fold: int | str) -> list[str]:
        predictions: list[str] = []
        for value in partition:
            index = int(value)
            record = records[index]
            result = classifier.predict(record.text)
            predictions.append(result.label)
            prediction_rows[index] = {
                "record_id": record.record_id,
                "platform": record.platform,
                "split": split,
                "fold": fold,
                "gold_label": record.gold_label,
                "predicted_label": result.label,
                "sentiment_score": result.score,
                "sentiment_evidence": json.dumps(result.matched_terms, ensure_ascii=False),
                "model_version": classifier.version,
            }
        return predictions

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_training_positions, validation_positions) in enumerate(
        splitter.split(development, development_labels),
        start=1,
    ):
        # Training positions deliberately unused: the lexicon is immutable.
        validation = development[validation_positions]
        actual = [labels[int(index)] for index in validation]
        predicted = predict_partition(validation, "development", fold)
        for value in validation:
            validation_folds[int(value)] = fold
        fold_metrics.append(
            _append_diagnostics(
                actual,
                predicted,
                "cv_fold",
                fold,
                metric_rows,
                report_rows,
                confusion_rows,
            )
        )
    actual_development = [labels[int(index)] for index in development]
    predicted_development = [
        prediction_rows[int(index)]["predicted_label"] for index in development
    ]
    _append_diagnostics(
        actual_development,
        predicted_development,
        "development_oof",
        "",
        metric_rows,
        report_rows,
        confusion_rows,
    )
    for metric in fold_metrics[0]:
        values = [fold[metric] for fold in fold_metrics]
        for split, value in (("cv_mean", np.mean(values)), ("cv_std", np.std(values))):
            metric_rows.append(
                {
                    "split": split,
                    "fold": "",
                    "n_records": len(development),
                    "metric": metric,
                    "value": float(value),
                    "model_version": classifier.version,
                    "notes": _EVALUATION_NOTE + " Unweighted fold mean; population SD (ddof=0).",
                }
            )
    # The fixed model and all CV predictions are already finalized at this point.
    predicted_holdout = predict_partition(holdout, "holdout", "")
    _append_diagnostics(
        [labels[int(index)] for index in holdout],
        predicted_holdout,
        "holdout",
        "",
        metric_rows,
        report_rows,
        confusion_rows,
    )
    holdout_set = set(int(index) for index in holdout)
    assignment_rows = [
        {
            "record_id": record.record_id,
            "platform": record.platform,
            "split": "holdout" if index in holdout_set else "development",
            "validation_fold": validation_folds.get(index, ""),
            "seed": seed,
            "holdout_fraction": holdout_fraction,
            "cv_folds": folds,
        }
        for index, record in enumerate(records)
    ]
    paths = {
        "evaluation_metrics": output_dir / "evaluation_metrics.csv",
        "classification_report": output_dir / "classification_report.csv",
        "confusion_matrix": output_dir / "confusion_matrix.csv",
        "split_assignments": output_dir / "split_assignments.csv",
        "evaluation_predictions": output_dir / "evaluation_predictions.csv",
    }
    write_csv(
        paths["evaluation_metrics"],
        metric_rows,
        (
            "split",
            "fold",
            "n_records",
            "metric",
            "value",
            "model_version",
            "notes",
        ),
    )
    write_csv(
        paths["classification_report"],
        report_rows,
        (
            "split",
            "fold",
            "label",
            "precision",
            "recall",
            "f1",
            "support",
        ),
    )
    write_csv(
        paths["confusion_matrix"],
        confusion_rows,
        (
            "split",
            "fold",
            "actual_label",
            "predicted_label",
            "count",
        ),
    )
    write_csv(
        paths["split_assignments"],
        assignment_rows,
        (
            "record_id",
            "platform",
            "split",
            "validation_fold",
            "seed",
            "holdout_fraction",
            "cv_folds",
        ),
    )
    write_csv(
        paths["evaluation_predictions"],
        (prediction_rows[i] for i in range(len(records))),
        (
            "record_id",
            "platform",
            "split",
            "fold",
            "gold_label",
            "predicted_label",
            "sentiment_score",
            "sentiment_evidence",
            "model_version",
        ),
    )
    return paths
