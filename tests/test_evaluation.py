"""Human-label metrics and disjoint stratified development/holdout partitions."""

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from ph_sentiment.errors import ConfigurationError, DataValidationError
from ph_sentiment.evaluation import _metrics, evaluate
from ph_sentiment.schema import LABELS, Record


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def labeled_corpus() -> list[Record]:
    cues = {"positive": "good", "negative": "bad", "neutral": "information"}
    return [
        Record(
            record_id=f"{label}-{index}",
            platform="reddit",
            created_at="2026-01-01T00:00:00Z",
            text=f"Government policy {cues[label]} sample{index}",
            gold_label=label,
        )
        for label in LABELS
        for index in range(20)
    ]


def test_metrics_have_known_values() -> None:
    values = _metrics(["negative", "neutral", "positive"], ["negative", "positive", "positive"])
    assert values["accuracy"] == pytest.approx(2 / 3)
    assert values["macro_precision"] == pytest.approx(1 / 2)
    assert values["macro_recall"] == pytest.approx(2 / 3)
    assert values["macro_f1"] == pytest.approx(5 / 9)
    assert values["balanced_accuracy"] == pytest.approx(2 / 3)


def test_holdout_is_excluded_from_every_cv_fold_and_predictions_are_complete(
    labeled_corpus: list[Record],
    tmp_path: Path,
) -> None:
    paths = evaluate(labeled_corpus, tmp_path)
    assignments = read_rows(paths["split_assignments"])
    assert len(assignments) == 60
    holdout = {row["record_id"] for row in assignments if row["split"] == "holdout"}
    development = {row["record_id"] for row in assignments if row["split"] == "development"}
    assert len(holdout) == 6
    assert len(development) == 54
    assert not holdout & development
    assert {row["validation_fold"] for row in assignments if row["split"] == "development"} == {
        str(index) for index in range(1, 11)
    }
    assert all(row["validation_fold"] == "" for row in assignments if row["split"] == "holdout")
    predictions = read_rows(paths["evaluation_predictions"])
    assert len(predictions) == len(labeled_corpus)
    assert len({row["record_id"] for row in predictions}) == len(labeled_corpus)
    assert {row["record_id"] for row in predictions if row["fold"]} == development
    assert all(row["gold_label"] == row["predicted_label"] for row in predictions)
    for split in ("holdout", "development"):
        assert {row["gold_label"] for row in predictions if row["split"] == split} == set(LABELS)
    metrics = read_rows(paths["evaluation_metrics"])
    assert len([row for row in metrics if row["split"] == "cv_fold"]) == 50
    assert all(float(row["value"]) == 1 for row in metrics if row["split"] == "holdout")
    assert all("no training or tuning" in row["notes"] for row in metrics)
    matrix = [row for row in read_rows(paths["confusion_matrix"]) if row["split"] == "holdout"]
    assert len(matrix) == 9
    assert sum(int(row["count"]) for row in matrix) == 6
    assert {row["actual_label"] for row in matrix} == set(LABELS)


def test_split_is_deterministic(labeled_corpus: list[Record], tmp_path: Path) -> None:
    first = evaluate(labeled_corpus, tmp_path / "first")
    second = evaluate(labeled_corpus, tmp_path / "second")
    assert first["split_assignments"].read_bytes() == second["split_assignments"].read_bytes()
    assert first["evaluation_metrics"].read_bytes() == second["evaluation_metrics"].read_bytes()


def test_evaluation_requires_independent_human_labels(
    labeled_corpus: list[Record], tmp_path: Path
) -> None:
    missing_label = [replace(labeled_corpus[0], gold_label=""), *labeled_corpus[1:]]
    with pytest.raises(DataValidationError, match="human gold_label"):
        evaluate(missing_label, tmp_path)
    with pytest.raises(DataValidationError, match="every class"):
        evaluate([record for record in labeled_corpus if record.gold_label != "neutral"], tmp_path)
    assert not list(tmp_path.glob("*.csv"))


def test_duplicates_are_rejected_before_any_split(
    labeled_corpus: list[Record], tmp_path: Path
) -> None:
    duplicate = replace(
        labeled_corpus[0],
        record_id="duplicate",
        platform="x",
        text=labeled_corpus[0].text.upper() + "! https://example.com/source @sample",
    )
    with pytest.raises(DataValidationError, match="prevent leakage"):
        evaluate([*labeled_corpus, duplicate], tmp_path)
    assert not list(tmp_path.glob("*.csv"))


def test_small_classes_fail_with_clear_message(
    labeled_corpus: list[Record], tmp_path: Path
) -> None:
    small = [record for record in labeled_corpus if int(record.record_id.rsplit("-", 1)[1]) < 10]
    with pytest.raises(DataValidationError, match="after reserving the holdout"):
        evaluate(small, tmp_path)


def test_invalid_configuration_and_empty_input_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="human-labeled"):
        evaluate([], tmp_path)
    with pytest.raises(ConfigurationError, match="strictly between"):
        evaluate([], tmp_path, holdout_fraction=1)
    with pytest.raises(ConfigurationError, match="two cross-validation"):
        evaluate([], tmp_path, folds=1)


def test_platform_qualified_ids_allow_cross_platform_collisions(
    labeled_corpus: list[Record],
    tmp_path: Path,
) -> None:
    collision = replace(labeled_corpus[1], platform="x", record_id=labeled_corpus[0].record_id)
    labeled_corpus[1] = collision
    rows = read_rows(evaluate(labeled_corpus, tmp_path)["split_assignments"])
    assert len({(row["platform"], row["record_id"]) for row in rows}) == len(labeled_corpus)
