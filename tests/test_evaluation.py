"""Reference-label metrics and disjoint stratified development/holdout partitions."""

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
    for path in paths.values():
        assert all(row["label_source"] == "human" for row in read_rows(path))


def test_ai_reference_provenance_is_carried_by_every_evaluation_csv(
    labeled_corpus: list[Record], tmp_path: Path
) -> None:
    # Synthetic fixture labels are intentionally different from model outputs.
    # The evaluator must keep supplied annotations rather than overwrite them.
    opposite = {"positive": "negative", "negative": "positive", "neutral": "neutral"}
    references = [
        replace(record, gold_label=opposite[record.gold_label]) for record in labeled_corpus
    ]
    paths = evaluate(references, tmp_path, label_source="ai_assisted")
    for path in paths.values():
        rows = read_rows(path)
        assert rows
        assert {row["label_source"] for row in rows} == {"ai_assisted"}
        assert all("not human-validated accuracy" in row["notes"] for row in rows)
        assert all("no training or tuning" in row["notes"] for row in rows)
        assert all("Holdout excluded from CV" in row["notes"] for row in rows)
        assert all("Human labels required" not in row["notes"] for row in rows)
    predictions = read_rows(paths["evaluation_predictions"])
    labels = {record.record_id: record.gold_label for record in references}
    assert all(row["gold_label"] == labels[row["record_id"]] for row in predictions)
    assert any(row["gold_label"] != row["predicted_label"] for row in predictions)
    for split in ("development", "holdout"):
        assert {row["gold_label"] for row in predictions if row["split"] == split} == set(LABELS)
    for fold in range(1, 11):
        assert {row["gold_label"] for row in predictions if row["fold"] == str(fold)} == set(LABELS)
    assignments = read_rows(paths["split_assignments"])
    holdout_ids = {row["record_id"] for row in assignments if row["split"] == "holdout"}
    cv_ids = {row["record_id"] for row in predictions if row["fold"]}
    assert not holdout_ids & cv_ids
    assert len(holdout_ids) == 6


def test_ai_references_require_complete_independent_annotations(
    labeled_corpus: list[Record], tmp_path: Path
) -> None:
    missing_label = [replace(labeled_corpus[0], gold_label=""), *labeled_corpus[1:]]
    with pytest.raises(DataValidationError, match="AI-assisted reference gold_label"):
        evaluate(missing_label, tmp_path, label_source="ai_assisted")
    with pytest.raises(DataValidationError, match="AI-assisted reference-labeled"):
        evaluate([], tmp_path, label_source="ai_assisted")
    assert not list(tmp_path.glob("*.csv"))


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


@pytest.mark.parametrize("label_source", ["human", "ai_assisted"])
def test_small_classes_fail_with_clear_message(
    labeled_corpus: list[Record], tmp_path: Path, label_source: str
) -> None:
    small = [record for record in labeled_corpus if int(record.record_id.rsplit("-", 1)[1]) < 10]
    with pytest.raises(DataValidationError, match="after reserving the holdout"):
        evaluate(small, tmp_path, label_source=label_source)


@pytest.mark.parametrize("label_source", ["human", "ai_assisted"])
def test_holdout_without_every_class_is_rejected_before_writing(
    labeled_corpus: list[Record], tmp_path: Path, label_source: str
) -> None:
    # A rare class can meet ten-fold development counts yet receive no holdout
    # examples with a small holdout fraction. Do not silently report that split.
    references = [
        record
        for record in labeled_corpus
        if record.gold_label != "positive" or int(record.record_id.rsplit("-", 1)[1]) < 11
    ]
    for label in ("negative", "neutral"):
        template = next(record for record in labeled_corpus if record.gold_label == label)
        references.extend(
            replace(
                template,
                record_id=f"{label}-additional-{index}",
                text=f"{template.text} uniqueadditional{index}",
            )
            for index in range(80)
        )
    with pytest.raises(DataValidationError, match="Holdout must include every class"):
        evaluate(references, tmp_path, holdout_fraction=0.025, label_source=label_source)
    assert not list(tmp_path.glob("*.csv"))


def test_invalid_configuration_and_empty_input_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="label_source"):
        evaluate([], tmp_path, label_source="model_predictions")
    with pytest.raises(DataValidationError, match="human-labeled"):
        evaluate([], tmp_path)
    with pytest.raises(ConfigurationError, match="strictly between"):
        evaluate([], tmp_path, holdout_fraction=1)
    with pytest.raises(ConfigurationError, match="two cross-validation"):
        evaluate([], tmp_path, folds=1)
    assert not list(tmp_path.glob("*.csv"))


def test_platform_qualified_ids_allow_cross_platform_collisions(
    labeled_corpus: list[Record],
    tmp_path: Path,
) -> None:
    collision = replace(labeled_corpus[1], platform="x", record_id=labeled_corpus[0].record_id)
    labeled_corpus[1] = collision
    rows = read_rows(evaluate(labeled_corpus, tmp_path)["split_assignments"])
    assert len({(row["platform"], row["record_id"]) for row in rows}) == len(labeled_corpus)
