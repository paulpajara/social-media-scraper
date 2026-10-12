"""Validate the actual committed labeled research inputs and benchmark provenance."""

from collections import Counter
from pathlib import Path

from ph_sentiment.annotation_review import adjudicate_annotations
from ph_sentiment.config import load_config
from ph_sentiment.csvio import read_records
from ph_sentiment.curation import curate_records
from ph_sentiment.handoff import _table, verify_integrity_manifest
from ph_sentiment.preprocessing import prepare_records

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "deliverables/labeled_2026"


def test_labeled_snapshot_is_complete_and_preserves_actual_public_acquisition():
    verify_integrity_manifest(SNAPSHOT)
    original = ROOT / "deliverables/public_2026"
    for path in (original / "raw").glob("*.csv"):
        assert path.read_bytes() == (SNAPSHOT / "raw" / path.name).read_bytes()
    config = load_config(SNAPSHOT / "config.toml")
    raw = [
        record
        for platform in ("bluesky", "mastodon")
        for record in read_records(SNAPSHOT / "raw" / f"{platform}.csv")
    ]
    candidates, _ = prepare_records(
        raw, config["research"]["start_date"], config["research"]["end_date"]
    )
    assert candidates == read_records(SNAPSHOT / "processed/candidates.csv")
    decisions = _table(SNAPSHOT / "labels/annotation_decisions.csv")[1]
    curated = curate_records(candidates, decisions)
    assert len(curated) >= 200
    assert {record.platform for record in curated} == {"bluesky", "mastodon"}
    assert curated == read_records(SNAPSHOT / "processed/posts.csv")
    assert curated == read_records(SNAPSHOT / "labels/annotation.csv")
    assert {r["label_source"] for r in decisions} == {"ai_assisted"}


def test_real_annotation_quality_and_adjudication_replay_exactly():
    folder = SNAPSHOT / "labels"
    candidates = read_records(SNAPSHOT / "processed/candidates.csv")
    primary = _table(folder / "primary_annotation_decisions.csv")[1]
    review = _table(folder / "independent_review.csv")[1]
    resolutions = _table(folder / "adjudication.csv")[1]
    final, quality = adjudicate_annotations(
        candidates, primary, review, resolutions, sampling_seed=20261012
    )
    assert final == _table(folder / "annotation_decisions.csv")[1]
    actual = _table(folder / "annotation_quality.csv")[1]
    assert len(actual) == len(quality)
    for expected, observed in zip(quality, actual, strict=True):
        assert expected["metric"] == observed["metric"]
        assert expected["numerator"] == int(observed["numerator"])
        assert expected["denominator"] == int(observed["denominator"])
        assert expected["value"] == float(observed["value"])


def test_reference_labels_are_distinct_from_predictions_and_holdout_excluded_from_cv():
    records = read_records(SNAPSHOT / "processed/posts.csv")
    predictions = _table(SNAPSHOT / "evaluation/evaluation_predictions.csv")[1]
    assignments = _table(SNAPSHOT / "evaluation/split_assignments.csv")[1]
    by_id = {record.record_id: record for record in records}
    assert len(predictions) == len(assignments) == len(records)
    assert {row["record_id"] for row in predictions} == set(by_id)
    assert any(row["gold_label"] != row["predicted_label"] for row in predictions)
    for row in predictions:
        assert row["gold_label"] == by_id[row["record_id"]].gold_label
    splits = Counter(row["split"] for row in assignments)
    assert splits["holdout"] + splits["development"] == len(records)
    assert splits["holdout"] >= int(0.1 * len(records))
    holdout = [row for row in assignments if row["split"] == "holdout"]
    development = [row for row in assignments if row["split"] == "development"]
    assert all(not row["validation_fold"] for row in holdout)
    assert {row["validation_fold"] for row in development} == {str(i) for i in range(1, 11)}
    assert {by_id[row["record_id"]].gold_label for row in holdout} == {
        "negative",
        "neutral",
        "positive",
    }
    for fold in range(1, 11):
        assert {
            by_id[row["record_id"]].gold_label
            for row in development
            if row["validation_fold"] == str(fold)
        } == {"negative", "neutral", "positive"}
    for path in (SNAPSHOT / "evaluation").glob("*.csv"):
        rows = _table(path)[1]
        assert {row["label_source"] for row in rows} == {"ai_assisted"}
        assert all("not human-validated accuracy" in row["notes"] for row in rows)


def test_reference_summary_counts_actual_reference_labels():
    records = read_records(SNAPSHOT / "processed/posts.csv")
    reference = _table(SNAPSHOT / "labels/reference_sentiment_summary.csv")[1]
    assert sum(int(row["count"]) for row in reference) == len(records)
    actual = Counter()
    for row in reference:
        actual[row["label"]] += int(row["count"])
        assert float(row["share"]) == int(row["count"]) / int(row["total"])
        assert row["label_source"] == "ai_assisted"
    assert actual == Counter(record.gold_label for record in records)
