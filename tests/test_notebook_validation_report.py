"""Provisional QA is traceable and cannot be mistaken for human sentiment labels."""

import csv
import hashlib
from collections import Counter
from pathlib import Path

from ph_sentiment.csvio import read_records

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "deliverables/validation_2026"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_provisional_review_covers_original_corpus_without_gold_labels():
    original = read_records(ROOT / "deliverables/public_2026/processed/posts.csv")
    review = rows(REPORT / "relevance_review.csv")
    assert len(review) == len(original) == 335
    assert {row["record_id"] for row in review} == {record.record_id for record in original}
    assert all(row["reviewer_type"] == "AI_assisted_provisional" for row in review)
    assert all(row["human_confirmation"] == "pending" for row in review)
    assert all("gold_label" not in row for row in review)
    assert all(not record.gold_label for record in original)
    counts = Counter(row["review_status"] for row in review)
    assert counts == {
        "clearly_relevant": 184,
        "clearly_irrelevant": 82,
        "uncertain/context_needed": 69,
    }
    summary = rows(REPORT / "review_summary.csv")
    assert {
        row["review_status"]: int(row["records"]) for row in summary if row["platform"] == "all"
    } == counts


def test_validation_report_csv_checksums_and_row_counts_match():
    manifest = rows(REPORT / "file_manifest.csv")
    assert {row["path"] for row in manifest} == {
        path.name for path in REPORT.glob("*.csv") if path.name != "file_manifest.csv"
    }
    for row in manifest:
        path = (REPORT / row["path"]).resolve()
        assert path.is_relative_to(REPORT.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert len(rows(path)) == int(row["rows"])


def test_sensitivity_report_distinguishes_total_tokens_and_model_features():
    sensitivity = rows(REPORT / "topic_sensitivity.csv")
    assert {row["topic_id"] for row in sensitivity} == {"0", "1", "2"}
    topic = next(row for row in sensitivity if row["topic_id"] == "1")
    assert int(topic["removed_document_all_tokens"]) == 2022
    assert int(topic["removed_document_feature_tokens"]) == 1240
    assert 0.49 < float(topic["share_of_assigned_topic_feature_tokens"]) < 0.51
    assert float(topic["npmi_all_documents"]) > 0
    assert float(topic["npmi_without_longest_document"]) < 0
    assert "no LDA refit" in topic["scope"]
