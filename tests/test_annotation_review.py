"""Annotation consistency uses blind pre-adjudication decisions and real denominators."""

from random import Random

import pytest

from ph_sentiment.annotation_review import adjudicate_annotations
from ph_sentiment.curation import text_digest
from ph_sentiment.errors import DataValidationError
from ph_sentiment.schema import Record


@pytest.fixture
def audits():
    candidates = [
        Record(
            f"bluesky:fixture-{i}", "bluesky", "2026-09-01T00:00:00Z", f"national policy report {i}"
        )
        for i in range(10)
    ]
    primary = [
        {
            "index": str(i),
            "record_id": r.record_id,
            "platform": r.platform,
            "relevance": "include",
            "label": "neutral",
            "reason": "Factual report.",
            "content_kind": "news_report",
            "context_needed": "no",
            "label_source": "ai_assisted",
            "annotator": "first_ai",
            "annotation_date": "2026-10-12",
            "rubric_version": "test",
            "text_sha256": text_digest(r.text),
        }
        for i, r in enumerate(candidates)
    ]
    indices = Random(42).sample(range(10), 2)
    review = [dict(primary[i], annotator="second_ai") for i in indices]
    review[0].update(relevance="exclude", label="")
    p = primary[indices[0]]
    resolutions = [
        {
            "index": p["index"],
            "record_id": p["record_id"],
            "platform": p["platform"],
            "primary_relevance": "include",
            "primary_label": "neutral",
            "review_relevance": "exclude",
            "review_label": "",
            "final_relevance": "exclude",
            "final_label": "",
            "reason": "Not a national-government target.",
            "adjudicator": "root_ai",
            "annotation_date": "2026-10-12",
            "text_sha256": p["text_sha256"],
        }
    ]
    return candidates, primary, review, resolutions


def test_quality_compares_initial_labels_and_reports_conditional_denominator(audits):
    final, quality = adjudicate_annotations(*audits, sampling_seed=42)
    by_metric = {r["metric"]: r for r in quality}
    assert by_metric["exact_decision_agreement"]["value"] == 0.5
    assert by_metric["relevance_agreement"]["denominator"] == 2
    assert by_metric["sentiment_agreement_both_included"]["denominator"] == 1
    assert by_metric["sentiment_agreement_both_included"]["value"] == 1.0
    original_index = int(audits[3][0]["index"])
    assert final[original_index]["relevance"] == "exclude"
    assert audits[1][original_index]["relevance"] == "include"


def test_unresolved_disagreements_fail(audits):
    with pytest.raises(DataValidationError, match="every independently"):
        adjudicate_annotations(*audits[:3], [], sampling_seed=42)


@pytest.mark.parametrize(
    "field,value",
    [
        ("text_sha256", "bad"),
        ("primary_label", "positive"),
        ("review_label", "negative"),
        ("record_id", "unknown"),
    ],
)
def test_mismatched_adjudication_fails(audits, field, value):
    audits[3][0][field] = value
    with pytest.raises(DataValidationError):
        adjudicate_annotations(*audits, sampling_seed=42)


def test_claimed_random_sample_is_verified(audits):
    with pytest.raises(DataValidationError, match="random sample"):
        adjudicate_annotations(*audits, sampling_seed=123)
