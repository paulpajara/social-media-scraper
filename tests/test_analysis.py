"""Verify sparse topic analysis, bounded proximity, and empty-feature handling."""

import csv
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ph_sentiment.analysis import _npmi_coherence, _sample_pairs, _topic_tokens, analyze
from ph_sentiment.errors import ConfigurationError, DataValidationError
from ph_sentiment.schema import Record
from ph_sentiment.sentiment import SentimentClassifier


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def corpus() -> list[Record]:
    texts = [
        "Good helpful healthcare clinic public service",
        "Efficient hospital health project medicine service",
        "Bad corrupt transport roads infrastructure budget",
        "Palpak railway roads traffic project budget",
        "School learning education classrooms information",
        "Teacher education books learning information",
    ]
    return [
        Record(str(index), "reddit", "2026-01-31T16:00:00Z", text)
        for index, text in enumerate(texts)
    ]


def test_analysis_exports_complete_datasets_and_valid_metrics(
    corpus: list[Record],
    tmp_path: Path,
) -> None:
    paths = analyze(corpus, tmp_path, max_pairs=7, top_terms=4)
    assert set(paths) == {
        "documents",
        "topic_terms",
        "topic_metrics",
        "sentiment_summary",
        "proximity_pairs",
    }
    documents = read_rows(paths["documents"])
    assert len(documents) == len(corpus)
    assert all(row["topic_status"] == "assigned" for row in documents)
    assert all(0 <= int(row["topic_id"]) < 3 for row in documents)
    assert all(0 <= float(row["topic_probability"]) <= 1 for row in documents)
    terms = read_rows(paths["topic_terms"])
    assert len(terms) == 12
    assert {row["topic_id"] for row in terms} == {"0", "1", "2"}
    metrics = read_rows(paths["topic_metrics"])
    perplexity = next(row for row in metrics if row["metric"] == "lda_perplexity")
    assert math.isfinite(float(perplexity["value"]))
    assert perplexity["evaluation_scope"] == "in_sample"
    assert "not held-out" in perplexity["notes"]
    assert all(
        -1 <= float(row["value"]) <= 1 for row in metrics if row["metric"] == "npmi_coherence"
    )
    summaries = read_rows(paths["sentiment_summary"])
    assert sum(int(row["count"]) for row in summaries) == len(corpus)
    assert {row["month"] for row in summaries} == {"2026-02"}  # Manila calendar.
    pairs = read_rows(paths["proximity_pairs"])
    assert len(pairs) == 7
    for row in pairs:
        cosine = float(row["cosine_similarity"])
        assert 0 <= cosine <= 1
        assert 0 <= float(row["jaccard_similarity"]) <= 1
        assert float(row["euclidean_distance"]) == pytest.approx(math.sqrt(2 - 2 * cosine))
        assert row["platform_a"] == row["platform_b"] == "reddit"


def test_zero_features_are_unassigned_and_pair_metrics_missing(
    corpus: list[Record],
    tmp_path: Path,
) -> None:
    corpus.append(Record("emptyfeatures", "x", "2026-01-01T00:00:00Z", "the at ang"))
    paths = analyze(corpus, tmp_path, max_pairs=100)
    empty = next(
        row for row in read_rows(paths["documents"]) if row["record_id"] == "emptyfeatures"
    )
    assert empty["topic_status"] == "zero_features"
    assert empty["topic_id"] == empty["topic_probability"] == ""
    pairs = [
        row
        for row in read_rows(paths["proximity_pairs"])
        if "emptyfeatures" in {row["record_id_a"], row["record_id_b"]}
    ]
    assert len(pairs) == 6
    assert all(row["pair_status"] == "zero_feature_document" for row in pairs)
    assert all(row["cosine_similarity"] == "" for row in pairs)


def test_pair_sampling_is_unique_deterministic_and_scales_without_all_pairs() -> None:
    assert list(_sample_pairs(6, 100, 42)) == [
        (left, right) for left in range(6) for right in range(left + 1, 6)
    ]
    pairs = list(_sample_pairs(1_000_000, 7, 5))
    assert len(set(pairs)) == 7
    assert all(0 <= left < right < 1_000_000 for left, right in pairs)
    assert pairs == list(_sample_pairs(1_000_000, 7, 5))


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        ([[1, 1], [0, 0]], 1.0),
        ([[1, 0], [0, 1]], -1.0),
        ([[1, 1], [1, 0], [0, 1], [0, 0]], 0.0),
        ([[1, 1], [1, 1]], 0.0),
    ],
)
def test_coherence_uses_binary_document_presence(matrix: list[list[int]], expected: float) -> None:
    assert _npmi_coherence(csr_matrix(np.array(matrix) * 5), [0, 1]) == pytest.approx(expected)


def test_analysis_rejects_insufficient_or_empty_vocabulary(
    corpus: list[Record], tmp_path: Path
) -> None:
    with pytest.raises(ConfigurationError, match="at least three"):
        analyze(corpus, tmp_path, n_topics=2)
    with pytest.raises(DataValidationError, match="as many documents"):
        analyze(corpus[:2], tmp_path)
    empty = [Record(str(i), "reddit", "2026-01-01T00:00:00Z", "the at ang") for i in range(3)]
    with pytest.raises(DataValidationError, match="No topic vocabulary"):
        analyze(empty, tmp_path)


def test_record_identity_includes_platform(corpus: list[Record], tmp_path: Path) -> None:
    corpus.append(Record("0", "x", "2026-01-01T00:00:00Z", "Transparent fair tax services"))
    assert len(read_rows(analyze(corpus, tmp_path)["documents"])) == 7


@pytest.mark.parametrize("apostrophe", ["'", "’"])
def test_topic_features_remove_exact_publisher_template_and_keep_content(apostrophe: str) -> None:
    template = (
        f"🤖 IA: It{apostrophe}s not clickbait ✅\n👥 USERS: It{apostrophe}s not clickbait ✅"
    )
    content = "Philippine government clickbait claims on 2026-10-12 budget 500 health policy"
    assert _topic_tokens(content + " " + template + " 👇👇👇") == _topic_tokens(content)
    assert "clickbait" in _topic_tokens(content)
    assert "2026" in _topic_tokens(content)
    assert "500" in _topic_tokens(content)
    assert "government" in _topic_tokens(content)
    assert _topic_tokens("😊 ✅ 👥 👇 🤖 !!! ...") == []


def test_partial_publisher_phrase_is_retained_as_content() -> None:
    # Do not globally blacklist these words or remove ordinary discussion.
    assert "clickbait" in _topic_tokens("Users say it is clickbait government reporting")
    assert "clickbait" in _topic_tokens("🤖 IA: It's not clickbait ✅")


def test_topic_filter_preserves_canonical_and_sentiment_text(
    corpus: list[Record], tmp_path: Path
) -> None:
    template = " 🤖 IA: It's not clickbait ✅ 👥 Users: It's not clickbait ✅ 👇👇👇 😊"
    decorated = [replace(record, text=record.text + template) for record in corpus]
    base_paths = analyze(corpus, tmp_path / "base", max_pairs=100)
    decorated_paths = analyze(decorated, tmp_path / "decorated", max_pairs=100)
    # Footer differences cannot become LDA or proximity features.
    for output in ("topic_terms", "topic_metrics", "proximity_pairs"):
        assert base_paths[output].read_bytes() == decorated_paths[output].read_bytes()
    documents = read_rows(decorated_paths["documents"])
    classifier = SentimentClassifier()
    for record, document in zip(decorated, documents, strict=True):
        assert document["text"] == record.text
        assert "😊" in document["clean_text"]
        assert "clickbait" in document["clean_text"]
        prediction = classifier.predict(record.text)
        assert document["sentiment_label"] == prediction.label
        assert float(document["sentiment_score"]) == prediction.score
    assert all(
        "document and sentiment text are preserved" in row["notes"]
        for row in read_rows(decorated_paths["topic_metrics"])
    )
