"""Sparse topic, sentiment, and document proximity analysis with CSV outputs."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfTransformer,
)

from .csvio import write_csv
from .errors import ConfigurationError, DataValidationError, PipelineError
from .preprocessing import clean_text, tokenize
from .schema import LABELS, Record, parse_timestamp
from .sentiment import SentimentClassifier

_FILIPINO_STOPWORDS = frozenset(
    "ako akin aking ang ano at ay ba bakit din dito doon isang ito iyang iyon ka "
    "kami kanila kanilang kayo ko kung lang lamang mga mo muna na namin natin "
    "ng nga ni nila ninyo nito noong o pa para rin sa siya sila tayo to upang "
    "wala walang yan yung iyong po opo naman dahil habang kapag kung saan".split()
)
_TOPIC_STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | _FILIPINO_STOPWORDS


def _topic_tokens(text: str) -> list[str]:
    return [word for word in tokenize(text) if word not in _TOPIC_STOPWORDS]


def _validate_inputs(
    records: Sequence[Record],
    n_topics: int,
    seed: int,
    max_features: int,
    top_terms: int,
    max_pairs: int,
) -> None:
    if n_topics < 3:
        raise ConfigurationError("Topic analysis requires at least three topics.")
    if max_features < n_topics:
        raise ConfigurationError("max_features must be at least n_topics.")
    if not 2 <= top_terms <= 50:
        raise ConfigurationError("top_terms must be between 2 and 50.")
    if max_pairs < 1:
        raise ConfigurationError("max_pairs must be positive.")
    if not 0 <= seed < 2**32:
        raise ConfigurationError("seed must be an integer between 0 and 2**32 - 1.")
    if len(records) < n_topics:
        raise DataValidationError("There must be at least as many documents as topics.")
    identifiers: set[tuple[str, str]] = set()
    for record in records:
        record.validate()
        identity = (record.platform, record.record_id)
        if identity in identifiers:
            raise DataValidationError(
                f"Duplicate record_id in {record.platform}: {record.record_id}"
            )
        identifiers.add(identity)


def _sample_pairs(n_documents: int, max_pairs: int, seed: int) -> Iterator[tuple[int, int]]:
    """Sample unordered pairs without constructing an O(n²) pair list."""
    total = n_documents * (n_documents - 1) // 2
    ranks = sorted(random.Random(seed).sample(range(total), min(total, max_pairs)))
    for rank in ranks:
        low, high = 0, n_documents - 1
        while low + 1 < high:
            middle = (low + high) // 2
            preceding = middle * (2 * n_documents - middle - 1) // 2
            if preceding <= rank:
                low = middle
            else:
                high = middle
        preceding = low * (2 * n_documents - low - 1) // 2
        yield low, low + 1 + rank - preceding


def _npmi_coherence(counts: csr_matrix, term_indices: Sequence[int]) -> float:
    """Mean NPMI from exact document presence, not token-frequency products.

    Unseen pairs have -1. Universal pairs have 0: their NPMI is undefined and
    they contain no information distinguishing documents. Only the at-most
    50 selected terms are densified; the document matrix always stays sparse.
    """
    if len(term_indices) < 2:
        return 0.0
    presence = counts[:, list(term_indices)].astype(np.int64).copy()
    presence.data.fill(1)
    frequency = np.asarray(presence.sum(axis=0)).ravel()
    cooccurrence = (presence.T @ presence).toarray()
    n_documents = counts.shape[0]
    scores: list[float] = []
    for left, right in combinations(range(len(term_indices)), 2):
        joint = float(cooccurrence[left, right]) / n_documents
        if joint == 0.0:
            scores.append(-1.0)
        elif joint == 1.0:
            scores.append(0.0)
        else:
            product = float(frequency[left] * frequency[right]) / n_documents**2
            scores.append(math.log(joint / product) / -math.log(joint))
    return float(np.mean(scores))


def _proximity_rows(
    records: Sequence[Record],
    counts: csr_matrix,
    tfidf: csr_matrix,
    max_pairs: int,
    seed: int,
) -> Iterator[dict[str, Any]]:
    for left, right in _sample_pairs(len(records), max_pairs, seed):
        left_vector = tfidf.getrow(left)
        right_vector = tfidf.getrow(right)
        row: dict[str, Any] = {
            "record_id_a": records[left].record_id,
            "platform_a": records[left].platform,
            "record_id_b": records[right].record_id,
            "platform_b": records[right].platform,
            "cosine_similarity": "",
            "jaccard_similarity": "",
            "euclidean_distance": "",
            "pair_status": "zero_feature_document",
        }
        if left_vector.nnz and right_vector.nnz:
            cosine = float(left_vector.multiply(right_vector).sum())
            cosine = float(np.clip(cosine, 0.0, 1.0))
            left_terms = set(counts.getrow(left).indices)
            right_terms = set(counts.getrow(right).indices)
            row.update(
                cosine_similarity=cosine,
                jaccard_similarity=len(left_terms & right_terms) / len(left_terms | right_terms),
                euclidean_distance=math.sqrt(max(0.0, 2.0 - 2.0 * cosine)),
                pair_status="ok",
            )
        yield row


def analyze(
    records: list[Record],
    output_dir: Path,
    n_topics: int = 3,
    seed: int = 42,
    max_features: int = 5000,
    top_terms: int = 10,
    max_pairs: int = 5000,
) -> dict[str, Path]:
    """Write descriptive analysis; callers enforce their required sample size.

    LDA uses counts and proximity uses L2-normalized TF-IDF in the same fixed
    vocabulary. Topic mixture weights and rule scores are not model confidence.
    """
    _validate_inputs(records, n_topics, seed, max_features, top_terms, max_pairs)
    cleaned = [clean_text(record.text) for record in records]
    vectorizer = CountVectorizer(
        tokenizer=_topic_tokens,
        token_pattern=None,
        lowercase=False,
        max_features=max_features,
        dtype=np.int64,
    )
    try:
        counts = vectorizer.fit_transform(cleaned).tocsr()
    except ValueError as exc:
        raise DataValidationError(
            "No topic vocabulary remains after cleaning and stopword removal."
        ) from exc
    valid_mask = np.asarray(counts.getnnz(axis=1)).ravel() > 0
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) < n_topics or counts.shape[1] < n_topics:
        raise DataValidationError(
            "Topic analysis needs at least n_topics documents with features and "
            "at least n_topics distinct vocabulary terms."
        )
    model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=seed,
        learning_method="batch",
        max_iter=20,
        n_jobs=1,
    )
    topic_weights = model.fit_transform(counts[valid_indices])
    perplexity = float(model.perplexity(counts[valid_indices]))
    if not np.isfinite(topic_weights).all() or not math.isfinite(perplexity):
        raise PipelineError("Topic model produced non-finite results; inspect the input corpus.")
    vocabulary = vectorizer.get_feature_names_out()
    selected_terms = [
        np.argsort(component, kind="stable")[::-1][: min(top_terms, len(vocabulary))]
        for component in model.components_
    ]
    topic_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    coherences: list[float] = []
    for topic_id, (component, selected) in enumerate(
        zip(model.components_, selected_terms, strict=True)
    ):
        denominator = float(component.sum())
        for rank, term_index in enumerate(selected, start=1):
            topic_rows.append(
                {
                    "topic_id": topic_id,
                    "rank": rank,
                    "term": vocabulary[term_index],
                    "probability": float(component[term_index]) / denominator,
                }
            )
        coherence = _npmi_coherence(counts[valid_indices], selected)
        coherences.append(coherence)
        metric_rows.append(
            {
                "metric": "npmi_coherence",
                "topic_id": topic_id,
                "value": coherence,
                "evaluation_scope": "in_sample_document_presence",
                "notes": "Unseen pairs=-1; universal pairs=0; all top-term pairs are averaged.",
            }
        )
    unique_terms = set(int(index) for topic in selected_terms for index in topic)
    diversity = len(unique_terms) / sum(len(topic) for topic in selected_terms)
    for metric, value, note in (
        ("lda_perplexity", perplexity, "In-sample fit diagnostic; not held-out topic quality."),
        ("mean_npmi_coherence", float(np.mean(coherences)), "Mean of topic NPMI values."),
        ("topic_diversity", diversity, "Unique top terms / all selected top-term slots."),
        ("documents_with_features", len(valid_indices), "Documents used to fit LDA."),
        ("documents_without_features", len(records) - len(valid_indices), "Not assigned topics."),
        ("vocabulary_size", len(vocabulary), "Retained topic and proximity vocabulary."),
    ):
        metric_rows.append(
            {
                "metric": metric,
                "topic_id": "",
                "value": value,
                "evaluation_scope": "in_sample",
                "notes": note,
            }
        )

    classifier = SentimentClassifier()
    local_zone = ZoneInfo("Asia/Manila")
    assignments = {
        int(index): (int(weights.argmax()), float(weights.max()))
        for index, weights in zip(valid_indices, topic_weights, strict=True)
    }
    documents: list[dict[str, Any]] = []
    # Store sums and counts, never all group members or an n-by-n matrix.
    groups: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {label: [0.0, 0.0] for label in LABELS}
    )
    for index, record in enumerate(records):
        result = classifier.predict(record.text)
        topic_id, probability = assignments.get(index, ("", ""))
        documents.append(
            {
                **asdict(record),
                "clean_text": cleaned[index],
                "topic_id": topic_id,
                "topic_probability": probability,
                "topic_status": "assigned" if valid_mask[index] else "zero_features",
                "sentiment_label": result.label,
                "sentiment_score": result.score,
                "sentiment_evidence": json.dumps(result.matched_terms, ensure_ascii=False),
                "sentiment_model_version": classifier.version,
            }
        )
        month = parse_timestamp(record.created_at).astimezone(local_zone).strftime("%Y-%m")
        group = groups[(str(topic_id), record.platform, record.record_type, month)]
        group[result.label][0] += 1
        group[result.label][1] += result.score
    summary: list[dict[str, Any]] = []
    for (topic_id, platform, record_type, month), values in sorted(groups.items()):
        total = int(sum(value[0] for value in values.values()))
        for label in LABELS:
            count, score_sum = values[label]
            summary.append(
                {
                    "topic_id": topic_id,
                    "platform": platform,
                    "record_type": record_type,
                    "month": month,
                    "label": label,
                    "count": int(count),
                    "total": total,
                    "share": count / total,
                    "mean_score": score_sum / count if count else "",
                }
            )
    paths = {
        "documents": output_dir / "analyzed_documents.csv",
        "topic_terms": output_dir / "topic_terms.csv",
        "topic_metrics": output_dir / "topic_metrics.csv",
        "sentiment_summary": output_dir / "sentiment_summary.csv",
        "proximity_pairs": output_dir / "proximity_pairs.csv",
    }
    write_csv(paths["documents"], documents, tuple(documents[0]))
    write_csv(paths["topic_terms"], topic_rows, ("topic_id", "rank", "term", "probability"))
    write_csv(
        paths["topic_metrics"],
        metric_rows,
        (
            "metric",
            "topic_id",
            "value",
            "evaluation_scope",
            "notes",
        ),
    )
    write_csv(
        paths["sentiment_summary"],
        summary,
        (
            "topic_id",
            "platform",
            "record_type",
            "month",
            "label",
            "count",
            "total",
            "share",
            "mean_score",
        ),
    )
    tfidf = TfidfTransformer(norm="l2").fit_transform(counts).tocsr()
    write_csv(
        paths["proximity_pairs"],
        _proximity_rows(records, counts, tfidf, max_pairs, seed),
        (
            "record_id_a",
            "platform_a",
            "record_id_b",
            "platform_b",
            "cosine_similarity",
            "jaccard_similarity",
            "euclidean_distance",
            "pair_status",
        ),
    )
    return paths
