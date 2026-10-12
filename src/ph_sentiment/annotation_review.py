"""Compute actual annotation consistency and apply documented AI adjudications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from random import Random

from .curation import curate_records, text_digest
from .errors import DataValidationError
from .schema import Record

ADJUDICATION_FIELDS = (
    "index",
    "record_id",
    "platform",
    "primary_relevance",
    "primary_label",
    "review_relevance",
    "review_label",
    "final_relevance",
    "final_label",
    "reason",
    "adjudicator",
    "annotation_date",
    "text_sha256",
)
QUALITY_FIELDS = ("metric", "numerator", "denominator", "value", "sampling_seed", "notes")


def _identity(row: Mapping[str, str]) -> tuple[str, str]:
    return row["platform"], row["record_id"]


def adjudicate_annotations(
    candidates: Sequence[Record],
    primary: Sequence[Mapping[str, str]],
    review: Sequence[Mapping[str, str]],
    resolutions: Sequence[Mapping[str, str]],
    *,
    sampling_seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str | int | float]]]:
    """Compare pre-adjudication decisions; resolve every observed disagreement.

    Agreement is measured on the independently selected review subset before
    any root changes. The conditional sentiment rate uses only rows both
    annotators included. These are AI consistency checks, not truth accuracy.
    """
    curate_records(candidates, primary)
    if (
        isinstance(sampling_seed, bool)
        or not isinstance(sampling_seed, int)
        or not 0 <= sampling_seed < 2**32
    ):
        raise DataValidationError("sampling_seed must be an integer from 0 to 2**32 - 1.")
    candidate_by_id = {(record.platform, record.record_id): record for record in candidates}
    primary_by_id = {_identity(row): row for row in primary}
    review_by_id = {_identity(row): row for row in review}
    if (
        not review
        or len(review_by_id) != len(review)
        or not set(review_by_id) <= set(primary_by_id)
    ):
        raise DataValidationError(
            "Independent review requires a unique, nonempty candidate subset."
        )
    expected_sample = {
        (candidates[i].platform, candidates[i].record_id)
        for i in Random(sampling_seed).sample(range(len(candidates)), len(review))
    }
    if set(review_by_id) != expected_sample:
        raise DataValidationError(
            "Independent review IDs do not match the declared random sample seed."
        )
    curate_records([candidate_by_id[_identity(row)] for row in review], review)
    both_included = [
        key
        for key in review_by_id
        if primary_by_id[key]["relevance"] == review_by_id[key]["relevance"] == "include"
    ]
    counts = (
        (
            "relevance_agreement",
            sum(
                primary_by_id[key]["relevance"] == row["relevance"]
                for key, row in review_by_id.items()
            ),
            len(review),
        ),
        (
            "exact_decision_agreement",
            sum(
                (primary_by_id[key]["relevance"], primary_by_id[key]["label"])
                == (row["relevance"], row["label"])
                for key, row in review_by_id.items()
            ),
            len(review),
        ),
        (
            "sentiment_agreement_both_included",
            sum(primary_by_id[key]["label"] == review_by_id[key]["label"] for key in both_included),
            len(both_included),
        ),
    )
    quality = [
        {
            "metric": metric,
            "numerator": numerator,
            "denominator": denominator,
            "value": numerator / denominator if denominator else "",
            "sampling_seed": sampling_seed,
            "notes": "Pre-adjudication blind AI sample agreement; not human validation or predictive accuracy. Sentiment denominator includes only posts both annotators retained.",
        }
        for metric, numerator, denominator in counts
    ]
    expected_disagreements = {
        key
        for key, row in review_by_id.items()
        if (primary_by_id[key]["relevance"], primary_by_id[key]["label"])
        != (row["relevance"], row["label"])
    }
    resolved = set()
    final = {key: dict(row) for key, row in primary_by_id.items()}
    for resolution in resolutions:
        if any(not isinstance(resolution.get(field), str) for field in ADJUDICATION_FIELDS):
            raise DataValidationError("Adjudication requires every audit column.")
        key = _identity(resolution)
        if key not in final or key in resolved:
            raise DataValidationError("Adjudication IDs must be known and unique.")
        original = primary_by_id[key]
        independently_reviewed = review_by_id.get(key)
        expected_review = (
            (independently_reviewed["relevance"], independently_reviewed["label"])
            if independently_reviewed
            else ("not_sampled", "")
        )
        if (
            (resolution["primary_relevance"], resolution["primary_label"])
            != (original["relevance"], original["label"])
            or (resolution["review_relevance"], resolution["review_label"]) != expected_review
            or resolution["text_sha256"] != text_digest(candidate_by_id[key].text)
        ):
            raise DataValidationError(
                "Adjudication does not match its original decisions and text."
            )
        if not resolution["reason"].strip() or not resolution["adjudicator"].strip():
            raise DataValidationError("Adjudication needs a rationale and reviewer provenance.")
        final[key].update(
            relevance=resolution["final_relevance"],
            label=resolution["final_label"],
            reason=resolution["reason"],
            annotator=resolution["adjudicator"],
            context_needed="yes" if resolution["final_relevance"] == "unresolved" else "no",
        )
        resolved.add(key)
    if not expected_disagreements <= resolved:
        raise DataValidationError("Resolve every independently observed annotation disagreement.")
    decisions = [final[_identity(row)] for row in primary]
    curate_records(candidates, decisions)
    return decisions, quality
