"""Apply audited reference annotations; never infer labels from model predictions."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import replace

from .errors import DataValidationError
from .schema import LABELS, Record

DECISION_FIELDS = (
    "index",
    "record_id",
    "platform",
    "relevance",
    "label",
    "reason",
    "content_kind",
    "context_needed",
    "label_source",
    "annotator",
    "annotation_date",
    "rubric_version",
    "text_sha256",
)


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def curate_records(
    candidates: Sequence[Record], decisions: Sequence[Mapping[str, str]]
) -> list[Record]:
    """Join complete, hash-bound decisions to unique prepared candidate records.

    Excluded and unresolved rows have no reference sentiment. Included records
    retain their frozen text and provenance and receive only the assigned label.
    An empty or incomplete audit is an error, never a neutral fallback.
    """
    if not candidates:
        raise DataValidationError("Curation requires prepared candidate records.")
    records: dict[tuple[str, str], Record] = {}
    for record in candidates:
        record.validate()
        identity = (record.platform, record.record_id)
        if identity in records:
            raise DataValidationError("Duplicate candidate record identity.")
        if record.gold_label:
            raise DataValidationError("Candidates must be unlabelled before independent curation.")
        records[identity] = record
    audited: dict[tuple[str, str], Mapping[str, str]] = {}
    for decision in decisions:
        if any(not isinstance(decision.get(field), str) for field in DECISION_FIELDS):
            raise DataValidationError("Annotation decision is missing a required string column.")
        identity = (decision["platform"], decision["record_id"])
        if identity not in records or identity in audited:
            raise DataValidationError("Annotation IDs must match candidates exactly once.")
        record = records[identity]
        if decision["text_sha256"] != text_digest(record.text):
            raise DataValidationError(
                "Annotation text checksum differs from the prepared candidate."
            )
        if not decision["reason"].strip() or not decision["annotator"].strip():
            raise DataValidationError(
                "Every decision requires a rationale and annotator provenance."
            )
        if decision["label_source"] not in {"human", "ai_assisted"}:
            raise DataValidationError("Annotation label_source must be human or ai_assisted.")
        relevance, label = decision["relevance"], decision["label"]
        if relevance not in {"include", "exclude", "unresolved"}:
            raise DataValidationError("Invalid annotation relevance decision.")
        if decision["context_needed"] not in {"yes", "no"}:
            raise DataValidationError("context_needed must be yes or no.")
        if relevance == "include":
            if label not in LABELS or decision["context_needed"] != "no":
                raise DataValidationError(
                    "Included records need a valid label and resolved context."
                )
        elif label:
            raise DataValidationError(
                "Excluded or unresolved records cannot carry sentiment labels."
            )
        audited[identity] = decision
    if set(audited) != set(records):
        raise DataValidationError("Annotation audit must cover every prepared candidate.")
    return [
        replace(record, gold_label=audited[identity]["label"])
        for identity, record in records.items()
        if audited[identity]["relevance"] == "include"
    ]
