"""Completed reference audits produce an immutable, credential-free offline handoff."""

import pytest

from ph_sentiment.annotation_context import CONTEXT_FIELDS
from ph_sentiment.config import load_config
from ph_sentiment.csvio import read_records, write_csv, write_records
from ph_sentiment.curation import DECISION_FIELDS, curate_records, text_digest
from ph_sentiment.errors import DataValidationError
from ph_sentiment.handoff import _table, verify_integrity_manifest, write_integrity_manifest
from ph_sentiment.labeled_handoff import export_labeled_handoff
from ph_sentiment.preprocessing import prepare_records
from ph_sentiment.schema import Record


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    (source / "raw").mkdir(parents=True)
    (source / "config.toml").write_text(
        """[research]
start_date="2026-01-01"
end_date="2027-01-01"
queries=["Philippine government"]
[analysis]
minimum_records=200
max_pairs=5
""",
        encoding="utf-8",
    )
    for platform in ("bluesky", "mastodon"):
        write_records(
            source / "raw" / f"{platform}.csv",
            [
                Record(
                    f"{platform}:fixture-{i}",
                    platform,
                    "2026-09-01T00:00:00Z",
                    f"Philippine government policy {platform} fixture{i}",
                )
                for i in range(105)
            ],
        )
    # An unexpected source file must never be copied to a deliverable.
    (source / "raw/.env").write_text("FAKE_SECRET=synthetic-private-fixture", encoding="utf-8")
    write_csv(source / "collection_report.csv", [{"platform": "bluesky"}], ("platform",))
    write_csv(
        source / "data_dictionary.csv",
        [{"field": "gold_label", "type": "string", "description": "Blank reference column."}],
        ("field", "type", "description"),
    )
    write_integrity_manifest(source)
    config = load_config(source / "config.toml")
    raw = [r for p in ("bluesky", "mastodon") for r in read_records(source / "raw" / f"{p}.csv")]
    candidates, _ = prepare_records(
        raw, config["research"]["start_date"], config["research"]["end_date"]
    )
    decisions = [
        {
            "index": str(i),
            "record_id": r.record_id,
            "platform": r.platform,
            "relevance": "include",
            "label": ("negative", "neutral", "positive")[i % 3],
            "reason": "Synthetic software-test reference; not research data.",
            "content_kind": "other",
            "context_needed": "no",
            "label_source": "ai_assisted",
            "annotator": "test_fixture",
            "annotation_date": "2026-10-12",
            "rubric_version": "test",
            "text_sha256": text_digest(r.text),
        }
        for i, r in enumerate(candidates)
    ]
    decision_path = tmp_path / "decisions.csv"
    write_csv(decision_path, decisions, DECISION_FIELDS)
    context_path = tmp_path / "context.csv"
    write_csv(
        context_path,
        [
            dict.fromkeys(CONTEXT_FIELDS, "")
            | {"record_id": r.record_id, "platform": r.platform, "context_status": "not_requested"}
            for r in candidates
        ],
        CONTEXT_FIELDS,
    )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic software-test protocol.", encoding="utf-8")
    return source, decision_path, context_path, protocol, tmp_path / "output"


def test_export_reproduces_reference_records_and_all_evaluation_artifacts(inputs):
    output = export_labeled_handoff(*inputs)
    verify_integrity_manifest(output)
    candidates = read_records(output / "processed/candidates.csv")
    _, decisions = _table(output / "labels/annotation_decisions.csv")
    expected = curate_records(candidates, decisions)
    assert len(expected) == 210
    assert read_records(output / "processed/posts.csv") == expected
    assert read_records(output / "labels/annotation.csv") == expected
    assert len(list((output / "analysis").glob("*.csv"))) == 5
    assert len(list((output / "evaluation").glob("*.csv"))) == 5
    for path in (output / "evaluation").glob("*.csv"):
        _, rows = _table(path)
        assert {r["label_source"] for r in rows} == {"ai_assisted"}
    assert not (output / "raw/.env").exists()
    with pytest.raises(DataValidationError, match="already exists"):
        export_labeled_handoff(*inputs)


def test_failed_split_does_not_publish_partial_handoff(inputs):
    source, decision_path, context_path, protocol, output = inputs
    _, decisions = _table(decision_path)
    for row in decisions:
        row["label"] = "neutral"
    write_csv(decision_path, decisions, DECISION_FIELDS)
    with pytest.raises(DataValidationError, match="every class"):
        export_labeled_handoff(source, decision_path, context_path, protocol, output)
    assert not output.exists()
    assert not list(output.parent.glob(".labeled-*"))


def test_source_tampering_fails_before_output(inputs):
    source, _, _, _, output = inputs
    (source / "config.toml").write_text("tampered", encoding="utf-8")
    with pytest.raises(DataValidationError, match="checksum"):
        export_labeled_handoff(*inputs)
    assert not output.exists()


def test_manifest_refresh_never_includes_itself(inputs):
    source = inputs[0]
    first = (source / "file_manifest.csv").read_bytes()
    write_integrity_manifest(source)
    verify_integrity_manifest(source)
    assert (source / "file_manifest.csv").read_bytes() == first
