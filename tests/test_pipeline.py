import csv
from pathlib import Path

import pytest

from ph_sentiment.cli import _prepared_input, main, parser
from ph_sentiment.config import DEFAULT_CONFIG, load_config
from ph_sentiment.csvio import read_records, write_records
from ph_sentiment.schema import Record


def sample_records():
    # Synthetic test fixtures only; never presented as observed public opinion.
    return [
        Record(str(index), "threads" if index % 2 else "x", "2026-06-01T00:00:00Z", text)
        for index, text in enumerate(
            [
                "good government health hospitals affordable medicine",
                "bad healthcare queues poor public hospital service",
                "neutral health budget announced today",
                "maganda edukasyon paaralan mahusay guro serbisyo",
                "bad schools poor education classroom shortage",
                "education teachers school budget public meeting",
                "good economy jobs improving transport wages",
                "bad inflation mahal presyo pagkain trabaho",
                "economy transport fuel budget government report",
            ]
        )
    ]


def test_analysis_validation_preserves_canonical_csv_text(tmp_path):
    # A once-decoded nested entity can still look like an entity. Validation
    # must not replace the canonical text with a second decoding of that text.
    records = [
        Record("1", "bluesky", "2026-06-01T00:00:00+00:00", "government health &amp; services"),
        Record("2", "mastodon", "2026-06-01T00:00:00+00:00", "public education and teachers"),
        Record("3", "bluesky", "2026-06-01T00:00:00+00:00", "government transport budgets"),
    ]
    path = write_records(tmp_path / "prepared.csv", records)
    assert _prepared_input(path, load_config(DEFAULT_CONFIG), 3) == records


def test_offline_prepare_analyze_label_pipeline(tmp_path: Path):
    raw, processed, audit = [tmp_path / f"{name}.csv" for name in ("raw", "processed", "audit")]
    write_records(raw, sample_records())
    common = ["--config", str(DEFAULT_CONFIG)]
    assert (
        main(
            [
                "prepare",
                "--input",
                str(raw),
                "--output",
                str(processed),
                "--audit-output",
                str(audit),
                *common,
            ]
        )
        == 0
    )
    assert len(read_records(processed)) == 9
    output = tmp_path / "analysis"
    assert (
        main(
            [
                "analyze",
                "--input",
                str(processed),
                "--output-dir",
                str(output),
                "--minimum-records",
                "3",
                *common,
            ]
        )
        == 0
    )
    assert len(list(output.glob("*.csv"))) >= 5
    with (output / "analyzed_documents.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 9
    labels = tmp_path / "labels.csv"
    assert main(["label", "--input", str(processed), "--output", str(labels), *common]) == 0
    assert all(not row.gold_label for row in read_records(labels))
    assert main(["label", "--input", str(processed), "--output", str(labels), *common]) == 2


def test_default_activity_minimum_and_input_protection(tmp_path: Path):
    raw = tmp_path / "raw.csv"
    write_records(raw, sample_records())
    assert main(["analyze", "--input", str(raw), "--output-dir", str(tmp_path / "analysis")]) == 2
    assert (
        main(
            [
                "prepare",
                "--input",
                str(raw),
                "--output",
                str(raw),
                "--audit-output",
                str(tmp_path / "audit.csv"),
                "--overwrite",
            ]
        )
        == 2
    )
    assert len(read_records(raw)) == 9


@pytest.mark.parametrize("label_source", ["human", "ai_assisted"])
def test_evaluation_cli_reserves_holdout_at_activity_scale(tmp_path: Path, label_source: str):
    # Independently specified synthetic software labels; these are not public observations.
    cues = {"positive": "good", "negative": "bad", "neutral": "information"}
    records = [
        Record(
            f"{label}-{index}",
            "threads" if index % 2 else "x",
            "2026-06-01T00:00:00Z",
            f"Government policy {cue} testexample{index}",
            gold_label=label,
        )
        for label, cue in cues.items()
        for index in range(70)
    ]
    source = tmp_path / "human_fixture.csv"
    write_records(source, records)
    output = tmp_path / "evaluation"
    arguments = ["evaluate", "--input", str(source), "--output-dir", str(output)]
    if label_source == "ai_assisted":
        arguments += ["--label-source", label_source]
    assert main(arguments) == 0
    with (output / "split_assignments.csv").open(encoding="utf-8-sig", newline="") as stream:
        assignments = list(csv.DictReader(stream))
    assert len(assignments) == 210
    assert sum(row["split"] == "holdout" for row in assignments) == 21
    assert {row["validation_fold"] for row in assignments if row["split"] == "development"} == {
        str(index) for index in range(1, 11)
    }
    assert {row["label_source"] for row in assignments} == {label_source}
    if label_source == "ai_assisted":
        assert all("not human-validated accuracy" in row["notes"] for row in assignments)


def test_evaluation_cli_rejects_unknown_label_provenance():
    with pytest.raises(SystemExit) as captured:
        parser().parse_args(
            [
                "evaluate",
                "--input",
                "labels.csv",
                "--output-dir",
                "evaluation",
                "--label-source",
                "model_predictions",
            ]
        )
    assert captured.value.code == 2


def test_collect_cli_writes_manifest_and_closes_client(tmp_path: Path, monkeypatch):
    from ph_sentiment import collectors

    class FakeCollector:
        warnings = ["Test-only bounded coverage"]
        closed = False

        def collect(self):
            return sample_records()

        def close(self):
            self.closed = True

    collector = FakeCollector()
    monkeypatch.setattr(collectors, "make_collector", lambda platform, config: collector)
    output = tmp_path / "threads.csv"
    assert main(["collect", "--platform", "threads", "--output", str(output)]) == 0
    assert collector.closed
    assert len(read_records(output)) == 9
    with (tmp_path / "threads_collection_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        manifest = list(csv.DictReader(stream))
    assert manifest[0]["collected_records"] == "9"
    assert manifest[0]["warnings"] == collector.warnings[0]


@pytest.mark.parametrize("command", ["collect", "collect-live"])
@pytest.mark.parametrize("platform", ["threads", "x", "bluesky", "mastodon"])
def test_collection_cli_accepts_the_four_main_platforms(command: str, platform: str):
    arguments = [command, "--platform", platform]
    if command == "collect":
        arguments += ["--output", "source.csv"]
    selected = parser().parse_args(arguments)
    assert selected.platform == (platform if command == "collect" else [platform])


@pytest.mark.parametrize("command", ["collect", "collect-live", "collect-public"])
@pytest.mark.parametrize("platform", ["reddit", "facebook"])
def test_collection_cli_rejects_removed_platforms(command: str, platform: str):
    with pytest.raises(SystemExit) as captured:
        parser().parse_args([command, "--platform", platform])
    assert captured.value.code == 2
