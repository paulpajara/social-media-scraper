import csv
from pathlib import Path

from ph_sentiment.cli import main
from ph_sentiment.config import DEFAULT_CONFIG
from ph_sentiment.csvio import read_records, write_records
from ph_sentiment.schema import Record


def sample_records():
    # Synthetic test fixtures only; never presented as observed public opinion.
    return [
        Record(str(index), "reddit" if index % 2 else "x", "2026-06-01T00:00:00Z", text)
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


def test_evaluation_cli_reserves_holdout_at_activity_scale(tmp_path: Path):
    # Independently specified synthetic software labels; these are not public observations.
    cues = {"positive": "good", "negative": "bad", "neutral": "information"}
    records = [
        Record(
            f"{label}-{index}",
            "reddit" if index % 2 else "x",
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
    assert main(["evaluate", "--input", str(source), "--output-dir", str(output)]) == 0
    with (output / "split_assignments.csv").open(encoding="utf-8-sig", newline="") as stream:
        assignments = list(csv.DictReader(stream))
    assert len(assignments) == 210
    assert sum(row["split"] == "holdout" for row in assignments) == 21
    assert {row["validation_fold"] for row in assignments if row["split"] == "development"} == {
        str(index) for index in range(1, 11)
    }


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
    output = tmp_path / "reddit.csv"
    assert main(["collect", "--platform", "reddit", "--output", str(output)]) == 0
    assert collector.closed
    assert len(read_records(output)) == 9
    with (tmp_path / "reddit_collection_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        manifest = list(csv.DictReader(stream))
    assert manifest[0]["collected_records"] == "9"
    assert manifest[0]["warnings"] == collector.warnings[0]
