"""Verify the committed research handoff survives downloads on every CI platform."""

import csv
import hashlib
from pathlib import Path

import nbformat

from ph_sentiment.config import load_config
from ph_sentiment.csvio import read_records
from ph_sentiment.live import PUBLIC_PLATFORMS
from ph_sentiment.preprocessing import prepare_records

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "deliverables/public_2026"


def test_committed_snapshot_checksums_and_csv_row_counts_match():
    with (SNAPSHOT / "file_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    assert len({row["path"] for row in manifest}) == len(manifest)
    assert {row["path"] for row in manifest} == {
        path.relative_to(SNAPSHOT).as_posix()
        for path in SNAPSHOT.rglob("*")
        if path.is_file() and path.name != "file_manifest.csv"
    }
    for row in manifest:
        path = (SNAPSHOT / row["path"]).resolve()
        assert path.is_relative_to(SNAPSHOT.resolve())
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]
        assert len(payload) == int(row["bytes"])
        if path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                assert len(list(csv.DictReader(stream))) == int(row["rows"])


def test_committed_snapshot_has_two_real_sources_and_blank_independent_labels():
    config = load_config(SNAPSHOT / "config.toml")
    research = config["research"]
    raw = [
        record
        for platform in PUBLIC_PLATFORMS
        for record in read_records(SNAPSHOT / "raw" / f"{platform}.csv")
    ]
    expected, _ = prepare_records(raw, research["start_date"], research["end_date"])
    assert len(expected) >= 200
    assert {record.platform for record in expected} == set(PUBLIC_PLATFORMS)
    assert read_records(SNAPSHOT / "processed/posts.csv") == expected
    assert read_records(SNAPSHOT / "labels/annotation.csv") == expected
    assert all(not record.gold_label for record in raw)
    assert not (SNAPSHOT / "evaluation").exists()


def test_committed_notebook_is_clean_and_defaults_to_offline_handoff():
    notebook = nbformat.read(ROOT / "notebooks/philippine_government_2026.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is None and not cell.outputs for cell in code_cells)
    sources = "\n".join(cell.source for cell in code_cells)
    assert '"deliverables" / "public_2026"' in sources
    assert "RUN_EVALUATION = False" in sources
    assert '"collect-live"' not in sources and '"collect-public"' not in sources
    assert "sha256" in sources
    for cell in code_cells:
        compile(cell.source, "handoff-notebook", "exec")
