from pathlib import Path

import pytest

from ph_sentiment.csvio import read_records, safe_cell, write_csv, write_records
from ph_sentiment.errors import DataValidationError
from ph_sentiment.schema import Record


def test_multilingual_multiline_formula_roundtrip(tmp_path: Path):
    record = Record(
        "r1", "reddit", "2026-06-01T00:00:00Z", '=HYPERLINK("bad")\nMagaling, pero mahal! 🇵🇭'
    )
    path = tmp_path / "posts.csv"
    write_records(path, [record])
    assert read_records(path) == [record]
    assert "'=HYPERLINK" in path.read_text(encoding="utf-8-sig")
    assert safe_cell(-0.5) == -0.5
    assert safe_cell("\t +CMD()") == "'\t +CMD()"


def test_failed_write_preserves_existing_dataset(tmp_path: Path):
    path = tmp_path / "posts.csv"
    path.write_text("original", encoding="utf-8")

    def broken_rows():
        yield {"text": "first"}
        raise RuntimeError("interrupted source")

    with pytest.raises(RuntimeError):
        write_csv(path, broken_rows(), ["text"])
    assert path.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("text", ["=hello", "'=hello", "''=hello", "'ordinary", "ordinary"])
def test_preexisting_apostrophes_roundtrip(tmp_path: Path, text: str):
    row = Record("id", "reddit", "2026-06-01T00:00:00Z", text)
    path = tmp_path / "apostrophes.csv"
    write_records(path, [row])
    assert read_records(path) == [row]


def test_rejects_ambiguous_csv_columns(tmp_path: Path):
    path = tmp_path / "posts.csv"
    path.write_text(
        "record_id,platform,created_at,text\nr,reddit,2026-01-01T00:00:00Z,text,extra\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="excess columns"):
        read_records(path)
