"""Atomic CSV persistence with explicit UTF-8 and spreadsheet formula protection."""

import csv
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .errors import DataValidationError
from .schema import RECORD_FIELDS, Record


def safe_cell(value: Any) -> Any:
    # Numeric metrics remain numeric; text capable of starting a formula is quoted as text.
    # Escape existing apostrophes as well, so canonical CSV round trips are reversible.
    if isinstance(value, str) and (
        value.startswith("'") or value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@"))
    ):
        return "'" + value
    return value


def write_csv(
    path: Path | str, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]
) -> Path:
    """Replace a complete file atomically, never leaving a truncated dataset after failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: safe_cell(value) for key, value in row.items()})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def write_records(path: Path | str, records: Iterable[Record]) -> Path:
    return write_csv(path, (record.to_dict() for record in records), RECORD_FIELDS)


def read_records(path: Path | str) -> list[Record]:
    """Read the canonical schema; validation/filtering is performed in preprocessing."""
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"record_id", "platform", "created_at", "text"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise DataValidationError(
                    "CSV must contain record_id, platform, created_at, and text."
                )
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise DataValidationError("CSV contains duplicate column names.")
            records = []
            for line_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise DataValidationError(
                        f"CSV row {line_number} has missing or excess columns."
                    )
                values = {key: row[key] for key in RECORD_FIELDS if key in row}
                for key in values:
                    # Reverse only our formula-protection marker for canonical round trips.
                    if values[key].startswith("'") and safe_cell(values[key][1:]) == values[key]:
                        values[key] = values[key][1:]
                records.append(Record(**values))
            return records
    except (csv.Error, UnicodeError) as exc:
        raise DataValidationError(
            "Could not read CSV. Use UTF-8 and correctly quoted text cells."
        ) from exc
