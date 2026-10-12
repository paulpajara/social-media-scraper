"""Check notebook protections against stale results and incompatible reference labels."""

import ast
import csv
import hashlib
import html
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

from ph_sentiment.csvio import read_records, write_csv, write_records
from ph_sentiment.schema import Record

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def helpers(tmp_path):
    notebook = nbformat.read(ROOT / "notebooks/philippine_government_2026.ipynb", as_version=4)
    setup = ast.parse(notebook.cells[2].source)
    functions = ast.Module(
        body=[node for node in setup.body if isinstance(node, ast.FunctionDef)], type_ignores=[]
    )
    namespace = {
        "Path": Path,
        "csv": csv,
        "hashlib": hashlib,
        "html": html,
        "sys": sys,
        "version": lambda name: "fixed-test-version",
        "replace": replace,
        "read_records": read_records,
        "write_csv": write_csv,
        "REPO": tmp_path,
        "DATA_DIR": tmp_path / "working",
        "FORCE_REBUILD": False,
    }
    exec(compile(functions, "notebook-helpers", "exec"), namespace)
    return namespace


def test_cached_outputs_require_unchanged_inputs_and_complete_output_hashes(helpers):
    output = helpers["DATA_DIR"] / "analysis" / "example.csv"
    write_csv(output, [{"value": 1}], ("value",))
    outputs = (output,)
    helpers["remember_stage"]("analysis", outputs, "same-input")
    assert helpers["cached_stage"]("analysis", outputs, "same-input")
    with pytest.raises(RuntimeError, match="inputs or outputs changed"):
        helpers["cached_stage"]("analysis", outputs, "different-input")
    write_csv(output, [{"value": 2}], ("value",))
    with pytest.raises(RuntimeError, match="inputs or outputs changed"):
        helpers["cached_stage"]("analysis", outputs, "same-input")
    helpers["FORCE_REBUILD"] = True
    assert not helpers["cached_stage"]("analysis", outputs, "same-input")


def test_partial_output_without_provenance_requires_explicit_rebuild(helpers):
    first = helpers["DATA_DIR"] / "evaluation" / "first.csv"
    second = first.with_name("second.csv")
    assert not helpers["cached_stage"]("evaluation", (first, second), "input")
    write_csv(first, [{"value": 1}], ("value",))
    with pytest.raises(RuntimeError, match="previous run is incomplete"):
        helpers["cached_stage"]("evaluation", (first, second), "input")


def test_stage_fingerprint_changes_with_input_implementation_and_label_source(helpers):
    repo = helpers["REPO"]
    module_dir = repo / "src" / "ph_sentiment"
    module_dir.mkdir(parents=True)
    for name in (
        "schema.py",
        "csvio.py",
        "preprocessing.py",
        "sentiment.py",
        "cli.py",
        "config.py",
        "evaluation.py",
    ):
        (module_dir / name).write_text("# first version", encoding="utf-8")
    inputs = repo / "labels.csv"
    inputs.write_text("first input", encoding="utf-8")
    fingerprint = helpers["stage_fingerprint"]
    original = fingerprint("evaluation", (inputs,), "ai_assisted")
    assert original != fingerprint("evaluation", (inputs,), "human")
    inputs.write_text("second input", encoding="utf-8")
    assert original != fingerprint("evaluation", (inputs,), "ai_assisted")
    inputs.write_text("first input", encoding="utf-8")
    (module_dir / "sentiment.py").write_text("# second version", encoding="utf-8")
    assert original != fingerprint("evaluation", (inputs,), "ai_assisted")


def test_stage_fingerprint_changes_with_runtime_and_dependency_versions(helpers):
    module_dir = helpers["REPO"] / "src" / "ph_sentiment"
    module_dir.mkdir(parents=True)
    for name in (
        "schema.py",
        "csvio.py",
        "preprocessing.py",
        "sentiment.py",
        "cli.py",
        "config.py",
        "analysis.py",
    ):
        (module_dir / name).write_text("# current version", encoding="utf-8")
    fingerprint = helpers["stage_fingerprint"]
    original = fingerprint("analysis", ())
    helpers["version"] = lambda name: "changed-version" if name == "numpy" else "fixed-test-version"
    assert original != fingerprint("analysis", ())
    helpers["version"] = lambda name: "fixed-test-version"
    helpers["sys"] = SimpleNamespace(version=f"{sys.version}-changed-runtime")
    assert original != fingerprint("analysis", ())
    helpers["sys"] = sys
    (module_dir / "config.py").write_text("# new version", encoding="utf-8")
    assert original != fingerprint("analysis", ())


def test_preview_filters_exact_scope_and_escapes_html(helpers):
    path = helpers["REPO"] / "metrics.csv"
    write_csv(
        path,
        [
            {"split": "cv_fold", "metric": "wrong-scope", "value": 1},
            {"split": "holdout", "metric": "<script>alert('x')</script>", "value": 2},
            {"split": "holdout_other", "metric": "also-wrong-scope", "value": 3},
        ],
        ("split", "metric", "value"),
    )
    rendered = []
    helpers["HTML"] = lambda payload: payload
    helpers["display"] = rendered.append
    helpers["preview_csv"](path, where={"split": "holdout"})
    assert len(rendered) == 1
    assert "wrong-scope" not in rendered[0]
    assert "<script>" not in rendered[0]
    assert "&lt;script&gt;" in rendered[0]
    with pytest.raises(ValueError, match="missing CSV column"):
        helpers["preview_csv"](path, where={"missing": "holdout"})


def test_reference_validation_preserves_human_changes_and_rejects_stale_text(helpers):
    record = Record(
        record_id="post-1",
        platform="bluesky",
        created_at="2026-10-01T00:00:00Z",
        text="Government policy discussion",
        gold_label="neutral",
    )
    annotation = helpers["REPO"] / "annotation.csv"
    revised = replace(record, gold_label="negative")
    write_records(annotation, [revised])
    before = annotation.read_bytes()
    assert helpers["verify_annotation_records"](annotation, [record]) == [revised]
    with pytest.raises(ValueError, match="labels do not match"):
        helpers["verify_annotation_records"](annotation, [record], require_supplied=True)
    assert annotation.read_bytes() == before
    write_records(annotation, [replace(revised, text="Different post text")])
    with pytest.raises(ValueError, match="text/provenance changed"):
        helpers["verify_annotation_records"](annotation, [record])


def test_reference_validation_rejects_blank_labels_and_duplicate_records(helpers):
    record = Record(
        record_id="post-1",
        platform="mastodon",
        created_at="2026-10-01T00:00:00Z",
        text="Government policy discussion",
        gold_label="positive",
    )
    annotation = helpers["REPO"] / "annotation.csv"
    write_records(annotation, [replace(record, gold_label="")])
    with pytest.raises(ValueError, match="Incomplete reference label"):
        helpers["verify_annotation_records"](annotation, [record])
    write_records(annotation, [record, record])
    with pytest.raises(ValueError, match="IDs/platforms must match"):
        helpers["verify_annotation_records"](annotation, [record])
