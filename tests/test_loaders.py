import json
from pathlib import Path

from docx import Document

from personal_search_layer.ingestion.loaders import load_document


def test_load_docx(tmp_path: Path) -> None:
    docx_path = tmp_path / "sample.docx"
    doc = Document()
    doc.add_paragraph("Hello Docx")
    doc.save(docx_path)

    loaded, report = load_document(docx_path)
    assert report.skip_reason is None
    assert loaded is not None
    assert "Hello Docx" in loaded.blocks[0].text


def test_load_ipynb(tmp_path: Path) -> None:
    ipynb_path = tmp_path / "sample.ipynb"
    payload = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Title\n", "Note"]},
            {"cell_type": "code", "source": ["print('hi')"]},
        ]
    }
    ipynb_path.write_text(json.dumps(payload))

    loaded, report = load_document(ipynb_path)
    assert report.skip_reason is None
    assert loaded is not None
    text = loaded.blocks[0].text
    assert "Title" in text
    assert "print('hi')" in text


def test_load_csv_and_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n")
    loaded_csv, report_csv = load_document(csv_path)
    assert report_csv.skip_reason is None
    assert loaded_csv is not None
    assert "a" in loaded_csv.blocks[0].text

    json_path = tmp_path / "sample.json"
    json_path.write_text(json.dumps({"key": "value"}))
    loaded_json, report_json = load_document(json_path)
    assert report_json.skip_reason is None
    assert loaded_json is not None
    assert "key" in loaded_json.blocks[0].text
