"""Tests for core/dataset.py - the test-set loader."""

from __future__ import annotations

import json

import pytest

from rag_score.core.dataset import load_dataset


class TestLoadDataset:
    def test_loads_valid_json(self, tmp_path):
        data = [
            {"question": "Q1?", "expected_doc_ids": ["doc_1"]},
            {"question": "Q2?", "expected_doc_ids": ["doc_2"]},
        ]
        path = tmp_path / "test_set.json"
        path.write_text(json.dumps(data))

        cases = load_dataset(path)
        assert len(cases) == 2
        assert cases[0].question == "Q1?"

    def test_infers_dataset_name_from_filename(self, tmp_path):
        path = tmp_path / "my_dataset.json"
        path.write_text(json.dumps([{"question": "Q?"}]))

        cases = load_dataset(path)
        assert cases[0].dataset_name == "my_dataset"

    def test_explicit_dataset_name_overrides_filename(self, tmp_path):
        path = tmp_path / "my_dataset.json"
        path.write_text(json.dumps([{"question": "Q?"}]))

        cases = load_dataset(path, dataset_name="custom")
        assert cases[0].dataset_name == "custom"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "does_not_exist.json")

    def test_missing_question_field_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([{"expected_doc_ids": ["doc_1"]}]))
        with pytest.raises(ValueError, match="question"):
            load_dataset(path)

    def test_non_list_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"question": "not a list"}))
        with pytest.raises(ValueError, match="list"):
            load_dataset(path)

    def test_optional_fields_default_correctly(self, tmp_path):
        path = tmp_path / "minimal.json"
        path.write_text(json.dumps([{"question": "Q?"}]))
        cases = load_dataset(path)
        assert cases[0].expected_doc_ids == []
        assert cases[0].ground_truth_answer is None
