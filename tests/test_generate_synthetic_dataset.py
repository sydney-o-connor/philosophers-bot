"""
Tests for generate_synthetic_dataset.py's JSON-extraction logic --
this is the part most likely to break, since it depends on parsing
imperfect output from a local LLM.

Run with: pytest
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generate_synthetic_dataset import extract_json


def test_extract_json_parses_clean_json():
    raw = '{"instruction": "What is duty?", "output": "Duty is acting from respect for the moral law, not from inclination."}'
    result = extract_json(raw)
    assert result is not None
    assert result["instruction"] == "What is duty?"


def test_extract_json_strips_markdown_fences():
    raw = '```json\n{"instruction": "What is virtue?", "output": "Virtue is a mean between two vices, one of excess and one of deficiency."}\n```'
    result = extract_json(raw)
    assert result is not None
    assert "virtue" in result["instruction"].lower()


def test_extract_json_handles_preamble_text():
    raw = 'Sure, here is the JSON:\n\n{"instruction": "Is pleasure the good?", "output": "Pleasure alone cannot be the good, since some pleasures degrade rather than fulfill us."}'
    result = extract_json(raw)
    assert result is not None


def test_extract_json_rejects_too_short_output():
    raw = '{"instruction": "What is good?", "output": "Yes."}'
    result = extract_json(raw)
    assert result is None


def test_extract_json_rejects_malformed_json():
    raw = '{"instruction": "broken" "output": "missing comma"}'
    result = extract_json(raw)
    assert result is None


def test_extract_json_rejects_wrong_keys():
    raw = '{"question": "wrong schema", "answer": "wrong schema"}'
    result = extract_json(raw)
    assert result is None