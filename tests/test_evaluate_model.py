"""
Tests for evaluate_model.py's judge-output parsing -- the part most
likely to break, since it depends on a local LLM reliably producing
well-formed JSON, which it doesn't always do.

Run with: pytest
"""

import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# evaluate_model.py imports chromadb/sentence_transformers via query.py
# at module load time -- mock them so tests don't need those heavy
# packages installed just to check the parsing logic.
sys.modules.setdefault("chromadb", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

import evaluate_model as em

VALID_SCORES = (
    '{"validity": {"score": 4, "note": "sound"}, '
    '"focus": {"score": 5, "note": "on topic"}, '
    '"groundedness": {"score": 3, "note": "mostly ok"}, '
    '"synthesis": {"score": 4, "note": "coherent"}, '
    '"progress": {"score": null, "note": "n/a"}, '
    '"error_recovery": {"score": 5, "note": "honest"}, '
    '"voice_consistency": {"score": 4, "note": "consistent"}}'
)


def test_extract_json_block_parses_clean_json():
    result = em.extract_json_block(VALID_SCORES)
    assert result is not None
    assert result["validity"]["score"] == 4


def test_extract_json_block_strips_markdown_fences():
    fenced = f"```json\n{VALID_SCORES}\n```"
    result = em.extract_json_block(fenced)
    assert result is not None
    assert result["focus"]["score"] == 5


def test_extract_json_block_handles_preamble_text():
    with_preamble = f"Here are the scores:\n\n{VALID_SCORES}"
    result = em.extract_json_block(with_preamble)
    assert result is not None


def test_extract_json_block_returns_none_for_garbage():
    assert em.extract_json_block("not valid json at all, sorry") is None


def test_extract_json_block_returns_none_for_malformed_json():
    broken = '{"validity": {"score": 4 "note": "missing comma"}}'
    assert em.extract_json_block(broken) is None


def test_judge_conversation_accepts_complete_valid_scores(monkeypatch):
    monkeypatch.setattr(
        em.ollama, "chat",
        lambda model, messages: {"message": {"content": VALID_SCORES}}
    )
    transcript = [{"speaker": "user", "text": "q"}, {"speaker": "bot", "text": "a"}]
    scores = em.judge_conversation(transcript, "fake-judge", is_multi_turn=True)
    assert scores is not None
    for key in em.RUBRIC:
        assert key in scores


def test_judge_conversation_forces_progress_none_for_single_turn(monkeypatch):
    monkeypatch.setattr(
        em.ollama, "chat",
        lambda model, messages: {"message": {"content": VALID_SCORES}}
    )
    transcript = [{"speaker": "user", "text": "q"}, {"speaker": "bot", "text": "a"}]
    scores = em.judge_conversation(transcript, "fake-judge", is_multi_turn=False)
    assert scores is not None
    assert scores["progress"]["score"] is None
    assert "single-turn" in scores["progress"]["note"]


def test_judge_conversation_returns_none_on_missing_rubric_key(monkeypatch):
    incomplete = '{"validity": {"score": 4, "note": "x"}}'
    monkeypatch.setattr(
        em.ollama, "chat",
        lambda model, messages: {"message": {"content": incomplete}}
    )
    transcript = [{"speaker": "user", "text": "q"}, {"speaker": "bot", "text": "a"}]
    scores = em.judge_conversation(transcript, "fake-judge", is_multi_turn=True)
    assert scores is None


def test_judge_conversation_returns_none_on_unparseable_output(monkeypatch):
    monkeypatch.setattr(
        em.ollama, "chat",
        lambda model, messages: {"message": {"content": "I refuse to output JSON today."}}
    )
    transcript = [{"speaker": "user", "text": "q"}, {"speaker": "bot", "text": "a"}]
    scores = em.judge_conversation(transcript, "fake-judge", is_multi_turn=True)
    assert scores is None


def test_format_transcript_labels_speakers_correctly():
    transcript = [
        {"speaker": "user", "text": "Is virtue teachable?"},
        {"speaker": "bot", "text": "That depends on what we mean by virtue."},
    ]
    formatted = em.format_transcript(transcript)
    assert "User: Is virtue teachable?" in formatted
    assert "Bot: That depends on what we mean by virtue." in formatted