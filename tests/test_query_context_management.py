"""
Tests for query.py's conversation-memory context management --
compact_message() and trim_history(). These exist to stop a long
conversation's context from growing unboundedly (each turn's retrieved
background text is ~2000+ tokens; without stripping it from history,
a handful of turns would blow past a local model's context window).

Run with: pytest
"""

import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

sys.modules.setdefault("chromadb", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

from query import compact_message, trim_history, build_context_message


def test_compact_message_drops_background_keeps_question():
    compacted = compact_message("What is virtue?")
    assert compacted == "QUESTION: What is virtue?"
    assert "BACKGROUND" not in compacted


def test_compact_message_much_shorter_than_full_context_message():
    passages = [f"[Author {i} — Work {i}]\n" + ("x" * 800) for i in range(10)]
    full = build_context_message("What is virtue?", passages)
    compacted = compact_message("What is virtue?")
    assert len(compacted) < len(full) / 10  # should be dramatically smaller


def test_trim_history_leaves_short_conversations_untouched():
    messages = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    result = trim_history(messages, max_exchanges=8)
    assert result == messages


def test_trim_history_caps_at_max_exchanges():
    # 10 exchanges (20 messages) with a cap of 3 exchanges (6 messages)
    system_msg = {"role": "system", "content": "persona"}
    conversation = []
    for i in range(10):
        conversation.append({"role": "user", "content": f"q{i}"})
        conversation.append({"role": "assistant", "content": f"a{i}"})
    messages = [system_msg] + conversation

    result = trim_history(messages, max_exchanges=3)

    assert result[0] == system_msg  # system message always preserved
    assert len(result) == 1 + (3 * 2)  # system + 3 exchanges worth of messages
    # Should keep the MOST RECENT exchanges, not the oldest
    assert result[1]["content"] == "q7"
    assert result[-1]["content"] == "a9"


def test_trim_history_preserves_system_message_even_when_trimming():
    system_msg = {"role": "system", "content": "unique persona text"}
    conversation = [{"role": "user", "content": f"q{i}"} for i in range(20)]
    messages = [system_msg] + conversation

    result = trim_history(messages, max_exchanges=2)
    assert result[0]["content"] == "unique persona text"


def test_repeated_compact_and_trim_keeps_context_bounded():
    """Simulates many turns of conversation and confirms total context
    size stabilizes instead of growing without bound."""
    system_msg = {"role": "system", "content": "persona"}
    messages = [system_msg]
    passages = [f"[Author {i} — Work {i}]\n" + ("x" * 800) for i in range(10)]

    sizes = []
    for turn_num in range(15):
        question = f"Question {turn_num}?"
        messages.append({"role": "user", "content": build_context_message(question, passages)})
        messages.append({"role": "assistant", "content": f"Answer {turn_num}."})
        messages[-2]["content"] = compact_message(question)
        messages = trim_history(messages, max_exchanges=8)
        sizes.append(sum(len(m["content"]) for m in messages))

    # After the cap kicks in (turn 8 onward), size should stop growing
    late_sizes = sizes[8:]
    assert max(late_sizes) - min(late_sizes) < 200, f"Context kept growing after cap: {late_sizes}"