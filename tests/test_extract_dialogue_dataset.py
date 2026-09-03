"""
Tests for extract_dialogue_dataset.py's speaker-turn parsing and
pair-building logic -- the trickiest part of the dataset pipeline,
since it depends on correctly interpreting Gutenberg's dialogue
formatting conventions.

Run with: pytest
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from extract_dialogue_dataset import extract_turns, build_pairs, parse_metadata


def test_parse_metadata_extracts_author_and_title():
    text = "AUTHOR: Plato\nTITLE: Euthyphro\n\nSOCRATES. Hello there."
    author, title, body = parse_metadata(text)
    assert author == "Plato"
    assert title == "Euthyphro"
    assert "SOCRATES" in body


def test_extract_turns_groups_multiline_speeches():
    body = """SOCRATES. This is the first line of a speech.
This continues on a second line, same speaker.

EUTHYPHRO. Now a different speaker responds here with enough length."""
    turns = extract_turns(body)
    assert len(turns) == 2
    assert turns[0][0] == "Socrates"
    assert "first line" in turns[0][1] and "second line" in turns[0][1]
    assert turns[1][0] == "Euthyphro"


def test_extract_turns_ignores_lines_before_first_speaker():
    body = """Some narration or stage direction with no speaker label.

SOCRATES. The actual dialogue starts here with this speaker."""
    turns = extract_turns(body)
    assert len(turns) == 1
    assert turns[0][0] == "Socrates"


def test_extract_turns_handles_multi_word_speaker_names():
    body = """FIRST CITIZEN. A speech from a multi-word speaker name here.

SOCRATES. A reply from a single-word speaker name in response."""
    turns = extract_turns(body)
    assert len(turns) == 2
    assert turns[0][0] == "First Citizen"


def test_extract_turns_empty_body_returns_no_turns():
    assert extract_turns("") == []
    assert extract_turns("Just plain narration, no speaker labels at all.") == []


def test_build_pairs_creates_pair_for_alternating_speakers():
    turns = [
        ("Socrates", "What is piety, can you tell me its nature clearly?"),
        ("Euthyphro", "Piety is that which is dear to the gods, Socrates, and impiety is not."),
    ]
    pairs = build_pairs(turns, "Plato", "Euthyphro")
    assert len(pairs) == 1
    assert pairs[0]["speaker_prompt"] == "Socrates"
    assert pairs[0]["speaker_response"] == "Euthyphro"
    assert pairs[0]["philosopher"] == "Plato"
    assert pairs[0]["work"] == "Euthyphro"


def test_build_pairs_skips_same_speaker_consecutive_turns():
    # Shouldn't happen given how extract_turns groups things, but
    # build_pairs should be defensive about it anyway.
    turns = [
        ("Socrates", "First remark, long enough to pass the length filter easily."),
        ("Socrates", "A second remark from the same speaker, also long enough."),
    ]
    pairs = build_pairs(turns, "Plato", "Test")
    assert len(pairs) == 0


def test_build_pairs_skips_short_responses():
    turns = [
        ("Socrates", "A substantive question worth asking about virtue and the good."),
        ("Euthyphro", "Yes."),  # too short -- an acknowledgment, not a real response
    ]
    pairs = build_pairs(turns, "Plato", "Test")
    assert len(pairs) == 0


def test_build_pairs_skips_short_questions():
    turns = [
        ("Socrates", "Hm?"),  # too short
        ("Euthyphro", "A full, substantive response that is long enough to count."),
    ]
    pairs = build_pairs(turns, "Plato", "Test")
    assert len(pairs) == 0


def test_build_pairs_handles_three_speaker_chain():
    # Each consecutive different-speaker pair should become its own
    # training example.
    turns = [
        ("Socrates", "A question posed to open the dialogue about justice."),
        ("Glaucon", "A response from the second speaker, long enough to count."),
        ("Adeimantus", "A third speaker joins in with their own substantive point."),
    ]
    pairs = build_pairs(turns, "Plato", "The Republic")
    assert len(pairs) == 2
    assert pairs[0]["speaker_prompt"] == "Socrates"
    assert pairs[0]["speaker_response"] == "Glaucon"
    assert pairs[1]["speaker_prompt"] == "Glaucon"
    assert pairs[1]["speaker_response"] == "Adeimantus"