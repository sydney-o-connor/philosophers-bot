"""
Basic tests for download_texts.py's text-cleaning logic.
Run with: pytest
"""

import sys
import os

# Make sure the project root (one level up from tests/) is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from download_texts import strip_gutenberg_boilerplate


def test_strip_gutenberg_boilerplate_removes_header_and_footer():
    raw = """Some Gutenberg legal header text here...
*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***

This is the actual book content.
It has multiple lines.

*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***
Some footer legal text here...
"""
    cleaned = strip_gutenberg_boilerplate(raw)

    assert "actual book content" in cleaned
    assert "START OF" not in cleaned
    assert "END OF" not in cleaned
    assert "legal header" not in cleaned
    assert "legal text" not in cleaned


def test_strip_gutenberg_boilerplate_handles_missing_markers():
    # If there are no START/END markers, the function should just return
    # the text unchanged (stripped of whitespace) rather than erroring.
    raw = "   Just some plain text with no Gutenberg markers.   "
    cleaned = strip_gutenberg_boilerplate(raw)
    assert cleaned == "Just some plain text with no Gutenberg markers."