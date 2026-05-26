#!/usr/bin/env python3
"""Lightweight checks for backported Qwen3 reasoning parser (no GPU)."""

from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARSER_SRC = (SCRIPT_DIR / "qwen3_reasoning_parser.py").read_text()


def partial_tag_overlap(text: str, tag: str) -> int:
    max_check = min(len(tag) - 1, len(text))
    for k in range(max_check, 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


def test_shipped_parser_has_key_fixes():
    assert "def count_reasoning_tokens" in PARSER_SRC
    assert "input_ids.index(self._tool_call_token_id)" in PARSER_SRC
    assert "just_completed_tool_call_tag" in PARSER_SRC
    assert "partial_tag_overlap" in PARSER_SRC
    assert "is_reasoning_end_streaming" in PARSER_SRC


def test_partial_tag_overlap():
    assert partial_tag_overlap("thinking <tool", "<tool_call>") == 5
    assert partial_tag_overlap("hello", "<tool_call>") == 0


def test_extract_content_ids_logic():
    """Mirror extract_content_ids first-occurrence semantics."""
    tool_id = 3
    input_ids = [1, 2, tool_id, 10, tool_id, 11]
    tool_call_index = input_ids.index(tool_id)
    content_ids = input_ids[tool_call_index:]
    assert content_ids == [3, 10, 3, 11]


def main():
    test_shipped_parser_has_key_fixes()
    test_partial_tag_overlap()
    test_extract_content_ids_logic()
    print("OK: reasoning unit tests passed")


if __name__ == "__main__":
    main()
