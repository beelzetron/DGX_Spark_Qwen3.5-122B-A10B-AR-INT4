#!/usr/bin/env python3
"""Apply vLLM PR #35687 + #40783 + serving port + partial_tag_overlap for v0.19.0.

Backports Qwen3 reasoning parser fixes for agentic workflows on DGX Spark without
rebuilding the vllm-sm121 NVCC base image. Idempotent — safe to re-run.
"""

from __future__ import annotations

import os
import shutil
import sys

MARKER = "DGX_SPARK_PR40783"

VLLM_PKG = os.environ.get(
    "VLLM_PKG",
    "/usr/local/lib/python3.12/dist-packages/vllm",
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REASONING_SRC = os.path.join(SCRIPT_DIR, "qwen3_reasoning_parser.py")

TARGETS = {
    "reasoning": os.path.join(VLLM_PKG, "reasoning", "qwen3_reasoning_parser.py"),
    "utils": os.path.join(VLLM_PKG, "tool_parsers", "utils.py"),
    "serving": os.path.join(
        VLLM_PKG, "entrypoints", "openai", "chat_completion", "serving.py"
    ),
    "qwen3xml": os.path.join(VLLM_PKG, "tool_parsers", "qwen3xml_tool_parser.py"),
}


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _write(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)


def apply_reasoning_parser() -> None:
    dst = TARGETS["reasoning"]
    if not os.path.isfile(REASONING_SRC):
        print(f"FAIL: missing {REASONING_SRC}")
        sys.exit(1)
    shutil.copy2(REASONING_SRC, dst)
    print(f"OK: installed {dst} (PR #35687 + #40783)")


def apply_partial_tag_overlap() -> None:
    path = TARGETS["utils"]
    content = _read(path)
    if "def partial_tag_overlap" in content:
        print(f"SKIP: partial_tag_overlap already in {path}")
        return
    anchor = "logger = init_logger(__name__)\n"
    if anchor not in content:
        print(f"FAIL: anchor not found in {path}")
        sys.exit(1)
    insert = '''logger = init_logger(__name__)


def partial_tag_overlap(text: str, tag: str) -> int:
    """Length of the longest prefix of *tag* that matches a suffix of *text*.

    E.g. text ending in ``"<tool_"`` returns 6 when tag is ``"<tool_call>"``.
    Returns 0 when there is no overlap.
    """
    max_check = min(len(tag) - 1, len(text))
    for k in range(max_check, 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0
'''
    _write(path, content.replace(anchor, insert, 1))
    print(f"OK: added partial_tag_overlap to {path}")


def _fix_serving_delta_message_shadow(content: str) -> tuple[str, bool]:
    """Remove inline DeltaMessage import that shadows module-level binding."""
    bad = """                                else:
                                    from vllm.entrypoints.openai.engine.protocol import (
                                        DeltaMessage,
                                    )

                                    delta_message = DeltaMessage(
                                        reasoning=reasoning_from_transition
                                    )"""
    good = """                                else:
                                    delta_message = DeltaMessage(
                                        reasoning=reasoning_from_transition
                                    )"""
    if bad not in content:
        return content, False
    return content.replace(bad, good, 1), True


def apply_serving() -> None:
    path = TARGETS["serving"]
    content = _read(path)
    if MARKER in content:
        content, fixed = _fix_serving_delta_message_shadow(content)
        if fixed:
            _write(path, content)
            print(f"OK: fixed DeltaMessage UnboundLocalError in {path}")
        else:
            print(f"SKIP: serving patch already applied in {path}")
        return

    old_end_check = """                                if reasoning_parser.is_reasoning_end(output_token_ids):
                                    reasoning_end_arr[i] = True
                                    current_token_ids = (
                                        reasoning_parser.extract_content_ids(
                                            output_token_ids
                                        )
                                    )"""

    new_end_check = f"""                                # {MARKER}: use streaming end check for MTP/single-delta tool_call
                                _reasoning_end_fn = getattr(
                                    reasoning_parser,
                                    "is_reasoning_end_streaming",
                                    None,
                                )
                                if _reasoning_end_fn is not None:
                                    _reasoning_ended = _reasoning_end_fn(
                                        current_token_ids, output_token_ids
                                    )
                                else:
                                    _reasoning_ended = reasoning_parser.is_reasoning_end(
                                        output_token_ids
                                    )
                                if _reasoning_ended:
                                    reasoning_end_arr[i] = True
                                    current_token_ids = (
                                        reasoning_parser.extract_content_ids(
                                            current_token_ids
                                        )
                                    )"""

    if old_end_check not in content:
        print(f"FAIL: is_reasoning_end anchor not found in {path}")
        sys.exit(1)
    content = content.replace(old_end_check, new_end_check, 1)

    old_tool_handoff = """                            delta_message = tool_parser.extract_tool_calls_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=delta_token_ids,
                                request=request,
                            )
                            if delta_message and delta_message.tool_calls:
                                tools_streamed[i] = True
                    # when only tool calls
                    elif tool_choice_auto:"""

    new_tool_handoff = f"""                            # {MARKER}: preserve reasoning text from transition delta
                            reasoning_from_transition = (
                                delta_message.reasoning
                                if delta_message is not None
                                else None
                            )
                            delta_message = tool_parser.extract_tool_calls_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=delta_token_ids,
                                request=request,
                            )
                            if reasoning_from_transition:
                                if delta_message is not None:
                                    delta_message.reasoning = (
                                        reasoning_from_transition
                                    )
                                else:
                                    delta_message = DeltaMessage(
                                        reasoning=reasoning_from_transition
                                    )
                            if delta_message and delta_message.tool_calls:
                                tools_streamed[i] = True
                    # when only tool calls
                    elif tool_choice_auto:"""

    if old_tool_handoff not in content:
        print(f"FAIL: tool handoff anchor not found in {path}")
        sys.exit(1)
    content = content.replace(old_tool_handoff, new_tool_handoff, 1)

    _write(path, content)
    print(f"OK: patched {path} ({MARKER})")


def apply_qwen3xml_partial_tag() -> None:
    """Minimal PR #40861 slice: withhold partial <tool_call> prefix in XML parser."""
    path = TARGETS["qwen3xml"]
    content = _read(path)
    if f"{MARKER}_XML" in content:
        print(f"SKIP: qwen3xml patch already applied in {path}")
        return

    import_anchor = (
        "from vllm.tool_parsers.abstract_tool_parser import (\n"
        "    Tool,\n"
        "    ToolParser,\n"
        ")\n"
    )
    import_replacement = (
        "from vllm.tool_parsers.abstract_tool_parser import (\n"
        "    Tool,\n"
        "    ToolParser,\n"
        ")\n"
        "from vllm.tool_parsers.utils import partial_tag_overlap\n"
    )
    if "partial_tag_overlap" not in content:
        if import_anchor not in content:
            print(f"FAIL: qwen3xml import anchor not found in {path}")
            sys.exit(1)
        content = content.replace(import_anchor, import_replacement, 1)

    old_emit = """            if self.text_content_buffer and self.tool_call_index == 0:
                # Has text content but no tool_call yet, output text content
                text_delta = DeltaMessage(content=self.text_content_buffer)
                self._emit_delta(text_delta)
                # Clear buffer to avoid duplicate output
                self.text_content_buffer = ""
                return text_delta"""

    new_emit = f"""            if self.text_content_buffer and self.tool_call_index == 0:
                # {MARKER}_XML: withhold suffix that could be start of <tool_call>
                overlap = partial_tag_overlap(
                    self.streaming_buffer, self.tool_call_start_token
                )
                emit_len = len(self.text_content_buffer) - overlap
                if emit_len > 0:
                    text_delta = DeltaMessage(
                        content=self.text_content_buffer[:emit_len]
                    )
                    self.text_content_buffer = self.text_content_buffer[emit_len:]
                    self._emit_delta(text_delta)
                    return text_delta
                if overlap > 0:
                    return DeltaMessage()
                text_delta = DeltaMessage(content=self.text_content_buffer)
                self._emit_delta(text_delta)
                self.text_content_buffer = ""
                return text_delta"""

    if old_emit not in content:
        print(f"FAIL: qwen3xml text_content_buffer anchor not found in {path}")
        sys.exit(1)
    content = content.replace(old_emit, new_emit, 1)

    _write(path, content)
    print(f"OK: patched {path} ({MARKER}_XML)")


def main() -> None:
    for name, path in TARGETS.items():
        if not os.path.isfile(path):
            print(f"FAIL: {name} target not found: {path}")
            sys.exit(1)

    apply_reasoning_parser()
    apply_partial_tag_overlap()
    apply_serving()
    apply_qwen3xml_partial_tag()
    print(f"OK: all {MARKER} patches applied")


if __name__ == "__main__":
    main()
