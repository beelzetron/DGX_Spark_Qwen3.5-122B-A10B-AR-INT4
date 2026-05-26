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
MARKER_PARAM_BODY = "DGX_SPARK_QWEN3XML_PARAM_BODY"

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


def revert_qwen3xml_partial_tag() -> None:
    """Remove minimal #40861 slice; it used the wrong buffer and broke XML parsing.

    Fragmented <tool_call> handling stays in qwen3_reasoning_parser.py only.
    """
    path = TARGETS["qwen3xml"]
    content = _read(path)
    if f"{MARKER}_XML" not in content:
        print(f"SKIP: no qwen3xml partial-tag patch to revert in {path}")
        return

    old_emit = """            if self.text_content_buffer and self.tool_call_index == 0:
                # Has text content but no tool_call yet, output text content
                text_delta = DeltaMessage(content=self.text_content_buffer)
                self._emit_delta(text_delta)
                # Clear buffer to avoid duplicate output
                self.text_content_buffer = ""
                return text_delta"""

    patched_emit_variants = [
        f"""            if self.text_content_buffer and self.tool_call_index == 0:
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
                return text_delta""",
        f"""            if self.text_content_buffer and self.tool_call_index == 0:
                # {MARKER}_XML: withhold suffix that could be start of <tool_call>
                overlap = partial_tag_overlap(
                    self.text_content_buffer, self.tool_call_start_token
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
                return text_delta""",
    ]

    reverted = False
    for patched_emit in patched_emit_variants:
        if patched_emit in content:
            content = content.replace(patched_emit, old_emit, 1)
            reverted = True
            break
    if not reverted:
        print(f"FAIL: {MARKER}_XML marker present but emit block not found in {path}")
        sys.exit(1)

    import_extra = "from vllm.tool_parsers.utils import partial_tag_overlap\n"
    if import_extra in content:
        content = content.replace(import_extra, "", 1)

    _write(path, content)
    print(f"OK: reverted qwen3xml partial-tag patch in {path}")


def apply_qwen3xml_param_body_fix() -> None:
    """Stop feeding JSON parameter bodies (and '<' in values) to expat.

    v0.19.0 _should_skip_element returns False for any text while a tool call is
    active, so _find_next_complete_element splits on '<' inside JSON and expat
    logs 'not well-formed (invalid token)'. Route parameter body to _pre_param_buffer.
    """
    path = TARGETS["qwen3xml"]
    content = _read(path)
    if MARKER_PARAM_BODY in content:
        print(f"SKIP: qwen3xml param-body fix already applied in {path}")
        return

    helper_anchor = "    def _should_skip_element(self, element: str) -> bool:"
    helper_method = f'''    def _is_tool_xml_element(self, element: str) -> bool:
        """{MARKER_PARAM_BODY}: tag fragment vs parameter JSON/plain body."""
        if not element:
            return False
        stripped = element.lstrip()
        prefixes = (
            self.tool_call_start_token,
            self.tool_call_end_token,
            self.function_start_token,
            self.function_end_token,
            self.parameter_start_token,
            self.parameter_end_token,
            "<function=",
            "<parameter=",
        )
        return any(stripped.startswith(p) for p in prefixes)

    def _should_skip_element(self, element: str) -> bool:'''
    if helper_anchor not in content:
        print(f"FAIL: _should_skip_element anchor not found in {path}")
        sys.exit(1)
    content = content.replace(helper_anchor, helper_method, 1)

    loop_anchor = """            if self._should_skip_element(element):
                self.last_processed_pos = end_pos
                continue

            # Found complete XML element, process it"""
    loop_replacement = f"""            if self._should_skip_element(element):
                self.last_processed_pos = end_pos
                continue

            # {MARKER_PARAM_BODY}: parameter JSON must not go through expat
            if self.current_call_id is not None and not self._is_tool_xml_element(
                element
            ):
                if self._pre_inside_parameter or self.current_param_name:
                    self._pre_param_buffer += element
                    found_any = True
                self.last_processed_pos = end_pos
                continue

            # Found complete XML element, process it"""
    if loop_anchor not in content:
        print(f"FAIL: _process_complete_xml_elements loop anchor not found in {path}")
        sys.exit(1)
    content = content.replace(loop_anchor, loop_replacement, 1)

    find_anchor = """        buffer = self.streaming_buffer[start_pos:]

        if not buffer:
            return None, start_pos

        if buffer.startswith("<"):"""
    find_replacement = f"""        buffer = self.streaming_buffer[start_pos:]

        if not buffer:
            return None, start_pos

        # {MARKER_PARAM_BODY}: do not treat '<' inside parameter values as new tags
        if self._pre_inside_parameter:
            close_pos = buffer.find(self.parameter_end_token)
            if close_pos != -1:
                return buffer[:close_pos], start_pos + close_pos
            return None, start_pos

        if buffer.startswith("<"):"""
    if find_anchor not in content:
        print(f"FAIL: _find_next_complete_element anchor not found in {path}")
        sys.exit(1)
    content = content.replace(find_anchor, find_replacement, 1)

    _write(path, content)
    print(f"OK: patched {path} ({MARKER_PARAM_BODY})")


def main() -> None:
    for name, path in TARGETS.items():
        if not os.path.isfile(path):
            print(f"FAIL: {name} target not found: {path}")
            sys.exit(1)

    apply_reasoning_parser()
    apply_partial_tag_overlap()
    apply_serving()
    revert_qwen3xml_partial_tag()
    apply_qwen3xml_param_body_fix()
    print(f"OK: all {MARKER} patches applied")


if __name__ == "__main__":
    main()
