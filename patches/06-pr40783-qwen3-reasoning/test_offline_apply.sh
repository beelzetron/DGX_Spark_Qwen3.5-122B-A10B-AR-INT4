#!/usr/bin/env bash
# Offline validation: apply patches to v0.19.0 source snapshots (no Docker/GPU).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

VLLM_REF="${VLLM_REF:-v0.19.0}"
BASE="https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_REF}"

mkdir -p "$TMP/vllm/reasoning" \
         "$TMP/vllm/tool_parsers" \
         "$TMP/vllm/entrypoints/openai/chat_completion"

curl -fsSL "${BASE}/vllm/reasoning/qwen3_reasoning_parser.py" \
  -o "$TMP/vllm/reasoning/qwen3_reasoning_parser.py"
curl -fsSL "${BASE}/vllm/tool_parsers/utils.py" \
  -o "$TMP/vllm/tool_parsers/utils.py"
curl -fsSL "${BASE}/vllm/tool_parsers/qwen3xml_tool_parser.py" \
  -o "$TMP/vllm/tool_parsers/qwen3xml_tool_parser.py"
curl -fsSL "${BASE}/vllm/entrypoints/openai/chat_completion/serving.py" \
  -o "$TMP/vllm/entrypoints/openai/chat_completion/serving.py"

export VLLM_PKG="$TMP/vllm"
python3 "$ROOT/patches/06-pr40783-qwen3-reasoning/apply_pr40783_patches.py"

grep -q partial_tag_overlap "$TMP/vllm/tool_parsers/utils.py"
grep -q count_reasoning_tokens "$TMP/vllm/reasoning/qwen3_reasoning_parser.py"
grep -q DGX_SPARK_PR40783 "$TMP/vllm/entrypoints/openai/chat_completion/serving.py"
! grep -q DGX_SPARK_PR40783_XML "$TMP/vllm/tool_parsers/qwen3xml_tool_parser.py"
! grep -q DGX_SPARK_QWEN3XML_PARAM_BODY "$TMP/vllm/tool_parsers/qwen3xml_tool_parser.py"

# Idempotent re-run
python3 "$ROOT/patches/06-pr40783-qwen3-reasoning/apply_pr40783_patches.py"

echo "OK: offline apply test passed (vLLM ${VLLM_REF})"
