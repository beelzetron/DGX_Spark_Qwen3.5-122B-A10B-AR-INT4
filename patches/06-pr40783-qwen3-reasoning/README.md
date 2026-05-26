# PR #40783: Qwen3 reasoning parser backport (v0.19.0)

Backports [vLLM PR #35687](https://github.com/vllm-project/vllm/pull/35687) and
[PR #40783](https://github.com/vllm-project/vllm/pull/40783) for agentic workflows
on the pinned **v0.19.0** DGX Spark stack, without rebuilding `vllm-sm121`.

## What it fixes

- Fragmented `<tool_call>` tags across streaming deltas
- Lost last reasoning fragment at reasoning→tool-call transition
- MTP/spec-decode delivering a complete `<tool_call>…</tool_call>` in one delta
- Multi-tool flows dropping all but the last tool call
- `count_reasoning_tokens` returning 0 for Qwen3.5+ outputs
- Withheld partial-tag bytes silently dropped when continuation is not a tool call

Also includes a minimal slice of [PR #40861](https://github.com/vllm-project/vllm/pull/40861)
for `qwen3_xml` partial-tag withholding (full #40861 requires v0.20+).

## Files

| File | Action |
|------|--------|
| `qwen3_reasoning_parser.py` | Replaces `vllm/reasoning/qwen3_reasoning_parser.py` |
| `apply_pr40783_patches.py` | Patches `utils.py`, `serving.py`, `qwen3xml_tool_parser.py` |

## Apply

Baked into `docker/Dockerfile.v2` when built with default settings (or
`install.sh` without `--no-pr40783`).

Manual apply inside a running container:

```bash
python3 /opt/patches/06-pr40783-qwen3-reasoning/apply_pr40783_patches.py
```

## Agentic launch flags

```bash
--reasoning-parser qwen3 \
--enable-auto-tool-choice \
--tool-call-parser qwen3_xml \
--speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
--attention-backend FLASHINFER
```

## Performance

Parser-only Python changes — expect **no measurable tok/s impact** on
`bench_qwen35.sh`. Verify with before/after bench within ±2%.

## When to remove

Delete this directory when upgrading to vLLM ≥0.20 where #35687 is merged and
#40783/#40861 are upstream (or no longer needed).

## Troubleshooting

**`UnboundLocalError: DeltaMessage` in `serving.py`:** An early patch version imported
`DeltaMessage` inside `chat_completion_stream_generator`, which shadows the module-level
import. Re-run the apply script (idempotent) or rebuild the image:

```bash
python3 /opt/patches/06-pr40783-qwen3-reasoning/apply_pr40783_patches.py
```

## TODO (post–multi-model review)

Validate on DGX with a real agentic workload before further code changes.

- [ ] Rebuild `vllm-qwen35-v2` with `WITH_PR40783=1` (default); use `configs/launch-v2-agentic.sh` flags
- [ ] Smoke test: streaming + `tool_choice=auto` + `enable_thinking=true`, multi-tool prompt — confirm structured `tool_calls` (not raw XML in `content` only)
- [ ] Run `bench_qwen35.sh` — expect ±2% vs pre-patch baseline
- [ ] If `tool_calls` empty but XML appears in stream: add text-based reasoning-end in `apply_serving()` (align with `just_completed_tool_call_tag` in parser)
- [ ] If pre-tool text is truncated: fix qwen3xml `partial_tag_overlap` to use `text_content_buffer` (not `streaming_buffer`), or drop minimal #40861 slice until full port
- [ ] Only after smoke passes: consider patching `tool_choice=required` / named-function serving branches; add real streaming unit tests (multi-delta tag, buffer desync)
