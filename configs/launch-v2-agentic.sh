#!/bin/bash
# v2 + agentic tool calling (PR #40783 backport, default-on in Dockerfile.v2)
# Requires: vllm-qwen35-v2 built with WITH_PR40783=1 (default)

docker run -d --name vllm-qwen35 \
  --gpus all --net=host --ipc=host \
  -v /path/to/models:/models \
  vllm-qwen35-v2 \
  serve /models/qwen35-122b-hybrid-int4fp8 \
  --served-model-name qwen \
  --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --attention-backend FLASHINFER \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
