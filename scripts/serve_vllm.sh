#!/usr/bin/env bash
# Serve an LLM with an OpenAI-compatible API on port 8000 (vLLM).
#
# Default: Llama-3.1-8B-Instruct in bf16 on a dedicated 40 GB A100.
#   - bf16 = same precision as the published i-MedRAG / MedRAG baselines (no quantization).
#   - 0.85 of 40 GB = 34 GB for vLLM: 16 GB weights + ~18 GB KV cache, which lets
#     ~128 requests run at once. The remaining ~6 GB is for the MedCPT retrieval models
#     (~1 GB) that the benchmark script loads on the same GPU.
#   - Use a separate virtualenv for vLLM: it pins its own torch version.
#
# Usage:
#   bash scripts/serve_vllm.sh                                  # Llama-3.1-8B, bf16
#   GPU_UTIL=0.45 QUANT=fp8 bash scripts/serve_vllm.sh          # if the GPU is shared again
#   MODEL=google/medgemma-27b-text-it NAME=medgemma-27b QUANT=fp8 GPU_UTIL=0.88 \
#       bash scripts/serve_vllm.sh                              # optional 2nd model (~27 GB in FP8)
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
NAME="${NAME:-llama31-8b}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_SEQS="${MAX_SEQS:-128}"
QUANT="${QUANT:-}"              # empty = full bf16; "fp8" = weight-only FP8 (Marlin kernels on A100)
PORT="${PORT:-8000}"

EXTRA_ARGS=()
if [[ -n "$QUANT" ]]; then
  EXTRA_ARGS+=(--quantization "$QUANT")
fi

vllm serve "$MODEL" \
  --served-model-name "$NAME" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs "$MAX_SEQS" \
  --enable-prefix-caching \
  --port "$PORT" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
