#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

MODEL="${MODEL:-hunyuanvideo-community/HunyuanVideo-I2V}"
MODEL_CLASS_NAME="${MODEL_CLASS_NAME:-HunyuanVideoImageToVideoPipeline}"
PORT="${PORT:-8097}"
FLOW_SHIFT="${FLOW_SHIFT:-7.0}"
CACHE_BACKEND="${CACHE_BACKEND:-none}"

echo "Starting HunyuanVideo-I2V server..."
echo "Model: $MODEL"
echo "Model class: $MODEL_CLASS_NAME"
echo "Port: $PORT"
echo "Flow shift: $FLOW_SHIFT"
echo "Cache backend: $CACHE_BACKEND"

CACHE_BACKEND_FLAG=""
if [ "$CACHE_BACKEND" != "none" ]; then
  CACHE_BACKEND_FLAG="--cache-backend $CACHE_BACKEND"
fi

vllm serve "$MODEL" --omni \
  --model-class-name "$MODEL_CLASS_NAME" \
  --port "$PORT" \
  --flow-shift "$FLOW_SHIFT" \
  $CACHE_BACKEND_FLAG
