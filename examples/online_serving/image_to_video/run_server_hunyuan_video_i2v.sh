#!/bin/bash
# HunyuanVideo-I2V (original, 13B) image-to-video online serving startup script.
#
# Guidance-distilled model: serve as-is and pass true_cfg_scale=1.0 from the client
# (distilled guidance only). For larger resolutions / longer clips, shard weights
# across GPUs with HSDP (--use-hsdp --hsdp-shard-size 2).

MODEL="${MODEL:-hunyuanvideo-community/HunyuanVideo-I2V}"
MODEL_CLASS_NAME="${MODEL_CLASS_NAME:-HunyuanVideoImageToVideoPipeline}"
PORT="${PORT:-8099}"
FLOW_SHIFT="${FLOW_SHIFT:-7.0}"
CACHE_BACKEND="${CACHE_BACKEND:-none}"
ENABLE_CPU_OFFLOAD="${ENABLE_CPU_OFFLOAD:-false}"

echo "Starting HunyuanVideo-I2V server..."
echo "Model: $MODEL"
echo "Model class: $MODEL_CLASS_NAME"
echo "Port: $PORT"
echo "Flow shift: $FLOW_SHIFT"
echo "Cache backend: $CACHE_BACKEND"
echo "CPU offload: $ENABLE_CPU_OFFLOAD"

EXTRA_FLAGS=""
if [ "$CACHE_BACKEND" != "none" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --cache-backend $CACHE_BACKEND"
fi
if [ "$ENABLE_CPU_OFFLOAD" = "true" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --enable-cpu-offload"
fi

vllm serve "$MODEL" --omni \
    --model-class-name "$MODEL_CLASS_NAME" \
    --port "$PORT" \
    --flow-shift "$FLOW_SHIFT" \
    $EXTRA_FLAGS
