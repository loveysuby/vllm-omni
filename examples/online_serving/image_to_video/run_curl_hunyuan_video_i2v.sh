#!/bin/bash
# HunyuanVideo-I2V (original, 13B) image-to-video curl example using the async video job API.
#
# Guidance-distilled model: pass true_cfg_scale=1.0 (distilled guidance only).

set -euo pipefail

INPUT_IMAGE="${INPUT_IMAGE:-test_input.jpg}"
BASE_URL="${BASE_URL:-http://localhost:8099}"
OUTPUT_PATH="${OUTPUT_PATH:-hunyuan_video_i2v.mp4}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"

if [ ! -f "$INPUT_IMAGE" ]; then
    echo "Input image not found: $INPUT_IMAGE"
    echo "Provide an image via INPUT_IMAGE env var."
    exit 1
fi

create_response=$(
  curl -sS -X POST "${BASE_URL}/v1/videos" \
    -H "Accept: application/json" \
    -F "prompt=Cinematic slow push-in on a curious rabbit in a sunny meadow, smooth camera motion." \
    -F "input_reference=@${INPUT_IMAGE}" \
    -F "height=480" \
    -F "width=832" \
    -F "num_frames=97" \
    -F "num_inference_steps=50" \
    -F "guidance_scale=6.0" \
    -F "true_cfg_scale=1.0" \
    -F "flow_shift=7.0" \
    -F "vae_use_tiling=true" \
    -F "seed=42"
)

video_id="$(echo "${create_response}" | jq -r '.id')"
if [ -z "${video_id}" ] || [ "${video_id}" = "null" ]; then
  echo "Failed to create video job:"
  echo "${create_response}" | jq .
  exit 1
fi

echo "Created video job ${video_id}"
echo "${create_response}" | jq .

while true; do
  status_response="$(curl -sS "${BASE_URL}/v1/videos/${video_id}")"
  status="$(echo "${status_response}" | jq -r '.status')"

  case "${status}" in
    queued|in_progress)
      echo "Video job ${video_id} status: ${status}"
      sleep "${POLL_INTERVAL}"
      ;;
    completed)
      echo "${status_response}" | jq .
      break
      ;;
    failed)
      echo "Video generation failed:"
      echo "${status_response}" | jq .
      exit 1
      ;;
    *)
      echo "Unexpected status response:"
      echo "${status_response}" | jq .
      exit 1
      ;;
  esac
done

curl -sS -L "${BASE_URL}/v1/videos/${video_id}/content" -o "${OUTPUT_PATH}"
echo "Saved video to ${OUTPUT_PATH}"
