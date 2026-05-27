# HunyuanVideo-I2V Online Serving Example

This example shows how to serve `hunyuanvideo-community/HunyuanVideo-I2V` with vLLM-Omni and call the async video API.

## 1) Start Server

```bash
cd examples/online_serving/hunyuan_video_i2v
bash run_server.sh
```

Environment overrides:

- `MODEL` (default: `hunyuanvideo-community/HunyuanVideo-I2V`)
- `MODEL_CLASS_NAME` (default: `HunyuanVideoImageToVideoPipeline`)
- `PORT` (default: `8097`)
- `FLOW_SHIFT` (default: `7.0`)
- `CACHE_BACKEND` (default: `none`)

## 2) Send Request

```bash
python openai_chat_client.py \
  --server http://localhost:8097 \
  --image ./input.jpg \
  --prompt "A cinematic orbit camera movement with smooth temporal motion." \
  --negative-prompt "blurry, distorted, low quality" \
  --height 320 \
  --width 480 \
  --num-frames 45 \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --true-cfg-scale 6.0 \
  --seed 42 \
  --output hunyuan_i2v_online.mp4
```

## API Notes

- This path uses `POST /v1/videos` and polls `GET /v1/videos/{video_id}` until completion.
- Download uses `GET /v1/videos/{video_id}/content`.
- The model requires an image input (`input_reference` in multipart request).
