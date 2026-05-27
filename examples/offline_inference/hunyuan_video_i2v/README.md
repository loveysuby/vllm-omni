# HunyuanVideo-I2V Offline Example

This example shows how to run `hunyuanvideo-community/HunyuanVideo-I2V` with vLLM-Omni in offline mode.

## Quickstart

```bash
cd examples/offline_inference/hunyuan_video_i2v
python end2end.py \
  --model hunyuanvideo-community/HunyuanVideo-I2V \
  --model-class-name HunyuanVideoImageToVideoPipeline \
  --image ./input.jpg \
  --prompt "A cinematic camera move around the subject, soft motion blur." \
  --height 320 \
  --width 480 \
  --num-frames 45 \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --true-cfg-scale 6.0 \
  --flow-shift 7.0 \
  --seed 42 \
  --output hunyuan_i2v_output.mp4
```

## Notes

- `--image` is required for this pipeline.
- `--model-class-name HunyuanVideoImageToVideoPipeline` is recommended for explicit routing.
- `--flow-shift 7.0` is a common starting point for this model family.
- If your GPU memory is limited, reduce `--num-frames`, resolution, or steps first.
