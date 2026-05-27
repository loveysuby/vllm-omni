#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline end-to-end example for HunyuanVideo-I2V."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import PIL.Image
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput


def _extract_video_frames(output: object) -> object:
    """Normalize pipeline outputs to a frame container."""
    frames = output

    if isinstance(frames, list):
        frames = frames[0] if frames else None

    if isinstance(frames, OmniRequestOutput):
        if frames.images:
            if len(frames.images) == 1:
                first = frames.images[0]
                if isinstance(first, dict):
                    frames = first.get("frames") or first.get("video")
                elif isinstance(first, tuple) and len(first) == 2:
                    frames = first[0]
                else:
                    frames = first
            else:
                frames = frames.images
        elif frames.request_output is not None:
            frames = frames.request_output

    if isinstance(frames, OmniRequestOutput):
        if frames.images:
            frames = frames.images[0]
        else:
            frames = None

    if isinstance(frames, tuple) and len(frames) == 2:
        frames = frames[0]
    if isinstance(frames, dict):
        frames = frames.get("frames") or frames.get("video")

    return frames


def _to_frame_list(video: object) -> list[np.ndarray]:
    """Convert various frame containers to a list of HWC numpy arrays."""
    if isinstance(video, torch.Tensor):
        tensor = video.detach().cpu()
        if tensor.dim() == 5:
            tensor = tensor[0]
        if tensor.dim() == 4 and tensor.shape[0] in (3, 4):
            tensor = tensor.permute(1, 2, 3, 0)
        arr = tensor.float().numpy()
        if arr.ndim == 4:
            return [frame for frame in arr]
        raise ValueError(f"Unsupported tensor shape for video output: {tuple(tensor.shape)}")

    if isinstance(video, np.ndarray):
        if video.ndim == 5:
            video = video[0]
        if video.ndim == 4:
            return [frame for frame in video]
        raise ValueError(f"Unsupported ndarray shape for video output: {video.shape}")

    if isinstance(video, list):
        if not video:
            raise ValueError("Empty frame list in output.")
        normalized: list[np.ndarray] = []
        for frame in video:
            if isinstance(frame, PIL.Image.Image):
                normalized.append(np.asarray(frame).astype(np.float32) / 255.0)
            elif isinstance(frame, np.ndarray):
                arr = frame.astype(np.float32)
                if np.issubdtype(frame.dtype, np.integer):
                    arr = arr / 255.0
                normalized.append(arr)
            elif isinstance(frame, torch.Tensor):
                arr = frame.detach().cpu().float().numpy()
                if arr.ndim == 3 and arr.shape[0] in (3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                normalized.append(arr)
            else:
                raise TypeError(f"Unsupported frame type: {type(frame)}")
        return normalized

    raise TypeError(f"Unsupported output type: {type(video)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline HunyuanVideo-I2V generation")
    parser.add_argument("--model", default="hunyuanvideo-community/HunyuanVideo-I2V")
    parser.add_argument("--model-class-name", default="HunyuanVideoImageToVideoPipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Prompt for image-to-video generation")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num-frames", type=int, default=45)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--true-cfg-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="hunyuan_i2v_output.mp4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = PIL.Image.open(args.image).convert("RGB")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    omni = Omni(
        model=args.model,
        model_class_name=args.model_class_name,
        flow_shift=args.flow_shift,
    )

    output = omni.generate(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "multi_modal_data": {"image": image},
        },
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            true_cfg_scale=args.true_cfg_scale,
            seed=args.seed,
            generator=generator,
        ),
    )

    frames = _extract_video_frames(output)
    if frames is None:
        raise ValueError("No video frames found in generation output.")

    frame_list = _to_frame_list(frames)

    from diffusers.utils import export_to_video

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frame_list, str(output_path), fps=16)
    print(f"Saved video to: {output_path}")


if __name__ == "__main__":
    main()
