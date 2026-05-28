#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Async /v1/videos client for HunyuanVideo-I2V."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests


def create_video_job(args: argparse.Namespace) -> str:
    with open(args.image, "rb") as image_file:
        files = {"input_reference": (Path(args.image).name, image_file, "image/jpeg")}
        data: dict[str, str] = {
            "prompt": args.prompt,
            "height": str(args.height),
            "width": str(args.width),
            "num_frames": str(args.num_frames),
            "num_inference_steps": str(args.num_inference_steps),
            "guidance_scale": str(args.guidance_scale),
            "true_cfg_scale": str(args.true_cfg_scale),
            "seed": str(args.seed),
            "vae_use_tiling": str(args.vae_use_tiling).lower(),
        }
        if args.negative_prompt:
            data["negative_prompt"] = args.negative_prompt

        response = requests.post(
            f"{args.server}/v1/videos",
            files=files,
            data=data,
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        video_id = payload.get("id")
        if not isinstance(video_id, str):
            raise ValueError(f"Unexpected create response: {payload}")
        return video_id


def wait_until_completed(args: argparse.Namespace, video_id: str) -> None:
    deadline = time.time() + args.timeout
    while True:
        response = requests.get(f"{args.server}/v1/videos/{video_id}", timeout=30)
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        if status == "completed":
            return
        if status == "failed":
            raise RuntimeError(f"Video generation failed: {payload}")
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for job completion: {video_id}")
        time.sleep(args.poll_interval)


def download_video(args: argparse.Namespace, video_id: str) -> Path:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"{args.server}/v1/videos/{video_id}/content", timeout=300)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HunyuanVideo-I2V /v1/videos client")
    parser.add_argument("--server", default="http://localhost:8097")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num-frames", type=int, default=45)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--true-cfg-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vae-use-tiling",
        action="store_true",
        default=False,
        help="Enable VAE tiling to reduce peak VRAM (recommended for GPUs with <80GB VRAM)",
    )
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--output", default="hunyuan_i2v_online.mp4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_id = create_video_job(args)
    print(f"Created video job: {video_id}")
    wait_until_completed(args, video_id)
    output = download_video(args, video_id)
    print(f"Saved output video to: {output}")


if __name__ == "__main__":
    main()
