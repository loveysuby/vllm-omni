# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests of common diffusion feature combinations in online serving mode
for HunyuanVideo-I2V (original, 13B dual-stream + single-stream).

Coverage (H100, since model cannot fit L4):
- CacheDiT + Layerwise CPU offloading (1 GPU)
- CacheDiT + TP=2 + VAE patch parallel=2 (2 GPUs)
"""

import pytest

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
    generate_synthetic_image,
)
from tests.utils import hardware_marks

PROMPT = "The camera slowly pans around a cat sitting on a windowsill, sunlight streaming in."
NEGATIVE_PROMPT = "low quality, blurry, distorted"

MODEL = "hunyuanvideo-community/HunyuanVideo-I2V"

SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _get_diffusion_feature_cases(model: str):
    """Return L4 diffusion feature cases for HunyuanVideo-I2V.

    Designed for 2x H100 environment per issue #1832.
    """
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--enable-layerwise-offload",
                ],
            ),
            id="single_card_cachedit_layerwise",
            marks=SINGLE_CARD_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--tensor-parallel-size",
                    "2",
                    "--vae-patch-parallel-size",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_cachedit_tp2_vae2",
            marks=PARALLEL_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_hunyuan_video_i2v(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """L4 diffusion feature coverage for HunyuanVideo-I2V on H100."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(480, 320)['base64']}"

    messages = dummy_messages_from_mix_data(
        image_data_url=image_data_url,
        content_text=PROMPT,
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 320,
            "width": 480,
            "num_frames": 5,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
