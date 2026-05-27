# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_omni.diffusion.models.hunyuan_video_i2v.pipeline_hunyuan_video_i2v import (
    _resolve_last_token_positions,
)


def test_resolve_last_token_positions_prefers_last_match():
    input_ids = torch.tensor(
        [
            [1, 7, 2, 7, 0, 0],  # last token_id=7 at idx 3
            [7, 3, 4, 0, 0, 0],  # last token_id=7 at idx 0
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ]
    )

    positions = _resolve_last_token_positions(input_ids, attention_mask, token_id=7)
    assert positions.tolist() == [3, 0]


def test_resolve_last_token_positions_fallback_to_last_non_pad():
    input_ids = torch.tensor(
        [
            [1, 2, 3, 0, 0],  # no token_id=9 -> fallback idx 2
            [4, 5, 0, 0, 0],  # no token_id=9 -> fallback idx 1
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ]
    )

    positions = _resolve_last_token_positions(input_ids, attention_mask, token_id=9)
    assert positions.tolist() == [2, 1]
