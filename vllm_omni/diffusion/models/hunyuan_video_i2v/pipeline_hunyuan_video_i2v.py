# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import PIL.Image
import torch
from diffusers import AutoencoderKLHunyuanVideo
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    LlamaTokenizerFast,
    LlavaForConditionalGeneration,
)
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.hunyuan_video_i2v.hunyuan_video_transformer import HunyuanVideoTransformer3DModel
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)


def _expand_input_ids_with_image_tokens(
    text_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    max_sequence_length: int,
    image_token_index: int,
    image_emb_len: int,
    image_emb_start: int,
    image_emb_end: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    special_image_token_mask = text_input_ids == image_token_index
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
    batch_indices, non_image_indices = torch.where(text_input_ids != image_token_index)

    max_expanded_length = max_sequence_length + (num_special_image_tokens.max() * (image_emb_len - 1))
    max_expanded_length = max(max_expanded_length, image_emb_end + max_sequence_length - image_emb_start)
    new_token_positions = torch.cumsum((special_image_token_mask * (image_emb_len - 1) + 1), -1) - 1
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

    expanded_input_ids = torch.full(
        (text_input_ids.shape[0], max_expanded_length), pad_token_id,
        dtype=text_input_ids.dtype, device=text_input_ids.device,
    )
    expanded_input_ids[batch_indices, text_to_overwrite] = text_input_ids[batch_indices, non_image_indices]
    expanded_input_ids[:, image_emb_start:image_emb_end] = image_token_index

    expanded_attention_mask = torch.zeros(
        (text_input_ids.shape[0], max_expanded_length),
        dtype=prompt_attention_mask.dtype, device=prompt_attention_mask.device,
    )
    attn_batch_indices, attention_indices = torch.where(expanded_input_ids != pad_token_id)
    expanded_attention_mask[attn_batch_indices, attention_indices] = 1.0
    expanded_attention_mask = expanded_attention_mask.to(prompt_attention_mask.dtype)
    position_ids = (expanded_attention_mask.cumsum(-1) - 1).masked_fill_((expanded_attention_mask == 0), 1)

    return {
        "input_ids": expanded_input_ids,
        "attention_mask": expanded_attention_mask,
        "position_ids": position_ids,
    }


def _retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def get_hunyuan_video_i2v_post_process_func(od_config: OmniDiffusionConfig):
    vae_scale_factor_temporal = 4
    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(output: DiffusionOutput) -> DiffusionOutput:
        if output.output.dim() == 5:
            output.output = output.output[:, :, vae_scale_factor_temporal:, :, :]
            output.output = video_processor.postprocess_video(output.output, output_type="np")[0]
        return output

    return post_process_func


def get_hunyuan_video_i2v_pre_process_func(od_config: OmniDiffusionConfig):
    max_area = 720 * 1280
    divisor = 16

    def pre_process_func(req: OmniDiffusionRequest) -> OmniDiffusionRequest:
        if not req.prompts:
            return req

        prompt_data = req.prompts[0]
        if isinstance(prompt_data, str):
            return req

        multi_modal_data = prompt_data.get("multi_modal_data", {})
        raw_image = multi_modal_data.get("image", None)
        if raw_image is None:
            return req
        if isinstance(raw_image, list):
            raw_image = raw_image[0]

        if isinstance(raw_image, str):
            image = PIL.Image.open(raw_image).convert("RGB")
        elif isinstance(raw_image, PIL.Image.Image):
            image = raw_image.convert("RGB")
        else:
            return req

        height = req.sampling_params.height
        width = req.sampling_params.width
        if height is None or width is None:
            w, h = image.size
            aspect = w / h
            target_h = int((max_area / aspect) ** 0.5)
            target_w = int(target_h * aspect)
            height = height or ((target_h + divisor - 1) // divisor * divisor)
            width = width or ((target_w + divisor - 1) // divisor * divisor)
            req.sampling_params.height = height
            req.sampling_params.width = width
        return req

    return pre_process_func


class HunyuanVideoI2VPipeline(nn.Module, CFGParallelMixin, SupportImageInput):
    support_image_input = True
    color_format = "RGB"

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        model = od_config.model
        local_files_only = os.path.exists(model)

        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only,
        )
        self.text_encoder = LlavaForConditionalGeneration.from_pretrained(
            model, subfolder="text_encoder", torch_dtype=dtype, local_files_only=local_files_only,
        ).to(self.device)

        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            model, subfolder="tokenizer_2", local_files_only=local_files_only,
        )
        self.text_encoder_2 = CLIPTextModel.from_pretrained(
            model, subfolder="text_encoder_2", torch_dtype=dtype, local_files_only=local_files_only,
        ).to(self.device)

        self.image_processor = CLIPImageProcessor.from_pretrained(
            model, subfolder="image_processor", local_files_only=local_files_only,
        )

        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(
            model, subfolder="vae", torch_dtype=torch.float32, local_files_only=local_files_only,
        ).to(self.device)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only,
        )
        if od_config.flow_shift is not None:
            self.scheduler._shift = od_config.flow_shift

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, HunyuanVideoTransformer3DModel)
        self.transformer = HunyuanVideoTransformer3DModel(od_config=od_config, **transformer_kwargs)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        self.vae_scaling_factor = self.vae.config.scaling_factor if hasattr(self.vae, "config") else 0.476986
        self.vae_scale_factor_temporal = (
            self.vae.temporal_compression_ratio if hasattr(self.vae, "temporal_compression_ratio") else 4
        )
        self.vae_scale_factor_spatial = (
            self.vae.spatial_compression_ratio if hasattr(self.vae, "spatial_compression_ratio") else 8
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # fmt: off
        self.prompt_template = (
            "<|start_header_id|>system<|end_header_id|>\n\n \nDescribe the video by detailing the following aspects "
            "according to the reference image: "
            "1. The main content and theme of the video."
            "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
            "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
            "4. background environment, light, style and atmosphere."
            "5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n"
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        # fmt: on
        self.prompt_template_crop_start = 103
        self.prompt_template_image_emb_start = 5
        self.prompt_template_image_emb_end = 581
        self.prompt_template_image_emb_len = 576
        self.prompt_template_double_return_token_id = 271
        self.default_image_embed_interleave = 4

        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @torch.no_grad()
    def _get_llama_prompt_embeds(
        self,
        image: PIL.Image.Image,
        prompt: list[str],
        device: torch.device,
        dtype: torch.dtype,
        max_sequence_length: int = 256,
        num_hidden_layers_to_skip: int = 2,
        image_embed_interleave: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = [self.prompt_template.format(p) for p in prompt]
        crop_start = self.prompt_template_crop_start
        image_emb_len = self.prompt_template_image_emb_len
        image_emb_start = self.prompt_template_image_emb_start
        image_emb_end = self.prompt_template_image_emb_end
        double_return_token_id = self.prompt_template_double_return_token_id
        max_seq = max_sequence_length + crop_start

        text_inputs = self.tokenizer(
            prompt, max_length=max_seq, padding="max_length", truncation=True,
            return_tensors="pt", return_attention_mask=True,
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)

        image_embeds = self.image_processor(image, return_tensors="pt").pixel_values.to(device)

        image_token_index = self.text_encoder.config.image_token_index
        pad_token_id = self.text_encoder.config.pad_token_id

        expanded = _expand_input_ids_with_image_tokens(
            text_input_ids, prompt_attention_mask, max_seq,
            image_token_index, image_emb_len, image_emb_start, image_emb_end, pad_token_id,
        )

        print(f"[DEBUG] prompt: {prompt}")
        print(f"[DEBUG] text_input_ids shape: {text_input_ids.shape}")
        print(f"[DEBUG] expanded input_ids shape: {expanded['input_ids'].shape}")
        print(f"[DEBUG] num image tokens in expanded: {(expanded['input_ids'] == image_token_index).sum().item()}")
        print(f"[DEBUG] pixel_values shape: {image_embeds.shape}")
        print(f"[DEBUG] image_token_index: {image_token_index}, image_emb_len: {image_emb_len}")

        prompt_embeds = self.text_encoder(
            **expanded, pixel_values=image_embeds, output_hidden_states=True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        text_crop_start = crop_start - 1 + image_emb_len
        batch_indices, last_double_return = torch.where(text_input_ids == double_return_token_id)

        if last_double_return.shape[0] == 3:
            last_double_return = torch.cat([last_double_return, torch.tensor([text_input_ids.shape[-1]])])
            batch_indices = torch.cat([batch_indices, torch.tensor([0])])

        last_double_return = last_double_return.reshape(text_input_ids.shape[0], -1)[:, -1]

        assistant_crop_start = last_double_return - 1 + image_emb_len - 4
        assistant_crop_end = last_double_return - 1 + image_emb_len
        attn_mask_crop_start = last_double_return - 4
        attn_mask_crop_end = last_double_return

        prompt_embed_list = []
        prompt_mask_list = []
        image_embed_list = []
        image_mask_list = []

        for i in range(text_input_ids.shape[0]):
            prompt_embed_list.append(torch.cat([
                prompt_embeds[i, text_crop_start:assistant_crop_start[i].item()],
                prompt_embeds[i, assistant_crop_end[i].item():],
            ]))
            prompt_mask_list.append(torch.cat([
                prompt_attention_mask[i, crop_start:attn_mask_crop_start[i].item()],
                prompt_attention_mask[i, attn_mask_crop_end[i].item():],
            ]))
            image_embed_list.append(prompt_embeds[i, image_emb_start:image_emb_end])
            image_mask_list.append(
                torch.ones(image_emb_end - image_emb_start, device=device, dtype=prompt_attention_mask.dtype)
            )

        prompt_embed_list = torch.stack(prompt_embed_list)
        prompt_mask_list = torch.stack(prompt_mask_list)
        image_embed_list = torch.stack(image_embed_list)
        image_mask_list = torch.stack(image_mask_list)

        if 0 < image_embed_interleave < 6:
            image_embed_list = image_embed_list[:, ::image_embed_interleave, :]
            image_mask_list = image_mask_list[:, ::image_embed_interleave]

        prompt_embeds = torch.cat([image_embed_list, prompt_embed_list], dim=1)
        prompt_attention_mask = torch.cat([image_mask_list, prompt_mask_list], dim=1)

        return prompt_embeds, prompt_attention_mask

    @torch.no_grad()
    def _get_clip_prompt_embeds(
        self,
        prompt: list[str],
        device: torch.device,
        dtype: torch.dtype,
        max_sequence_length: int = 77,
    ) -> torch.Tensor:
        text_inputs = self.tokenizer_2(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, return_tensors="pt",
        )
        pooled = self.text_encoder_2(text_inputs.input_ids.to(device), output_hidden_states=False).pooler_output
        return pooled.to(dtype=dtype)

    def encode_prompt(
        self,
        image: PIL.Image.Image,
        prompt: str | list[str],
        device: torch.device,
        dtype: torch.dtype,
        max_sequence_length: int = 256,
        image_embed_interleave: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        interleave = image_embed_interleave or self.default_image_embed_interleave

        prompt_embeds, prompt_attention_mask = self._get_llama_prompt_embeds(
            image, prompt, device, dtype,
            max_sequence_length=max_sequence_length,
            image_embed_interleave=interleave,
        )
        pooled_prompt_embeds = self._get_clip_prompt_embeds(prompt, device, dtype)

        return prompt_embeds, pooled_prompt_embeds, prompt_attention_mask

    def prepare_latents(
        self,
        image: PIL.Image.Image,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_channels = self.vae.config.latent_channels if hasattr(self.vae, "config") else 16
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        shape = (batch_size, num_channels, num_latent_frames, latent_height, latent_width)

        image_tensor = self.video_processor.preprocess(image, height, width)
        image_tensor = image_tensor.unsqueeze(2).to(device=device, dtype=self.vae.dtype)

        image_latents = _retrieve_latents(self.vae.encode(image_tensor), generator, "argmax")
        image_latents = image_latents.to(dtype) * self.vae_scaling_factor
        image_latents = image_latents.repeat(1, 1, num_latent_frames, 1, 1)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        t_init = torch.tensor([0.999], device=device)
        latents = latents * t_init + image_latents * (1 - t_init)

        image_latents = image_latents[:, :, :1]
        return latents, image_latents

    def predict_noise(self, **kwargs: Any) -> torch.Tensor:
        return self.transformer(**kwargs)[0]

    def forward(
        self,
        req: OmniDiffusionRequest,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        if len(req.prompts) < 1:
            raise ValueError("Prompt is required for HunyuanVideo I2V generation.")
        if len(req.prompts) > 1:
            raise ValueError("Only a single prompt per request is supported.")

        prompt_data = req.prompts[0]
        prompt = prompt_data if isinstance(prompt_data, str) else prompt_data.get("prompt")
        negative_prompt = None if isinstance(prompt_data, str) else prompt_data.get("negative_prompt")

        multi_modal_data = prompt_data.get("multi_modal_data", {}) if not isinstance(prompt_data, str) else {}
        raw_image = multi_modal_data.get("image", None)
        if isinstance(raw_image, list):
            if len(raw_image) > 1:
                logger.warning("Multiple images received; only the first will be used.")
            raw_image = raw_image[0]

        if raw_image is None:
            raise ValueError(
                "Image is required for HunyuanVideo I2V. Pass via multi_modal_data={'image': <image>}"
            )

        if isinstance(raw_image, str):
            image = PIL.Image.open(raw_image).convert("RGB")
        elif isinstance(raw_image, PIL.Image.Image):
            image = raw_image.convert("RGB")
        else:
            image = cast(PIL.Image.Image, raw_image)

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames_val = req.sampling_params.num_frames or num_frames
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps

        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale
        self._guidance_scale = guidance_scale

        true_cfg_scale = getattr(req.sampling_params, "true_cfg_scale", 1.0) or 1.0
        has_neg = negative_prompt is not None
        do_true_cfg = true_cfg_scale > 1.0 and has_neg

        device = self.device
        dtype = next(self.transformer.parameters()).dtype

        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
            image=image, prompt=prompt, device=device, dtype=dtype,
        )
        prompt_embeds = prompt_embeds.to(dtype)
        prompt_attention_mask = prompt_attention_mask.to(dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype)

        batch_size = prompt_embeds.shape[0]

        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        negative_prompt_attention_mask = None
        if do_true_cfg:
            black_image = PIL.Image.new("RGB", (width, height), 0)
            negative_prompt_embeds, negative_pooled_prompt_embeds, negative_prompt_attention_mask = (
                self.encode_prompt(
                    image=black_image, prompt=negative_prompt, device=device, dtype=dtype,
                )
            )
            negative_prompt_embeds = negative_prompt_embeds.to(dtype)
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(dtype)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(dtype)

        latents, image_latents = self.prepare_latents(
            image, batch_size, height, width, num_frames_val, dtype, device, generator,
            req.sampling_params.latents,
        )

        sigmas = np.linspace(1.0, 0.0, num_steps + 1)[:-1]
        self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        guidance = None
        if self.transformer.guidance_embeds:
            guidance = torch.tensor(
                [guidance_scale] * batch_size, dtype=dtype, device=device,
            ) * 1000.0

        for i, t in enumerate(timesteps):
            self._current_timestep = t
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            latent_model_input = torch.cat([image_latents, latents[:, :, 1:]], dim=2).to(dtype)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                pooled_projections=pooled_prompt_embeds,
                guidance=guidance,
                return_dict=False,
            )[0]

            if do_true_cfg and negative_prompt_embeds is not None:
                neg_noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_attention_mask=negative_prompt_attention_mask,
                    pooled_projections=negative_pooled_prompt_embeds,
                    guidance=guidance,
                    return_dict=False,
                )[0]
                noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

            denoised = self.scheduler.step(noise_pred[:, :, 1:], t, latents[:, :, 1:], return_dict=False)[0]
            latents = torch.cat([image_latents, denoised], dim=2)

        self._current_timestep = None

        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()

        if output_type == "latent":
            output = latents
        else:
            latents = latents.to(self.vae.dtype) / self.vae_scaling_factor
            output = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(output=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
