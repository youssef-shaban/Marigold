

from __future__ import annotations

import math
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers import DDIMScheduler, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.utils import BaseOutput
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize, to_pil_image
from tqdm.auto import tqdm
import timm  # type: ignore[import]
from timm.data import resolve_model_data_config  # type: ignore[import]
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD  # type: ignore[import]
from timm.data.transforms_factory import create_transform  # type: ignore[import]

from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_normals
from .util.image_util import chw2hwc, get_tv_resample_method, resize_max_res


class BaselineNormalsOutput(BaseOutput):
    """
    Output class for the baseline surface normals estimation pipeline.

    Args:
        normals_np (`np.ndarray`):
            Predicted normals map of shape [3, H, W] with values in the range of [-1, 1] (unit length vectors).
        normals_img (`PIL.Image.Image`):
            Normals image, with the shape of [H, W, 3] and values in [0, 255].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty (MAD, median absolute deviation) coming from ensembling.
    """

    normals_np: np.ndarray
    normals_img: Image.Image
    uncertainty: Union[None, np.ndarray]


class BaselineNormalsPipeline(DiffusionPipeline):
    """
    Baseline diffusion pipeline that predicts surface normals directly in pixel space.

    This pipeline replaces the VAE/text-conditioning stack of the original Marigold implementation
    with a frozen DINOv2 encoder that provides conditioning features from the input RGB image.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: Union[DDPMScheduler, DDIMScheduler],
        image_encoder: Optional[nn.Module] = None,
        image_processor: Optional[torch.nn.Module] = None,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()

        if image_encoder is None:
            image_encoder = timm.create_model(
                "vit_small_patch14_dinov2.lvd142m",
                pretrained=True,
                num_classes=0,
                global_pool="",
            )
            try:
                config = resolve_model_data_config(model=image_encoder)
                image_processor = create_transform(**config, is_training=False)
            except Exception:  # pragma: no cover - best-effort fallback
                image_processor = None
        else:
            if image_processor is None:
                try:
                    config = resolve_model_data_config(model=image_encoder)
                    image_processor = create_transform(**config, is_training=False)
                except Exception:  # pragma: no cover - best-effort fallback
                    image_processor = None

        image_encoder.requires_grad_(False)
        image_encoder.eval()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
        )

        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        # Store configuration for serialization
        self.register_to_config(
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        # Conditioning projection to the cross-attention dimension.
        cross_attention_dim = getattr(unet.config, "cross_attention_dim", None)
        if cross_attention_dim is None:
            raise ValueError("UNet must define `cross_attention_dim` for conditioning.")

        encoder_dim = getattr(self.image_encoder, "embed_dim", None)
        if encoder_dim is None:
            raise ValueError("DINOv2 encoder is expected to expose `embed_dim` attribute.")

        if encoder_dim == cross_attention_dim:
            self.condition_proj = nn.Identity()
        else:
            proj = nn.Linear(encoder_dim, cross_attention_dim, bias=False)
            nn.init.kaiming_uniform_(proj.weight, a=math.sqrt(5))
            self.condition_proj = proj

        # Normalization is optional; keep it identity to avoid additional trainable params.
        self.condition_norm = nn.Identity()

        # Transforms and normalization for the conditioning encoder
        self.condition_transform = image_processor
        self.condition_mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=torch.float32).view(1, -1, 1, 1)
        self.condition_std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=torch.float32).view(1, -1, 1, 1)

        # Cached dtype for inference convenience
        self.empty_condition = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
    ) -> BaselineNormalsOutput:
        """
        Predict normals for the provided RGB image.
        """
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert denoising_steps is not None, "denoising_steps must be provided or in config."
        assert processing_res is not None, "processing_res must be provided or in config."

        assert processing_res >= 0
        assert ensemble_size >= 1

        resample_method_enum: InterpolationMode = get_tv_resample_method(resample_method)

        rgb = self._prepare_input_tensor(input_image)
        input_size = rgb.shape

        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method_enum,
            )

        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0
        rgb_norm = rgb_norm.to(self.dtype)

        condition_tokens = self.encode_condition(rgb_norm)

        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        duplicated_condition = condition_tokens.expand(ensemble_size, -1, -1)

        single_rgb_dataset = TensorDataset(duplicated_rgb, duplicated_condition)

        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[-2:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=_bs, shuffle=False)

        target_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc="  Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader

        for batch in iterable:
            batched_img, batched_condition = batch
            target_pred_raw = self.single_infer(
                rgb_in=batched_img,
                condition_tokens=batched_condition,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
            )
            target_pred_ls.append(target_pred_raw.detach())

        target_preds = torch.concat(target_pred_ls, dim=0)
        torch.cuda.empty_cache()

        if ensemble_size > 1:
            final_pred, pred_uncert = ensemble_normals(
                target_preds,
                **(ensemble_kwargs or {}),
            )
        else:
            final_pred = target_preds
            pred_uncert = None

        if match_input_res:
            final_pred = resize(
                final_pred,
                input_size[-2:],
                interpolation=resample_method_enum,
                antialias=True,
            )

        final_pred = final_pred.squeeze()
        final_pred = final_pred.cpu().numpy()
        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()

        final_pred = final_pred.clip(-1, 1)
        normals_img = ((final_pred + 1) * 127.5).astype(np.uint8)
        normals_img = chw2hwc(normals_img)
        normals_img = Image.fromarray(normals_img)

        return BaselineNormalsOutput(
            normals_np=final_pred,
            normals_img=normals_img,
            uncertainty=pred_uncert,
        )

    def _prepare_input_tensor(
        self, input_image: Union[Image.Image, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")

        assert (
            4 == rgb.dim() and 3 == rgb.shape[1]
        ), f"Wrong input shape {rgb.shape}, expected [1, 3, H, W]"
        return rgb

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        condition_tokens: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
    ) -> torch.Tensor:
        device = self.device
        rgb_in = rgb_in.to(device=device, dtype=self.dtype)
        condition_tokens = condition_tokens.to(device=device, dtype=self.dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        target = torch.randn_like(
            rgb_in,
            generator=generator,
        ).to(device=device, dtype=self.dtype)

        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc="    Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for _, t in iterable:
            unet_input = torch.cat([rgb_in, target], dim=1)
            noise_pred = self.unet(
                unet_input,
                t,
                encoder_hidden_states=condition_tokens,
            ).sample

            target = self.scheduler.step(
                noise_pred, t, target, generator=generator
            ).prev_sample

        normals = torch.clip(target, -1.0, 1.0)
        norm = torch.norm(normals, dim=1, keepdim=True)
        normals = normals / norm.clamp(min=1e-6)
        return normals

    @torch.no_grad()
    def encode_condition(self, rgb_norm: torch.Tensor) -> torch.Tensor:
        """
        Extract conditioning tokens from the input RGB tensor using the frozen DINOv2 encoder.
        """
        device = self.device
        b, c, h, w = rgb_norm.shape
        assert c == 3, f"Expected 3-channel RGB tensor, got shape {rgb_norm.shape}"

        rgb_01 = (rgb_norm + 1.0) * 0.5
        rgb_01 = rgb_01.clamp(0.0, 1.0)

        if self.condition_transform is not None:
            processed_list = []
            for img in rgb_01:
                pil_img = to_pil_image(img.cpu())
                processed_list.append(self.condition_transform(pil_img))
            cond_in = torch.stack(processed_list, dim=0)
        else:
            mean = self.condition_mean.to(rgb_01.device)
            std = self.condition_std.to(rgb_01.device)
            cond_in = (rgb_01 - mean) / std

        cond_in = cond_in.to(device=device, dtype=torch.float32)
        self.image_encoder.to(device)

        features = self.image_encoder.forward_features(cond_in)
        if isinstance(features, dict):
            cls_token = features.get("x_norm_clstoken", None)
            if cls_token is None:
                cls_token = features.get("x_norm_clstoken_per_layer", None)
                if cls_token is not None and isinstance(cls_token, list):
                    cls_token = cls_token[-1]
            if cls_token is None:
                # fall back to mean of patch tokens
                patch_tokens = features.get("x_norm_patchtokens", None)
                if patch_tokens is None:
                    raise ValueError("Unexpected output from DINO encoder.")
                cls_token = patch_tokens.mean(dim=1)
        else:
            cls_token = features

        if cls_token.dim() == 2:
            cls_token = cls_token.unsqueeze(1)

        condition = self.condition_proj(cls_token.to(dtype=self.dtype))
        condition = self.condition_norm(condition)
        return condition

    def to_condition_tokens(self, rgb_norm: torch.Tensor) -> torch.Tensor:
        """
        Helper exposed for trainer to obtain conditioning tokens.
        """
        return self.encode_condition(rgb_norm)


