
from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from typing import List, Union

import numpy as np
import torch
from diffusers import DDPMScheduler, DDIMScheduler
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from safetensors.torch import load_file

from marigold.baseline_normals_pipeline import (
    BaselineNormalsPipeline,
    BaselineNormalsOutput,
)
from src.util.image_util import img_chw2hwc
from src.util import metric
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dict_to_text
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker, compute_cosine_error
from src.util.multi_res_noise import multi_res_noise_like
from src.util.seeding import generate_seed_sequence


class BaselineNormalsTrainer:
    """
    Trainer for the baseline Conditional2DUnet pipeline that operates directly in pixel space.
    """

    def __init__(
        self,
        cfg: OmegaConf,
        model: BaselineNormalsPipeline,
        train_dataloader: DataLoader,
        device,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
    ):
        self.cfg: OmegaConf = cfg
        self.model: BaselineNormalsPipeline = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders or []
        self.vis_loaders: List[DataLoader] = vis_dataloaders or []
        self.accumulation_steps: int = accumulation_steps

        # Ensure UNet input matches expected channels (RGB + target = 6)
        expected_in_channels = 6
        unet_in_channels = getattr(self.model.unet.config, "in_channels", None)
        if expected_in_channels != unet_in_channels:
            raise ValueError(
                f"Baseline UNet must have in_channels={expected_in_channels}, "
                f"got {unet_in_channels}"
            )

        self.model.to(self.device)
        # if hasattr(self.model.unet, "enable_xformers_memory_efficient_attention"):
        #     try:
        #         self.model.unet.enable_xformers_memory_efficient_attention()
        #     except Exception:
        #         logging.warning("XFormers memory efficient attention could not be enabled.")
        # if hasattr(self.model.unet, "enable_gradient_checkpointing"):
        #     try:
        #         self.model.unet.enable_gradient_checkpointing()
        #     except Exception:
        #         logging.warning("Gradient checkpointing could not be enabled for the UNet.")

        # Freeze everything except the UNet
        self.model.image_encoder.requires_grad_(False)
        if hasattr(self.model.condition_proj, "parameters"):
            for param in self.model.condition_proj.parameters():
                param.requires_grad_(True)
        if hasattr(self.model.condition_norm, "parameters"):
            for param in self.model.condition_norm.parameters():
                param.requires_grad_(True)

        self.model.unet.requires_grad_(True)

        # Optimizer and scheduler
        lr = self.cfg.lr
        trainable_parameters = list(self.model.unet.parameters())
        if hasattr(self.model.condition_proj, "parameters"):
            proj_params = [p for p in self.model.condition_proj.parameters() if p.requires_grad]
            if proj_params:
                trainable_parameters.extend(proj_params)

        if hasattr(self.model.condition_norm, "parameters"):
            norm_params = [p for p in self.model.condition_norm.parameters() if p.requires_grad]
            if norm_params:
                trainable_parameters.extend(norm_params)

        self.optimizer = Adam(trainable_parameters, lr=lr)

        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)

        # Training scheduler (diffusion)
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_config(
            self.model.scheduler.config,
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
        )

        logging.info(
            "DDPM training noise scheduler config is updated: "
            f"rescale_betas_zero_snr = {self.training_noise_scheduler.config.rescale_betas_zero_snr}, "
            f"timestep_spacing = {self.training_noise_scheduler.config.timestep_spacing}"
        )

        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        # Inference DDIM scheduler (used for validation / visualization)
        self.model.scheduler = DDIMScheduler.from_config(
            self.training_noise_scheduler.config,
        )

        # Metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]
        self.train_metrics = MetricTracker(*["loss"])
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])

        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal
        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.gt_normals_type = self.cfg.gt_normals_type
        self.gt_mask_type = self.cfg.gt_mask_type
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # Multi-resolution noise
        self.apply_multi_res_noise = self.cfg.multi_res_noise is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed
            self.mr_noise_downscale_strategy = (
                self.cfg.multi_res_noise.downscale_strategy
            )

        # Internal state
        self.epoch = 1
        self.n_batch_in_epoch = 0
        self.effective_iter = 0
        self.in_evaluation = False
        self.global_seed_sequence: List = []

    def train(self, t_end=None):
        logging.info("Start training")
        logging.info(f"Training configuration: max_epoch={self.max_epoch}, max_iter={self.max_iter}, "
                    f"gradient_accumulation_steps={self.gradient_accumulation_steps}")
        logging.info(f"Checkpoint periods: save={self.save_period}, backup={self.backup_period}, "
                    f"validation={self.val_period}, visualization={self.vis_period}")

        device = self.device
        self.model.to(device)

        if self.in_evaluation:
            logging.info(
                "Last evaluation was not finished, will do evaluation before continue training."
            )
            self.validate()

        self.train_metrics.reset()
        accumulated_step = 0

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.info(f"Starting epoch: {self.epoch}")

            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                logging.info(f"Batch {self.n_batch_in_epoch + 1} received from dataloader")
                self.model.unet.train()

                # Consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None
                logging.debug("Random generator initialized")

                logging.debug("Loading batch data to device")
                rgb = batch["rgb_norm"].to(device=device, dtype=self.model.dtype)
                normals_gt = batch[self.gt_normals_type].to(device=device, dtype=self.model.dtype)
                logging.debug(f"Data loaded: rgb shape={rgb.shape}, normals_gt shape={normals_gt.shape}")

                if self.gt_mask_type is not None:
                    logging.debug("Processing ground truth mask")
                    valid_mask_for_latent = batch[self.gt_mask_type].to(device=device)
                    invalid_mask = ~valid_mask_for_latent
                    valid_mask_down = ~torch.max_pool2d(
                        invalid_mask.float(), 8, 8
                    ).bool()
                    valid_mask_down = valid_mask_down.repeat((1, 3, 1, 1))
                    logging.debug(f"Mask processed: valid_mask_down shape={valid_mask_down.shape}")
                else:
                    valid_mask_down = None

                batch_size = rgb.shape[0]

                logging.debug("Encoding RGB to condition tokens")
                with torch.no_grad():
                    condition_tokens = self.model.to_condition_tokens(rgb)
                logging.debug(f"Condition tokens encoded: shape={condition_tokens.shape}")

                logging.debug("Generating timesteps")
                timesteps = torch.randint(
                    0,
                    self.scheduler_timesteps,
                    (batch_size,),
                    device=device,
                    generator=rand_num_generator,
                ).long()
                logging.debug(f"Timesteps generated: {timesteps}")

                logging.debug("Generating noise")
                if self.apply_multi_res_noise:
                    strength = self.mr_noise_strength
                    if self.annealed_mr_noise:
                        strength = strength * (timesteps / self.scheduler_timesteps)
                    noise = multi_res_noise_like(
                        normals_gt,
                        strength=strength,
                        downscale_strategy=self.mr_noise_downscale_strategy,
                        generator=rand_num_generator,
                        device=device,
                    )
                    logging.debug(f"Multi-res noise generated with strength={strength}")
                else:
                    noise = torch.randn(
                        normals_gt.shape,
                        device=device,
                        generator=rand_num_generator,
                        dtype=self.model.dtype,
                    )
                    logging.debug("Standard noise generated")

                logging.debug("Adding noise to targets")
                noisy_targets = self.training_noise_scheduler.add_noise(
                    normals_gt, noise, timesteps
                )
                logging.debug(f"Noisy targets created: shape={noisy_targets.shape}")

                logging.debug("Preparing UNet input and running forward pass")
                unet_input = torch.cat([rgb, noisy_targets], dim=1).float()
                logging.debug(f"UNet input prepared: shape={unet_input.shape}")
                
                model_pred = self.model.unet(
                    unet_input,
                    timesteps,
                    encoder_hidden_states=condition_tokens,
                ).sample
                logging.debug("UNet forward pass completed")
                logging.debug(f"Model prediction shape={model_pred.shape}")

                logging.debug("Computing loss target")
                if "sample" == self.prediction_type:
                    target = normals_gt
                elif "epsilon" == self.prediction_type:
                    target = noise
                elif "v_prediction" == self.prediction_type:
                    target = self.training_noise_scheduler.get_velocity(
                        normals_gt, noise, timesteps
                    )
                else:
                    raise ValueError(f"Unknown prediction type {self.prediction_type}")
                logging.debug(f"Loss target prepared (prediction_type={self.prediction_type})")

                logging.debug("Computing loss")
                if valid_mask_down is not None:
                    loss_value = self.loss(
                        model_pred[valid_mask_down].float(),
                        target[valid_mask_down].float(),
                    )
                    logging.debug("Loss computed with mask")
                else:
                    loss_value = self.loss(model_pred.float(), target.float())
                    logging.debug("Loss computed without mask")

                loss = loss_value.mean()
                self.train_metrics.update("loss", loss.item())
                logging.debug(f"Loss value: {loss.item():.5f}")

                logging.debug("Running backward pass")
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                logging.debug("Backward pass completed")
                accumulated_step += 1
                self.n_batch_in_epoch += 1

                if accumulated_step >= self.gradient_accumulation_steps:
                    logging.debug(f"Gradient accumulation complete ({self.gradient_accumulation_steps} steps), applying optimizer")
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    logging.debug("Optimizer step completed")
                    accumulated_step = 0
                    self.effective_iter += 1

                    logging.debug("Logging metrics to tensorboard")
                    accumulated_loss = self.train_metrics.result()["loss"]
                    tb_logger.log_dict(
                        {
                            f"train/{k}": v
                            for k, v in self.train_metrics.result().items()
                        },
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "lr",
                        self.lr_scheduler.get_last_lr()[0],
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "n_batch_in_epoch",
                        self.n_batch_in_epoch,
                        global_step=self.effective_iter,
                    )
                    logging.info(
                        f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                    )
                    self.train_metrics.reset()

                    logging.debug("Calling train step callbacks")
                    self._train_step_callback()
                    logging.debug("Train step callbacks completed")

                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        logging.info("Max iterations reached, saving final checkpoint")
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        logging.info("Training ended.")
                        return
                    elif t_end is not None and datetime.now() >= t_end:
                        logging.info("Time limit reached, saving checkpoint")
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

                    logging.info("Clearing CUDA cache")
                    torch.cuda.empty_cache()
                    logging.debug("CUDA cache cleared")

            self.n_batch_in_epoch = 0
            logging.info(f"Epoch {epoch} completed")

    def _train_step_callback(self):
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            logging.info(f"Backup period reached (every {self.backup_period} iters), saving backup checkpoint")
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )
            logging.debug("Backup checkpoint saved")

        _is_latest_saved = False
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            logging.info(f"Validation period reached (every {self.val_period} iters), starting validation")
            self.in_evaluation = True
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True
            self.validate()
            self.in_evaluation = False
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            logging.info("Validation completed")

        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            logging.info(f"Save period reached (every {self.save_period} iters), saving checkpoint")
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            logging.debug("Latest checkpoint saved")

        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            logging.info(f"Visualization period reached (every {self.vis_period} iters), running visualization")
            self.visualize()
            logging.info("Visualization completed")

    def validate(self):
        for i, val_loader in enumerate(self.val_loaders):
            val_dataset_name = val_loader.dataset.disp_name
            val_metric_dict = self.validate_single_dataset(
                data_loader=val_loader, metric_tracker=self.val_metrics
            )
            logging.info(
                f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dict}"
            )
            tb_logger.log_dict(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dict.items()},
                global_step=self.effective_iter,
            )

            eval_text = eval_dict_to_text(
                val_metrics=val_metric_dict,
                dataset_name=val_dataset_name,
                sample_list_path=val_loader.dataset.filename_ls_path,
            )
            _save_to = os.path.join(
                self.out_dir_eval,
                f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
            )
            with open(_save_to, "w+") as f:
                f.write(eval_text)

            if 0 == i:
                main_eval_metric = val_metric_dict[self.main_val_metric]
                if (
                    "minimize" == self.main_val_metric_goal
                    and main_eval_metric < self.best_metric
                    or "maximize" == self.main_val_metric_goal
                    and main_eval_metric > self.best_metric
                ):
                    self.best_metric = main_eval_metric
                    logging.info(
                        f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                    )
                    self.save_checkpoint(
                        ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                    )

    def visualize(self):
        for val_loader in self.vis_loaders:
            vis_dataset_name = val_loader.dataset.disp_name
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            _ = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                save_to_dir=vis_out_dir,
            )

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
    ):
        self.model.to(self.device)
        metric_tracker.reset()

        val_init_seed = self.cfg.validation.init_seed
        val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))

        for i, batch in enumerate(
            tqdm(data_loader, desc=f"evaluating on {data_loader.dataset.disp_name}"),
            start=1,
        ):
            assert 1 == data_loader.batch_size
            rgb_int = batch["rgb_int"]
            normals_gt = batch["normals"].to(self.device)

            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)

            pipe_out: BaselineNormalsOutput = self.model(
                rgb_int,
                denoising_steps=self.cfg.validation.denoising_steps,
                ensemble_size=self.cfg.validation.ensemble_size,
                processing_res=self.cfg.validation.processing_res,
                match_input_res=self.cfg.validation.match_input_res,
                generator=generator,
                batch_size=1,
                show_progress_bar=False,
                resample_method=self.cfg.validation.resample_method,
            )

            normals_pred = pipe_out.normals_np
            normals_pred_ts = (
                torch.from_numpy(normals_pred).unsqueeze(0).to(self.device)
            )
            cosine_error = compute_cosine_error(
                normals_pred_ts, normals_gt, masked=True
            )

            for met_func in self.metric_funcs:
                _metric_name = met_func.__name__
                _metric = met_func(cosine_error).item()
                metric_tracker.update(_metric_name, _metric)

            if save_to_dir is not None:
                img_name = batch["rgb_relative_path"][0].replace("/", "_")
                img_stem, _ = os.path.splitext(img_name)

                pred_save_path = os.path.join(save_to_dir, f"{img_stem}_pred.png")
                normals_pred_img = img_chw2hwc(((normals_pred + 1) * 127.5)).astype(
                    np.uint8
                )
                Image.fromarray(normals_pred_img).save(pred_save_path)

                normals_gt_np = (
                    normals_gt.squeeze(0).detach().cpu().numpy()
                )
                normals_gt_img = img_chw2hwc(
                    ((normals_gt_np + 1) * 127.5)
                ).astype(np.uint8)
                gt_save_path = os.path.join(save_to_dir, f"{img_stem}_gt.png")
                Image.fromarray(normals_gt_img).save(gt_save_path)

        return metric_tracker.result()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        os.makedirs(ckpt_dir, exist_ok=True)

        unet_path = os.path.join(ckpt_dir, "unet")
        self.model.unet.save_pretrained(unet_path, safe_serialization=True)
        logging.info(f"UNet is saved to: {unet_path}")

        scheduler_path = os.path.join(ckpt_dir, "scheduler")
        self.model.scheduler.save_pretrained(scheduler_path)
        logging.info(f"Scheduler is saved to: {scheduler_path}")

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            with open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w"):
                pass
            logging.info(f"Trainer state is saved to: {train_state_path}")

        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        _model_path = os.path.join(ckpt_path, "unet", "diffusion_pytorch_model.safetensors")
        state_dict = load_file(_model_path, device=self.device)
        self.model.unet.load_state_dict(state_dict)
        self.model.unet.to(self.device)
        logging.info(f"UNet parameters are loaded from {_model_path}")

        if load_trainer_state:
            checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"))
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"


