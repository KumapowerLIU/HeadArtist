import random
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    ControlNetModel

)
from diffusers.utils.import_utils import is_xformers_available
import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *
from threestudio.utils.perceptual.perceptual import PerceptualLoss


class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)


@threestudio.register("ssd-head-guidance")
class SSDGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
        controlnet_name_or_path: str = "/home/liuhongyu/code/ControlNetMediaPipeFace"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        time_prior: Optional[Any] = None  # [w1,w2,s1,s2]
        half_precision_weights: bool = True
        condition_scale: float = 1.5

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        view_dependent_prompting: bool = True
        use_perceptual: bool = False
        # calculate loss in x_0
        use_recon: bool = False

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        @dataclass
        class SubModules:
            pipe: StableDiffusionControlNetPipeline

        controlnet_base = ControlNetModel.from_pretrained(
            self.cfg.controlnet_name_or_path,
            torch_dtype=self.weights_dtype
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, controlnet=controlnet_base,
            **pipe_kwargs,
        ).to(self.device)

        ###############

        cleanup()
        self.submodules = SubModules(pipe=pipe)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder

        cleanup()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.control_base.parameters():
            p.requires_grad_(False)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.scheduler_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.pipe.scheduler = self.scheduler
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value
        if self.cfg.time_prior is not None:
            m1, m2, s1, s2 = self.cfg.time_prior
            weights = torch.cat(
                (
                    torch.exp(
                        -(torch.arange(self.num_train_timesteps, m1, -1) - m1 ** 2)
                        / (2 * s1 ** 2)
                    ),
                    torch.ones(m1 - m2 + 1),
                    torch.exp(-(torch.arange(m2 - 1, 0, -1) - m2 ** 2) / (2 * s2 ** 2)),
                )
            )
            weights = weights / torch.sum(weights)
            self.time_prior_acc_weights = torch.cumsum(weights, dim=0)

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        self.perceptual = {}
        self.grad_clip_val: Optional[float] = None
        if self.cfg.use_perceptual:
            perceptual_loss = PerceptualLoss().to(self.device)
            self.perceptual['loss'] = perceptual_loss
        threestudio.info(f"Loaded Stable Diffusion!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def unet(self):
        return self.submodules.pipe.unet

    @property
    def vae(self):
        return self.submodules.pipe.vae

    @property
    def control_base(self):
        return self.submodules.pipe.controlnet


    @torch.cuda.amp.autocast(enabled=False)
    def forward_control_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        cross_attention_kwargs,
        down_block_additional_residuals,
        mid_block_additional_residual,
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_controlnet(
        self,
        controlnet,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        image_cond: Float[Tensor, "..."],
        condition_scale: float,
        cross_attention_kwargs,
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None

    ) -> Float[Tensor, "..."]:
        return controlnet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            controlnet_cond=image_cond.to(self.weights_dtype),
            conditioning_scale=condition_scale,
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    @contextmanager
    def disable_control_class_embedding(self, controlnet: ControlNetModel):
        class_embedding = controlnet.class_embedding
        try:
            controlnet.class_embedding = None
            yield controlnet
        finally:
            controlnet.class_embedding = class_embedding

    def compute_grad_ssd(
        self,
        img_cond,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings_vd: Float[Tensor, "BB 77 768"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        current_step_ratio=None,

    ):
        B = latents.shape[0]
        noise_pred_pretrain_normal = None
        with torch.no_grad():
            # random timestamp
            if self.cfg.time_prior is not None:
                time_index = torch.where(
                    (self.time_prior_acc_weights - current_step_ratio) > 0
                )[0][0]
                if time_index == 0 or torch.abs(
                    self.time_prior_acc_weights[time_index] - current_step_ratio
                ) < torch.abs(
                    self.time_prior_acc_weights[time_index - 1] - current_step_ratio
                ):
                    t = self.num_train_timesteps - time_index
                else:
                    t = self.num_train_timesteps - time_index + 1
                t = torch.clip(t, self.min_step, self.max_step + 1)
                t = torch.full((B,), t, dtype=torch.long, device=self.device)
            else:
                # random timestamp
                t = torch.randint(
                    self.min_step,
                    self.max_step + 1,
                    [B],
                    dtype=torch.long,
                    device=self.device,
                )
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            # pred_pretrain_noise
            (
                down_block_res_samples,
                mid_block_res_sample,
            ) = self.forward_controlnet(
                self.control_base,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings_vd,
                image_cond=torch.cat([img_cond] * 2, dim=0),
                condition_scale=self.cfg.condition_scale,
                cross_attention_kwargs=None
            )
            noise_pred_pretrain = self.forward_control_unet(
                self.unet,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings_vd,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample
            )

        # pred_3D_noise, need grad for recon
        (
            down_block_res_samples,
            mid_block_res_sample,
        ) = self.forward_controlnet(
            self.control_base,
            latent_model_input,
            torch.cat([t] * 2),
            encoder_hidden_states=text_embeddings,
            image_cond=torch.cat([img_cond] * 2, dim=0),
            condition_scale=self.cfg.condition_scale,
            cross_attention_kwargs=None
        )
        noise_pred_est = self.forward_control_unet(
            self.unet,
            latent_model_input,
            torch.cat([t] * 2),
            encoder_hidden_states=text_embeddings,
            cross_attention_kwargs=None,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample
        )

        (
            noise_pred_pretrain_text,
            noise_pred_pretrain_uncond,
        ) = noise_pred_pretrain.chunk(2)
        (
            noise_pred_est_text,
            noise_pred_est_uncond,
        ) = noise_pred_est.chunk(2)
        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_pretrain = noise_pred_pretrain_uncond + self.cfg.guidance_scale * (
            noise_pred_pretrain_text - noise_pred_pretrain_uncond
        )


        if self.cfg.use_recon:
            # reconstruct x0
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

            latents_recon = self.scheduler.predict_start_from_noise(latents_noisy, t, noise_pred_pretrain)

            latents = self.scheduler.predict_start_from_noise(latents_noisy, t, noise_pred_est_text)

            loss = 0.5 * F.mse_loss(latents, latents_recon.detach(), reduction="sum") / latents.shape[0]
            grad = w * torch.autograd.grad(loss, latents, retain_graph=True)[0]

        else:
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            grad = w * (noise_pred_pretrain - noise_pred_est_text)
            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            grad = torch.nan_to_num(grad)

            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]


        return loss, t, latents_noisy, noise_pred_pretrain, grad.norm()

    def get_latents(
        self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)
        return latents

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        landmark_condition: Float[Tensor, "B C H W"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        current_step=None,
        **kwargs,
    ):
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents = self.get_latents(rgb_BCHW, rgb_as_latents=rgb_as_latents)
        self.current_step = current_step
        # view-dependent text embeddings
        if prompt_utils.use_perp_neg:
            (
                text_embeddings_vd,
                neg_guidance_weights_vd,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
        else:
            neg_guidance_weights_vd = None
            text_embeddings_vd = prompt_utils.get_text_embeddings(
                elevation,
                azimuth,
                camera_distances,
                view_dependent_prompting=self.cfg.view_dependent_prompting,
            )

        # input text embeddings, view-independent
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        )
        loss, t, latents_noisy, noise, grad = self.compute_grad_ssd(landmark_condition,
                                                                        latents, text_embeddings_vd, text_embeddings
                                                                        )

        if self.cfg.use_perceptual:
            image_out = self.sample(landmark_condition,  text_embeddings_vd)
            loss_perceptual = self.perceptual['loss'](rgb_BCHW * 2 - 1,
                                                      image_out.detach()).squeeze()

        else:
            loss_perceptual = None
        if loss_perceptual is None:
            return {
                "loss_ssd": loss,
                "grad_norm": grad.norm(),
                "min_step": self.min_step,
                "max_step": self.max_step,
            }

        else:

            return {
                "loss_ssd": loss,
                "loss_perceptual": loss_perceptual,
                "grad_norm": grad.norm(),
                "min_step": self.min_step,
                "max_step": self.max_step,
            }




    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        image_cond,
        latents_noisy,
        t,
        text_embeddings,
    ):

        # pred noise
        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)

        with self.disable_control_class_embedding(self.control_base) as controlnet:
            cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
            (
                down_block_res_samples,
                mid_block_res_sample,
            ) = self.forward_controlnet(
                controlnet,
                latent_model_input,
                torch.cat([t.unsqueeze(0)] * 2),
                encoder_hidden_states=text_embeddings,
                image_cond=torch.cat([image_cond] * 2, dim=0),
                condition_scale=self.cfg.condition_scale,
                cross_attention_kwargs=cross_attention_kwargs,

            )
        with self.disable_unet_class_embedding(self.unet) as unet:
            cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
            noise_pred_pretrain = self.forward_control_unet(
                unet,
                latent_model_input,
                torch.cat([t.unsqueeze(0)] * 2),
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,

            )
        noise_pred_text, noise_pred_uncond = noise_pred_pretrain.chunk(2)
        noise_pred_pretrain = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        return noise_pred_pretrain
    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _sample(
        self,
        pipe: StableDiffusionControlNetPipeline,
        sample_scheduler: DPMSolverMultistepScheduler,
        text_embeddings: Float[Tensor, "BB N Nf"],
        landmark_condition: Float[Tensor, "B C H W"],
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Float[Tensor, "B H W 3"]:
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        height = height or pipe.unet.config.sample_size * vae_scale_factor
        width = width or pipe.unet.config.sample_size * vae_scale_factor
        batch_size = text_embeddings.shape[0] // 2
        device = self.device

        sample_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = sample_scheduler.timesteps
        num_channels_latents = pipe.unet.config.in_channels

        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.weights_dtype,
            device,
            generator,
        )

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = sample_scheduler.scale_model_input(
                latent_model_input, t
            )
            (
                down_block_res_samples,
                mid_block_res_sample,
            ) = self.forward_controlnet(
                self.control_base,
                latent_model_input,
                torch.cat([t.unsqueeze(0)] * 2),
                encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                image_cond=torch.cat([landmark_condition] * 2, dim=0),
                condition_scale=self.cfg.condition_scale,
                cross_attention_kwargs=None
            )
            noise_pred = self.forward_control_unet(
                self.unet,
                latent_model_input,
                torch.cat([t.unsqueeze(0)] * 2),
                encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample
            )


            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = sample_scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / pipe.vae.config.scaling_factor * latents
        images = pipe.vae.decode(latents).sample
        return images

    def sample(
        self,
        landmark_condition: Float[Tensor, "B C H W"],
        text_embeddings_vd,
        seed: int = 42,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        # view-dependent text embeddings
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self._sample(
            pipe=self.pipe,
            sample_scheduler=self.scheduler_sample,
            landmark_condition=landmark_condition,
            text_embeddings=text_embeddings_vd,
            num_inference_steps=25,
            guidance_scale=self.cfg.guidance_scale,
            generator=generator,
        )

    # @torch.cuda.amp.autocast(enabled=False)
    # def sample_controlnet(
    #     self,
    #     landmark_condition,
    #     t_orig,
    #     text_embeddings,
    #     latents_noisy,
    #     noise_pred,
    #     use_perp_neg=False
    # ):
    #     # use only 50 timesteps, and find nearest of those to t
    #     self.scheduler.set_timesteps(25)
    #     self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
    #     bs = (latents_noisy.shape[0]
    #     )  # batch size
    #     large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
    #                                                                         :bs
    #                                                                         ].unsqueeze(
    #         -1
    #     )  # sized [bs,50] > [bs,1]
    #     idxs = torch.min(large_enough_idxs, dim=1)[1]
    #
    #     t = self.scheduler.timesteps_gpu[idxs]
    #
    #     # get prev latent
    #     latents_1step = []
    #     pred_1orig = []
    #     for b in range(bs):
    #         step_output = self.scheduler.step(
    #             noise_pred[b: b + 1], t[b], latents_noisy[b: b + 1]
    #         )
    #         latents_1step.append(step_output["prev_sample"])
    #         pred_1orig.append(step_output["pred_original_sample"])
    #     latents_1step = torch.cat(latents_1step)
    #     latents_final = []
    #     for b, i in enumerate(idxs):
    #         latents = latents_1step[b: b + 1]
    #         text_emb = (
    #             text_embeddings[
    #                 [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
    #             ]
    #             if use_perp_neg
    #             else text_embeddings[[b, b + len(idxs)], ...]
    #         )
    #         for t in self.scheduler.timesteps[i + 1:]:
    #             # pred noise
    #             noise_pred = self.get_noise_pred(landmark_condition,
    #                                              latents, t.to(self.device), text_emb
    #                                              )
    #             # get prev latent
    #             latents = self.scheduler.step(noise_pred, t.to(self.device), latents)[
    #                 "prev_sample"
    #             ]
    #         latents_final.append(latents)
    #     latents_final = torch.cat(latents_final)
    #     imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)
    #
    #     return imgs_final

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
