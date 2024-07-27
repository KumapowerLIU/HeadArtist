from dataclasses import dataclass, field
import math
import cv2
import PIL
from PIL import Image
import torch
import numpy
import threestudio
import torch.nn.functional as F
from einops import rearrange, repeat
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.models.landmark_control.landmark_rotation import LandmarkRotation
from threestudio.models.depth_warping.warping import DepthWarping
from controlnet_aux import CannyDetector
from threestudio.utils.misc import get_device, load_module_weights
from threestudio.utils.perceptual.perceptual import PerceptualLoss
from PIL import ImageFilter
import kornia


@threestudio.register("dheadvsd-edit-system")
class DheadVsdDreamFusion(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        latent_steps: int = 1000
        stage: str = "coarse"
        visualize_samples: bool = False
        landmark_dir: str = ""
        depth_dir: str = ""
        use_sdf: bool = False
        use_edge_and_color: bool = False
        canny_lower_bound: int = 50
        canny_upper_bound: int = 100
        use_guidance: str = ""
        retrain: bool = False
        normal_save_test: bool = True
        front_dir: str = ""
        back_dir: str = ""
        front_input_dir: str = ""
        back_input_dir: str = ""

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.pre_process(self.cfg.landmark_dir, self.cfg.depth_dir, self.cfg.front_dir, self.cfg.back_dir,
                         self.cfg.front_input_dir, self.cfg.back_input_dir)
        self.base_out = None
        self.depth_out = None
        self.prompt_utils = self.prompt_processor()
        self.canny = CannyDetector()

    def get_color(self, nerf_out):
        # nerf_out_numpy = nerf_out.data.detach().cpu().numpy()
        # nerf_out_numpy = nerf_out_numpy.clip(min=0, max=1)
        # nerf_out_numpy = (
        #     nerf_out_numpy * 255.0
        # ).astype(numpy.uint8)
        # stoke = Image.fromarray(nerf_out_numpy)
        # stoke = stoke.filter(ImageFilter.MedianFilter(size=25))
        # stoke = numpy.array(stoke)
        # stoke = torch.from_numpy(stoke).float().to(get_device()) / 255.0  # 0 1
        # stoke = stoke.permute(2, 0, 1)
        stroke = nerf_out.unsqueeze(0).permute(0, 3, 1, 2)
        stroke = kornia.filters.median_blur(stroke, (25, 25))
        stroke = stroke.permute(0, 2, 3, 1)
        return stroke[0]

    def get_edge(self, nerf_out):
        # cond_rgb = (
        #     ( nerf_out.cpu().numpy() * 255).astype(numpy.uint8).copy()
        # )
        # blurred_img = cv2.blur(cond_rgb, ksize=(5, 5))
        # detected_map = self.canny(
        #     blurred_img, self.cfg.canny_lower_bound, self.cfg.canny_upper_bound
        # )
        # edge = (
        #     torch.from_numpy(numpy.array(detected_map)).float().to(self.device) / 255.0
        # )
        # print(edge.shape)
        # edge = edge.unsqueeze(-1).repeat(1, 1, 3)
        # edge = edge.permute(2, 0, 1)
        edge = kornia.filters.sobel(nerf_out.unsqueeze(0).permute(0, 3, 1, 2))
        return edge.permute(0, 2, 3, 1)[0]

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.cfg.stage == "geometry":
            render_out = self.renderer(**batch, render_rgb=False)
        else:
            render_out = self.renderer(**batch)

        return {
            **render_out,
        }

    def condition_process(self, image, name='base'):
        if name == 'base':
            if isinstance(image, PIL.Image.Image) is False:
                raise ValueError('check the base output')
            else:
                image = image.convert("RGB")
                image = image.resize((512, 512))
                image = numpy.array(image).astype(numpy.float32)
                return image
        elif name == 'warping':
            # batch 1
            return (2. * image - 1.).to(get_device())
        elif name == 'landmark':
            image = numpy.moveaxis(image, 2, 0)  # h, w, c -> c, h, w
            image = torch.from_numpy(image.copy()).float().to(get_device()) / 255.0
            image = image.unsqueeze(0)
            return image
        elif name == 'ref':
            if isinstance(image, str) is False:
                raise ValueError('check the base output')
            else:
                image = Image.open(image)
                image = image.convert("RGB")
                image = image.resize((512, 512))
                image = numpy.array(image).astype(numpy.float32)
                image = numpy.moveaxis(image, 2, 0)  # h, w, c -> c, h, w
                image = torch.from_numpy(image.copy()).float().to(get_device()) / 255.0
                image = image.unsqueeze(0)
                return image

    def pre_process(self, cond_rgb: str, depthmodel_dir: str, front_img_dir: str, back_img_dir: str,
                    front_input_dir: str, back_input_dir: str, image_size=512):
        self.landmark_process = LandmarkRotation(cond_rgb, image_size, image_size, get_device())
        self.depth_process = DepthWarping(get_device(), image_size, depthmodel_dir)
        self.front_image = self.condition_process(front_img_dir, name='ref')
        self.back_image = self.condition_process(back_img_dir, name='ref')
        self.front_input = torch.load(front_input_dir)
        self.back_input = torch.load(back_input_dir)
        threestudio.info(f"Loaded depth and landmark model and front and back!")

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.cfg.use_sdf:
            self.geometry.initialize_shape()

    def camera_setting(self, elevation_deg, azimuth_deg, fov, camera_distances=2, image_size=512, thre=90.):
        B = elevation_deg.shape[0]
        f_out = []
        elevation_out = []
        azimuth_out = []
        camera_positions_out = []
        only_head_out = []
        for i in range(B):
            if azimuth_deg[i] > thre or azimuth_deg[i] < -thre:
                only_head = True
            else:
                only_head = False
            azimuth = azimuth_deg[i] * math.pi / 180
            elevation = elevation_deg[i] * math.pi / 180
            camera_positions = numpy.array(
                [camera_distances * math.sin(azimuth) * math.cos(elevation), camera_distances * math.sin(elevation),
                 camera_distances * math.cos(azimuth) * math.cos(elevation)])
            # you need return eye and
            f = fov[i] / 180 * math.pi
            f = torch.tensor(0.5 * image_size) / (torch.tan(torch.tensor(f / 2)))
            f_out.append(f)
            elevation_out.append(elevation)
            azimuth_out.append(azimuth)
            camera_positions_out.append(camera_positions)
            only_head_out.append(only_head)
        return camera_positions_out, azimuth_out, elevation_out, f_out, only_head_out

    def training_step(self, batch, batch_idx):
        out = self(batch)
        guidance_eval = (self.cfg.guidance_eval > 0
                         and self.true_global_step % self.cfg.guidance_eval == 0
                         )
        camera_positions, azimuth, elevation, f, only_head = self.camera_setting(batch['elevation'], batch['azimuth'],
                                                                                 fov=batch['fov'])

        batch_size = batch['elevation'].shape[0]
        landmark_condition_batch = torch.zeros(batch_size, 3, 512, 512, device=self.device, dtype=torch.float32)
        warping_condition = None
        with torch.no_grad():
            for k in range(len(elevation)):
                landmark_condition = self.landmark_process.changing(elevation[k], azimuth[k], 0, f[k], only_head[k])
                landmark_condition = self.condition_process(landmark_condition, name='landmark')
                landmark_condition_batch[k] = landmark_condition[0]
        if self.cfg.stage == "geometry":
            guidance_inp = out["comp_normal"]
            guidance_out = self.guidance(
                guidance_inp, landmark_condition_batch, self.prompt_utils, **batch, rgb_as_latents=False,
                current_step_ratio=self.true_global_step / self.trainer.max_steps, current_step=self.true_global_step
            )
        else:
            guidance_inp = out["comp_rgb"]
            if self.cfg.use_edge_and_color:
                edge_batch = torch.zeros(batch_size, 512, 512, 3, device=self.device, dtype=torch.float32)
                color_batch = torch.zeros(batch_size, 512, 512, 3, device=self.device, dtype=torch.float32)
                for k in range(batch_size):
                    guidance_inp_edge = self.get_edge(guidance_inp[k])
                    guidance_inp_color = self.get_color(guidance_inp[k])
                    edge_batch[k] = guidance_inp_edge
                    color_batch[k] = guidance_inp_color
                edge_batch = edge_batch * out["opacity"]
                guidance_out_edge = self.guidance(
                    edge_batch, warping_condition, landmark_condition_batch, self.prompt_utils, **batch,
                    rgb_as_latents=False,
                    current_step_ratio=self.true_global_step / self.trainer.max_steps,
                    current_step=self.true_global_step
                )
                guidance_out = self.guidance(
                    color_batch, warping_condition, landmark_condition_batch, self.prompt_utils, **batch,
                    rgb_as_latents=False,
                    current_step_ratio=self.true_global_step / self.trainer.max_steps,
                    current_step=self.true_global_step
                )
                guidance_out["loss_vsd"] = 0.5 * guidance_out_edge["loss_vsd"] + 0.5 * guidance_out["loss_vsd"]
                guidance_out["loss_lora"] = 0.5 * guidance_out_edge["loss_lora"] + 0.5 * guidance_out["loss_lora"]
                guidance_out["grad_norm"] = 0.5 * guidance_out_edge["grad_norm"] + 0.5 * guidance_out["grad_norm"]
            else:
                if self.cfg.use_guidance == 'vsd':
                    guidance_out = self.guidance(
                        guidance_inp, self.prompt_utils, **batch,
                        rgb_as_latents=False,
                    )
                elif self.cfg.use_guidance == 'vsdonelandmark':
                    guidance_out = self.guidance(
                        guidance_inp, landmark_condition_batch, self.prompt_utils, **batch, rgb_as_latents=False
                    )
                else:
                    guidance_out = self.guidance(
                        guidance_inp, warping_condition, landmark_condition_batch, self.prompt_utils, **batch,
                        rgb_as_latents=False,
                        current_step_ratio=self.true_global_step / self.trainer.max_steps,
                        current_step=self.true_global_step
                    )

        loss = 0.0

        for name, value in guidance_out.items():
            if name == 'eval':
                continue
            else:
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.cfg.stage == "coarse":
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                                  out["weights"].detach()
                                  * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                              ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            if "z_variance" in out:
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                self.log("train/loss_z_variance", loss_z_variance)
                loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)
            if "sdf_grad" in out:
                loss_eikonal = (
                    (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
                ).mean()
                self.log("train/loss_eikonal", loss_eikonal)
                loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)
                self.log("train/inv_std", out["inv_std"], prog_bar=True)
        elif self.cfg.stage == "geometry":
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )

            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )
        elif self.cfg.stage == "texture":
            out_front = self(self.front_input)
            out_back = self(self.back_input)
            # loss_front = F.mse_loss(out_front["comp_rgb"].permute(0, 3, 1, 2), self.front_image )
            # loss_back = F.mse_loss(out_back["comp_rgb"].permute(0, 3, 1, 2), self.back_image )

            loss_front = 0
            loss_back = 0
            self.log("train/loss_front", loss_front)
            self.log("train/loss_back", loss_back)
            loss += loss_front * self.C(
                self.cfg.loss.lambda_front_view
            )
            loss += loss_back * self.C(
                self.cfg.loss.lambda_back_view
            )

        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}")

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def guidance_evaluation_save(self, comp_rgb, guidance_eval_out):
        B, size = comp_rgb.shape[:2]
        resize = lambda x: F.interpolate(
            x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        filename = f"it{self.true_global_step}-train.png"

        def merge12(x):
            return x.reshape(-1, *x.shape[2:])

        self.save_image_grid(
            filename,
            [
                {
                    "type": "rgb",
                    "img": merge12(comp_rgb),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["gt_imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ) +
            (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["gt_imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ) +
            (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["gt_imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ) +
            (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["gt_imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ) +
            (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["landmark"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ) +
            (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["warping"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ),
            name="train_step",
            step=self.true_global_step,
            texts=guidance_eval_out["texts"],
        )

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )
        if self.cfg.visualize_samples:
            camera_positions, azimuth, elevation, f = self.camera_setting(batch['elevation'], batch['azimuth'],
                                                                          fov=batch['fov'])
            landmark_condition = self.landmark_process.changing(elevation, azimuth, 0, f)
            landmark_condition = self.condition_process(landmark_condition, name='landmark')
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}-sample.png",
                [
                    {
                        "type": "rgb",
                        "img": self.guidance.sample(
                            landmark_condition, self.prompt_utils, **batch, seed=self.global_step
                        )[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": self.guidance.sample_lora(
                            landmark_condition.clone().to(dtype=self.guidance.control_lora.dtype), self.prompt_utils,
                            **batch)[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="validation_step_samples",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normall" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    }
                ]
                if "opacityy" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )
        if self.cfg.normal_save_test:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                ),
                name="test_step",
                step=self.true_global_step,
                if_normal=True
            )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
