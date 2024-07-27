from dataclasses import dataclass, field
import math
import numpy
import torch
import torch.nn.functional as F
import PIL
from PIL import Image
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.models.landmark_control.landmark_rotation import LandmarkRotation
from threestudio.utils.misc import get_device, load_module_weights


@threestudio.register("fantasia3d-dhead-deform-system")
class Fantasia3DDeform(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        latent_steps: int = 1000
        texture: bool = False
        landmark_dir: str = ""
        depth_dir: str = ""

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        # self.pre_process(self.cfg.landmark_dir, self.cfg.depth_dir)
        self.base_out = None
        self.depth_out = None

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch, render_rgb=self.cfg.texture)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # if not self.cfg.texture:
        #     # initialize SDF
        #     # FIXME: what if using other geometry types?
        #     self.geometry.initialize_shape()

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

    # def pre_process(self, cond_rgb: str, depthmodel_dir: str, image_size=512):
    #
    #     self.landmark_process = LandmarkRotation(cond_rgb, image_size, image_size, get_device())
    #     threestudio.info(f"Loaded depth and landmark model!")
    def training_step(self, batch, batch_idx):
        loss = 0.0

        out = self(batch)
        prompt_utils = self.prompt_processor()
        camera_positions, azimuth, elevation, f, only_head = self.camera_setting(batch['elevation'], batch['azimuth'],
                                                                                 fov=batch['fov'])
        batch_size = batch['elevation'].shape[0]
        landmark_condition_batch = torch.zeros(batch_size, 3, 512, 512, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            for k in range(len(elevation)):
                cond_rgb = batch['exp_dir'][k]
                landmark_process = LandmarkRotation(cond_rgb, 512, 512, get_device())
                landmark_condition = landmark_process.changing(elevation[k], azimuth[k], 0, f[k], only_head[k])
                landmark_condition = self.condition_process(landmark_condition, name='landmark')
                landmark_condition_batch[k] = landmark_condition[0]
        if not self.cfg.texture:  # geometry training
            if self.true_global_step < self.cfg.latent_steps:
                # guidance_inp = torch.cat(
                #     [out["comp_normal"] * 2.0 - 1.0, out["opacity"]], dim=-1
                # )
                guidance_inp = torch.cat(
                    [out["comp_gan_normal"] * 2.0 - 1.0, out["opacity_gan"]], dim=-1
                )
                guidance_out = self.guidance(
                    guidance_inp, landmark_condition_batch, prompt_utils, **batch, rgb_as_latents=True,
                    current_step_ratio=self.true_global_step / self.trainer.max_steps
                )
            else:
                guidance_inp = out["comp_gan_normal"]
                guidance_out = self.guidance(
                    guidance_inp, landmark_condition_batch, prompt_utils, **batch, rgb_as_latents=False,
                    current_step_ratio=self.true_global_step / self.trainer.max_steps
                )

            loss_normal_consistency_batch = 0
            loss_laplacian_smoothness_batch = 0
            for i in range(batch_size):
                loss_normal_consistency = out["mesh_deform"][i].normal_consistency()
                loss_normal_consistency_batch += loss_normal_consistency * self.C(
                    self.cfg.loss.lambda_normal_consistency
                )
                if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                    loss_laplacian_smoothness = out["mesh_deform"][i].laplacian()
                    self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                    loss_laplacian_smoothness_batch += loss_laplacian_smoothness * self.C(
                        self.cfg.loss.lambda_laplacian_smoothness
                    )
            loss_normal_consistency_batch = loss_normal_consistency_batch / batch_size
            loss_laplacian_smoothness_batch = loss_laplacian_smoothness_batch / batch_size
            loss = loss + loss_normal_consistency_batch + loss_laplacian_smoothness_batch
            self.log("train/loss_normal_consistency", loss_normal_consistency_batch)
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                self.log("train/loss_normal_consistency", loss_laplacian_smoothness_batch)
        else:  # texture training
            guidance_inp = out["comp_rgb"]
            guidance_out = self.guidance(
                guidance_inp, landmark_condition_batch, prompt_utils, **batch, rgb_as_latents=False,
                current_step_ratio=self.true_global_step / self.trainer.max_steps
            )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        exp_rgb = batch["exp"]  # 0~1 512
        B, H, W, C = exp_rgb.shape
        # exp_pra = self.get_flame_pra(exp_rgb.permute(0, 3, 1, 2), batch_size=B, is_exp=True)
        # batch['exp_pra'] = exp_pra
        out = self(batch)
        rgb = batch["exp"][0]
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_gan_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if self.cfg.texture
                else []
            )
            + [{
                "type": "rgb",
                "img": rgb,
                "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
            },
                {
                    "type": "grayscale",
                    "img": out["opacity_gan"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["comp_gan_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        rgb = batch["exp"][0]
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_gan_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if self.cfg.texture
                else []
            )
            + [{
                "type": "rgb",
                "img": rgb,
                "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
            },
                {
                    "type": "grayscale",
                    "img": out["opacity_gan"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["comp_gan_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
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
