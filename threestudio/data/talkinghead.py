import bisect
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *


@dataclass
class TalkingImageDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 96
    width: Any = 96
    image_path_json: str = ""
    exp_image_json: str = ""
    image_test_path: str = ""
    exp_images_dir: str = ""
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    batch_size: int = 1
    exp_pra_dir: str = "/home/liuhongyu/code/dataset/ffhq1024/exp"
    exp_test_path: str = ''


class TalkingImageDataBase:
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: TalkingImageDataModuleConfig = cfg

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)
        with open(self.cfg.image_path_json, 'r') as fcc_file:
            json_load = json.load(fcc_file)
        self.image_data = json_load['images']

        self.batch_size = self.random_pose_generator.batch_size
        self.n_frames = len(self.image_data)
        with open(self.cfg.exp_image_json, 'r') as fcc_file:
            self.json_load_exp = json.load(fcc_file)

        self.exp_image_data = self.json_load_exp['happiness']
        self.select_exp = 'happiness'
        self.exp_n_frames = len(self.exp_image_data)
        self.exp_list = ["contempt", "disgust", "fear", "happiness", "sadness", "surprise", "anger"]
        self.exp_idex = {"anger": 1., "contempt": 2., "disgust": 3., "fear": 4., "happiness": 5., "sadness": 6.,
                         "surprise": 7.}
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]

    def load_images(self, image_path):
        # load image
        assert os.path.exists(image_path)
        rgb = cv2.cvtColor(
            cv2.imread(image_path), cv2.COLOR_BGR2RGB
        )
        rgb = (
                  cv2.resize(
                      rgb, (self.width, self.height), interpolation=cv2.INTER_AREA
                  ).astype(np.float32)
              ) / 255.0
        tensor_rgb = torch.from_numpy(rgb).contiguous().to(self.rank)
        numpy_rgb = rgb
        return [tensor_rgb, numpy_rgb]


class TalkingImageIterableDataset(IterableDataset, TalkingImageDataBase, Updateable):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def collate(self, batch) -> Dict[str, Any]:
        index_image = np.random.randint(0, self.n_frames, size=self.batch_size)
        index_exp = np.random.randint(0, self.exp_n_frames, size=self.batch_size)
        batch = self.random_pose_generator.collate(None)
        out_rgb_numpy = []
        out_rgb_tensor = []
        out_exp_numpy = []
        out_exp_tensor = []
        out_exp_label = []
        out_exp_dir = []
        out_exp_pra = []
        for i in range(self.batch_size):
            rgb = self.load_images(self.image_data[index_image[i]])
            exp = self.load_images(
                os.path.join(self.cfg.exp_images_dir, self.exp_image_data[index_exp[i]]['name'] + '.png'))
            out_exp_pra.append(
                torch.load(os.path.join(self.cfg.exp_pra_dir, self.exp_image_data[index_exp[i]]['name'] + '.pt')).to(
                    self.rank))
            out_rgb_tensor.append(rgb[0])
            # out_rgb_numpy.append(rgb[1])
            out_exp_tensor.append(exp[0])
            out_exp_label.append(torch.tensor(self.exp_idex[self.select_exp]).to(self.rank))
            out_exp_dir.append(
                os.path.join(self.cfg.exp_images_dir, self.exp_image_data[index_exp[i]]['name'] + '.png'))
            out_exp_numpy.append(exp[1])

            # rgb = self.load_images(self.image_data[index_image[i]])
            # exp = self.load_images('/home/liuhongyu/code/dataset/ffhq1024/ffhq1024/00001.png')
            # out_exp_pra.append(torch.load('/home/liuhongyu/code/dataset/ffhq1024//exp/00001.pt').to(self.rank))
            # out_rgb_tensor.append(rgb[0])
            # # out_rgb_numpy.append(rgb[1])
            # out_exp_tensor.append(exp[0])
            # out_exp_label.append(torch.tensor(self.exp_idex['happiness']).to(self.rank))
            # out_exp_dir.append('/home/liuhongyu/code/dataset/ffhq1024/ffhq1024/00001.png')

        batch['rgb_real'] = torch.stack(out_rgb_tensor, dim=0)
        batch['exp'] = torch.stack(out_exp_tensor, dim=0)
        batch['exp_label'] = torch.stack(out_exp_label, dim=0)
        batch['exp_dir'] = out_exp_dir
        batch['exp_pra'] = torch.stack(out_exp_pra, dim=0)
        # batch['rgb_numpy'] = out_rgb_numpy
        # batch['exp_numpy'] = out_exp_numpy
        return batch

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.select_exp = self.exp_list[global_step % len(self.exp_list)]
        # self.select_exp = "happiness"
        self.exp_image_data = self.json_load_exp[self.select_exp]
        self.exp_n_frames = len(self.exp_image_data)
        self.random_pose_generator.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}


class TalkingImageDataset(Dataset, TalkingImageDataBase):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def __len__(self):
        return len(self.random_pose_generator)

    def __getitem__(self, index):
        out = self.random_pose_generator[index]
        exp_rgb = self.load_images(self.cfg.image_test_path)[0]
        out['exp_pra'] = torch.load(self.cfg.exp_test_path)
        out['exp'] = exp_rgb
        return out
        # if index == 0:
        #     return {
        #         'rays_o': self.rays_o[0],
        #         'rays_d': self.rays_d[0],
        #         'mvp_mtx': self.mvp_mtx[0],
        #         'camera_positions': self.camera_position[0],
        #         'light_positions': self.light_position[0],
        #         'elevation': self.elevation_deg[0],
        #         'azimuth': self.azimuth_deg[0],
        #         'camera_distances': self.camera_distance[0],
        #         'rgb': self.rgb[0],
        #         'depth': self.depth[0],
        #         'mask': self.mask[0]
        #     }
        # else:
        #     return self.random_pose_generator[index - 1]


@register("talking-image-datamodule")
class TalkingImageDataModule(pl.LightningDataModule):
    cfg: TalkingImageDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(TalkingImageDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = TalkingImageIterableDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = TalkingImageDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = TalkingImageDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)
