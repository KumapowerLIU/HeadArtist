import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import DeformationNetwork
from threestudio.utils.ops import dot, get_activation
from threestudio.utils.typing import *
from threestudio.utils.config import config_to_primitive


@threestudio.register("no-material-deform")
class NoMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        n_output_dims: int = 3
        color_activation: str = "sigmoid"
        input_feature_dims: Optional[int] = None
        deformation_network_config: dict = field(
            default_factory=lambda: {
                "dim_in_mapping": 50,
                "n_neurons_mapping": 64,
                "n_hidden_layers_mapping": 3,
                "n_neurons_film": 64,
                "n_hidden_layers_film": 3,
            }
        )
        use_network: bool = True

    cfg: Config

    def configure(self) -> None:
        self.use_network = self.cfg.use_network
        deformation_net_cfg = config_to_primitive(self.cfg.deformation_network_config)
        if self.use_network:
            self.network = DeformationNetwork(dim_in_film=32, dim_out_film=3,
                                              config=deformation_net_cfg)

    def forward(
        self, features: Float[Tensor, "B ... Nf"], enc_can, exp, is_can=False, **kwargs
    ) -> Float[Tensor, "B ... Nc"]:
        if is_can:
            color = get_activation(self.cfg.color_activation)(features)
        else:
            if not self.use_network:
                assert (
                    features.shape[-1] == self.cfg.n_output_dims
                ), f"Expected {self.cfg.n_output_dims} output dims, only got {features.shape[-1]} dims input."
                color = get_activation(self.cfg.color_activation)(features)
            else:
                color_deform_features = self.network(enc_can, exp).view(
                    *enc_can.shape[:-1], self.cfg.n_output_dims
                )
                features_deform = color_deform_features + features
                color = get_activation(self.cfg.color_activation)(features_deform)
        return color

    def export(self, features: Float[Tensor, "*N Nf"], **kwargs) -> Dict[str, Any]:
        color = self(features, **kwargs).clamp(0, 1)
        assert color.shape[-1] >= 3, "Output color must have at least 3 channels"
        if color.shape[-1] > 3:
            threestudio.warn(
                "Output color has >3 channels, treating the first 3 as RGB"
            )
        return {"albedo": color[..., :3]}
