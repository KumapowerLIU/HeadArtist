import math

import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.utils.base import Updateable
from threestudio.utils.config import config_to_primitive
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *


class ProgressiveBandFrequency(nn.Module, Updateable):
    def __init__(self, in_channels: int, config: dict):
        super().__init__()
        self.N_freqs = config["n_frequencies"]
        self.in_channels, self.n_input_dims = in_channels, in_channels
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2 ** torch.linspace(0, self.N_freqs - 1, self.N_freqs)
        self.n_output_dims = self.in_channels * (len(self.funcs) * self.N_freqs)
        self.n_masking_step = config.get("n_masking_step", 0)
        self.update_step(
            None, None
        )  # mask should be updated at the beginning each step

    def forward(self, x):
        out = []
        for freq, mask in zip(self.freq_bands, self.mask):
            for func in self.funcs:
                out += [func(freq * x) * mask]
        return torch.cat(out, -1)

    def update_step(self, epoch, global_step, on_load_weights=False):
        if self.n_masking_step <= 0 or global_step is None:
            self.mask = torch.ones(self.N_freqs, dtype=torch.float32)
        else:
            self.mask = (
                            1.0
                            - torch.cos(
                            math.pi
                            * (
                                global_step / self.n_masking_step * self.N_freqs
                                - torch.arange(0, self.N_freqs)
                            ).clamp(0, 1)
                        )
                        ) / 2.0
            threestudio.debug(
                f"Update mask: {global_step}/{self.n_masking_step} {self.mask}"
            )


class TCNNEncoding(nn.Module):
    def __init__(self, in_channels, config, dtype=torch.float32) -> None:
        super().__init__()
        self.n_input_dims = in_channels
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, config, dtype=dtype)
        self.n_output_dims = self.encoding.n_output_dims

    def forward(self, x):
        return self.encoding(x)


class ProgressiveBandHashGrid(nn.Module, Updateable):
    def __init__(self, in_channels, config, dtype=torch.float32):
        super().__init__()
        self.n_input_dims = in_channels
        encoding_config = config.copy()
        encoding_config["otype"] = "Grid"
        encoding_config["type"] = "Hash"
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, encoding_config, dtype=dtype)
        self.n_output_dims = self.encoding.n_output_dims
        self.n_level = config["n_levels"]
        self.n_features_per_level = config["n_features_per_level"]
        self.start_level, self.start_step, self.update_steps = (
            config["start_level"],
            config["start_step"],
            config["update_steps"],
        )
        self.current_level = self.start_level
        self.mask = torch.zeros(
            self.n_level * self.n_features_per_level,
            dtype=torch.float32,
            device=get_rank(),
        )

    def forward(self, x):
        enc = self.encoding(x)
        enc = enc * self.mask
        return enc

    def update_step(self, epoch, global_step, on_load_weights=False):
        current_level = min(
            self.start_level
            + max(global_step - self.start_step, 0) // self.update_steps,
            self.n_level,
        )
        if current_level > self.current_level:
            threestudio.debug(f"Update current level to {current_level}")
        self.current_level = current_level
        self.mask[: self.current_level * self.n_features_per_level] = 1.0


class CompositeEncoding(nn.Module, Updateable):
    def __init__(self, encoding, include_xyz=False, xyz_scale=2.0, xyz_offset=-1.0):
        super(CompositeEncoding, self).__init__()
        self.encoding = encoding
        self.include_xyz, self.xyz_scale, self.xyz_offset = (
            include_xyz,
            xyz_scale,
            xyz_offset,
        )
        self.n_output_dims = (
            int(self.include_xyz) * self.encoding.n_input_dims
            + self.encoding.n_output_dims
        )

    def forward(self, x, *args):
        return (
            self.encoding(x, *args)
            if not self.include_xyz
            else torch.cat(
                [x * self.xyz_scale + self.xyz_offset, self.encoding(x, *args)], dim=-1
            )
        )


def get_encoding(n_input_dims: int, config) -> nn.Module:
    # input suppose to be range [0, 1]
    encoding: nn.Module
    if config.otype == "ProgressiveBandFrequency":
        encoding = ProgressiveBandFrequency(n_input_dims, config_to_primitive(config))
    elif config.otype == "ProgressiveBandHashGrid":
        encoding = ProgressiveBandHashGrid(n_input_dims, config_to_primitive(config))
    else:
        encoding = TCNNEncoding(n_input_dims, config_to_primitive(config))
    encoding = CompositeEncoding(
        encoding,
        include_xyz=config.get("include_xyz", False),
        xyz_scale=2.0,
        xyz_offset=-1.0,
    )  # FIXME: hard coded
    return encoding


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dim_in_mapping=0, n_hidden_layers=0, n_neuron=0, groups=8):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.GroupNorm(groups, hidden_dim)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # mapping_layers = [
        #     self.make_linear(dim_in_mapping, n_neurons, is_first=True, is_last=False),
        #     self.make_activation(),
        #     self.make_linear(n_neurons, hidden_dim * 2, is_first=False, is_last=True)
        # ]

        # self.mapping_network = nn.Sequential(*mapping_layers)

    def forward(self, x, scale, shift):
        x = self.layer(x)
        x = self.norm(x)
        # mapping_out = self.mapping_network(exp)
        # scale = mapping_out[..., :mapping_out.shape[-1] // 2]
        # shift = mapping_out[..., mapping_out.shape[-1] // 2:]
        x = x * (scale + 1) + shift
        x = self.act(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=False)
        return layer

    def make_activation(self):
        return nn.LeakyReLU(0.2, inplace=True)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv3d') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class DeformationNetwork(nn.Module):
    def __init__(self, dim_in_film, dim_out_film, config):
        super().__init__()
        dim_in_mapping = config["dim_in_mapping"]
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons_mapping"],
            config["n_hidden_layers_mapping"],
        )

        self.n_neurons_film, self.n_hidden_layers_film = (
            config["n_neurons_film"],
            config["n_hidden_layers_film"],
        )
        dim_out_mapping = self.n_neurons_film * 2

        layers = [
            self.make_linear(dim_in_mapping, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        layers += [
            self.make_linear(self.n_neurons, dim_out_mapping, is_first=False, is_last=True)
        ]
        self.mapping_network = nn.Sequential(*layers)

        layers_film = [
            FiLMLayer(dim_in_film, self.n_neurons_film)
        ]
        for i in range(self.n_hidden_layers_film - 1):
            layers_film += [
                FiLMLayer(self.n_neurons_film, self.n_neurons_film)
            ]
        self.layers_film = nn.ModuleList(layers_film)
        self.final_layer = self.make_linear(self.n_neurons_film, dim_out_film, is_first=False, is_last=True)
        self.mapping_network.apply(kaiming_leaky_init)
        self.layers_film.apply(kaiming_leaky_init)
        nn.init.zeros_(self.final_layer.weight)

    def forward(self, x, exp_condition):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP

        scale_shift = self.mapping_network(exp_condition)
        scale = scale_shift[..., :scale_shift.shape[-1] // 2]
        shift = scale_shift[..., scale_shift.shape[-1] // 2:]
        for index, layer in enumerate(self.layers_film):
            x = layer(x, scale, shift)
        x = self.final_layer(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=False)
        return layer

    def make_activation(self):
        return nn.LeakyReLU(0.2, inplace=True)


class DeformationNetworkConcat(nn.Module):
    def __init__(self, dim_in_film, dim_out_film, config):
        super().__init__()
        dim_in_mapping = config["dim_in_mapping"]
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons_mapping"],
            config["n_hidden_layers_mapping"],
        )
        layers = [
            self.make_linear(dim_in_mapping + dim_in_film, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]

        self.mapping_network = nn.Sequential(*layers)
        self.final_layer = self.make_linear(self.n_neurons, dim_out_film, is_first=False, is_last=True)
        self.mapping_network.apply(kaiming_leaky_init)
        nn.init.zeros_(self.final_layer.weight)

    def forward(self, x, exp_condition):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        x = self.mapping_network(torch.concat([x, exp_condition.repeat(x.shape[0], 1)], dim=-1))
        x = self.final_layer(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=False)
        return layer

    def make_activation(self):
        return nn.LeakyReLU(0.2, inplace=True)


class DeformationNetworkExp(nn.Module):
    def __init__(self, dim_in_film, dim_out_film, config):
        super().__init__()
        dim_in_mapping = config["dim_in_mapping"]
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons_mapping"],
            config["n_hidden_layers_mapping"],
        )

        self.n_neurons_film, self.n_hidden_layers_film = (
            config["n_neurons_film"],
            config["n_hidden_layers_film"],
        )

        layers_film = [
            FiLMLayer(dim_in_film, self.n_neurons_film, dim_in_mapping, self.n_hidden_layers, self.n_neurons)
        ]
        for i in range(self.n_hidden_layers_film - 1):
            layers_film += [
                FiLMLayer(self.n_neurons_film, self.n_neurons_film, dim_in_mapping, self.n_hidden_layers,
                          self.n_neurons)
            ]
        self.layers_film = nn.ModuleList(layers_film)
        self.final_layer = self.make_linear(self.n_neurons_film, dim_out_film, is_first=False, is_last=True)
        self.layers_film.apply(kaiming_leaky_init)
        nn.init.zeros_(self.final_layer.weight)

    def forward(self, x, exp_condition):
        # disable autocast
        with torch.cuda.amp.autocast(enabled=False):
            for index, layer in enumerate(self.layers_film):
                x = layer(x, exp_condition)
            x = self.final_layer(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=False)
        return layer

    def make_activation(self):
        return nn.LeakyReLU(0.2, inplace=True)


class VanillaMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: dict):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons"],
            config["n_hidden_layers"],
        )
        layers = [
            self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        layers += [
            self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(config.get("output_activation", None))

    def forward(self, x):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        with torch.cuda.amp.autocast(enabled=False):
            x = self.layers(x)
            x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=False)
        return layer

    def make_activation(self):
        return nn.ReLU(inplace=True)


class SphereInitVanillaMLP(nn.Module):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons"],
            config["n_hidden_layers"],
        )
        self.sphere_init, self.weight_norm = True, True
        self.sphere_init_radius = config["sphere_init_radius"]
        self.sphere_init_inside_out = config["inside_out"]

        self.layers = [
            self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        self.layers += [
            self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)
        ]
        self.layers = nn.Sequential(*self.layers)
        self.output_activation = get_activation(config.get("output_activation", None))

    def forward(self, x):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        with torch.cuda.amp.autocast(enabled=False):
            x = self.layers(x)
            x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=True)

        if is_last:
            if not self.sphere_init_inside_out:
                torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(
                    layer.weight,
                    mean=math.sqrt(math.pi) / math.sqrt(dim_in),
                    std=0.0001,
                )
            else:
                torch.nn.init.constant_(layer.bias, self.sphere_init_radius)
                torch.nn.init.normal_(
                    layer.weight,
                    mean=-math.sqrt(math.pi) / math.sqrt(dim_in),
                    std=0.0001,
                )
        elif is_first:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
            torch.nn.init.normal_(
                layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dim_out)
            )
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dim_out))

        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer

    def make_activation(self):
        return nn.Softplus(beta=100)


class TCNNNetwork(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: dict) -> None:
        super().__init__()
        with torch.cuda.device(get_rank()):
            self.network = tcnn.Network(dim_in, dim_out, config)

    def forward(self, x):
        return self.network(x).float()  # transform to float32


def get_mlp(n_input_dims, n_output_dims, config) -> nn.Module:
    network: nn.Module
    if config.otype == "VanillaMLP":
        network = VanillaMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    elif config.otype == "SphereInitVanillaMLP":
        network = SphereInitVanillaMLP(
            n_input_dims, n_output_dims, config_to_primitive(config)
        )
    else:
        assert (
            config.get("sphere_init", False) is False
        ), "sphere_init=True only supported by VanillaMLP"
        network = TCNNNetwork(n_input_dims, n_output_dims, config_to_primitive(config))
    return network


class NetworkWithInputEncoding(nn.Module, Updateable):
    def __init__(self, encoding, network):
        super().__init__()
        self.encoding, self.network = encoding, network

    def forward(self, x):
        return self.network(self.encoding(x))


class TCNNNetworkWithInputEncoding(nn.Module):
    def __init__(
        self,
        n_input_dims: int,
        n_output_dims: int,
        encoding_config: dict,
        network_config: dict,
    ) -> None:
        super().__init__()
        with torch.cuda.device(get_rank()):
            self.network_with_input_encoding = tcnn.NetworkWithInputEncoding(
                n_input_dims=n_input_dims,
                n_output_dims=n_output_dims,
                encoding_config=encoding_config,
                network_config=network_config,
            )

    def forward(self, x):
        return self.network_with_input_encoding(x).float()  # transform to float32


def create_network_with_input_encoding(
    n_input_dims: int, n_output_dims: int, encoding_config, network_config
) -> nn.Module:
    # input suppose to be range [0, 1]
    network_with_input_encoding: nn.Module
    if encoding_config.otype in [
        "VanillaFrequency",
        "ProgressiveBandHashGrid",
    ] or network_config.otype in ["VanillaMLP", "SphereInitVanillaMLP"]:
        encoding = get_encoding(n_input_dims, encoding_config)
        network = get_mlp(encoding.n_output_dims, n_output_dims, network_config)
        network_with_input_encoding = NetworkWithInputEncoding(encoding, network)
    else:
        network_with_input_encoding = TCNNNetworkWithInputEncoding(
            n_input_dims=n_input_dims,
            n_output_dims=n_output_dims,
            encoding_config=config_to_primitive(encoding_config),
            network_config=config_to_primitive(network_config),
        )
    return network_with_input_encoding


if __name__ == "__main__":
    deformation_net_cfg = {
        "dim_in_mapping": 100,
        "n_neurons_mapping": 128,
        "n_hidden_layers_mapping": 4,
        "n_neurons_film": 64,
        "n_hidden_layers_film": 4, }
    deformation_net = DeformationNetwork(dim_in_film=32, dim_out_film=3, config=deformation_net_cfg).to("cuda:1")

    exp = torch.rand(1, 100).to("cuda:1")
    points = torch.rand(277410, 32).to("cuda:1")
    out = deformation_net(points, exp)
