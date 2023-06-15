from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

import modulus.models.layers as layers
from modulus.models.arch import Arch
from modulus.key import Key


class ModifiedFourierNetArch(Arch):
    """
    A modified Fourier Network which enables multiplicative interactions
    betweeen the Fourier features and hidden layers.
    References:
    (1) Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S.,
    Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J.T. and Ng, R., 2020.
    Fourier features let networks learn high frequency functions in low dimensional domains.
    arXiv preprint arXiv:2006.10739.
    (2) Wang, S., Teng, Y. and Perdikaris, P., 2020.
    Understanding and mitigating gradient pathologies in physics-informed
    neural networks. arXiv preprint arXiv:2001.04536.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list
    output_keys : List[Key]
        Output key list
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    frequencies : Tuple[str, List[float]] = ("axis", [i for i in range(10)])
        A tuple that describes the Fourier encodings to use any inputs in
        the list `['x', 'y', 'z', 't']`.
        The first element describes the type of frequency encoding
        with options, `'gaussian', 'full', 'axis', 'diagonal'`.
        `'gaussian'` samples frequency of Fourier series from Gaussian.
        `'axis'` samples along axis of spectral space with the given list range of frequencies.
        `'diagonal'` samples along diagonal of spectral space with the given list range of frequencies.
        `'full'` samples along entire spectral space for all combinations of frequencies in given list.
    frequencies_params : Tuple[str, List[float]] = ("axis", [i for i in range(10)])
        Same as `frequencies` except these are used for encodings
        on any inputs not in the list `['x', 'y', 'z', 't']`.
    activation_fn : layers.Activation = layers.Activation.SILU
        Activation function used by network.
    layer_size : int = 512
        Layer size for every hidden layer of the model.
    nr_layers : int = 6
        Number of hidden layers of the model.
    skip_connections : bool = False
        If true then apply skip connections every 2 hidden layers.
    weight_norm : bool = True
        Use weight norm on fully connected layers.
    adaptive_activations : bool = False
        If True then use an adaptive activation function as described here
        https://arxiv.org/abs/1906.01170.
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        frequencies=("axis", [i for i in range(10)]),
        frequencies_params=("axis", [i for i in range(10)]),
        activation_fn=layers.Activation.SILU,
        layer_size: int = 512,
        nr_layers: int = 6,
        skip_connections: bool = False,
        weight_norm: bool = True,
        adaptive_activations: bool = False,
    ) -> None:
        super().__init__(
            input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys
        )

        self.skip_connections = skip_connections

        if adaptive_activations:
            activation_par = nn.Parameter(torch.ones(1))
        else:
            activation_par = None

        self.xyzt_var = [x for x in self.input_key_dict if x in ["x", "y", "z", "t"]]
        # Prepare slice index
        xyzt_slice_index = self.prepare_slice_index(self.input_key_dict, self.xyzt_var)
        self.register_buffer("xyzt_slice_index", xyzt_slice_index, persistent=False)

        self.params_var = [
            x for x in self.input_key_dict if x not in ["x", "y", "z", "t"]
        ]
        params_slice_index = self.prepare_slice_index(
            self.input_key_dict, self.params_var
        )
        self.register_buffer("params_slice_index", params_slice_index, persistent=False)

        in_features_xyzt = sum(
            (v for k, v in self.input_key_dict.items() if k in self.xyzt_var)
        )
        in_features_params = sum(
            (v for k, v in self.input_key_dict.items() if k in self.params_var)
        )
        in_features = in_features_xyzt + in_features_params
        out_features = sum(self.output_key_dict.values())

        in_features = in_features_xyzt + in_features_params

        if in_features_xyzt > 0:
            self.fourier_layer_xyzt = layers.FourierLayer(
                in_features=in_features_xyzt, frequencies=frequencies
            )
            in_features += self.fourier_layer_xyzt.out_features()
        else:
            self.fourier_layer_xyzt = None

        if in_features_params > 0:
            self.fourier_layer_params = layers.FourierLayer(
                in_features=in_features_params, frequencies=frequencies_params
            )
            in_features += self.fourier_layer_params.out_features()
        else:
            self.fourier_layer_params = None

        self.fc_u = layers.FCLayer(
            in_features=in_features,
            out_features=layer_size,
            activation_fn=activation_fn,
            weight_norm=weight_norm,
            activation_par=activation_par,
        )

        self.fc_v = layers.FCLayer(
            in_features=in_features,
            out_features=layer_size,
            activation_fn=activation_fn,
            weight_norm=weight_norm,
            activation_par=activation_par,
        )

        self.fc_0 = layers.FCLayer(
            in_features,
            layer_size,
            activation_fn,
            weight_norm,
            activation_par=activation_par,
        )

        self.fc_layers = nn.ModuleList()

        for i in range(nr_layers - 1):
            self.fc_layers.append(
                layers.FCLayer(
                    layer_size,
                    layer_size,
                    activation_fn,
                    weight_norm,
                    activation_par=activation_par,
                )
            )

        self.final_layer = layers.FCLayer(
            in_features=layer_size,
            out_features=out_features,
            activation_fn=layers.Activation.IDENTITY,
            weight_norm=False,
            activation_par=None,
        )

    def _tensor_forward(self, x: Tensor) -> Tensor:
        x = self.process_input(
            x, self.input_scales_tensor, input_dict=self.input_key_dict, dim=-1
        )
        if self.fourier_layer_xyzt is not None:
            in_xyzt_var = self.slice_input(x, self.xyzt_slice_index, dim=-1)
            fourier_xyzt = self.fourier_layer_xyzt(in_xyzt_var)
            x = torch.cat((x, fourier_xyzt), dim=-1)
        if self.fourier_layer_params is not None:
            in_params_var = self.slice_input(x, self.params_slice_index, dim=-1)
            fourier_params = self.fourier_layer_params(in_params_var)
            x = torch.cat((x, fourier_params), dim=-1)

        xu = self.fc_u(x)
        xv = self.fc_v(x)

        x = self.fc_0(x)

        x_skip: Optional[Tensor] = None
        for i, layer in enumerate(self.fc_layers, 1):
            x = layer(x)
            x = xu - x * xu + x * xv
            if self.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    x, x_skip = x + x_skip, x
                else:
                    x_skip = x

        x = self.final_layer(x)
        x = self.process_output(x, self.output_scales_tensor)
        return x

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.concat_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        y = self._tensor_forward(x)
        return self.split_output(y, self.output_key_dict, dim=-1)

    def _dict_forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        This is the original forward function, left here for the correctness test.
        """
        x = self.prepare_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
        )

        if self.fourier_layer_xyzt is not None:
            in_xyzt_var = self.prepare_input(
                in_vars,
                self.xyzt_var,
                detach_dict=self.detach_key_dict,
                dim=-1,
                input_scales=self.input_scales,
            )
            fourier_xyzt = self.fourier_layer_xyzt(in_xyzt_var)
            x = torch.cat((x, fourier_xyzt), dim=-1)
        if self.fourier_layer_params is not None:
            in_params_var = self.prepare_input(
                in_vars,
                self.params_var,
                detach_dict=self.detach_key_dict,
                dim=-1,
                input_scales=self.input_scales,
            )
            fourier_params = self.fourier_layer_params(in_params_var)
            x = torch.cat((x, fourier_params), dim=-1)

        xu = self.fc_u(x)
        xv = self.fc_v(x)

        x = self.fc_0(x)

        x_skip: Optional[Tensor] = None
        for i, layer in enumerate(self.fc_layers, 1):
            x = layer(x)
            x = xu - x * xu + x * xv
            if self.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    x, x_skip = x + x_skip, x
                else:
                    x_skip = x

        x = self.final_layer(x)
        return self.prepare_output(
            x, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )
