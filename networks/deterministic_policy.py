import torch as t
from copy import deepcopy
from .base import Network

class DeterministicPolicy(Network[t.Tensor, t.Tensor]):

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int,
        num_layers: int,
        hidden_activation: t.nn.Module,
        output_activation: t.nn.Module,
    ) -> None:
        super().__init__()

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(t.nn.Linear(in_size, hidden_size))
                layers.append(deepcopy(hidden_activation))
            elif i == num_layers - 1:
                layers.append(t.nn.Linear(hidden_size, out_size))
                layers.append(output_activation)
            else:
                layers.append(t.nn.Linear(hidden_size, hidden_size))
                layers.append(deepcopy(hidden_activation))

        self._layers = t.nn.Sequential(*layers)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self._layers(x)
