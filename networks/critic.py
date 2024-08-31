import torch as t
from .base import Network


class Critic(Network[t.Tensor, t.Tensor]):

    def __init__(
        self,
        observation_flatdim: int,
        action_flatdim: int,
        hidden_size: int,
        num_layers: int,
        hidden_activation: t.nn.Module,
    ) -> None:
        super().__init__()

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(
                    t.nn.Linear(observation_flatdim + action_flatdim, hidden_size)
                )
                layers.append(hidden_activation)
            elif i == num_layers - 1:
                layers.append(t.nn.Linear(hidden_size, 1))
            else:
                layers.append(t.nn.Linear(hidden_size, hidden_size))
                layers.append(hidden_activation)

        self._layers = t.nn.Sequential(*layers)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self._layers(x)
