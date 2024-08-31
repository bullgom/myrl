import torch as t
from typing import SupportsFloat


class ReplayBuffer:

    def __init__(self, capacity: int, observation_dim: int, action_dim: int) -> None:
        self._capacity = capacity
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._observations = t.zeros((capacity, observation_dim))
        self._actions = t.zeros((capacity, action_dim))
        self._rewards = t.zeros((capacity, 1))
        self._next_observations = t.zeros((capacity, observation_dim))
        self._dones = t.zeros((capacity, 1))
        self._size = 0
        self._index = 0

    def add(
        self,
        observation: t.Tensor,
        action: t.Tensor,
        reward: SupportsFloat,
        next_observation: t.Tensor,
        done: bool,
    ) -> None:
        self._observations[self._index] = observation
        self._actions[self._index] = action
        self._rewards[self._index] = float(reward)
        self._next_observations[self._index] = next_observation
        self._dones[self._index] = done
        self._index = (self._index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
        indices = t.randint(0, self._size, (batch_size,))
        return (
            self._observations[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_observations[indices],
            self._dones[indices],
        )

    def __len__(self) -> int:
        return self._size
