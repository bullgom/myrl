from ..trainer import Trainer
from .types import NetworkSet
from networks import Critic, DeterministicPolicy
import torch as t
from replay_buffer import ReplayBuffer
from .functional import update_policy, update_critic, update_target_network
from torch.optim.adam import Adam


class TD3Trainer(Trainer[None, tuple[float, float]]):

    def __init__(
        self,
        policy: NetworkSet[DeterministicPolicy, Adam],
        critics: list[NetworkSet[Critic, Adam]],
        relply_buffer: ReplayBuffer,
        device: t.device,
        batch_size: int,
        discount: float,
        target_update_rate: float,
        target_policy_smoothing_rate: float,
    ) -> None:
        self._policy = policy
        self._critics = critics

        self._replay_buffer = relply_buffer
        self._device = device
        self._batch_size = batch_size
        self._discount = discount
        self._target_update_rate = target_update_rate
        self._target_policy_smoothing = target_policy_smoothing_rate

    def step(self) -> tuple[float, float]:
        batch = self._replay_buffer.sample(self._batch_size)

        o, a, r, no, te = tuple(data.to(self._device) for data in batch)

        policy_loss = update_policy(
            self._policy.optimizer, self._policy.learning, self._critics[0].learning, o
        )
        critic_loss = update_critic(
            self._critics,
            self._policy.target,
            o,
            a,
            r,
            te,
            no,
            self._discount,
            self._target_policy_smoothing,
        )

        for network_set in (self._policy, *self._critics):
            update_target_network(
                network_set.target, network_set.learning, self._target_update_rate
            )

        return policy_loss, critic_loss
