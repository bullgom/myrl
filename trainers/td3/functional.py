import torch as t
from networks import DeterministicPolicy, Critic
from .types import NetworkSet
from typing_extensions import Sequence
from torch.optim.optimizer import Optimizer


def get_policy_loss(
    observation: t.Tensor, policy: DeterministicPolicy, target_critic: Critic
) -> t.Tensor:
    target_critic.eval()

    action = policy(observation)
    return -target_critic(t.cat([observation, action], dim=-1)).mean()


def update_policy(
    policy_optimizer: Optimizer,
    policy: DeterministicPolicy,
    critic: Critic,
    observation: t.Tensor,
) -> float:
    policy.train()
    
    policy_optimizer.zero_grad()
    loss = get_policy_loss(observation, policy, critic)
    loss.backward()
    policy_optimizer.step()
    return loss.item()


@t.no_grad()
def get_critic_target(
    critics: Sequence[NetworkSet[Critic, Optimizer]],
    policy: DeterministicPolicy,
    next_observation: t.Tensor,
    r: t.Tensor,
    terminated: t.Tensor,
    gamma: float,
    target_policy_smoothing: float = 0.2,
) -> t.Tensor:
    policy.eval()

    next_action = policy(next_observation)
    next_action += t.randn_like(next_action) * target_policy_smoothing

    qs = []
    for critic in critics:
        critic.target.eval()
        qs.append(critic.target(t.cat([next_observation, next_action], dim=-1)))

    target = r + gamma * (1 - terminated) * t.stack(qs).min(dim=0).values
    return target


def update_critic(
    critics: Sequence[NetworkSet[Critic, Optimizer]],
    target_policy: DeterministicPolicy,
    observation: t.Tensor,
    action: t.Tensor,
    reward: t.Tensor,
    terminated: t.Tensor,
    next_observation: t.Tensor,
    gamma: float,
    target_policy_smoothing: float,
) -> float:
    critic_target = get_critic_target(
        critics,
        target_policy,
        next_observation,
        reward,
        terminated,
        gamma,
        target_policy_smoothing,
    )

    sum = 0.0
    for critic in critics:
        critic.learning.train()
        critic.optimizer.zero_grad()
        loss = t.nn.functional.mse_loss(
            critic.learning(t.cat([observation, action], dim=-1)), critic_target
        )
        loss.backward()
        critic.optimizer.step()
        sum += loss.item()

    return sum / len(critics)


def polyak_average(target: t.nn.Parameter, source: t.Tensor, tau: float = 0.05) -> None:
    target.data = tau * source.data + (1 - tau) * target.data


def update_target_network(
    target: t.nn.Module, source: t.nn.Module, tau: float = 0.05
) -> None:
    """Update target network using polyak averaging.

    Args:
        target (t.nn.Module): Target network.
        source (t.nn.Module): Source network.
        tau (float, optional): Polyak averaging coefficient. Defaults to 0.05.
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        polyak_average(target_param, source_param.data, tau)
