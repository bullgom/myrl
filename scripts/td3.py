if __name__ != "__main__":
    raise ImportError("This script should not be imported. Run it directly.")
from networks import Critic, DeterministicPolicy
from replay_buffer import ReplayBuffer
from trainers.td3 import TD3Trainer, NetworkSet
import gymnasium as gym
import torch as t
from torch.optim.adam import Adam
from copy import deepcopy

# parameters

total_steps = 100_000
batch_size = 128
discount = 0.99
target_update_rate = 0.005
target_policy_smoothing = 0.2
num_critics = 2
learning_rate = 1e-4
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# prepare
env = gym.make("MountainCarContinuous-v0", render_mode="human")
assert env.observation_space.shape is not None
assert env.action_space.shape is not None

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Running on device:", device)

_policy = DeterministicPolicy(
    env.observation_space.shape[0],
    env.action_space.shape[0],
    hidden_size=128,
    num_layers=2,
    hidden_activation=t.nn.LeakyReLU(),
    output_activation=t.nn.Tanh(),
).to(device)
_target_policy = deepcopy(_policy).to(device)
policy = NetworkSet(
    _policy, _target_policy, Adam(_policy.parameters(), lr=learning_rate)
)

_critic = Critic(
    env.observation_space.shape[0],
    env.action_space.shape[0],
    hidden_size=128,
    num_layers=2,
    hidden_activation=t.nn.LeakyReLU(),
)
critics: list[NetworkSet[Critic, Adam]] = []
for i in range(num_critics):
    _critic = deepcopy(_critic).to(device)
    _target_critic = deepcopy(_critic).to(device)
    critics.append(
        NetworkSet(
            _critic, _target_critic, Adam(_critic.parameters(), lr=learning_rate)
        )
    )

replay_buffer = ReplayBuffer(
    1_000_000, env.observation_space.shape[0], env.action_space.shape[0]
)

trainer = TD3Trainer(
    policy,
    critics,
    replay_buffer,
    device,
    batch_size,
    discount,
    target_update_rate,
    target_policy_smoothing,
)

for step in range(total_steps):
    observation, info = env.reset()
    observation = t.tensor(observation, dtype=t.float32).to(device)
    terminated = False

    while not terminated:
        with t.no_grad():
            action = policy.learning(observation).cpu()
        next_observation, reward, terminated, _, _ = env.step(action.numpy())
        next_observation = t.tensor(next_observation, dtype=t.float32).to(device)
        replay_buffer.add(observation, action, reward, next_observation, terminated)
        observation = next_observation

        policy_loss, critic_loss = trainer.step()

env.close()
