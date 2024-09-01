if __name__ != "__main__":
    raise ImportError("This script should not be imported. Run it directly.")
from networks import Critic, DeterministicPolicy
from replay_buffer import ReplayBuffer
from trainers.td3 import TD3Trainer, NetworkSet
import gymnasium as gym
import torch as t
from torch.optim.adam import Adam
from copy import deepcopy
import plotly.express as px
import dash
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import threading
import pandas as pd

# parameters

total_steps = 100_000
batch_size = 1024
discount = 0.99
target_update_rate = 0.005
target_policy_smoothing = 0.2
num_critics = 2
learning_rate = 1e-4
exploration_noise = 0.1
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# etc parameters
plot_interval = 100
plot_ma = 100
render_mode = None

# prepare
env = gym.make("Pendulum-v1", render_mode=render_mode)
env = gym.experimental.wrappers.StickyActionV0(env, 0.3)
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
    output_activation=None
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

# Create a Dash web application
app = Dash(__name__)

# Define the layout of the application
app.layout = html.Div(
    [
        dcc.Graph(id="real-time-plot"),
        dcc.Interval(
            id="interval-component",
            interval=1000,  # in milliseconds
            n_intervals=0,
        ),
    ]
)


rewards = 0
mas = []
ys = []


# Define callback to update the plot in real-time
@app.callback(
    Output("real-time-plot", "figure"), [Input("interval-component", "n_intervals")]
)
def update_graph(n):

    # Update the plot# Create a DataFrame for Plotly Express
    df = pd.DataFrame({"Index": ys, "Moving Average": mas})
    fig = px.line(
        df,
        x="Index",
        y=["Moving Average"],
        title="Reward and Moving Average Over Time",
    )

    return fig


def run_dash():
    app.run_server(debug=True, use_reloader=False)


threading.Thread(target=run_dash, daemon=True).start()

step = 0
running = True
while running:
    observation, info = env.reset()
    observation = t.tensor(observation, dtype=t.float32).to(device)
    terminated = False
    truncated = False

    while not terminated and not truncated:
        with t.no_grad():
            action = policy.learning(observation).cpu()

        action += t.randn_like(action) * exploration_noise
        next_observation, reward, terminated, truncated, _ = env.step(action.numpy())
        next_observation = t.tensor(next_observation, dtype=t.float32).to(device)
        replay_buffer.add(observation, action, reward, next_observation, terminated)
        observation = next_observation
        policy_loss, critic_loss = trainer.step()

        rewards = rewards * (1 - 1 / plot_ma) + reward / plot_ma  # type: ignore

        if step % plot_interval == 0:
            mas.append(rewards)
            ys.append(step)
        if step > total_steps:
            running = False
            break
        step += 1


env.close()
