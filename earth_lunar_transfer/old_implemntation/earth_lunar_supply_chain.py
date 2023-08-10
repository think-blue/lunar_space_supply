import pykep as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import torch
from torch import nn
from torch.nn import functional as F


def update_params(worker_opt, values, logprobs, rewards, G, clc=0.1, gamma=0.95):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    Returns = []
    ret_ = G  # for n step actor critic
    # ret_ = torch.Tensor([0])
    for r in range(rewards.shape[0]):  # B
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)
    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns, dim=0)
    actor_loss = -1 * logprobs * (Returns - values.detach())
    critic_loss = torch.pow(values - Returns, 2)
    loss = actor_loss.sum() + clc * critic_loss.sum()
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss, rewards


def run_episode(spacecraft, worker_model, params, delta_epoch, source, dest, batch, device=None):
    spacecraft_state = spacecraft.state
    state = torch.from_numpy(spacecraft_state).float().to(device) if device is not None else torch.from_numpy(
        spacecraft_state).float()
    values, logprobs, rewards = [], [], []
    reward_components_list = []
    done = False
    j = 0
    G = torch.Tensor([0])
    N_steps = params['n_time_steps']
    start_epoch = params['start_mjd2000_epoch']
    while j < N_steps and not done:
        j += 1
        policy, value = worker_model(state)
        values.append(value)
        action = policy.sample()
        logprob_ = policy.log_prob(action)
        logprobs.append(logprob_.mean())  # taking mean since there are three actions?
        reward, reward_components, _ = spacecraft.accelerate(thrust=action.detach().cpu().numpy(),
                                                             next_epoch=pk.epoch(
                                                                 spacecraft.epoch.mjd2000 + delta_epoch),
                                                             target_epoch=pk.epoch(start_epoch + params['num_days']))
        state = torch.from_numpy(spacecraft.state).float().to(device) if device is not None else torch.from_numpy(
            spacecraft.state).float()

        if len(spacecraft.epoch_history) >= N_steps:
            done = True
        elif np.linalg.norm(
                spacecraft.position) <= pk.AU * 0.25 or np.linalg.norm(spacecraft.position) >= pk.AU * 1.75:
            done = True
            print(f"epoch: {batch} :ended because spacecraft went to long distances")

        if done:
            spacecraft.reset_state(payload=params['mass'], fuel_mass=params['fuel_mass'],
                                   source=source, destination=dest,
                                   specific_impulse=params['specific_impulse'],
                                   epoch=pk.epoch(start_epoch))
        else:
            G = value.detach()

        rewards.append(reward)
        reward_components_list.append(reward_components)
    avg_reward_components = np.array(reward_components_list).mean(axis=0)
    return values, logprobs, rewards, G, avg_reward_components


class ActorCritic(nn.Module):
    def __init__(self, state_dim=14, n_actions=3):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(state_dim, 25)  # first layer
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, n_actions)  # action to take ... mean and std.
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)

        # adding fixed standard deviation as a parameter of the network
        logstds_param = nn.Parameter(torch.full((n_actions,), -0.6931))  # log(-0.6931) = 0.5
        self.register_parameter("logstds", logstds_param)

    def forward(self, x):
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        means = torch.tanh(self.actor_lin1(y))  # limiting output to -1 to 1 range
        stds = torch.clamp(self.logstds.exp(), 1e-3, 50)
        actor = torch.distributions.Normal(means, stds)
        # actor = F.log_softmax(self.actor_lin1(y), dim=0)

        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic