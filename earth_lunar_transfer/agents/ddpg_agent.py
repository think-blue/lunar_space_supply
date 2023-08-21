# refered from https://github.com/germain-hug/Deep-RL-Keras
# Source DDPG: https://raw.githubusercontent.com/ghliu/pytorch-ddpg/master/ddpg.py
# Source DDPG Training: https://raw.githubusercontent.com/ghliu/pytorch-ddpg/master/main.py

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from earth_lunar_transfer.network.actor import ActorNN
from earth_lunar_transfer.network.critic import CriticNN
from .constants import AgentConstants
from earth_lunar_transfer.config import Config

from .metrices import Metric
from .memory import Memory
from .utils import soft_update, hard_update, to_tensor, to_numpy, batchify, \
    LONG, INT32, USE_CUDA, use_device
from .noise import OrnsteinUhlenbeckActionNoise
import logging

criterion = nn.MSELoss()


class DDPGAgentGNN:
    ACTOR_LEARNING_RATE = AgentConstants.ACTOR_LEARNING_RATE
    CRITIQ_LEARNING_RATE = AgentConstants.CRITIQ_LEARNING_RATE
    TAU = AgentConstants.TAU

    GAMMA = AgentConstants.GAMMA

    BATCH_SIZE = AgentConstants.BATCH_SIZE
    BUFFER_SIZE = AgentConstants.BUFFER_SIZE
    EXPLORATION_EPISODES = AgentConstants.EXPLORATION_EPISODES

    def __init__(
            self, env, weights_path, complex_path=None,
            warmup=32, prate=0.00005, is_training=1
    ):
        config = Config.get_instance()
        self.metrices = Metric.get_instance()
        self.input_shape = env.input_shape
        self.action_shape = env.action_space.n_outputs
        self.eps = 1.0
        self.decay_epsilon = 1/50000
        self.action_bounds = env.action_space.action_bounds
        self.memory = Memory(self.BUFFER_SIZE)
        self.env = env

        self.noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(self.action_shape),
            dimension=self.action_shape,
            num_steps=config.n_episodes
        )

        self.warm_up_steps = warmup
        self.is_training = is_training

        self._actor = ActorNN(
            self.input_shape,
            self.action_shape,
            self.ACTOR_LEARNING_RATE,
            self.TAU
        )
        self._actor_target = ActorNN(
            self.input_shape,
            self.action_shape,
            self.ACTOR_LEARNING_RATE,
            self.TAU
        )
        self._actor_optim = Adam(self._actor.parameters(), lr=prate)

        self._critiq = CriticNN(
            self.input_shape,
            self.action_shape,
            self.CRITIQ_LEARNING_RATE,
            self.TAU
        )
        self._critiq_target = CriticNN(
            self.input_shape,
            self.action_shape,
            self.CRITIQ_LEARNING_RATE,
            self.TAU
        )
        self._critiq_optim = Adam(
            self._critiq.parameters(),
            lr=prate,
            weight_decay=1e-2
        )

        if USE_CUDA:
            self._actor.cuda()
            self._actor_target.cuda()
            self._critiq.cuda()
            self._critiq_target.cuda()

        hard_update(self._actor_target, self._actor)  # Make sure target is with the same weight
        hard_update(self._critiq_target, self._critiq)

        self.weights_path = weights_path
        self.complex_path = complex_path

    def memorize(self, state, action, reward, next_state, done):
        self.memory.add_sample({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

    def compute_q_loss(self, state, action, reward, next_state, terminal):
        q_value = self._critiq([state, action])
        with torch.no_grad():
            q_target_value = self._critiq_target([
                next_state,
                self._actor_target(next_state)
            ])

            expected_q = reward + self.GAMMA * (1 - terminal) * q_target_value
        return criterion(q_value, expected_q)

    def compute_policy_loss(self, state):
        return -self._critiq([
            state, self._actor(state)
        ])

    def batch_update(self, batches, loss_function):
        training_loss = None
        
        for batch in batches:
            complex_batched = batchify(batch.states)
            next_complex_batched = batchify(batch.next_states)

            actions = to_tensor(batch.actions, dtype=LONG)
            rewards = to_tensor(batch.rewards)
            terminals = to_tensor(batch.terminals, dtype=INT32)
            complex_batched = use_device(complex_batched)
            next_complex_batched = use_device(next_complex_batched)

            loss = loss_function(
                complex_batched,
                actions,
                rewards,
                next_complex_batched,
                terminals
            )
            
            training_loss = loss if training_loss is None else training_loss + loss
            complex_batched.cpu()
            next_complex_batched.cpu()

        return training_loss / len(batches)

    def update_network(self):
        batches = self.memory.sample(self.BATCH_SIZE)

        self._critiq_optim.zero_grad()
        mean_q_loss = self.batch_update(batches, self.compute_q_loss)
        mean_q_loss.backward()
        self._critiq_optim.step()

        self._actor_optim.zero_grad()
        mean_policy_loss = self.batch_update(
            batches,
            lambda state, action, reward, next_state, done: self.compute_policy_loss(state)
        )
        mean_policy_loss.backward()
        self._actor_optim.step()

        # Target update
        soft_update(self._actor_target, self._actor, self.TAU)
        soft_update(self._critiq_target, self._critiq, self.TAU)

        print("Value Loss: ", mean_q_loss, " policy loss: ", mean_policy_loss)
        return to_numpy(mean_q_loss), to_numpy(mean_policy_loss)

    def get_predicted_action(self, data, step=None, decay_epsilon=True):
        # Explore AdaptiveParamNoiseSpec, with normalized action space
        # https://github.com/l5shi/Multi-DDPG-with-parameter-noise/blob/master/Multi_DDPG_with_parameter_noise.ipynb
        self._actor.eval()
        action = to_numpy(
            self._actor(data)[0]
        )
        
        self._actor.train()

        if step is not None:
            action += self.noise.step()

        return action

    def get_action(self, action):
        action *= self.action_bounds[1]  # Remove as the multiplier has been removed
        return np.clip(action, *self.action_bounds)

    def play(self, num_train_episodes):
        config = Config.get_instance()
        returns = []
        num_steps = 0
        max_episode_length = config.episode_length
        max_reward = 0
        i_episode = 0

        while i_episode < num_train_episodes:
            state, _ = self.env.reset()
            critic_losses = []
            actor_losses = []

            molecule_index = 0
            self.metrices.init_score(
                i_episode, f'run_{i_episode}',
                0,
                [0, 0, 0]
            )

            episode_return, episode_length, d_store = 0, 0, False

            while not d_store:
                
                data = torch.tensor(state)
                data = use_device(data)
                predicted_action = self.get_predicted_action(
                    data, episode_length
                )
                data.cpu()
                action = self.get_action(predicted_action)

                # reward, terminal = self.env.step(action)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # self.metrices.cache_score(i_episode, self.env.reward.score, molecule_index)
                self.metrices.cache_action_reward(
                    i_episode, action, reward, molecule_index
                )
                # self.metrices.cache_divergence(i_episode, terminal, molecule_index)

                d_store = terminated or truncated
                # reward = 0 if episode_length == max_episode_length else reward
                self.memorize(data, [predicted_action], reward, torch.tensor(next_state), d_store)

                episode_return += reward
                episode_length += 1

                self.log(action, np.round(reward, 4), episode_length, i_episode)

                if num_steps > self.warm_up_steps:
                    critic_loss, actor_loss = self.update_network()
                    self.metrices.cache_loss(
                        i_episode, actor_loss, critic_loss
                    )

                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)

                num_steps += 1

            returns.append([
                i_episode + 1,
                episode_length,
                episode_return,
                np.mean(critic_losses),
                np.mean(actor_losses)
            ])

            max_reward = max_reward if max_reward > episode_return else episode_return

            print(
                f"Episode: {i_episode + 1} \
                Return: {episode_return} \
                episode_length: {episode_length} \
                Max Reward; {max_reward} \
                Critic Loss: {np.mean(critic_losses)} \
                Actor loss: {np.mean(actor_losses)}"
            )

            logging.info(
                f"Episode: {i_episode + 1} \
                Return: {episode_return} \
                episode_length: {episode_length} \
                Max Reward; {max_reward} \
                Critic Loss: {np.mean(critic_losses)} \
                Actor loss: {np.mean(actor_losses)}"
            )

            if i_episode >= config.save_model_from and i_episode % 10 == 0:
                self.save_weights(self.weights_path, i_episode)
                self.metrices.plot_score_trend(i_episode, self.complex_path)
                Metric.save(self.metrices, self.complex_path)

            i_episode += 1

        return returns

    def log(self, action, reward, episode_length, i_episode):
        print(
            "Action:", np.round(np.array(action), 4),
            "Reward:", np.round(reward, 4),
            "E_i:", episode_length,
            self.env.reward,
            "E:", i_episode
        )
        logging.info(
            f"Action: {np.round(np.array(action), 4)},\
              Reward: {np.round(reward, 4)}, \
              E_i: {episode_length}, E: {i_episode}, \
              {self.env.reward}"
        )

    def save_weights(self, path, episode):
        torch.save(self._actor.state_dict(), f'{path}_{episode}_actor')
        torch.save(self._critiq.state_dict(), f'{path}_{episode}_critic')

    def load_weights(self, path_actor, path_critic):
        if USE_CUDA:
            self._actor.load_state_dict(torch.load(path_actor))
            self._critiq.load_state_dict(torch.load(path_critic))
        else:
            self._actor.load_state_dict(torch.load(
                path_actor, map_location=torch.device('cpu')
            ))
            self._critiq.load_state_dict(torch.load(
                path_critic, map_location=torch.device('cpu')
            ))
