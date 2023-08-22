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

from .memory import Memory
from .utils import soft_update, hard_update, to_tensor, to_numpy, \
    LONG, INT32, USE_CUDA, use_device
from .noise import OrnsteinUhlenbeckActionNoise
import logging
import mlflow
from tensorboardX import SummaryWriter

criterion = nn.MSELoss()

class DDPGAgentGNN:
    ACTOR_LEARNING_RATE = AgentConstants.ACTOR_LEARNING_RATE
    CRITIQ_LEARNING_RATE = AgentConstants.CRITIQ_LEARNING_RATE
    TAU = AgentConstants.TAU

    GAMMA = AgentConstants.GAMMA

    BATCH_SIZE = AgentConstants.BATCH_SIZE
    BUFFER_SIZE = AgentConstants.BUFFER_SIZE
    # EXPLORATION_EPISODES = AgentConstants.EXPLORATION_EPISODES

    def __init__(
            self, env, agent_config, tb_log_dir, weights_path,
            complex_path=None, warmup=32, prate=0.00005, is_training=1
    ):
        self.config = agent_config
        self.input_shape = env.input_shape
        self.action_shape = env.action_space.n_outputs
        self.eps = 1.0
        self.decay_epsilon = 1/50000
        self.action_bounds = env.action_space.action_bounds
        self.memory = Memory(self.BUFFER_SIZE)
        self.tb_writer = SummaryWriter(tb_log_dir)
        self.env = env

        self.noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(self.action_shape),
            dimension=self.action_shape,
            num_steps=self.config["n_episodes"]
        )

        self.warm_up_steps = max(warmup, self.BATCH_SIZE)
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
            complex_batched = batch.states[0]
            next_complex_batched = batch.next_states[0]

            actions = to_tensor(batch.actions)
            rewards = to_tensor(batch.rewards)
            terminals = to_tensor(batch.terminals, dtype=INT32)
            actions = use_device(actions)
            complex_batched = use_device(complex_batched)
            next_complex_batched = use_device(next_complex_batched)
            # import pdb; pdb.set_trace()
            loss = loss_function(
                complex_batched,
                actions,
                rewards,
                next_complex_batched,
                terminals
            )
            
            training_loss = loss if training_loss is None else training_loss + loss
            actions.cpu()
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

        print("Critic Loss: ", mean_q_loss, " Actor loss: ", mean_policy_loss)
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

    def play(self, num_train_episodes, mlflow_run_id=None):
        with mlflow.start_run(run_id=mlflow_run_id, nested=True) as child_run:
            returns = []
            num_steps = 0
            max_reward = 0
            i_episode = 0

            while i_episode < num_train_episodes:
                state, _ = self.env.reset()
                critic_losses = []
                actor_losses = []

                episode_return, episode_length, d_store = [], 0, False

                while not d_store:
                    data = torch.tensor(state.astype(np.float32).reshape(1,self.input_shape))
                    data = use_device(data)
                    predicted_action = self.get_predicted_action(
                        data, episode_length
                    )
                    data.cpu()
                    action = self.get_action(predicted_action.copy())
                    # import pdb; pdb.set_trace()

                    # reward, terminal = self.env.step(action)
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    mlflow.log_metric("step_reward", reward, num_steps)

                    d_store = terminated or truncated
                    # reward = 0 if episode_length == max_episode_length else reward
                    next_data = torch.tensor(next_state.astype(np.float32).reshape(1,self.input_shape))
                    self.memorize(data, [action], reward, next_data, d_store)

                    state = next_state.copy()
                    episode_return.append(reward)
                    episode_length += 1

                    self.log(action, np.round(reward, 4), episode_length, i_episode)

                    if num_steps > self.warm_up_steps:
                        critic_loss, actor_loss = self.update_network()
                        self.log_gradients_in_model(self._critiq, num_steps)
                        self.log_gradients_in_model(self._critiq_target, num_steps)
                        self.log_gradients_in_model(self._actor, num_steps)
                        self.log_gradients_in_model(self._actor_target, num_steps)
                        
                        mlflow.log_metric("step_critic_loss", critic_loss, num_steps)
                        mlflow.log_metric("step_actor_loss", actor_loss, num_steps)

                        critic_losses.append(critic_loss)
                        actor_losses.append(actor_loss)

                    num_steps += 1

                mlflow.log_metric("episode_reward", sum(episode_return), i_episode)
                mlflow.log_metric("episode_average_reward", np.mean(episode_return), i_episode)
                mlflow.log_metric("episode_min_reward", min(episode_return), i_episode)
                mlflow.log_metric("episode_max_reward", max(episode_return), i_episode)
                mlflow.log_metric("episode_average_critic_loss", critic_loss, num_steps)
                mlflow.log_metric("episode_average_actor_loss", actor_loss, num_steps)
                
                returns.append([
                    i_episode + 1,
                    episode_length,
                    sum(episode_return),
                    np.mean(critic_losses),
                    np.mean(actor_losses)
                ])

                max_reward = max_reward if max_reward > sum(episode_return) else sum(episode_return)

                print(
                    f"Episode: {i_episode + 1} \
                    Return: {sum(episode_return)} \
                    episode_length: {episode_length} \
                    Max Reward; {max_reward} \
                    Critic Loss: {np.mean(critic_losses)} \
                    Actor loss: {np.mean(actor_losses)}"
                )

                logging.info(
                    f"Episode: {i_episode + 1} \
                    Return: {sum(episode_return)} \
                    episode_length: {episode_length} \
                    Max Reward; {max_reward} \
                    Critic Loss: {np.mean(critic_losses)} \
                    Actor loss: {np.mean(actor_losses)}"
                )

                if i_episode >= self.config["save_model_from"] and i_episode % 10 == 0:
                    self.save_weights(self.weights_path, i_episode)

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

    def log_gradients_in_model(self, model, step):
        for tag, value in model.named_parameters():
            if value.grad is not None:
                self.tb_writer.add_histogram(tag + "/grad", value.grad.cpu(), step)
    
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
