import mlflow
import numpy as np
import pykep as pk
import torch
from torch import optim

from earth_lunar_transfer.old_implemntation.Spacecraft import Spacecraft
from archived_code.old_implemntation.rl_agent import run_episode, update_params


def create_planets(planet_names=('mercury', 'venus', 'earth', 'mars', 'jupiter')):
    return np.array([pk.planet.spice(planet) for planet in planet_names])


def create_position_data(jpl_planets, epoch_list, states):
    loc_array = np.empty((len(epoch_list), len(jpl_planets) + 1, 3))
    vel_array = np.empty((len(epoch_list), len(jpl_planets) + 1, 3))
    for e_count, epoch in enumerate(epoch_list):
        for p_count, planet in enumerate(jpl_planets):
            location, velocity = planet.eph(epoch)
            loc_array[e_count, p_count, :] = location
            vel_array[e_count, p_count, :] = velocity

    loc_array[:, -1, :] = states[:, :3]
    vel_array[:, -1, :] = states[:, 3:6]
    return loc_array, vel_array


def worker_process(t, worker_model, counter, params, delta_epoch, source, dest, training_results):
    spacecraft = Spacecraft(payload=params['mass'], fuel_mass=params['fuel_mass'], source=source, destination=dest,
                            specific_impulse=params['specific_impulse'],
                            epoch=pk.epoch(params['start_mjd2000_epoch']))
    spacecraft_opt = optim.Adam(lr=1e-4, params=worker_model.parameters())
    spacecraft_opt.zero_grad()
    for i in range(params['epochs']):
        spacecraft_opt.zero_grad()
        values, logprobs, rewards, g, rewards_components = run_episode(spacecraft=spacecraft, worker_model=worker_model,
                                                                       params=params,
                                                                       delta_epoch=delta_epoch,
                                                                       source=source,
                                                                       dest=dest,
                                                                       batch=counter.value)
        actor_loss, critic_loss, eplen = update_params(spacecraft_opt, values, logprobs, rewards, g)
        # not capturing training results separately
        mlflow.log_metric(key="mean_batch_actor_loss", value=actor_loss.mean().detach().numpy().item(),
                          step=counter.value)
        mlflow.log_metric(key="mean_batch_critic_loss", value=critic_loss.mean().detach().numpy().item(),
                          step=counter.value)
        mlflow.log_metric(key="mean_batch_returns", value=eplen.mean().detach().numpy().item(), step=counter.value)
        mlflow.log_metrics(dict(actor_loss=actor_loss.mean(),
                                critic_loss=critic_loss.mean(),
                                reward=eplen.mean(),
                                reward_0=rewards_components[0],
                                reward_1=rewards_components[1],
                                reward_2=rewards_components[2],
                                reward_3=rewards_components[3],
                                reward_4=rewards_components[4],
                                reward_5=rewards_components[5]))
        counter.value = counter.value + 1
        if counter.value > 0 and counter.value % 200 == 0:
            print(f" trained on {counter.value} batches")
    return training_results


def create_epoch_list(start_epoch, end_epoch, n_steps):
    epoch_range = np.linspace(start_epoch, end_epoch, n_steps)
    epoch_list = [pk.epoch(epoch_num) for epoch_num in epoch_range]
    return epoch_list


def run_spacecraft(spacecraft, time_delta, num_time_steps, guidance_model):
    guidance_model.eval()
    states = np.zeros((num_time_steps, len(spacecraft.state)))
    for time_step in range(num_time_steps):
        state = torch.from_numpy(spacecraft.state).float()
        states[time_step] = state
        # todo: verify if it is the best way to take action in eval mode (mean value as reward - deterministic model)
        with torch.no_grad():
            action = guidance_model(state)[0].loc
        spacecraft.accelerate(thrust=action, next_epoch=pk.epoch(spacecraft.epoch.mjd2000 + time_delta))
    return states
