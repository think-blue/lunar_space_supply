"""
To be run on local environment
"""
import pykep as pk
import torch

from Spacecraft import Spacecraft
from earth_lunar_transfer.earth_lunar_old_implemntation.common_module import run_spacecraft
from earth_lunar_transfer.earth_lunar_old_implemntation.rl_agent import ActorCritic
from earth_lunar_transfer.earth_lunar_old_implemntation.visualise import simulate_spacecraft
import json

if __name__ == "__main__":
    # config options
    # mpl.use('TkAgg')
    with open("params.json", "rb") as config_file:
        params = json.load(config_file)["gpu"]

    pk.util.load_spice_kernel("../../kernels/de441.bsp")
    earth = pk.planet.spice("earth")
    moon = pk.planet.spice('moon')

    delta_time_step = params['num_days'] / params['n_time_steps']

    processes = []
    MasterNode = ActorCritic()
    MasterNode.load_state_dict(torch.load("../saved_models/model_gpu.pt"))
    MasterNode.eval()

    spacecraft = Spacecraft(payload=params['mass'], fuel_mass=params['fuel_mass'], source=earth,
                            epoch=pk.epoch(params['start_mjd2000_epoch']),
                            destination=moon, specific_impulse=params['specific_impulse'])
    states = run_spacecraft(spacecraft=spacecraft, time_delta=delta_time_step, num_time_steps=params['n_time_steps'],
                            guidance_model=MasterNode)
    simulate_spacecraft(states, spacecraft)
    pass
