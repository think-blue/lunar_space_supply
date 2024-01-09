"""
This script is used to visualise a single episode data saved during training. Please import the relevant experiment
class, and load the required csv data to visualise.
"""

# load the expriment class here
from earth_lunar_transfer.experiments.reference_exp.lunarenvironment import LunarEnvironment
import json
import pandas as pd

# load the config file for the experiment
with open("earth_lunar_transfer/configs/env_config_train.json", "rb") as config_file:
    env_config = json.load(config_file)

# load the csv dara for the episode
file_name = "140044978035472.csv"
data_path = "/nvme/lunar_space_supply/data/test_data"
simulation_save_path = "/nvme/lunar_space_supply/data/simulation_figures"

simulation_data = pd.read_csv(f"{data_path}/{file_name}", header=None)
lunar_env = LunarEnvironment(env_config)

save_file = file_name.split(sep=".")[0]
path = f"{simulation_save_path}/{save_file}.html"
lunar_env.animate(simulation_data[0],
                  position_history=simulation_data.loc[:, 1:],
                  source_planet=lunar_env.source_planet,
                  destination_planet=lunar_env.destination_planet,
                  path=path)
