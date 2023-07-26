from lunarenvironment import LunarEnvironment
import json
import pandas as pd

with open("env_config.json", "rb") as config_file:
    env_config = json.load(config_file)

file_name = "140346085163664.csv"
simulation_data = pd.read_csv(f"./training_data/{file_name}", header=None)
lunar_env = LunarEnvironment(env_config)

save_file = file_name.split(sep=".")[0]
path = "./simulation_figures/" + save_file + ".html"
lunar_env.simulate(simulation_data[0],
                   position_history=simulation_data.loc[:, 1:],
                   source_planet=lunar_env.source_planet,
                   destination_planet=lunar_env.destination_planet,
                   path=path)
