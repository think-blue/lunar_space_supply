import json
from earth_lunar_transfer.rl_lib_implementation.lunarenvironment import LunarEnvironment
from earth_lunar_transfer.agents.ddpg_agent import DDPGAgentGNN
from earth_lunar_transfer.agents.metrices import Metric
from earth_lunar_transfer.config import Config

import os
import logging
import time
import numpy as np

# read config_files
with open("env_config_train.json", "rb") as config_file:
    env_config = json.load(config_file)

class TrainingAgents:
    @staticmethod
    def get_agent(algorithm: str):
        agents = {
            'ddpg': DDPGAgentGNN
        }

        return agents.get(algorithm)


run_id = os.getenv('RUNID', 0)
folder = os.getenv('OUTPUT', "./model/")

config_path = os.getenv('CONFIG', './config.json')
Config.init(config_path)
config = Config.get_instance()
metric = Metric.get_instance()

path_prefix = f"{folder}/run{run_id}_"
log_filename = path_prefix+"training_log.log"

TRAINING_AGENT = TrainingAgents.get_agent(config.training_algorithm)

logging.basicConfig(
            filename=log_filename,
            filemode='w',
            format='%(message)s',
            datefmt='%I:%M:%S %p',
            level=logging.DEBUG
        )

env = LunarEnvironment(env_config=env_config)
agent = TRAINING_AGENT(
    env,
    weights_path=path_prefix+"weights_intermediate",
    complex_path=path_prefix+"intermediate"
)


if __name__ == '__main__':
    actions = agent.play(config.n_episodes)
    agent.save_weights(path_prefix+"weights", "final")

    metric.plot_loss_trend(path_prefix)
    Metric.save(metric, f'{path_prefix}_final')
    env.gnn_explainer.end()
    time.sleep(1)
