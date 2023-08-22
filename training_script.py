import json
from earth_lunar_transfer.rl_lib_implementation.lunarenvironment import LunarEnvironment
from earth_lunar_transfer.agents.ddpg_agent import DDPGAgentGNN
from earth_lunar_transfer.agents.constants import AgentConstants
import mlflow

import os
import logging
import time
import numpy as np

# read config_files
with open("earth_lunar_transfer/env_config_train.json", "rb") as config_file:
    env_config = json.load(config_file)

class TrainingAgents:
    @staticmethod
    def get_agent(algorithm: str):
        agents = {
            'ddpg': DDPGAgentGNN
        }

        return agents.get(algorithm)

exp_description = "state: state with fixed time period of 3 days\n" \
                  "reward: reward based on position, time penalty, and discounted reward\n" \
                  "environment: continuous action space\n" \
                  "algorithm: A2C"

agent_params = {k.title():v for k, v in AgentConstants.__dict__.items()
                                if not k.startswith("__")}

run_id = os.getenv('RUNID', 0)
folder = os.getenv('OUTPUT', "./model/")
tb_log_path = f"{env_config['data_path']}/tb_logs"
if not os.path.exists(tb_log_path):
    os.mkdir(tb_log_path)

path_prefix = f"{folder}/run{run_id}_"
log_filename = path_prefix+"training_log.log"
agent_config = env_config["agent_config"]
TRAINING_AGENT = TrainingAgents.get_agent(agent_config["training_algorithm"])

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
    env_config["agent_config"],
    tb_log_path,
    weights_path=path_prefix+"weights_intermediate",
    complex_path=path_prefix+"intermediate"
    )

if __name__ == '__main__':
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")
    mlflow.set_experiment(experiment_name=env_config["exp_name"])
    with mlflow.start_run(description=exp_description) as current_run:
        run_id = current_run.info.run_id
        env_config["agent_params"] = agent_params
        env_config["mlflow_run_id"] = run_id
        mlflow.log_params(env_config)
        
        actions = agent.play(agent_config["n_episodes"], run_id)
        agent.save_weights(path_prefix+"weights", "final")
        time.sleep(1)
