"""
script to be used for tuning the hyperparameters of the model.
"""

from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
import json
from ray import air, tune

# import sys
# sys.path.append("/nvme/lunar_space_supply")
# sys.path.append("/nvme/lunar_space_supply/earth_lunar_transfer")
# from earth_lunar_transfer.reference_exp.lunarenvironment import LunarEnvironment
from earth_lunar_transfer.experiments.exp_time_step.exp_no_gravity.lunarenvironment import LunarEnvNoGravity

import mlflow
from datetime import datetime

# read config_files
with open("../configs/env_config_train.json", "rb") as config_file:
    env_config = json.load(config_file)

exp_description = "state: state with fixed time period of 3 days\n" \
                  "reward: reward based on position, time penalty, and discounted reward\n" \
                  "environment: continuous action space\n" \
                  "algorithm: A2C"

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment(experiment_name=env_config["exp_name"])
with mlflow.start_run(description=exp_description) as current_run:
    run_id = current_run.info.run_id
    env_config["mlflow_run_id"] = run_id
    mlflow.log_params(env_config)

    # model_config = dict(fcnet_hiddens=[256, 256, 256], fcnet_activation="relu")
    a2c_config = (
        PPOConfig()
        .environment(env=LunarEnvNoGravity, env_config=env_config)
        .training(grad_clip=3, lr=tune.grid_search([1e-9, 1e-8, 1e-6]),
                  gamma=tune.grid_search([1, 0.99, 0.90]))
        .rollouts(num_rollout_workers=5, num_envs_per_worker=5)
        .evaluation(evaluation_num_workers=1)
        .framework("torch")
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=a2c_config,
        run_config=air.RunConfig(
            stop={"episode_reward_mean": 100},
            checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True)
        )
    )

    results = tuner.fit()

    # Get the best result based on a particular metric.
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")

    # Get the best checkpoint corresponding to the best result.
    best_checkpoint = best_result.checkpoint
