import os

print(os.environ['PYTHONPATH'])

import sys

print(sys.path)

from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig

import json

import sys
# sys.path.append("/nvme/lunar_space_supply")
# sys.path.append("/nvme/lunar_space_supply/earth_lunar_transfer")
# from earth_lunar_transfer.reference_exp.lunarenvironment import LunarEnvironment
# # from earth_lunar_transfer.exp_time_step.exp_position.lunarenvironment import LunarEnvPosition
# from earth_lunar_transfer.exp_time_step.exp_no_gravity.lunarenvironment_negative import LunarEnvNoGravity
# from earth_lunar_transfer.exp_time_step.exp_gravity_states.lunarenvironment_direction_based import LunarEnvForceHelper
from earth_lunar_transfer.exp_time_step.exp_gravity_states.lunarenvironment_direction_based_position import LunarEnvForceHelper

import mlflow
from datetime import datetime

# read config_files
with open("env_config_train.json", "rb") as config_file:
    env_config = json.load(config_file)

exp_description = """
        - reward based on velocity mag.
        - reward based on position direction and mag
        - previous position history is defined
        - reward is based on mag. change error"""

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment(experiment_name=env_config["exp_name"])
with mlflow.start_run(description=exp_description) as current_run:
    run_id = current_run.info.run_id
    env_config["mlflow_run_id"] = run_id
    mlflow.log_params(env_config)

    # model_config = dict(fcnet_hiddens=[128, 128, 128], fcnet_activation="relu")
    # a2c_config = (
    #     # A2CConfig()
    #     PPOConfig()
    #     .environment(env=LunarEnvForceHelper, env_config=env_config)
    #     .training(grad_clip=3, lr_schedule=[[0, 1e-6], #0.00002 this works
    #                                         [2e6, 0.00000006],
    #                                         [20000000, 0.000000000001]],
    #               gamma=0.9)
    #     .rollouts(num_rollout_workers=5, num_envs_per_worker=5)
    #     .evaluation(evaluation_num_workers=1)
    #     .framework("torch")
    #     .debugging(log_level="INFO")
    #     .offline_data(output="/nvme/lunar_space_supply/data/training_data/logs")
    # )
    agent_params = env_config["agent_params"]
    a2c_config = (
        # A2CConfig()
        PPOConfig()
        .environment(env=LunarEnvForceHelper, env_config=env_config)
        .training(grad_clip=agent_params["grad_clip"],
                  lr=agent_params["lr"],
                  gamma=agent_params["gamma"],
                  vf_loss_coeff=agent_params["vf_loss_coeff"],
                  clip_param=agent_params["clip_param"],
                  entropy_coeff=agent_params["entropy_coeff"],
                  lambda_=agent_params["lambda_"],
                  vf_clip_param=agent_params["vf_clip_param"]
                  )
        .rollouts(num_rollout_workers=5, num_envs_per_worker=5)
        .evaluation(evaluation_num_workers=1)
        .framework("torch")
        .debugging(log_level="INFO")
        # .offline_data(output="/nvme/lunar_space_supply/data/training_data/logs")
    )

    a2c_algo = a2c_config.build()

    print("training")
    for iteration in range(8005):
        train_results = a2c_algo.train()
        # print(train_results)
        mlflow.log_metric("episode_reward_max", train_results["episode_reward_max"], iteration)
        mlflow.log_metric("episode_reward_min", train_results["episode_reward_min"], iteration)
        mlflow.log_metric("episode_reward_mean", train_results["episode_reward_mean"], iteration)
        mlflow.log_metric("episodes_this_iter", train_results["episodes_this_iter"], iteration)
        mlflow.log_metric("episode_len_mean", train_results['episode_len_mean'], iteration)
        mlflow.log_metric("policy_loss",
                          train_results['info']['learner']['default_policy']['learner_stats']['policy_loss'], iteration)
        mlflow.log_metric("vf_loss", train_results['info']['learner']['default_policy']['learner_stats']['vf_loss'],
                          iteration)
        mlflow.log_metric("vf_explained_var", train_results['info']['learner']['default_policy']['learner_stats']['vf_explained_var'],
                          iteration)
        mlflow.log_metric("entropy", train_results['info']['learner']['default_policy']['learner_stats']['entropy'], iteration)
        mlflow.log_metric("mean_kl_loss", train_results['info']['learner']['default_policy']['learner_stats']['kl'], iteration)

        if iteration % 50 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f"{current_time}: trained {iteration + 1} epochs")
        if iteration % 100 == 0:
            path = a2c_algo.save()
            print(f"saved to {path}")
            mlflow.log_param(f"path_{iteration}", path)
            print(iteration)

    evaluation_results = a2c_algo.evaluate()

pass
print()
# visualise using
# tensorboard --logdir=~/ray_results
