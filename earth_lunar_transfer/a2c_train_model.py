"""
This script is used to train the model using the ray library.
Before training, import the relevant experiment class and load it's corresponding training environment
"""

from ray.rllib.algorithms.ppo import PPOConfig
import json

# import sys
# sys.path.append("/nvme/lunar_space_supply")
# sys.path.append("/nvme/lunar_space_supply/earth_lunar_transfer")

# load the environment class here
from earth_lunar_transfer.experiments.exp_time_step.exp_directed_force.lunarenvironment_directed_force import \
    LunarEnvForceHelper
import mlflow
from datetime import datetime

# read the config file here
with open("earth_lunar_transfer/configs/env_config_train.json", "rb") as config_file:
    env_config = json.load(config_file)

exp_description = """"mention the experiment description to be logged here"""

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment(experiment_name=env_config["exp_name"])
with mlflow.start_run(description=exp_description) as current_run:
    run_id = current_run.info.run_id
    env_config["mlflow_run_id"] = run_id
    mlflow.log_params(env_config)

    # model_config = dict(fcnet_hiddens=[128, 128, 128], fcnet_activation="relu")
    a2c_config = (
        # A2CConfig()
        PPOConfig()
        .environment(env=LunarEnvForceHelper, env_config=env_config)
        .training(grad_clip=3, lr_schedule=[[0, 1e-6],  # 0.00002 this works
                                            [2e6, 0.00000006],
                                            [20000000, 0.000000000001]],
                  gamma=0.95)
        .rollouts(num_rollout_workers=5, num_envs_per_worker=5)
        .evaluation(evaluation_num_workers=1)
        .framework("torch")
        .debugging(log_level="INFO")
        .offline_data(output="/nvme/lunar_space_supply/data/training_data/logs")
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
        mlflow.log_metric("vf_explained_var",
                          train_results['info']['learner']['default_policy']['learner_stats']['vf_explained_var'],
                          iteration)
        mlflow.log_metric("entropy", train_results['info']['learner']['default_policy']['learner_stats']['entropy'],
                          iteration)
        mlflow.log_metric("mean_kl_loss", train_results['info']['learner']['default_policy']['learner_stats']['kl'],
                          iteration)

        if iteration % 25 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f"{current_time}: trained {iteration + 1} epochs")
        if iteration % 50 == 0:
            path = a2c_algo.save()
            print(f"saved to {path}")
            mlflow.log_param(f"path_{iteration}", path)
            print(iteration)

    evaluation_results = a2c_algo.evaluate()
