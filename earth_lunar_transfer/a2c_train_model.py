import os
# print (os.environ['PYTHONPATH'])

import sys
print (sys.path)

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
import json
import time

# sys.path.append("/nvme/lunar_space_supply")
# sys.path.append("/nvme/lunar_space_supply/earth_lunar_transfer")
# from earth_lunar_transfer.reference_exp.lunarenvironment import LunarEnvironment
from exp_time_step.exp_position_neg_norm_pos_reward.lunarenvironment import LunarEnvPositionNegNormPosReward

import mlflow
from datetime import datetime

root_path="/home/e4r/Documents/Projects/IAC/lunar_space_supply/earth_lunar_transfer"

# read config_files
with open(f"{root_path}/env_config_train.json", "rb") as config_file:
    env_config = json.load(config_file)

exp_description = "state: state with fixed time period of 3 days\n" \
                  "reward: reward based on position, time penalty, and discounted reward\n" \
                  "environment: continuous action space\n" \
                  "algorithm: A2C"

mlflow.set_tracking_uri(uri="http://0.0.0.0:8000")
mlflow.set_experiment(experiment_name=env_config["exp_name"])
with mlflow.start_run(description=exp_description) as current_run:
    run_id = current_run.info.run_id
    env_config["mlflow_run_id"] = run_id
    mlflow.log_params(env_config)
    # import pdb; pdb.set_trace()
    # model_config = dict(fcnet_hiddens=[128, 128, 128], fcnet_activation="relu")
    ppo_config = (
        PPOConfig()
        .environment(env=LunarEnvPositionNegNormPosReward, env_config=env_config)
        .training(**env_config["agent_params"])
        .rollouts(num_rollout_workers=2, num_envs_per_worker=2)
        .resources(num_gpus=1)
        .evaluation(evaluation_num_workers=1)
    )

    ppo_algo = ppo_config.build()

    print("training")
    for iteration in range(8005):
        t0 = time.time()
        train_results = ppo_algo.train()
        print(f"Time taken for training in iteration {iteration}: {time.time()-t0}sec")
        # import pdb; pdb.set_trace()
        # print(train_results)
        mlflow.log_metric("episode_reward_max", train_results["episode_reward_max"], iteration)
        mlflow.log_metric("episode_reward_min", train_results["episode_reward_min"], iteration)
        mlflow.log_metric("episode_reward_mean", train_results["episode_reward_mean"], iteration)
        mlflow.log_metric("episodes_this_iter", train_results["episodes_this_iter"], iteration)
        mlflow.log_metric("episode_len_mean", train_results['episode_len_mean'], iteration)
        mlflow.log_metric("policy_loss",
                          train_results['info']['learner']['default_policy']['policy_loss'], iteration)
        mlflow.log_metric("vf_loss", train_results['info']['learner']['default_policy']['vf_loss'],
                          iteration)
        mlflow.log_metric("vf_explained_var", train_results['info']['learner']['default_policy']['vf_explained_var'],
                          iteration)
        mlflow.log_metric("entropy", train_results['info']['learner']['default_policy']['entropy'],
                          iteration)
        mlflow.log_metric("mean_kl_loss", train_results['info']['learner']['default_policy']['mean_kl_loss'],
                          iteration)
        # mlflow.log_metric("episode_rewards", train_results['hist_stats']['episode_reward'])
        if iteration % 50 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f"{current_time}: trained {iteration + 1} epochs")
        if iteration % 100 == 0:
            path = ppo_algo.save()
            print(f"saved to {path}")
            mlflow.log_param(f"path_{iteration}", path)
            print(iteration)

    evaluation_results = ppo_algo.evaluate()

pass
# visualise using
# tensorboard --logdir=~/ray_results
