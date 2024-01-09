from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.dqn import DQNConfig
import json
from lunarenvironment import LunarEnvironment
import mlflow
from datetime import datetime

# read config_files
with open("env_config_train.json", "rb") as config_file:
    env_config = json.load(config_file)

exp_description = "state: state with fixed time period of 3 days\n" \
                  "reward: reward based on position, velocity, and fuel errors\n" \
                  "environment: continuous action space\n" \
                  "algorithm: A2C"

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment(experiment_name=env_config["exp_name"])
with mlflow.start_run(description=exp_description) as current_run:
    run_id = current_run.info.run_id
    env_config["mlflow_run_id"] = run_id
    mlflow.log_params(env_config)

    a2c_config = (
        A2CConfig()
        .environment(env=LunarEnvironment, env_config=env_config)
        .rollouts(num_rollout_workers=2, num_envs_per_worker=2)
        .resources(num_gpus=1)
        .training(train_batch_size=500)
        .evaluation(evaluation_num_workers=1)
    )

    a2c_algo = a2c_config.build()

    print("training")
    for iteration in range(1002):
        train_results = a2c_algo.train()
        print(train_results)
        mlflow.log_metric("episode_reward_max", train_results["episode_reward_max"], iteration)
        mlflow.log_metric("episode_reward_min", train_results["episode_reward_min"], iteration)
        mlflow.log_metric("episode_reward_mean", train_results["episode_reward_mean"], iteration)
        mlflow.log_metric("episodes_this_iter", train_results["episodes_this_iter"], iteration)
        mlflow.log_metric("episode_len_mean", train_results['episode_len_mean'], iteration)
        # mlflow.log_metric("episode_rewards", train_results['hist_stats']['episode_reward'])
        if iteration % 50 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f"{current_time}: trained {iteration + 1} epochs")
        if iteration % 400 == 0:
            path = a2c_algo.save()
            print(f"saved to {path}")
            mlflow.log_param(f"path_{iteration}", path)
        print(iteration)

    evaluation_results = a2c_algo.evaluate()

pass
# visualise using
# tensorboard --logdir=~/ray_results
