from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.dqn import DQNConfig
import json
from lunarenvironment import LunarEnvironment
import mlflow
from datetime import datetime



with open("env_config.json", "rb") as config_file:
    env_config = json.load(config_file)


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
for i in range(3201):
    train_results = a2c_algo.train()
    if i % 50 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"{current_time}: trained {i + 1} epochs")
    if i % 400 == 0:
        path = a2c_algo.save()
        print(f"saved to {path}")

evaluation_results = a2c_algo.evaluate()
pass
# visualise using
# tensorboard --logdir=~/ray_results
