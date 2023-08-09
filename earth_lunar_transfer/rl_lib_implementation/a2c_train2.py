from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.dqn import DQNConfig
import json

from lunarenvironment_2 import LunarEnvironment2
from lunarenvironment import LunarEnvironment
from datetime import datetime

with open("env_config.json", "rb") as config_file:
    env_config = json.load(config_file)


a2c_config = (
    A2CConfig()
    .environment(env=LunarEnvironment2, env_config=env_config)
    .rollouts(num_rollout_workers=2, num_envs_per_worker=2)
    .resources(num_gpus=1)
    .training(train_batch_size=500)
    .evaluation(evaluation_num_workers=1)
)

algo = a2c_config.build()

# algo = a2c.A2C(env=LunarEnvironment, config={
#     "env_config": env_config})

for i in range(4001):
    algo.train()
    print("Current Episode", i)
    if i % 50 == 0:
        path = algo.save()
        print(f"saved to {path}")

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"{current_time}: trained {i + 1} episode")



