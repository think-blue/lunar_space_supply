from ray.rllib.algorithms import Algorithm
from lunarenvironment import LunarEnvironment
import json
from datetime import datetime
from ray.rllib.algorithms.a2c import A2CConfig


checkpoint = "/home/chinmayd/ray_results/A2C_LunarEnvironment_2023-07-28_13-23-0146g1wieu/checkpoint_000401"
a2c_config = (
    A2CConfig()
    .environment(env=LunarEnvironment, env_config=env_config)
    .rollouts(num_rollout_workers=2, num_envs_per_worker=2)
    .resources(num_gpus=1)
    .training(train_batch_size=500)
    .evaluation(evaluation_num_workers=1)
)

a2c_algo = a2c_config.build()
a2c_algo.restore(checkpoint_path=checkpoint)

for i in range(3201):
    train_results = a2c_algo.train()
    if i % 50 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"{current_time}: trained {i + 1} epochs")
    if i % 400 == 0:
        path = a2c_algo.save()
        print(f"saved to {path}")