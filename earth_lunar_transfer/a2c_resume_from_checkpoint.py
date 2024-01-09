"""
Use this training script to resume training from a checkpoint.
"""

from ray.rllib.algorithms import Algorithm
from earth_lunar_transfer.exp_time_step.exp_position.lunarenvironment import LunarEnvPosition
from earth_lunar_transfer.exp_time_step.exp_no_gravity.lunarenvironment_negative import LunarEnvNoGravity
import json
from datetime import datetime
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.ppo import PPOConfig

with open("earth_lunar_transfer/configs/env_config_train.json", "rb") as config_file:
    env_config = json.load(config_file)


# mention the file path of the checkpoint here
checkpoint = "/home/chinmayd/ray_results/PPO_LunarEnvNoGravity_2023-08-23_19-33-27a_9w5khq/checkpoint_000301"
# /home/chinmayd/ray_results/A2C_LunarEnvNoGravity_2023-08-21_19-30-12qja6p5h8/checkpoint_007201

a2c_config = (
    PPOConfig()
    .environment(env=LunarEnvNoGravity, env_config=env_config)
    .training(grad_clip=3, lr_schedule=[[0, 0.00002],  # 0.00002 this works
                                        [2e6, 0.00000004],
                                        [20000000, 0.000000000001]])
    .rollouts(num_rollout_workers=5, num_envs_per_worker=5)
    .evaluation(evaluation_num_workers=1)
    .framework("torch")
    .debugging(log_level="INFO")
    .offline_data(output="/nvme/lunar_space_supply/data/training_data/logs")
)

a2c_algo = a2c_config.build()
a2c_algo.restore(checkpoint_path=checkpoint)

# resume training from checkpoint

num_epochs = 8005
for i in range(num_epochs):
    train_results = a2c_algo.train()
    if i % 50 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"{current_time}: trained {i + 1} epochs")
    if i % 100 == 0:
        path = a2c_algo.save()
        print(f"saved to {path}")
