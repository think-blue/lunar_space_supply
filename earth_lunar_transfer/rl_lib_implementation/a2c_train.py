import ray
from ray.rllib.algorithms.a2c import A2CConfig
import json
from gym_compatible_env import LunarEnvironment

with open("env_config.json", "rb") as config_file:
    env_config = json.load(config_file)

algo = (
    A2CConfig()
    .rollouts(num_rollout_workers=10)
    .resources(num_gpus=0)
    .environment(env=LunarEnvironment, env_config=env_config)
    .build()

)

# algo = a2c.A2C(env=LunarEnvironment, config={
#     "env_config": env_config})

for _ in range(1000):
    print(algo.train())
