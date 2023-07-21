from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.dqn import DQNConfig
import json
from gym_compatible_env import LunarEnvironment

with open("env_config.json", "rb") as config_file:
    env_config = json.load(config_file)


algo = (
    A2CConfig()
    .environment(env=LunarEnvironment, env_config=env_config)
    .build()
)

# algo = a2c.A2C(env=LunarEnvironment, config={
#     "env_config": env_config})

for i in range(8000):
    print(algo.train())
    if i % 100 == 0:
        path = algo.save()
        print(f"saved to {path}")


