from ray.rllib.algorithms import Algorithm
from lunarenvironment import LunarEnvironment
import json
from datetime import datetime

checkpoint = "/home/chinmayd/ray_results/A2C_LunarEnvironment_2023-07-28_13-23-0146g1wieu/checkpoint_000401"
a2c_algo = Algorithm.from_checkpoint(checkpoint=checkpoint)

for i in range(3201):
    train_results = a2c_algo.train()
    if i % 50 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"{current_time}: trained {i + 1} epochs")
    if i % 400 == 0:
        path = a2c_algo.save()
        print(f"saved to {path}")


