import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8-paper")

data = pd.read_csv("./negative run/metrics.csv")

data_run = data[data["run_id"] == "8c944831de8f407ebf3d44e687bd932e"]

plt.plot(data_run.loc[data_run["key"] == "episode_reward_max", "value"].values, label="max. reward")
plt.plot(data_run.loc[data_run["key"] == "episode_reward_min", "value"].values, label="min. reward")
plt.plot(data_run.loc[data_run["key"] == "episode_reward_mean", "value"].values, label="mean reward")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Episodic Reward")
plt.savefig("./negative run/training_plot_negative.png", dpi=1200)
plt.show()
