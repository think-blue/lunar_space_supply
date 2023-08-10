import pykep as pk
from Spacecraft import Spacecraft
import mlflow
import sys
import json
import torch.optim as optim
import torch

sys.path.append("//")
from earth_lunar_transfer.old_implemntation.common_module import run_episode, update_params
from earth_lunar_transfer.old_implemntation.rl_agent import ActorCritic


if __name__ == "__main__":
    # load config data
    with open("params.json", "rb") as config_file:
        params = json.load(config_file)["gpu"]

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name=params["exp_name"])

    # set
    with mlflow.start_run():
        # setting the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        mlflow.log_params(params=params)
        delta_time_step = params['num_days'] / params['n_time_steps']

        # loading earth moon data
        pk.util.load_spice_kernel("../../kernels/de441.bsp")
        earth = pk.planet.spice("earth")
        moon = pk.planet.spice('moon')

        # setting source and destination points
        source = moon
        destination = earth

        # actor critic model initialisation
        actor_critic = ActorCritic().to(device=device)
        spacecraft_opt = optim.Adam(lr=1e-4, params=actor_critic.parameters())
        spacecraft_opt.zero_grad()

        # initialising spacecraft
        spacecraft = Spacecraft(payload=params['mass'], fuel_mass=params['fuel_mass'], source=source, destination=destination,
                                specific_impulse=params['specific_impulse'],
                                epoch=pk.epoch(params['start_mjd2000_epoch']))

        for epoch in range(params['epochs']):
            # 1. gradients to zero
            spacecraft_opt.zero_grad()

            # 2. forward pass
            values, logprobs, rewards, g, rewards_components = run_episode(spacecraft=spacecraft,
                                                                           worker_model=actor_critic,
                                                                           params=params,
                                                                           delta_epoch=delta_time_step,
                                                                           source=earth,
                                                                           dest=moon,
                                                                           device=device,
                                                                           batch=epoch)

            # 3. update params
            actor_loss, critic_loss, eplen = update_params(spacecraft_opt, values, logprobs, rewards, g)

            # 4. log results
            mlflow.log_metrics(dict(actor_loss=actor_loss.mean().item(),
                                    critic_loss=critic_loss.mean().item(),
                                    reward=eplen.mean().item(),
                                    reward_0=rewards_components[0],
                                    reward_1=rewards_components[1],
                                    reward_2=rewards_components[2]),
                                    step=epoch)

            if epoch > 0 and epoch % 50 == 0:
                print(f" trained for {epoch} epochs")

        torch.save(actor_critic.state_dict(), "../../saved_models/model_gpu.pt")
        pass

