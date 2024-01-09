import pykep as pk
from Spacecraft import Spacecraft
import multiprocessing as mp
import mlflow
import sys
import json

sys.path.append("/nvme/ganymede/project")
from earth_lunar_transfer.old_implemntation.common_module import worker_process, run_spacecraft
from archived_code.old_implemntation.rl_agent import ActorCritic

if __name__ == "__main__":
    # config options
    with open("params.json", "rb") as config_file:
        params = json.load(config_file)["multiprocessing"]

    mlflow.set_tracking_uri(uri="http://localhost:5000")
    mlflow.set_experiment(experiment_name=params["exp_name"])

    with mlflow.start_run():
        mlflow.log_params(params=params)
        delta_time_step = params['num_days'] / params['n_time_steps']

        pk.util.load_spice_kernel("../../kernels/de441.bsp")
        moon = pk.planet.spice("moon")
        earth = pk.planet.spice("earth")
        sun = pk.planet.spice("sun")

        processes = []
        MasterNode = ActorCritic()
        manager = mp.Manager()
        training_results = manager.dict()
        training_results["mean_batch_loss_actor"] = []
        training_results["mean_batch_loss_critic"] = []
        training_results["mean_batch_return"] = []
        MasterNode.share_memory()
        counter = mp.Value('i', 0)
        for i in range(params['n_workers']):
            p = mp.Process(target=worker_process,
                           args=(i, MasterNode, counter, params, delta_time_step, earth, moon, training_results))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        for p in processes:
            p.terminate()

        print(counter.value, processes[0].exitcode)
        actor_losses = training_results["mean_batch_loss_actor"]
        critic_losses = training_results["mean_batch_loss_critic"]

        spacecraft = Spacecraft(payload=params['mass'], fuel_mass=params['fuel_mass'], source=earth,
                                epoch=pk.epoch(params['start_mjd2000_epoch']),
                                destination=moon, specific_impulse=params['specific_impulse'])
        states = run_spacecraft(spacecraft=spacecraft, time_delta=delta_time_step,
                                num_time_steps=params['n_time_steps'],
                                guidance_model=MasterNode)
        # simulate_spacecraft(states, spacecraft)
        pass
