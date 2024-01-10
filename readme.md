
# CisLunar Autonomous Navigation Control
This project presents a Reinforcement based implementation of an autonomous navigation control mechanism in cis lunar space. An agent is trained to navigate from an orbit around moon to reach a desired orbit around earth in an automated and adaptive manner.
The results obtained via conducting experiments by utilising the code from this repo has been published in the [International Astronautical Congress 2023](https://dl.iafastro.directory/event/IAC-2023/paper/79728/). An open-sourced copy of the same can be accessed at [ResearchGate](https://www.researchgate.net/publication/375640838_Reinforcement_Learning_based_Optimization_of_the_Earth-Moon_Space_Supply_Network).

This repository aims to provide the codebase to reproduce the results published in the conference as well as to enable further research in this area. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

### Requirements

- Python.3.8+
- Required Packages can be installed using conda
```commandline
conda env create -n ENVNAME --file requirements_general.yaml
conda activate ENVNAME
```

### Clone the repository
```commandline
git clone https://github.com/think-blue/lunar_space_supply
```

## Usage


### Training the model

To train the model:
```commandline
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/"
```
```commandline
python a2c_train_model.py --config_file "earth_lunar_transfer/configs/env_config_train.json"
```

### Testing the model
To test the model:
```commandline
python a2c_test_model.py --config_file "earth_lunar_transfer/configs/env_config_test.json" --checkpoint ""/home/chinmayd/ray_results/PPO_LunarEnvForceHelper_2023-09-12_22-31-15r58fiom2/checkpoint_000051"
```
The test script displays the plots regarding velocity and positional error norms, and also plots a simulation of the trajectory. The same is saved under data/simulation_figures/<experiment_name>
A sample simulation plot should like this.

## License

This project is licensed under the terms of the MIT license.