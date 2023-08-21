import json
from os import path


class RunConfig:
    def __init__(self, run_config):
        self.n_run = 10
        self.episodes_per_run = 10
        self.steps_per_episode = 100
        self.run_mode = 'serial'
        self.actor_model = None
        self.critic_model = None
        self.n_nodes = 4
        self.save_complex = False
        self.window_size = 50
        self.convergence_std = 0.3
        self.skip_if_exists = True
        self.analysis = True

        self.__dict__.update(run_config)


class Config:
    __instance = None

    def __init__(self, config_path):

        # pre initialize to avoid errors
        self.single_step = [1, 1, 1, 1, 1, 1]
        self.datastore_path = f"{path.dirname(path.abspath(__file__))}/../datastore"
        self.test_dataset = f"{path.dirname(path.abspath(__file__))}/../data/test_dataset"
        self.train_dataset = f"{path.dirname(path.abspath(__file__))}/../data/train_dataset"
        self.run_config = None

        self.training_algorithm = "dqn"
        self.node_features = [
            'coord',
            'atom_type_encoding',
            'atom_named_features',
            'is_heavy_atom',
            'VDWr',
            'molecule_type',
            'smarts_patterns',
            'residue_labels',
            'z_scores',
            'kd_hydophobocitya',
            'conformational_similarity',
            'func_group'
        ]

        self.edge_features = [
           "bond_distance"
        ]

        self.gauss_basis_divisor = 0.75
        self.gauss_basis_variances = 15
        self.profiler_port = 8181
        self.gnn_explainer = False
        self.run_tests = True
        self.test_from_episode = 70
        self.profile_from_episode = 400
        self.divergence_slope = 0.005
        self.episode_length = 100
        self.save_model_from = 700
        self.test_episode_length = 100
        self.reward_function = 'rmsd'
        self.n_episodes = 1500

        config = {}
        with open(config_path, 'r') as config_f:
            config = json.load(config_f)

        self.__dict__.update(config)

        run_config = config.get('run_config', {})
        self.run_config = RunConfig(run_config)

    @classmethod
    def init(cls, config_path):
        cls.__instance = Config(config_path)

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            raise Exception('Instance not initialized, use Config.init to initialize the config')

        return cls.__instance
