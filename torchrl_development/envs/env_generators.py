# get directory of torchrl_development
import os
from torchrl.envs.transforms import CatTensors, TransformedEnv, Compose, RewardSum, RewardScaling, StepCounter, ActionMask, UnsqueezeTransform, SignTransform
from torchrl_development.custom_transforms import SymLogTransform, InverseReward, ReverseSignTransform
from torchrl_development.envs.SingleHop import SingleHop
from torchrl_development.envs.SingleHopGraph1 import SingleHopGraph
from copy import deepcopy
import numpy as np



PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
TORCHRL_DEVELOPMENT_PATH = os.path.dirname(CURR_FILE_PATH)
CONFIG_FILE_PATH = os.path.join(TORCHRL_DEVELOPMENT_PATH, "config", "environments")

def parse_env_json(json_path, config_args = None):
    import json
    import os
    print(f"CURR_FILE_PATH: {CURR_FILE_PATH}")
    print(f"TORCHRL_DEVELOPMENT_PATH: {TORCHRL_DEVELOPMENT_PATH}")
    print(f"CONFIG_FILE_PATH: {CONFIG_FILE_PATH}")
    full_path = os.path.join(CONFIG_FILE_PATH, json_path)
    para = json.load(open(full_path, "r"))
    env_para = para["problem_instance"]
    if config_args is not None:
        if hasattr(config_args,'env'):
            for key, value in env_para.items():
                setattr(config_args.env, f"{key}", value)
        else:
            for key, value in env_para.items():
                setattr(config_args, f"env.{key}", value)
    return env_para





def make_env(env_params,
             graph= False,
             observe_lambda = False,
             seed=0,
             terminal_backlog=None,
             observation_keys=["Q", "Y"],
             negative_keys = None,
             symlog_obs = True,
             symlog_reward = False,
             inverse_reward= False,
             cost_based: bool = False,
             terminate_on_convergence = False,
             convergence_threshold = 0.1,
             stat_window_size = 100000,
             terminate_on_lta_threshold = False,):
    """
    Makes a single environment based on the parameters in env_params
    :param env_params:
    :param max_steps: Used to truncate the episode length
    :param seed:
    :param device:
    :param terminal_backlog: Used for premature termination if the backlog is greater than this value
    :param observation_keys: Specifies the keys to concatenate to form the observation for neural network policies
    :return:
    """

    env_params = deepcopy(env_params)
    if terminal_backlog is not None:
        env_params["terminal_backlog"] = terminal_backlog
    if observe_lambda and "lambda":
        env_params["obs_lambda"] = True
        if "lambda" not in observation_keys: observation_keys.append("lambda")

    env_params["terminate_on_convergence"] = terminate_on_convergence
    env_params["convergence_threshold"] = convergence_threshold
    env_params["stat_window_size"] = stat_window_size
    env_params["terminate_on_lta_threshold"] = terminate_on_lta_threshold
    if graph:
        base_env = SingleHopGraph(env_params, seed)
    else:
        base_env = SingleHop(env_params, seed)
    if negative_keys is not None:
        base_env = TransformedEnv(base_env, ReverseSignTransform(in_keys=negative_keys, out_keys=negative_keys))
    if cost_based:
        base_env = TransformedEnv(base_env, ReverseSignTransform(in_keys=["reward"], out_keys=["reward"]))
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ActionMask(action_key="action", mask_key="mask"),
            CatTensors(in_keys=observation_keys, out_key="observation", del_keys=False),
            #SymLogTransform(in_keys=["observation"], out_keys=["observation"]),
            # StepCounter()
        )

    )
    if symlog_obs:
        env = TransformedEnv(env, SymLogTransform(in_keys=["observation"], out_keys=["observation"]))
    if symlog_reward:
        env = TransformedEnv(env, SymLogTransform(in_keys=["reward"], out_keys=["reward"]))
    if graph:
        "Create x key from concatentation of observation keys and do symlog transform"
        env = TransformedEnv(env, UnsqueezeTransform(in_keys = observation_keys, out_keys = observation_keys, unsqueeze_dim = -1))
        env = TransformedEnv(env, CatTensors(in_keys=observation_keys, out_key="x", del_keys=False, dim = 1))
        env = TransformedEnv(env, SymLogTransform(in_keys=["x"], out_keys=["x"]))
    if inverse_reward:
        env = TransformedEnv(env, InverseReward())
    # else:
    #     env = TransformedEnv(env, RewardScaling(loc=0, scale=0.01))

    return env



def create_training_test_generators(input_params,
                                    make_env_keywords = None,
                                    env_generator_seed = 0,
                                    test_size = 0.2,
                                    ):
    """
    Takes the env parameters and creates a training and test environment generator based on split coefficient in test_size
    :param env_params:
    :param make_env_keywords:
    :param seed:
    :param test_size:
    :return:
    """
    if "num_envs" not in input_params.keys():
        raise ValueError("input_params must have a key 'num_envs'")
    else:
        num_envs = input_params["num_envs"]
    all_env_params = input_params["all_env_params"]
    num_training = int(num_envs * (1 - test_size))
    training_inds = np.random.choice(num_envs, num_training, replace = False)
    test_inds = np.array([i for i in range(num_envs) if i not in training_inds])

    # create
    training_all_env_params = {key: all_env_params[str(key)] for key in training_inds}
    training_network_loads = {key: all_env_params[str(key)]["network_load"] for key in training_inds}
    training_ltas = {key: all_env_params[str(key)]["lta"] for key in training_inds}
    training_num_envs = len(training_inds)
    training_input_params = {"all_env_params": training_all_env_params,
                                "num_envs": training_num_envs,
                                "network_loads": training_network_loads,
                                "ltas": training_ltas,
                                }
    training_generator = EnvGenerator(training_input_params, make_env_keywords, env_generator_seed)


    test_all_env_params = {key: all_env_params[str(key)] for key in test_inds}
    test_network_loads = {key: all_env_params[str(key)]["network_load"] for key in test_inds}
    test_ltas = {key: all_env_params[str(key)]["lta"] for key in test_inds}
    test_num_envs = len(test_inds)
    test_input_params = {"all_env_params": test_all_env_params,
                            "num_envs": test_num_envs,
                            "network_loads": test_network_loads,
                            "ltas": test_ltas,
                            }
    test_generator = EnvGenerator(test_input_params, make_env_keywords, env_generator_seed)


    return training_generator, test_generator

def create_scaled_lambda_generator(env_params,
                                   make_env_keywords = None,
                                   env_generator_seed = 0,
                                   lambda_scale = 1.0):
    """
    Creates an environment from the env_params, but scales the arrival rates by lambda_scale
    :param env_params: 
    :param make_env_keywords: 
    :param seed: 
    :param lambda_scale: 
    :return: 
    """
    env_params = deepcopy(env_params)
    for key, value in env_params["X_params"].items():
        value["probability"] = value["probability"] * lambda_scale

    return EnvGenerator(env_params, make_env_keywords, env_generator_seed)

def create_scaled_lambda_params(env_params, lambda_scale = 1.0):
    """
    Takes in the environment parameters and scales the arrival rates by lambda_scale
    :param enb_params:
    :param lambda_scale:
    :return: scaled_env_params
    """
    env_params = deepcopy(env_params)
    for key, value in env_params["X_params"].items():
        value["probability"] = value["probability"] * lambda_scale
    return env_params

def make_gen_from_params(env_params_list,
                         make_env_keywords = None,
                         env_generator_seed = 0,
                         cycle_sample = False):
    """
    Creates an environment generator from a list of environment parameters
    :param env_params_list:
    :param make_env_keywords:
    :param env_generator_seed:
    :return:
    """
    env_params = {i: env_params_list[i] for i in range(len(env_params_list))}
    input_params = {"all_env_params": env_params,
                      "num_envs": len(env_params_list),
                      }
    return EnvGenerator(input_params, make_env_keywords, env_generator_seed, cycle_sample = cycle_sample)


def make_scaled_gen_from_params(env_params_list,
                                lambda_scales: list= [1.0],
                                make_env_keywords = None,
                                env_generator_seed = 0):
    """
    Creates a generator from the combination of all env_params in env_params_list and all lambda_scales"""
    scaled_env_params_list = []
    for env_params in env_params_list:
        for lambda_scale in lambda_scales:
            scaled_env_params_list.append(create_scaled_lambda_params(env_params, lambda_scale))
    return make_gen_from_params(scaled_env_params_list, make_env_keywords, env_generator_seed)

class EnvGenerator:
    """
    Takes in either a single environment parameter file and returns an instance of the environment on sample()
    or takes in a generated env parameter file which contains many different instances of the environment
    """

    def __init__(self, input_params,
                 make_env_keywords = None,
                 env_generator_seed = 0,
                 cycle_sample = False
                 ):
        # if env_params has a key "key_params" then is the parameters of many environments
        self.context_dicts = None

        # set np seed
        self.env_generator_seed = env_generator_seed
        self.seed_generator = np.random.default_rng(env_generator_seed)

        # check if env_params is a single environment or many
        if "num_envs" in input_params.keys():
            self.context_dicts = input_params["context_dicts"]
            # if all keys of context_dicts are str, then convert to int
            if all([isinstance(key, str) for key in self.context_dicts.keys()]):
                self.context_dicts = {int(key): value for key, value in self.context_dicts.items()}
            self.num_envs = input_params["num_envs"]
            if not cycle_sample:
                self.sample = self.sample_from_multi
            else:
                self.sample = self.cycle_sample
                self.last_sample_ind = -1
        else:
            self.context_dicts = {0: {
                                    "env_params":input_params,
                                   "admissible": None,
                                   "arrival_rates": input_params.get("arrival_rates", None),
                                   "lta": input_params.get("lta", None),
                                   "network_load": input_params.get("network_load", None)}}
            self.num_envs = 1
            self.sample = self.sample_from_multi
        self._make_env_keywords = make_env_keywords
        self.history = []

    def clear_history(self):
        self.history = []

    def reseed(self, seed = None):
        if seed is None:
            seed = self.env_generator_seed
        self.seed_generator = np.random.default_rng(seed)
        self.env_generator_seed = seed

    def gen_seeds(self, n):
        return self.seed_generator.integers(low = 0, high = 100000, size = n)


    def sample_from_multi(self, rel_ind = None, true_ind = None, seed = None):
        """
        If given rel_ind, then samples the rel_ind-th environment from the context_dicts
        If given true_ind, then samples the true_ind environment from the context_dicts
        If neither are given, then samples from the context_dicts with a random index uniformly
        :param rel_ind:
        :param true_ind:
        :return:
        """

        if rel_ind is None and true_ind is None:
            rel_ind = self.seed_generator.choice(self.num_envs)
        if rel_ind is not None and true_ind is not None:
            raise ValueError("Only one of rel_ind or true_ind can be specified")
        try:
            if rel_ind is not None:
                env_params_ind = list(self.context_dicts.keys())[rel_ind]
            if true_ind is not None:
                env_params_ind = true_ind
            env_params = self.context_dicts[env_params_ind]["env_params"]
            env_params["context_id"] = env_params_ind
            env_params["baseline_lta"] = self.context_dicts[env_params_ind]["lta"]
        # if ind is not in the keys
        except KeyError:
            raise ValueError(f"Index {rel_ind} is not in the keys of the environment parameters")
        if seed is None:
            seed = self.seed_generator.integers(low = 0, high = 100000)
        env = make_env(env_params, seed = seed, **self._make_env_keywords)
        # env.base_env.baseline_lta = self.context_dicts[env_params_ind]["lta"]
        self.history.append(env_params_ind)
        return env

    def cycle_sample(self):
        """On each call, the next environment is sampled"""
        self.last_sample_ind += 1
        if self.last_sample_ind >= self.num_envs:
            self.last_sample_ind = 0
        return self.sample_from_multi(self.last_sample_ind)

    def sample_from_solo(self):
        env = make_env(self.context_dicts, seed = self.seed_generator.integers(low = 0, high = 100000), **self._make_env_keywords)
        env.baseline_lta = self.baseline_lta
        self.history.append(0)
        return env


    def create_all_envs(self):
        envs = {}
        for i in range(self.num_envs):
            key = list(self.context_dicts.keys())[i]
            env_params = self.context_dicts[key]
            if "env_params" in env_params.keys():
                env_params = env_params["env_params"]
            env = make_env(env_params, **self._make_env_keywords)
            envs[i] = {"env":env,
                      "env_params":env_params,
                      "ind": i,
                      "arrival_rates":key}
        return envs

    def add_env_params(self, env_params, ind = None):
        if ind is None:
            ind = self.num_envs
        self.context_dicts[ind] = env_params
        self.num_envs += 1











if __name__ == "__main__":
    import json
    raise Exception("The test code is not up to date. Needs to be updated for context_set_dict and context_dict format")
    env_name = "SH1"
    single_env_params = parse_env_json(f"{env_name}.json")

    make_env_keywords = {"observe_lambda": True,
                            "device": "cpu",
                            "terminal_backlog": 100,
                            }
    # Test single environment generations
    env_generator = EnvGenerator(single_env_params, make_env_keywords)
    single_env = env_generator.sample()


    # Training and test split test
    generated_params = json.load(open("C:\\Users\\Jerrod\\PycharmProjects\\IA-DRL_4SQN\\torchrl_development\\envs\\sampling\\SH1_generated_params.json", 'rb'))
    if "env_params" in generated_params.keys(): # if true, change the key to "all_env_params"
        generated_params["all_env_params"] = generated_params.pop("env_params")

    training_env_generator, test_env_generator = create_training_test_generators(generated_params,
                                                                                 make_env_keywords,
                                                                                 env_generator_seed = 0,
                                                                                 test_size = 0.2)
    #training_env = training_env_generator.sample() # need to fix indexing with the gene

    # Scaling lambda test
    env_name = "SH1_NA"
    single_env_params = parse_env_json(f"{env_name}.json")
    lambda_scale = 0.6
    scaled_lambda_generator = create_scaled_lambda_generator(single_env_params,
                                                             make_env_keywords,
                                                             env_generator_seed = 0,
                                                             lambda_scale = lambda_scale)
    scaled_lambda_env = scaled_lambda_generator.sample()

    # Make generator from list of parameters
    env_params1 = parse_env_json(f"SH1_NA.json")
    env_params2 = parse_env_json(f"SH1_NA2.json")

    env_params_list = [env_params1, env_params2]
    env_generator = make_gen_from_params(env_params_list, make_env_keywords, env_generator_seed = 0)
    sampled_envs = env_generator.create_all_envs()

    # Make generator from list of parameters and lambda scales
    lambda_scales = [0.95, 0.85]
    scaled_gen = make_scaled_gen_from_params(env_params_list, lambda_scales, make_env_keywords, env_generator_seed = 0)
    sampled_scaled_envs = scaled_gen.create_all_envs()
    env_params1_scaled = create_scaled_lambda_params(env_params1, 0.98)
    scaled_gen.add_env_params(env_params1_scaled)
    sampled_scaled_envs = scaled_gen.create_all_envs()

    # Cycle sample test
    all_env_params = json.load(open("../experiments/experiment4/experiment4_envs_params.json", 'rb'))
    input_params = {"num_envs": len(all_env_params.keys()),
                    "all_env_params": all_env_params,}
    cycle_env_generator = EnvGenerator(input_params, make_env_keywords, env_generator_seed=1010110, cycle_sample = True)
    sampled_cycle_envs = [cycle_env_generator.sample() for i in range(10)]

