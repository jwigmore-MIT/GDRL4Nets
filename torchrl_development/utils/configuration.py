
import yaml
import os
import numpy as np
CURR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
TORCHRL_DEVELOPMENT_PATH = os.path.dirname(CURR_FILE_PATH)
CONFIG_FILE_PATH = os.path.join(TORCHRL_DEVELOPMENT_PATH, "config", "experiments")

class ConfigObject:
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, dict):
                v = ConfigObject(v)
            self.__dict__[k] = v

    def __repr__(self):
        return str(self.__dict__)

    # return the config as a dictionary
    def as_dict(self):
        # create copy of oneself
        new_self = self.__dict__.copy()
        for k, v in new_self.items():
            if isinstance(v, ConfigObject):
                new_self[k] = v.as_dict()
        return new_self


def load_config(yaml_file_path = None, full_path = None, lib_rel_path = None):
    if lib_rel_path is not None:
        # get parent of TORCHRL_DEVELOPMENT_PATH
        temp_path = os.path.dirname(TORCHRL_DEVELOPMENT_PATH)
        full_path = os.path.join(temp_path, lib_rel_path)
    if full_path is None:
        full_path = os.path.join(CONFIG_FILE_PATH, yaml_file_path)
    with open(full_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return ConfigObject(config_dict)

def create_config_from_dict(config_dict):
    return ConfigObject(config_dict)

def make_serializable(input_dict):
    """Takes a dictionary and makes it json serializable by converting all keys to strings and all np.arrays to list"""
    new_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, dict):
            v = make_serializable(v)
        if isinstance(v, np.ndarray):
            v = v.tolist()
        new_dict[str(k)] = v
    return new_dict

if __name__ == "__main__":
    # test the config object
    cfg = load_config("scaled_lambda_experiments.yaml")
    test_dict = cfg.as_dict()
