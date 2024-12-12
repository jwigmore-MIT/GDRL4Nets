

from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
import json
# ignnore DeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)


    return data


# Example usage
file_path = 'env1.json'
env_info = parse_json_file(file_path)


net = MultiClassMultiHop(**env_info)