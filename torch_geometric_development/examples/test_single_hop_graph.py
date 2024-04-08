from torchrl_development.envs.SingleHopGraph import SingleHopGraph
from torchrl_development.envs.env_generators import parse_env_json, make_env
from torchrl.envs.utils import check_env_specs
from torchrl_development.actors import MaxWeightActor
import torch
from torch_geometric_development.conversions import lazy_stacked_tensor_dict_to_batch as lstd2b



if __name__ == "__main__":
    env_config_path = "SH2u.json"
    env_para = parse_env_json(env_config_path)
    env = SingleHopGraph(env_para)
    check_env_specs(env)

    # Create Max Weight Actor to collect Rollout
    actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])

    td = env.rollout(max_steps = 10, policy = actor)

    # Now collect a rollout from another env
    env_config_path2 = "SH3.json"
    env_para2 = parse_env_json(env_config_path2)
    env2 = SingleHopGraph(env_para2)
    check_env_specs(env2)

    td2 = env2.rollout(max_steps = 10, policy = actor)

    # Now try stacking
    stacked_td = torch.stack([td, td2])

    batch = lstd2b(stacked_td['graph'])

