import wandb
import os
import json
from modules.torchrl_development.utils.configuration import load_config
from modules.torchrl_development.envs.env_creation import EnvGenerator
from torchrl.envs.utils import check_env_specs
from torchrl.envs import ExplorationType, set_exploration_type
from experiments.MultiClassMultiHopDevelopment.agents.mcmh_link_sage import MCHCGraphSage, GNN_Actor, GNN_Critic
from torchrl.modules import ProbabilisticActor
import torch
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_PATH = os.path.dirname(SCRIPT_PATH)

LOGGING_PATH = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_PATH)), "logs")

device = "cpu"


"""
PARAMETERS
"""
cfg = load_config(os.path.join(EXPERIMENT_PATH, 'config', 'MCMH_GNN_PPO_settings.yaml'))

# cfg.collector.total_frames = int(cfg.collector.frames_per_batch* 10)

cfg.agent.num_layers=3
cfg.agent.hidden_channels= 8

"""
Get Environment
"""
env_name= "grid_3x3"

file_path = f"../envs/{env_name}.json"
# Example usage
with open(file_path, 'r') as file:
    env_info = json.load(file)

env_generator = EnvGenerator(input_params=env_info,
                             make_env_keywords = cfg.training_make_env_kwargs.as_dict(),
                             env_generator_seed = 0,
                             mcmh=True)

env = env_generator.sample()
check_env_specs(env)

gnn_module = MCHCGraphSage(in_channels=env.observation_spec["X"].shape[-1],
                            hidden_channels=cfg.agent.hidden_channels,
                            num_layers=cfg.agent.num_layers,
                            normalize=False,
                            activate_last_layer=False
                            )

actor = GNN_Actor(module = gnn_module,
                  feature_key="X", edge_index_key="edge_index",
                  class_edge_index_key="class_edge_index",
                  out_keys=["logits", "probs"],
                  valid_action = False)

gnn_critic_module = MCHCGraphSage(in_channels=env.observation_spec["X"].shape[-1],
                            hidden_channels=cfg.agent.hidden_channels,
                            num_layers=cfg.agent.num_layers,
                            normalize=False,
                            activate_last_layer=False
                            )

critic = GNN_Critic(module = gnn_critic_module,
                            feature_key="X", edge_index_key="edge_index",
                            class_edge_index_key="class_edge_index", out_keys=["state_value"],
                        )


from modules.torchrl_development.agents.utils import  MaskedOneHotCategorical
from torchrl.modules import ActorCriticWrapper
actor = ProbabilisticActor(actor,
                           in_keys = ["logits", "mask"],
                           distribution_class= MaskedOneHotCategorical,
                           spec = env.action_spec,
                           default_interaction_type = ExplorationType.RANDOM,
                           return_log_prob=True,
                            )

agent = ActorCriticWrapper(actor, critic)

# Load artifact from Wandb
run = wandb.init()
artifact = run.use_artifact('jwigmore-research/MCMH_Development/trained_actor_module.pt_MCMH_GNN_PPO_grid_3x3_81147f9c_24_12_16-08_33_32:v5', type='model')
artifact_dir = artifact.download()

# load model
agent.load_state_dict(torch.load(os.path.join(artifact_dir, 'trained_actor_module.pt'),weights_only=True))


# Test model
td = env.rollout(max_steps = 500, policy = agent)

# plot results
from matplotlib import pyplot as plt
from modules.torchrl_development.utils.metrics import compute_lta
with torch.no_grad():
    fig, ax = plt.subplots()
    ax.plot(compute_lta(-td["next","reward"]))
    plt.show()
