from experiments.MultiClassMultiHopDevelopment.agents.mcmh_graph_sage import MCHCGraphSage
from experiments.MultiClassMultiHopDevelopment.agents.mcmh_agents import GNN_ActorTensorDictModule
from modules.torchrl_development.envs.custom_transforms import MCMHPygTransform
from torchrl.envs.transforms import TransformedEnv
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
from modules.torchrl_development.utils.gnn_utils import tensors_to_batch
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



file_path = "../envs/env2.json"
# Example usage
with open(file_path, 'r') as file:
    env_info = json.load(file)

# init env
env = MultiClassMultiHop(**env_info)
env = TransformedEnv(env, MCMHPygTransform(in_keys=["Q"], out_keys=["X"], env=env))
td = env.reset()

gnn = MCHCGraphSage(in_channels =td["X"].shape[-1], hidden_channels=32, num_layers=2)
actor = GNN_ActorTensorDictModule(module=gnn, x_key="X", edge_index_key="edge_index", class_edge_index_key="class_edge_index", out_keys=["probs", "logits"])

# Test with only a single data point
td = actor(td)

# Test with batch of data
td = env.rollout(max_steps=5)
batch = tensors_to_batch(td["X"], td["edge_index"], td["class_edge_index"], K = env.K)
logits, probs = actor(batch)

# Test will td with multiple batches of data
# td = env.rollout(max_steps=5)
td = actor(td)

all_equal = td["probs"] == probs

