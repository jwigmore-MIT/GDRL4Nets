import tensordict

from NetworkRunner import create_network_runner
from modules.torchrl_development.envs.MultiClassMultihopBP import MultiClassMultiHopBP
import json
from experiments.GNNBiasedBackpressureDevelopment.models.node_attention_gnn import DeeperNodeAttentionGNN
from experiments.GNNBiasedBackpressureDevelopment.modules.modules import NormalWrapper
from torchrl.modules import ProbabilisticActor, IndependentNormal, TanhNormal
from tensordict.nn import InteractionType
# import optimizer
import torch.optim as optim
import tensordict
import torch
from torchrl.envs import ExplorationType, set_exploration_type
from torch.utils.tensorboard import SummaryWriter



import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



file_path = "../envs/grid_3x3.json"
env_info = json.load(open(file_path, 'r'))
env_info["action_func"] = "bpi"
env = MultiClassMultiHopBP(**env_info)
network_specs = env.get_rep()

# Create the Model
model = DeeperNodeAttentionGNN(
    node_channels = network_specs["X"].shape[-1],
    edge_channels = network_specs["edge_attr"].shape[-1],
    hidden_channels =8,
    num_layers = 4,
    output_channels=2,
    output_activation=None,
    edge_decoder=True
)

norm_module = NormalWrapper(model)
actor = ProbabilisticActor(norm_module,
                            in_keys = ["loc", "scale"],
                            out_keys = ["bias"],
                            distribution_class=IndependentNormal,
                           distribution_kwargs={"tanh_loc": True},
                            return_log_prob=True,
                            default_interaction_type = InteractionType.RANDOM
                            )
optimizer = optim.Adam(actor.parameters(), lr=1e-10)

output = actor(network_specs)
runner = create_network_runner(env = env, max_steps = 2000, actor = actor)

writer = SummaryWriter()

with set_exploration_type(ExplorationType.DETERMINISTIC):
    base_td = runner.get_run(bias = env.bias.clone())
    print(f"Baseline Reward: {base_td['reward']}")
    baseline_reward = base_td["reward"]

xdata = []
losses = []
rewards = []

for epoch in range(50):
    sample = runner.get_run()
    dist = actor.get_dist(sample)
    log_prob = dist.log_prob(sample["bias"].unsqueeze(-1))
    loss = log_prob * (sample["reward"] - baseline_reward)
    mean_loss = loss.mean()
    optimizer.zero_grad()
    mean_loss.backward()
    optimizer.step()
    xdata.append(epoch)
    losses.append(mean_loss.item())
    rewards.append(sample["reward"])


    writer.add_scalar('Loss', mean_loss.item(), epoch)
    writer.add_scalar('Reward', sample["reward"], epoch)
    writer.add_scalar('Location Variation', sample["loc"].std().item(), epoch)
    writer.add_scalar('Average Log Prob', sample['sample_log_prob'].mean().item(), epoch)
    writer.add_scalar('Average loc logits', sample['logits'][0,:].mean().item(), epoch)
    writer.add_scalar('Average scale logits', sample['logits'][1,:].mean().item(), epoch)
    writer.add_scalar('Average loc', sample['loc'].mean(), epoch)
    writer.add_scalar('Average scale', sample['scale'].mean().item(), epoch)

    print(f"------Epoch {epoch} ----- \n"
          f"Loss: {mean_loss:.2f} Reward: {sample['reward']:.2f}\n"
          f"Location Variation {sample['loc'].std().item():.2f} "
          f"Average Log Prob {sample['sample_log_prob'].mean().item():.2f}\n"
          f"Average loc logits {sample['logits'][0,:].mean().item():.2f} "
          f"Average scale logits {sample['logits'][1,:].mean().item():.2f}\n"
          f"Average loc {sample['loc'].mean():.2f} Average scale {sample['scale'].mean().item():.2f}\n")
