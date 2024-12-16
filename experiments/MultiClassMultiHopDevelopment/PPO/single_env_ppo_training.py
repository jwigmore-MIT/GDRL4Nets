

"""
Train on a single environment using PPO algorithm
"""


import os

import torch

from modules.torchrl_development.envs.custom_transforms import MCMHPygLinkGraphTransform, SymLogTransform
from torchrl.envs.transforms import TransformedEnv
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
from modules.torchrl_development.utils.gnn_utils import tensors_to_batch
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from modules.torchrl_development.utils.configuration import load_config
from modules.torchrl_development.envs.env_creation import EnvGenerator
from torchrl.envs.utils import check_env_specs
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
from torchrl.record.loggers import get_logger, generate_exp_name
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
import wandb
import tqdm
import time
import sys
from copy import deepcopy
from torchrl.objectives.value.functional import generalized_advantage_estimate
from tensordict import TensorDict
from experiment_utils import evaluate_agent


from experiments.MultiClassMultiHopDevelopment.development.backpressure import BackpressureActor
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
from torchrl.objectives.value import GAE


from modules.torchrl_development.utils.metrics import compute_lta
from torchrl.modules import ProbabilisticActor, ActorCriticWrapper
import pickle




SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_PATH = os.path.dirname(SCRIPT_PATH)

LOGGING_PATH = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_PATH)), "logs")

device = "cpu"


"""
PARAMETERS
"""
cfg = load_config(os.path.join(EXPERIMENT_PATH, 'config', 'MCMH_GNN_PPO_settings.yaml'))

# cfg.collector.total_frames = int(cfg.collector.frames_per_batch* 10)

cfg.agent.num_layers=7
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

eval_env_generator = EnvGenerator(input_params=env_info,
                                    make_env_keywords = cfg.training_make_env_kwargs.as_dict(),
                                    env_generator_seed = 1,
                                    mcmh=True)

env = env_generator.sample()
check_env_specs(env)


"""
CREATE GNN ACTOR CRITIC 
"""
from experiments.MultiClassMultiHopDevelopment.agents.mcmh_link_sage import MCHCGraphSage, GNN_Actor, GNN_Critic

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

# td = env.rollout(max_steps=1000, policy=gnn_agent) # test rollout
# GAE Module
adv_module = GAE(
        gamma=0.99,
        lmbda=0.95,
        value_network=critic,
        average_gae=False,
        vectorized=False
    )

# Collector
collector = SyncDataCollector(
    create_env_fn=env_generator.sample,
    policy=actor,
    frames_per_batch=cfg.collector.frames_per_batch,
    total_frames=cfg.collector.total_frames,
    device = "cpu",
    max_frames_per_traj=cfg.collector.max_frames_per_traj,
    split_trajs=True,
    reset_when_done=True,
)


sampler = SamplerWithoutReplacement()
data_buffer = TensorDictReplayBuffer(
    storage=LazyMemmapStorage(cfg.collector.frames_per_batch),
    sampler=sampler,
    batch_size=cfg.loss.mini_batch_size,
)

lifetime_buffer = TensorDictReplayBuffer(
    storage=LazyMemmapStorage(cfg.collector.total_frames),
)

## Create PPO loss module
loss_module = ClipPPOLoss(
    actor_network=actor,
    critic_network=critic,
    clip_epsilon=cfg.loss.clip_epsilon,
    entropy_coef=cfg.loss.entropy_coef,
    normalize_advantage=cfg.loss.norm_advantage,
    loss_critic_type=cfg.loss.loss_critic_type,
)

## Create the optimizer
optimizer = torch.optim.Adam(
    loss_module.parameters(),
    lr= cfg.optim.lr,
    weight_decay=0,
)


experiment_name = generate_exp_name(f"MCMH_GNN_PPO", env_name)
logger = get_logger(
        "wandb",
        logger_name=LOGGING_PATH,
        experiment_name= experiment_name,
        wandb_kwargs={
            "config": cfg.as_dict(),
            "project": cfg.logger.project,
        },
    )

## Initialize variables for training loop
collected_frames = 0 # count of environment interactions
sampling_start = time.time()
num_updates = cfg.loss.num_updates # count of update epochs to the network
num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size # number of mini batches per update

total_network_updates = (
    (cfg.collector.total_frames // cfg.collector.frames_per_batch) * cfg.loss.num_updates * num_mini_batches
) # total number of network updates to be performed
pbar = tqdm.tqdm(total=cfg.collector.total_frames, file = sys.stdout)
num_network_updates = 0

# initialize the artifact saving params
best_eval_backlog = -10000000
artifact_name = logger.exp_name

# create a tracker to track the running average of a metric
running_sum = torch.zeros(env_generator.num_envs, device=device)
running_average = torch.zeros(env_generator.num_envs, device=device)
running_step_counter = torch.zeros(env_generator.num_envs, device=device)


## Main Training Loop
for i, data in enumerate(collector): # iterator that will collect frames_per_batch from the environments
    log_info = {} # dictionary to store all of the logging information for this iteration

    # Evaluate the agent (if conditions are met)
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):

        final = collected_frames >= collector.total_frames # check if this is the final evaluation epoch
        prev_test_frame = ((i - 1) * cfg.collector.frames_per_batch) // cfg.collector.test_interval
        cur_test_frame = (i * cfg.collector.frames_per_batch) // cfg.collector.test_interval
        if (i >= 1 and (prev_test_frame < cur_test_frame)) or final:
            actor.eval()
            eval_start = time.time()
            training_env_ids = list(env_generator.context_dicts.keys())
            eval_log_info, eval_tds = evaluate_agent(actor, eval_env_generator, training_env_ids, pbar, cfg,
                                                         device)

            log_info.update(eval_log_info)

            # Save the agent if the eval backlog is the best
            if eval_log_info["eval/lta_backlog_training_envs"] < best_eval_backlog:
                best_eval_backlog = eval_log_info["eval/lta_backlog_training_envs"]
                torch.save(agent.state_dict(), os.path.join(logger.experiment.dir, f"trained_actor_module.pt"))
                agent_artifact = wandb.Artifact(f"trained_actor_module_{artifact_name}", type="model")
                agent_artifact.add_file(os.path.join(logger.experiment.dir, f"trained_actor_module.pt"))
                agent_artifact.add_file(os.path.join(logger.experiment.dir, f"config.yaml"))
                wandb.log_artifact(agent_artifact, aliases=["best", "latest"])
            else:
                torch.save(agent.state_dict(), os.path.join(logger.experiment.dir, f"trained_actor_module.pt"))
                agent_artifact = wandb.Artifact(f"trained_actor_module.pt_{artifact_name}", type="model")
                agent_artifact.add_file(os.path.join(logger.experiment.dir, f"trained_actor_module.pt"))
                agent_artifact.add_file(os.path.join(logger.experiment.dir, f"config.yaml"))
                wandb.log_artifact(agent_artifact, aliases=["latest"])
            # log all of the state action figures to wandb

            actor.train()

    # Process collected data for logging before updates
    sampling_time = time.time() - sampling_start
    data = data[data["collector", "mask"]]

    pbar.update(data.numel())
    current_frames = data.numel()
    collected_frames += current_frames

    # ensure that the logits and actions are the correct shape
    # if data["logits"].dim() > 2:
    #     data["logits"] = data["logits"].squeeze()
    # if data["action"].dim() > 2:
    #     data["action"] = data["action"].squeeze()

    # loop through data from each training environment to track the per/env metrics
    for context_id in data["context_id"].unique():
        env_data = data[(data["context_id"] == context_id).squeeze()]
        env_data = env_data[env_data["collector","mask"]]
        env_data_count = env_data.shape[0]
        context_id = env_data.get("context_id", None)
        if context_id is not None:
            context_id = int(context_id[0].item())
        # baseline_lta = env_data.get("baseline_lta", None)
        # if baseline_lta is not None:
        #     env_lta = baseline_lta[-1]
        mean_episode_reward = env_data["next", "reward"].sum(dim = 1).mean()
        mean_episode_backlog = env_data["Q"].sum((1,2)).mean()
        running_sum[context_id] += mean_episode_backlog
        running_step_counter[context_id] += env_data["collector", "mask"].sum()
        running_average[context_id] = running_sum[context_id] / (i+1)
        # interference_factor = (env_data["action"] - env_data["next", "valid_action"]).mean().float()
        # interference_percentage = ((env_data["action"]- env_data["next", "valid_action"]).sum(dim = 1) > 0).float().mean()
        # mean_backlog = env_data["next", "ta_mean"][-1]
        # std_backlog = env_data["next", "ta_stdev"][-1]
        # # mean_backlog = env_data["next", "backlog"].float().mean()
        # normalized_backlog = mean_backlog / env_lta
        # valid_action_fraction = (env_data["mask"] * env_data["action"]).sum().float() / env_data["mask"].shape[0]
        log_header = f"train/context_id_{context_id}"

        log_info.update({f'{log_header}/mean_episode_reward': mean_episode_reward.item(),
                         f'{log_header}/running_average': running_average[context_id].item(),
                         f'{log_header}/mean_episode_backlog': mean_episode_backlog.item(),
                         # f'{log_header}/std_backlog': std_backlog.item(),
                         # f'{log_header}/mean_normalized_backlog': normalized_backlog.item(),
                         # f'{log_header}/global_interference_fraction': interference_percentage.item(),
                         # f'{log_header}/per_node_interference_fraction': interference_factor.item(),
                            })


    # Get average mean_normalized_backlog across each context
    # avg_mean_normalized_backlog = np.mean([log_info[f'train/context_id_{i}/mean_normalized_backlog'] for i in env_generator.context_dicts.keys()])
    # log_info.update({"train/avg_mean_normalized_backlog": avg_mean_normalized_backlog})

    # only keep the data that is valid
    data = data[data["collector", "mask"]]
    graph = tensors_to_batch(data["X"], data["edge_index"], data["class_edge_index"], K=data["Q"].shape[-1])
    next_graph = tensors_to_batch(data["next", "X"], data["next", "edge_index"], data["next", "class_edge_index"], K=data["Q"].shape[-1])
    # lifetime_buffer.extend(data)
    # optimization steps
    training_start = time.time()
    losses = TensorDict({}, batch_size=[cfg.loss.num_updates, num_mini_batches])
    value_estimates = torch.zeros(num_updates, device=device)
    q_value_estimates = torch.zeros(num_updates, device=device)
    for j in range(num_updates):
        # Compute GAE
        with torch.no_grad():
            data["state_value"] = critic(graph)
            data["next", "state_value"] = critic(next_graph)
            data["advantage"], data["value_target"] = generalized_advantage_estimate(
                gamma = cfg.loss.gamma,
                lmbda = cfg.loss.gae_lambda,
                state_value = data["state_value"],
                next_state_value = data["next", "state_value"],
                reward = data["next", "reward"],
                done = data["done"],
                terminated = data["terminated"])
            # data = adv_module(data.to(device, non_blocking=True))
            value_estimates[j] = data["state_value"].mean()
            q_value_estimates[j] = data["value_target"].mean()
        data_reshape = data.reshape(-1)
        # Update the data buffer
        data_buffer.extend(data_reshape)
        for k, batch in enumerate(data_buffer):

            # Linearly decrease the learning rate and clip epsilon
            alpha = 1.0
            if cfg.optim.anneal_lr:
                alpha = 1 - (num_network_updates / total_network_updates)
                for group in optimizer.param_groups:
                    group["lr"] = cfg.optim.lr * alpha

            num_network_updates += 1

            batch["sample_log_prob"] = batch["sample_log_prob"]
            batch["advantage"] = batch["advantage"].expand(batch["sample_log_prob"].shape).unsqueeze(-1)
            batch = batch.to(device, non_blocking=True)

            # Forward pass PPO loss
            loss = loss_module(batch)
            losses[j, k] = loss.select(
                "loss_critic", "loss_entropy", "loss_objective", "entropy", "ESS"
            )
            loss_sum = (
                    loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
            )
            # Backward pass
            loss_sum.backward()
            # loss["loss_objective"].backward()
            # loss["loss_critic"].backward()
            # loss["loss_entropy"].backward()

            torch.nn.utils.clip_grad_norm_(
                list(loss_module.parameters()), max_norm=cfg.optim.max_grad_norm
            )

            # Update the networks
            optimizer.step()
            optimizer.zero_grad()
    pbar.set_description("Training")

    training_time = time.time() - training_start

    # Get and log q-values, loss, epsilon, sampling time and training time
    for key, value in loss.items():
        if key not in ["loss_critic", "loss_entropy", "loss_objective"]:
            log_info.update({f"train/{key}": value.mean().item()})
        else:
            log_info.update({f"train/{key}": value.sum().item()})
    log_info.update(
        {
            "train/lr": alpha * cfg.optim.lr,
            "train/sampling_time": sampling_time,
            "train/training_time": training_time,
            "train/q_values": q_value_estimates.mean().item(),
            "train/value_estimate": value_estimates.mean().item(),
        }
    )



    try: # try block in case wandb logging fails
        log_info["trainer/step"] = collected_frames
        wandb.log(log_info, step=collected_frames)
    except Exception as e:
        print(e)
    prev_log_info = deepcopy(log_info)
    collector.update_policy_weights_()
    sampling_start = time.time()

# Save the final agent model and configuration files
torch.save(agent.state_dict(), os.path.join(logger.experiment.dir, f"trained_actor_module.pt"))
agent_artifact = wandb.Artifact(f"trained_actor_module.pt_{artifact_name}", type="model")
agent_artifact.add_file(os.path.join(logger.experiment.dir, f"trained_actor_module.pt"))
agent_artifact.add_file(os.path.join(logger.experiment.dir, f"config.yaml"))
wandb.log_artifact(agent_artifact, aliases=["latest"])