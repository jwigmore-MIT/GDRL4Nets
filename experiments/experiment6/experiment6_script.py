import json
from torchrl_development.envs.env_generator import create_scaled_lambda_generator, EnvGenerator
from torchrl_development.utils.configuration import load_config
from torchrl_development.actors import create_actor_critic
from datetime import datetime
from torchrl_development.utils.metrics import compute_lta
import numpy as np
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
import torch
from torchrl.record.loggers import get_logger
import os
import yaml
import wandb
import time
from tensordict import TensorDict
from tqdm import tqdm
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type

import argparse
# how to write this as a script that can be run from the command line and takes in an argument from the command line

parser = argparse.ArgumentParser(description='Run experiment 6')
parser.add_argument('--training_set', type=str, help='indices of the environments to train on', default="a")
#parser.add_argument('--test_envs_ind', nargs = '+', type=int, help='indices of the environments to test on', default=[4,5])
parser.add_argument('--env_json', type=str, help='json file that contains the set of environment context parameters', default="exp6_env_contexts.json")
parser.add_argument('--experiment_name', type=str, help='what the experiment will be titled for wandb', default="Experiment6a")

train_sets =  {"a": {"train": [4,6,8], "test": [3,5,9]},
               "b": {"train": [5,6,7], "test": [4,8,9]},
               "c": {"train": [6,8,9], "test": [3,4,5]},
               }

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_PATH, "experiment6.yaml")

args = parser.parse_args()
training_envs_ind = train_sets[args.training_set]["train"]
test_envs_ind = train_sets[args.training_set]["test"]

# add all training_envs_ind to test_envs_ind
test_envs_ind.extend(training_envs_ind)
env_json = args.env_json

#
cfg = load_config(full_path= CONFIG_PATH)
cfg.exp_name = f"Experiment6{args.training_set}_{datetime.now().strftime('%y_%m_%d-%H_%M_%S')}"
cfg.training_env.envs_ind = training_envs_ind
cfg.env_json = env_json

training_make_env_parameters = {"observe_lambda": cfg.agent.observe_lambda,
                   "device": cfg.device,
                   "terminal_backlog": cfg.training_env.terminal_backlog,
                   "inverse_reward": cfg.training_env.inverse_reward,
                   }

# creating eval env_generators
eval_make_env_parameters = {"observe_lambda": cfg.agent.observe_lambda,
                        "device": cfg.device,
                        "terminal_backlog": cfg.eval_envs.terminal_backlog,
                        "inverse_reward": cfg.eval_envs.inverse_reward,
                        }



env_context_dict = json.load(open(cfg.env_json, 'rb'))
all_env_params = env_context_dict["env_params"]
training_env_generator_input_params = {"num_envs": len(training_envs_ind),
                    "all_env_params": {str(i): all_env_params[str(i)] for i in training_envs_ind}}


eval_env_generator_input_params = {"num_envs": len(test_envs_ind),
            "all_env_params": {str(i): all_env_params[str(i)] for i in test_envs_ind}}

training_env_generator = EnvGenerator(training_env_generator_input_params,
                                training_make_env_parameters,
                                env_generator_seed=cfg.training_env.env_generator_seed)

eval_env_generator = EnvGenerator(eval_env_generator_input_params,
                                eval_make_env_parameters,
                                env_generator_seed=cfg.eval_envs.env_generator_seed)



# # Create base env for agent generation
base_env= training_env_generator.sample()
training_env_generator.clear_history()
check_env_specs(base_env)
#
# # Create agent
input_shape = base_env.observation_spec["observation"].shape
output_shape = base_env.action_spec.space.n
#
agent = create_actor_critic(
input_shape,
output_shape,
in_keys=["observation"],
action_spec=base_env.action_spec,
temperature=cfg.agent.temperature,
)

# Set device
device = cfg.device
# # Create Actor Critic Agent
base_env = training_env_generator.sample()
#
# # Specify actor and critic
actor = agent.get_policy_operator().to(device)
critic = agent.get_value_operator().to(device)
#
# # Create Generalized Advantage Estimation module
adv_module = GAE(
gamma=cfg.loss.gamma,
lmbda=cfg.loss.gae_lambda,
value_network=critic,
average_gae=False,
)
#
# # Create PPO loss module
loss_module = ClipPPOLoss(
actor_network=actor,
critic_network=critic,
clip_epsilon=cfg.loss.clip_epsilon,
entropy_coef=cfg.loss.entropy_coef,
normalize_advantage=cfg.loss.norm_advantage,
loss_critic_type="l2"
)
#
collector = SyncDataCollector(
policy=actor,
create_env_fn=training_env_generator.sample(),
frames_per_batch=cfg.collector.frames_per_batch,
total_frames=cfg.collector.total_training_frames,
device=device,
storing_device=device,
reset_at_each_iter=False,
)
#
# # create data buffer
sampler = SamplerWithoutReplacement()
data_buffer = TensorDictReplayBuffer(
storage=LazyMemmapStorage(cfg.collector.frames_per_batch),
sampler=sampler,
batch_size=cfg.loss.mini_batch_size, # amount of samples to be sampled when sample is called
)
#
# # Create optimizer
optim = torch.optim.Adam(
loss_module.parameters(),
lr=cfg.optim.lr,
weight_decay=cfg.optim.weight_decay,
eps=cfg.optim.eps,
)
#
# # Create logger
#
logger = get_logger(
"wandb",
logger_name="..\\logs",
experiment_name=getattr(cfg, "exp_name", None),
wandb_kwargs={
    "config": cfg.as_dict(),
    "project": cfg.logger.project_name,
},
)
# # Save the cfg as a yaml file and upload to wandb
cfg_dict = cfg.as_dict()
with open(os.path.join(logger.experiment.dir, "config.yaml"), "w") as file:
    yaml.dump(cfg_dict, file)
wandb.save("config.yaml")
# Save the env_json file
with open(os.path.join(logger.experiment.dir, cfg.env_json), "w") as file:
    json.dump(env_context_dict, file)
wandb.save(cfg.env_json) # this returns a WinError 1314 when running on windows

#wandb.watch(actor.module[0], log = "all", log_freq=10)
#
# # Main Loop
collected_frames = 0
num_network_updates = 0
sampling_start = time.time()

total_training_frames = cfg.collector.total_training_frames
pbar = tqdm(total=total_training_frames)


num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
total_network_updates = (
    (total_training_frames // cfg.collector.frames_per_batch) * cfg.loss.ppo_epochs * num_mini_batches
)




for i, data in enumerate(collector):
    #actor.train()
    log_info = {}
    sampling_time = time.time() - sampling_start
    frames_in_batch = data.numel()
    collected_frames += frames_in_batch * cfg.collector.frame_skip
    pbar.update(data.numel())
    # Training performance logging
    mean_episode_reward = data["next", "reward"].mean()
    mean_backlog = data["next","backlog"].float().mean()
    num_trajectories = data["done"].sum()
    log_info.update(
        {
            "train/mean_episode_reward": mean_episode_reward.item(),
            "train/mean_backlog": mean_backlog.item(),
            "train/num_trajectories": num_trajectories.item(),
        }
    )
    # Save the policy if it has the best mean_episode_reward


    training_start = time.time()
    for j in range(cfg.loss.ppo_epochs):

        # Compute GAE
        with torch.no_grad():
            data = adv_module(data.to(device, non_blocking=True))
        data_reshape = data.reshape(-1)
        # Update the data buffer
        data_buffer.extend(data_reshape)
        losses = TensorDict({}, batch_size=[cfg.loss.ppo_epochs, num_mini_batches])

        for k, batch in enumerate(data_buffer):
            # Linearly decrease the learning rate and clip epsilon
            alpha = 1.0
            if cfg.optim.anneal_lr:
                alpha = 1 - (num_network_updates / total_network_updates)
                for group in optim.param_groups:
                    group["lr"] = cfg.optim.lr * alpha


            num_network_updates += 1
            # Get a data batch
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

            ## Collect all gradient information if debugging
            # for name, param in loss_module.named_parameters():
            #     if param.grad is None:
            #         print(f"None gradient in actor {name}")
            #     else:
            #         print(f"{name} gradient: {param.grad.mean()}")
            torch.nn.utils.clip_grad_norm_(
                list(loss_module.parameters()), max_norm=cfg.optim.max_grad_norm
            )

            # Update the networks
            optim.step()
            optim.zero_grad()

    # Get training losses and times
    training_time = time.time() - training_start
    losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
    for key, value in losses_mean.items():
        log_info.update({f"train/{key}": value.item()})
    log_info.update(
        {
            "train/lr": alpha*cfg.optim.lr,
            "train/sampling_time": sampling_time,
            "train/training_time": training_time,
        }
    )
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        if ((
                    i - 1) * frames_in_batch * cfg.collector.frame_skip) // cfg.eval.eval_interval < (
                i * frames_in_batch * cfg.collector.frame_skip
        ) // cfg.eval.eval_interval:
            actor.eval()
            eval_start = time.time()
            # update pbar to say that we are evaluating
            pbar.set_description("Evaluating")
            # Want to evaluate the policy on all environments from gen_env_generator
            lta_backlogs = {}
            final_mean_lta_backlogs = {}
            normalized_final_mean_lta_backlogs = {}
            num_evals = 0
            num_eval_envs = eval_env_generator.num_envs # gen_env_generator.num_envs
            for i in eval_env_generator.env_parameters.keys(): # i =1,2 are scaled, 3-6 are general, and 0 is the same as training
                lta_backlogs[i] = []
                for n in range(cfg.eval.num_eval_envs):
                    num_evals+= 1
                    # update pbar to say that we are evaluating num_evals/gen_env_generator.num_envs*cfg.eval.num_eval_envs
                    pbar.set_description(f"Evaluating {num_evals}/{eval_env_generator.num_envs*cfg.eval.num_eval_envs} eval environment")
                    eval_env = eval_env_generator.sample(true_ind = i)
                    eval_td = eval_env.rollout(cfg.eval.traj_steps, actor)
                    eval_backlog = eval_td["next", "backlog"].numpy()
                    eval_lta_backlog = compute_lta(eval_backlog)
                    lta_backlogs[i].append(eval_lta_backlog)
                final_mean_lta_backlogs[i] = np.mean([t[-1] for t in lta_backlogs[i]])
                # get MaxWeight LTA from gen_env_generator.env_parameters[i]["lta]
                max_weight_lta = eval_env_generator.env_parameters[i]["lta"]
                normalized_final_mean_lta_backlogs[i] = final_mean_lta_backlogs[i] / max_weight_lta
            eval_time = time.time() - eval_start

            # add individual final_mean_lta_backlogs to log_info
            for i, lta in final_mean_lta_backlogs.items():
                log_info.update({f"eval/lta_backlog_lambda({i})": lta})
            # add individual normalized_final_mean_lta_backlogs to log_info
            for i, lta in normalized_final_mean_lta_backlogs.items():
                log_info.update({f"eval_normalized/normalized_lta_backlog_lambda({i})": lta})


            # log the performanec of the policy on the same environment
            # if all training inds are in test inds then we can do this
            if all([i in test_envs_ind for i in training_envs_ind]):
                training_env_lta_backlogs = np.mean([final_mean_lta_backlogs[i] for i in training_envs_ind])
                log_info.update({f"eval/lta_backlog_training_envs": training_env_lta_backlogs})
                # add the normalized lta backlog for the same environment
                normalized_training_mean_lta_backlogs = np.mean([normalized_final_mean_lta_backlogs[i] for i in training_envs_ind])
                log_info.update({f"eval_normalized/normalized_lta_backlog_training_envs": normalized_training_mean_lta_backlogs})


            # log the performance of the policy on the non-training environments
            non_training_inds = [i for i in test_envs_ind if i not in training_envs_ind]
            general_lta_backlogs = np.mean([final_mean_lta_backlogs[i] for i in non_training_inds])
            log_info.update({"eval/lta_backlog_non_training_envs": general_lta_backlogs})
            # add the normalized lta backlog for the general environments
            normalized_general_lta_backlogs = np.mean([normalized_final_mean_lta_backlogs[i] for i in non_training_inds])
            log_info.update({"eval_normalized/normalized_lta_backlog_non_training_envs": normalized_general_lta_backlogs})

            # log the performance of the policy on all environments
            all_lta_backlogs = np.mean([final_mean_lta_backlogs[i] for i in test_envs_ind])
            log_info.update({"eval/lta_backlog_all_envs": all_lta_backlogs})
            # add the normalized lta backlog for all environments
            normalized_all_lta_backlogs = np.mean([normalized_final_mean_lta_backlogs[i] for i in test_envs_ind])
            log_info.update({"eval_normalized/normalized_lta_backlog_all_envs": normalized_all_lta_backlogs})


            # Save the current agent, as model_{training_steps}
            torch.save(agent.state_dict(), os.path.join(logger.experiment.dir, f"model_{int(collected_frames)}.pt"))
            wandb.save(f"model_{int(collected_frames)}.pt")
            actor.train()
            # update pbar to say that we are training
            pbar.set_description("Training")





    #
            #
            # # Plot all lta backlogs
            # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            # for lta_backlog in test_lta_backlogs:
            #     ax.plot(lta_backlog)
            # # plot a horizontal line at the mean of the the test_baseline_ltas
            # ax.axhline(y=np.mean(test_baseline_ltas), color='r', linestyle='--')
            # ax.set_ylabel("Test Backlog")
            # ax.set_xlabel("Time")
            # ax.set(title = f"Eval on training step {collected_frames}")
            # #ax.legend()
            # wandb.log({f"eval_plots/training_step {collected_frames}": wandb.Image(fig)})
            # # plt.show()
            # plt.close(fig)
    if logger:
        for key, value in log_info.items():
            logger.log_scalar(key, value, collected_frames)

    collector.update_policy_weights_()
    sampling_start = time.time()
