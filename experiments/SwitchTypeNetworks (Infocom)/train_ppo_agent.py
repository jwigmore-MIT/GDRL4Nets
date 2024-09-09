# %%
import os
import pandas as pd
from copy import deepcopy
from tensordict import TensorDict
import torch
import time
import tqdm
import numpy as np
import wandb


from torchrl.collectors import MultiSyncDataCollector
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
from torchrl.record.loggers import get_logger, generate_exp_name
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss


from modules.torchrl_development.agents.actors import create_actor_critic
from modules.torchrl_development.agents.actors import create_independent_actor_critic

from experiment_utils import evaluate_agent

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
LOGGING_PATH = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_PATH)), "logs")

""" Script Description
This script is used to train a PPO agent on a set of environments. The script will train the agent on the specified environments and log the results to wandb.
"""



def train_ppo_agent(cfg, training_env_generator, eval_env_generator, device, logger = None, disable_pbar = False):

    # Get common input shape and output shape for all environments for the training_env_generator
    base_env = training_env_generator.sample()
    training_env_generator.clear_history()
    # Create DQN Agent
    input_shape = base_env.observation_spec["observation"].shape
    output_shape = base_env.action_spec.space.n
    action_spec = base_env.action_spec
    N = int(base_env.base_env.N) # number of nodes
    D = int(input_shape[0]//N) # number of features (e.g. queue length, link state, arrival rate, link_rate) per node

   # create actor and critic
    if getattr(cfg.agent, "actor_type", "MLP") == "MLP":
        agent = create_actor_critic(
            input_shape,
            output_shape,
            in_keys=["observation"],
            action_spec=action_spec,
            temperature=cfg.agent.temperature,
            actor_depth=cfg.agent.hidden_sizes.__len__(),
            actor_cells=cfg.agent.hidden_sizes[-1],
        )
    elif getattr(cfg.agent, "actor_type", "MLP") == "STN_shared":
        agent = create_independent_actor_critic(number_nodes=N,
                                                actor_input_dimension=D,
                                                actor_in_keys=["Q", "Y", "lambda", "mu"],
                                                critic_in_keys=["observation"],
                                                action_spec=action_spec,
                                                temperature=cfg.agent.temperature,
                                                actor_depth=cfg.agent.hidden_sizes.__len__(),
                                                actor_cells=cfg.agent.hidden_sizes[-1],
                                                type=3,
                                                network_type="STN",
                                                relu_max=getattr(cfg, "relu_max", 10),
                                                add_zero = base_env.base_env.__str__() != 'MultipathRouting()')



    actor = agent.get_policy_operator().to(device)
    critic = agent.get_value_operator().to(device)

    # Create the GAE module
    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
    )

    ## Create the collector
    make_env_funcs = [lambda i=i: training_env_generator.sample(true_ind = i) for i in training_env_generator.context_dicts.keys()]
    # Get lta backlog of MaxWeight policy for each environment
    training_env_info = {}
    for (e,i) in enumerate(training_env_generator.context_dicts.keys()):
        training_env_info[e] = {"lta": training_env_generator.context_dicts[i]["lta"],
                                "context_id": i}


    collector = MultiSyncDataCollector(
        create_env_fn= make_env_funcs,
        policy=agent.get_policy_operator(),
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        storing_device=device,
        env_device="cpu",
        max_frames_per_traj=cfg.collector.max_frames_per_traj,
        split_trajs=True,
        reset_when_done=True,)



    ## create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.collector.frames_per_batch),
        sampler=sampler,
        batch_size=cfg.loss.mini_batch_size,  # amount of samples to be sampled when sample is called
    )

    # TODO: is this used?
    long_term_data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.collector.map_interval),
        sampler=sampler,
        batch_size=cfg.collector.map_interval,  # amount of samples to be sampled when sample is called
    )

    ## Create PPO loss module
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coef=cfg.loss.entropy_coef,
        normalize_advantage=cfg.loss.norm_advantage,
        loss_critic_type="l2"
    )


    ## Create the optimizer
    optimizer = torch.optim.Adam(
        loss_module.parameters(),
        lr= cfg.optim.lr,
        weight_decay=0,
    )

    ## create wandb logger
    if logger is None:
        experiment_name = generate_exp_name(f"PPO_MLP", "Solo")
        logger = get_logger(
                "wandb",
                logger_name="..\\wandb",
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
    pbar = tqdm.tqdm(total=cfg.collector.total_frames, disable = disable_pbar)
    num_network_updates = 0

    # initialize the artifact saving params
    best_eval_backlog = np.inf
    artifact_name = logger.exp_name


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
                training_env_ids = list(training_env_generator.context_dicts.keys())
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
        if data["logits"].dim() > 2:
            data["logits"] = data["logits"].squeeze()
        if data["action"].dim() > 2:
            data["action"] = data["action"].squeeze()

        # loop through data from each training environment to track the per/env metrics
        for context_id in data["context_id"].unique():
            env_data = data[data["context_id"] == context_id]
            env_data = env_data[env_data["collector", "mask"]]

            context_id = env_data.get("context_id", None)
            if context_id is not None:
                context_id = context_id[0].item()
            baseline_lta = env_data.get("baseline_lta", None)
            if baseline_lta is not None:
                env_lta = baseline_lta[-1]
            mean_episode_reward = env_data["next", "reward"].mean()
            mean_backlog = env_data["next", "ta_mean"][-1]
            std_backlog = env_data["next", "ta_stdev"][-1]
            # mean_backlog = env_data["next", "backlog"].float().mean()
            normalized_backlog = mean_backlog / env_lta
            valid_action_fraction = (env_data["mask"] * env_data["action"]).sum().float() / env_data["mask"].shape[0]
            log_header = f"train/context_id_{context_id}"

            log_info.update({f'{log_header}/mean_episode_reward': mean_episode_reward.item(),
                             f'{log_header}/mean_backlog': mean_backlog.item(),
                             f'{log_header}/std_backlog': std_backlog.item(),
                             f'{log_header}/mean_normalized_backlog': normalized_backlog.item(),
                             f'{log_header}/valid_action_fraction': valid_action_fraction.item(), })



        # Get average mean_normalized_backlog across each context
        avg_mean_normalized_backlog = np.mean([log_info[f'train/context_id_{i}/mean_normalized_backlog'] for i in training_env_generator.context_dicts.keys()])
        log_info.update({"train/avg_mean_normalized_backlog": avg_mean_normalized_backlog})

        # only keep the data that is valid
        data = data[data["collector", "mask"]]

        # optimization steps
        training_start = time.time()
        losses = TensorDict({}, batch_size=[cfg.loss.num_updates, num_mini_batches])
        value_estimates = torch.zeros(num_updates, device=device)
        q_value_estimates = torch.zeros(num_updates, device=device)
        long_term_data_buffer.extend(data)
        for j in range(num_updates):
            # Compute GAE
            with torch.no_grad():
                data = adv_module(data.to(device, non_blocking=True))
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

                batch["sample_log_prob"] = batch["sample_log_prob"].squeeze()
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


