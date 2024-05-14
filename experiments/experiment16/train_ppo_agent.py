# %%
from torchrl_development.envs.env_generators import EnvGenerator
from torchrl_development.utils.configuration import load_config
from torchrl_development.actors import create_maxweight_actor_critic, create_actor_critic
from torchrl_development.utils.configuration import smart_type
import argparse
import os
from datetime import datetime
from torchrl_development.envs.env_generators import parse_env_json
import torch
import json
from torchrl_development.envs.env_generators import make_env
from torchrl_development.actors import MaxWeightActor
from torchrl_development.utils.metrics import compute_lta
from torchrl.modules import MLP,QValueActor
from torchrl_development.actors import MinQValueActor
from torchrl.modules import EGreedyModule
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate
from torchrl.collectors import MultiaSyncDataCollector, MultiSyncDataCollector, SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
from tensordict import TensorDict
from torchrl.data import CompositeSpec
from tensordict.nn import TensorDictSequential
import tempfile
import wandb
from torchrl.record.loggers import get_logger, generate_exp_name
import time
import tqdm
import numpy as np
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
from torchrl_development.MultiEnvSyncDataCollector import MultiEnvSyncDataCollector
import matplotlib.pyplot as plt
from torchrl_development.utils.configuration import make_serializable
import monotonicnetworks as lmn
from torchrl_development.SMNN import PureMonotonicNeuralNetwork as PMN, MultiLayerPerceptron as MLP, ReLUUnit, ExpUnit, FCLayer_notexp,ReLUnUnit
from torchrl.envs import ParallelEnv
from torchrl.trainers.helpers import make_collector_onpolicy
from torchrl.collectors.utils import  split_trajectories
""" Script Description
This script is used to experiment with using DQN to solve for the optimal policy of SingleHop environments.
Its mostly based off the dqn_atary.py script from torchrl library


"""
# %%
def evaluate_dqn_agent(actor,
                       eval_env_generator,
                       training_envs_ind,
                       pbar,
                       cfg,
                       device):
    log_info = {}
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):

            actor.eval()
            print("actor device: ", actor.device)
            test_envs_ind = eval_env_generator.context_dicts.keys()
            eval_start = time.time()
            # update pbar to say that we are evaluating
            pbar.set_description("Evaluating")
            # Want to evaluate the policy on all environments from gen_env_generator
            lta_backlogs = {}
            valid_action_fractions = {}
            final_mean_vaf  = {}
            final_mean_lta_backlogs = {}
            normalized_final_mean_lta_backlogs = {}
            num_evals = 0
            eval_tds = []

            # Create Parallel Envs for Testing
            # make_env_funcs = [lambda i=i: eval_env_generator.sample(true_ind = i) for i in eval_env_generator.context_dicts.keys()]
            # # envs = ParallelEnv(num_workers=eval_env_generator.num_envs, create_env_fn=make_env_funcs)
            # # envs_rollout = envs.rollout(1000, actor, break_when_any_done=False, auto_reset = False)
            # # num_eval_envs = eval_env_generator.num_envs  # gen_env_generator.num_envs
            # # Create Async Collector
            # collector = MultiaSyncDataCollector(
            #     create_env_fn = make_env_funcs,
            #     policy = actor,
            #     total_frames=make_env_funcs.__len__()* 1000*3,
            #     device = device,
            #     frames_per_batch=make_env_funcs.__len__()* 1000*3,
            # )
            # for batch in collector:
            #     print("Here")

            for e, i in enumerate(eval_env_generator.context_dicts.keys()):  # i =1,2 are scaled, 3-6 are general, and 0 is the same as training
                pbar.set_description(f"Evaluating Environment {i} ({e+1}/{len(eval_env_generator.context_dicts.keys())})")
                lta_backlogs[i] = []
                valid_action_fractions[i] = []
                eval_env_generator.reseed()
                seeds = eval_env_generator.gen_seeds(cfg.eval.num_eval_envs)
                make_env_func = [lambda seed = seed: eval_env_generator.sample(true_ind=i, seed = seed) for seed in seeds]
                env = ParallelEnv(cfg.eval.num_eval_envs,
                                  make_env_func,
                                  # [make_env_func]*cfg.eval.num_eval_envs,
                                  device = device,)
                td = env.rollout(cfg.eval.traj_steps,
                                 actor,
                                 break_when_any_done = True,
                                 auto_cast_to_device=True).to("cpu")
                lta_backlogs[i] = [compute_lta(td["backlog"][i]) for i in range(cfg.eval.num_eval_envs)]
                valid_action_fractions[i] = [(td["mask"][i] * td["action"][i]).sum().float() / td["mask"][i].shape[0] for i in range(cfg.eval.num_eval_envs)]
                # print("Here")

                # for n in range(cfg.eval.num_eval_envs):
                #     # reset eval_env_generator
                #     num_evals += 1
                #     # update pbar to say that we are evaluating num_evals/gen_env_generator.num_envs*cfg.eval.num_eval_envs
                #     pbar.set_description(
                #         f"Evaluating {num_evals}/{eval_env_generator.num_envs * cfg.eval.num_eval_envs} eval environment")
                #     eval_env = eval_env_generator.sample(true_ind=i)
                #     eval_td = eval_env.rollout(cfg.eval.traj_steps, actor, auto_cast_to_device=True).to('cpu')
                #     eval_tds.append(eval_td)
                #     eval_backlog = eval_td["next", "backlog"].numpy()
                #     eval_lta_backlog = compute_lta(eval_backlog)
                #     vaf =  (eval_td["mask"] * eval_td["action"]).sum().float() / eval_td["mask"].shape[0]
                #     valid_action_fractions[i].append(vaf)
                #     lta_backlogs[i].append(eval_lta_backlog)
                final_mean_lta_backlogs[i] = np.mean([t[-1] for t in lta_backlogs[i]])
                # get MaxWeight LTA from gen_env_generator.context_dicts[i]["lta]
                max_weight_lta = eval_env_generator.context_dicts[i]["lta"]
                normalized_final_mean_lta_backlogs[i] = final_mean_lta_backlogs[i] / max_weight_lta
                final_mean_vaf[i] = np.mean(valid_action_fractions[i])
            eval_time = time.time() - eval_start

            # add individual final_mean_lta_backlogs to log_info
            for i, lta in final_mean_lta_backlogs.items():
                log_info.update({f"eval/lta_backlog_lambda({i})": lta})
            # add individual normalized_final_mean_lta_backlogs to log_info
            for i, lta in normalized_final_mean_lta_backlogs.items():
                log_info.update({f"eval_normalized/normalized_lta_backlog_lambda({i})": lta})

            # add individual final_mean_vaf to log_info
            for i, vaf in final_mean_vaf.items():
                log_info.update({f"eval/valid_action_fraction_lambda({i})": vaf})


            # log the performanec of the policy on the same environment
            # if all training inds are in test inds then we can do this
            if all([i in test_envs_ind for i in training_envs_ind]):
                training_env_lta_backlogs = np.mean([final_mean_lta_backlogs[i] for i in training_envs_ind])
                log_info.update({f"eval/lta_backlog_training_envs": training_env_lta_backlogs})
                # add the normalized lta backlog for the same environment
                normalized_training_mean_lta_backlogs = np.mean(
                    [normalized_final_mean_lta_backlogs[i] for i in training_envs_ind])
                log_info.update({
                    f"eval_normalized/normalized_lta_backlog_training_envs": normalized_training_mean_lta_backlogs})

            # log the performance of the policy on the non-training environments
            non_training_inds = [i for i in test_envs_ind if i not in training_envs_ind]
            general_lta_backlogs = np.mean([final_mean_lta_backlogs[i] for i in non_training_inds])
            log_info.update({"eval/lta_backlog_non_training_envs": general_lta_backlogs})
            # add the normalized lta backlog for the general environments
            normalized_general_lta_backlogs = np.mean(
                [normalized_final_mean_lta_backlogs[i] for i in non_training_inds])
            log_info.update(
                {"eval_normalized/normalized_lta_backlog_non_training_envs": normalized_general_lta_backlogs})

            # log the performance of the policy on all environments
            all_lta_backlogs = np.mean([final_mean_lta_backlogs[i] for i in test_envs_ind])
            log_info.update({"eval/lta_backlog_all_envs": all_lta_backlogs})
            # add the normalized lta backlog for all environments
            normalized_all_lta_backlogs = np.mean([normalized_final_mean_lta_backlogs[i] for i in test_envs_ind])
            log_info.update({"eval_normalized/normalized_lta_backlog_all_envs": normalized_all_lta_backlogs})

    return log_info, eval_tds


# %%

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

def train_ppo_agent(cfg, training_env_generator, eval_env_generator, device, logger = None, disable_pbar = False):


    base_env = training_env_generator.sample()
    training_env_generator.clear_history()
    # Create DQN Agent
    input_shape = base_env.observation_spec["observation"].shape
    output_shape = base_env.action_spec.space.n
    action_spec = base_env.action_spec

   # create actor and critic
    agent = create_actor_critic(
        input_shape,
        output_shape,
        in_keys=["observation"],
        action_spec=action_spec,
        temperature=cfg.agent.temperature,
        actor_depth=cfg.agent.hidden_sizes.__len__(),
        actor_cells=cfg.agent.hidden_sizes[-1],
    )
    actor = agent.get_policy_operator().to(device)
    critic = agent.get_value_operator().to(device)

    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
    )
    # Create the collector

    make_env_funcs = [lambda i=i: training_env_generator.sample(true_ind = i) for i in training_env_generator.context_dicts.keys()]
    # Get lta for each environment
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
        reset_when_done=True,
)

    # # create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.collector.frames_per_batch),
        sampler=sampler,
        batch_size=cfg.loss.mini_batch_size,  # amount of samples to be sampled when sample is called
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


    # Create the optimizer
    optimizer = torch.optim.Adam(
        loss_module.parameters(),
        lr= cfg.optim.lr,
    )

    # create wandb logger
    if logger is None:
        experiment_name = generate_exp_name(f"PPO_MLP", "Solo")
        logger = get_logger(
                "wandb",
                logger_name="..\\logs",
                experiment_name= experiment_name,
                wandb_kwargs={
                    "config": cfg.as_dict(),
                    "project": cfg.logger.project,
                },
            )
    #wandb.log({"init/init": True})
    #wandb.watch(q_module[0], log="parameters", log_freq=10)
    #wandb.watch(mono_nn, log="all", log_freq=100, log_graph=False)

    # # Save the cfg as a yaml file and upload to wandb
    # with open(os.path.join(logger.experiment.dir, cfg.context_set), "w") as file:
    #     json.dump(make_serializable(training_env_generator.context_dicts), file)
    # wandb.save(cfg.context_set)


    # Main loop
    collected_frames = 0
    start_time = time.time()
    sampling_start = time.time()
    num_updates = cfg.loss.num_updates
    max_grad = cfg.optim.max_grad_norm
    pol_losses = torch.zeros(num_updates, device=device)
    mask_losses = torch.zeros(num_updates, device = device)
    num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size

    total_network_updates = (
            (cfg.collector.total_frames // cfg.collector.frames_per_batch) * cfg.loss.num_updates * num_mini_batches
    )
    pbar = tqdm.tqdm(total=cfg.collector.total_frames, disable = disable_pbar)
    num_network_updates = 0
    # initialize the artifact saving params
    best_eval_backlog = np.inf
    artifact_name = logger.exp_name

    for i, data in enumerate(collector):

        # print(f"Device of data: {data.device}")
        log_info = {}
        sampling_time = time.time() - sampling_start
        data = data[data["collector", "mask"]]

        pbar.update(data.numel())
        current_frames = data.numel()
        collected_frames += current_frames
        # drop data if [collector, mask] is False

        #env_datas = split_trajectories(data)
        # Get and log training rewards and episode lengths
        #combine all data that has the same context_id
        # data = data[data["collector", "mask"]]
        for context_id in data["context_id"].unique():
            env_data = data[data["context_id"] == context_id]
            env_data = env_data[env_data["collector", "mask"]]

            # first check if all of the env_data is from the same environment
            # if not (env_data["collector", "traj_ids"] == env_data["collector", "traj_ids"][0]).all():
            #     raise ValueError("Data from multiple environments is being logged")
            context_id = env_data.get("context_id", None)
            if context_id is not None:
                context_id = context_id[0].item()
            baseline_lta = env_data.get("baseline_lta", None)
            if baseline_lta is not None:
                env_lta = baseline_lta[-1]
            mean_episode_reward = env_data["next", "reward"].mean()
            mean_backlog = env_data["next", "backlog"].float().mean()
            normalized_backlog = mean_backlog / env_lta
            valid_action_fraction = (env_data["mask"] * env_data["action"]).sum().float() / env_data["mask"].shape[0]
            log_header = f"train/context_id_{context_id}"

            log_info.update({f'{log_header}/mean_episode_reward': mean_episode_reward.item(),
                             f'{log_header}/mean_backlog': mean_backlog.item(),
                             f'{log_header}/mean_normalized_backlog': normalized_backlog.item(),
                             f'{log_header}/valid_action_fraction': valid_action_fraction.item(),})

        # Get average mean_normalized_backlog across each context
        avg_mean_normalized_backlog = np.mean([log_info[f'train/context_id_{i}/mean_normalized_backlog'] for i in training_env_generator.context_dicts.keys()])
        log_info.update({"train/avg_mean_normalized_backlog": avg_mean_normalized_backlog})
        # compute the fraction of times the chosen action was invalid
        """
        data["mask"] is a binary tensor for each possible action, where False means the action was invalid
        data["action"] is the action chosen by the agent which is a one-hot binary tensor
        data["mask"] * data["action"] will be a binary tensor where the action was invalid
        """
        data = data[data["collector", "mask"]]
        # optimization steps
        training_start = time.time()
        losses = TensorDict({}, batch_size=[cfg.loss.num_updates, num_mini_batches])
        value_estimates = torch.zeros(num_updates, device=device)
        q_value_estimates = torch.zeros(num_updates, device=device)
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
                # batch["action"] = batch["action"].squeeze()
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
                "train/q_values:": q_value_estimates.mean().item(),
                "train/value_estimate": value_estimates.mean().item(),
            }
        )


        # Get and log evaluation rewards and eval time
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            prev_test_frame = ((i - 1) * cfg.collector.frames_per_batch) // cfg.collector.test_interval
            cur_test_frame = (i * cfg.collector.frames_per_batch) // cfg.collector.test_interval
            final = current_frames >= collector.total_frames
            if (i >= 1 and (prev_test_frame < cur_test_frame)) or final:
                actor.eval()
                eval_start = time.time()
                training_env_ids = list(training_env_generator.context_dicts.keys())
                eval_log_info, eval_tds = evaluate_dqn_agent(actor, eval_env_generator, training_env_ids, pbar, cfg, device)

                eval_time = time.time() - eval_start
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
                actor.train()
        try:
            #for key, value in log_info.items():
                #logger.log(key, value, collected_frames)
            log_info["trainer/step"] = collected_frames
            wandb.log(log_info, step=collected_frames)
        except Exception as e:
            print(e)
        collector.update_policy_weights_()
        sampling_start = time.time()


