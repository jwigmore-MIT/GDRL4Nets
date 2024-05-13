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
from torchrl.data import CompositeSpec
from tensordict.nn import TensorDictSequential
import tempfile
import wandb
from torchrl.record.loggers import get_logger, generate_exp_name
import time
import tqdm
import numpy as np
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

            num_eval_envs = eval_env_generator.num_envs  # gen_env_generator.num_envs
            for i in eval_env_generator.context_dicts.keys():  # i =1,2 are scaled, 3-6 are general, and 0 is the same as training
                lta_backlogs[i] = []
                valid_action_fractions[i] = []
                eval_env_generator.reseed()
                for n in range(cfg.eval.num_eval_envs):
                    # reset eval_env_generator
                    num_evals += 1
                    # update pbar to say that we are evaluating num_evals/gen_env_generator.num_envs*cfg.eval.num_eval_envs
                    pbar.set_description(
                        f"Evaluating {num_evals}/{eval_env_generator.num_envs * cfg.eval.num_eval_envs} eval environment")
                    eval_env = eval_env_generator.sample(true_ind=i)
                    eval_td = eval_env.rollout(cfg.eval.traj_steps, actor, auto_cast_to_device=True).to('cpu')
                    eval_tds.append(eval_td)
                    eval_backlog = eval_td["next", "backlog"].numpy()
                    eval_lta_backlog = compute_lta(eval_backlog)
                    vaf =  (eval_td["mask"] * eval_td["action"]).sum().float() / eval_td["mask"].shape[0]
                    valid_action_fractions[i].append(vaf)
                    lta_backlogs[i].append(eval_lta_backlog)
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

def train_mono_dqn_agent(cfg, training_env_generator, eval_env_generator, device, logger = None, disable_pbar = False):


    base_env = training_env_generator.sample()
    training_env_generator.clear_history()
    # Create DQN Agent
    input_shape = base_env.observation_spec["observation"].shape
    output_shape = base_env.action_spec.space.n
    action_spec = base_env.action_spec

    if cfg.agent.type == "LMN":

        lip_nn = torch.nn.Sequential(
            lmn.LipschitzLinear(input_shape[0], 32, kind="one", lipschitz_const = 10),
            lmn.GroupSort(2),
            lmn.LipschitzLinear(32, 32, kind="one", lipschitz_const = 10),
            lmn.GroupSort(2),
            lmn.LipschitzLinear(32, output_shape, kind="one", lipschitz_const = 10),
        )
        mono_nn = lmn.MonotonicWrapper(lip_nn, monotonic_constraints=[-1,-1,1,1])
    elif cfg.agent.type == "PMN":
        mono_nn = PMN(input_size  = input_shape[0],
                      output_size = output_shape,
                      hidden_sizes = cfg.agent.hidden_sizes,
                      relu_max = getattr(cfg, "relu_max", 1),
                      )
    elif cfg.agent.type == "PMN_RELU":
        mono_nn = PMN(input_size  = input_shape[0],
                      output_size = output_shape,
                      hidden_sizes = cfg.agent.hidden_sizes,
                      exp_unit = ReLUUnit,
                      fc_layer = FCLayer_notexp,)
    elif cfg.agent.type == "MLP":
        mono_nn = MLP(input_size  = input_shape[0],
                      output_size = output_shape,
                      hidden_size = cfg.agent.hidden_sizes)
    else:

        Exception("Invalid agent type")



    q_module = MinQValueActor(module = mono_nn,
                           in_keys = ["observation"],
                           spec = CompositeSpec({"action": action_spec}),
                           action_mask_key = "mask" if getattr(cfg.agent, "mask", False) else None,)

    # Create Exploration Module
    greedy_module = EGreedyModule(
        annealing_num_steps= cfg.collector.annealing_frames,
        eps_init= cfg.collector.eps_init,
        eps_end= cfg.collector.eps_end,
        spec = q_module.spec,
        action_mask_key = "mask" if getattr(cfg.agent, "mask", False) else None,
    )

    model_explore = TensorDictSequential(
        q_module,
        greedy_module,).to(device)

    # Create the collector

    make_env_funcs = [lambda i=i: training_env_generator.sample(true_ind = i) for i in training_env_generator.context_dicts.keys()]
    # Get lta for each environment
    training_env_info = {}
    for (e,i) in enumerate(training_env_generator.context_dicts.keys()):
        training_env_info[e] = {"lta": training_env_generator.context_dicts[i]["lta"],
                                "context_id": i}


    collector = MultiSyncDataCollector(
        create_env_fn= make_env_funcs,
        policy=model_explore,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        storing_device=device,
        env_device="cpu",
        max_frames_per_traj=cfg.collector.max_frames_per_traj,
        split_trajs=True,
)

    tempdir = tempfile.TemporaryDirectory()
    scratch_dir = tempdir.name

    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=3,
        storage=LazyMemmapStorage(
            max_size=cfg.buffer.buffer_size,
            scratch_dir=scratch_dir,
            #device = device
        ),
        batch_size=cfg.buffer.batch_size,
    )

    loss_module = DQNLoss(
            value_network=q_module,
            loss_function="l2",
            delay_value=True,
            double_dqn=True,
        )
    loss_module.make_value_estimator(gamma=cfg.loss.gamma)
    target_net_updater = SoftUpdate(
        loss_module, eps = cfg.loss.soft_eps)
    # target_net_updater = HardUpdate(
    #     loss_module, value_network_update_interval=cfg.loss.hard_update_freq
    # )

    # Create the optimizer
    optimizer = torch.optim.Adam(
        model_explore.parameters(),
        lr= cfg.optim.lr,
    )

    # create wandb logger
    if logger is None:
        experiment_name = generate_exp_name(f"DQN_{cfg.agent.type}", "Solo")
        logger = get_logger(
                "wandb",
                logger_name="..\\logs",
                experiment_name= experiment_name,
                wandb_kwargs={
                    "config": cfg.as_dict(),
                    "project": cfg.logger.project,
                },
            )
    wandb.log({"init/init": True})
    #wandb.watch(q_module[0], log="parameters", log_freq=10)
    wandb.watch(mono_nn, log="all", log_freq=100, log_graph=False)

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
    q_losses = torch.zeros(num_updates, device=device)
    mask_losses = torch.zeros(num_updates, device = device)
    pbar = tqdm.tqdm(total=cfg.collector.total_frames, disable = disable_pbar)

    # initialize the artifact saving params
    best_eval_backlog = np.inf
    artifact_name = logger.exp_name

    for i, data in enumerate(collector):

        # print(f"Device of data: {data.device}")
        log_info = {}
        sampling_time = time.time() - sampling_start
        pbar.update(data.numel())
        current_frames = data.numel()
        collected_frames += current_frames
        greedy_module.step(current_frames)
        # drop data if [collector, mask] is False

        #env_datas = split_trajectories(data)
        # Get and log training rewards and episode lengths
        for env_data in data:

            env_data = env_data[env_data["collector", "mask"]]

            # first check if all of the env_data is from the same environment
            if not (env_data["collector", "traj_ids"] == env_data["collector", "traj_ids"][0]).all():
                raise ValueError("Data from multiple environments is being logged")
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
        data = data.reshape(-1)
        replay_buffer.extend(data)
        # optimization steps
        training_start = time.time()
        for j in range(num_updates):
            sampled_tensordict = replay_buffer.sample()
            sampled_tensordict = sampled_tensordict.to(device)

            loss_td = loss_module(sampled_tensordict)
            if cfg.agent.mask:
                mask_loss = torch.zeros_like(loss_td["loss"])
            else:
                mask_loss = cfg.loss.mask_loss*(sampled_tensordict["action"]*sampled_tensordict["mask"].logical_not()  * q_module(sampled_tensordict["observation"])[1]).sum()

            q_loss = loss_td["loss"] + mask_loss
            # Add loss for invalid actions

            optimizer.zero_grad()
            q_loss.backward()
            if max_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(loss_module.parameters()), max_norm=max_grad
                )
            optimizer.step()
            target_net_updater.step()
            q_losses[j].copy_(loss_td["loss"].detach())
            mask_losses[j].copy_(mask_loss.detach())
        pbar.set_description("Training")

        training_time = time.time() - training_start

        # Get and log q-values, loss, epsilon, sampling time and training time
        log_info.update(
            {
                "train/q_values": (data["action_value"] * data["action"]).sum().item(),
                "train/mask_loss": mask_losses.mean().item(),
                "train/q_loss": q_losses.mean().item(),
                "train/epsilon": greedy_module.eps,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
            }
        )


        # Get and log evaluation rewards and eval time
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            prev_test_frame = ((i - 1) * cfg.collector.frames_per_batch) // cfg.collector.test_interval
            cur_test_frame = (i * cfg.collector.frames_per_batch) // cfg.collector.test_interval
            final = current_frames >= collector.total_frames
            if (i >= 1 and (prev_test_frame < cur_test_frame)) or final:
                q_module.eval()
                eval_start = time.time()
                training_env_ids = list(training_env_generator.context_dicts.keys())
                eval_log_info, eval_tds = evaluate_dqn_agent(q_module, eval_env_generator, training_env_ids, pbar, cfg, device)

                eval_time = time.time() - eval_start
                log_info.update(eval_log_info)

                # Save the agent if the eval backlog is the best
                if eval_log_info["eval/lta_backlog_training_envs"] < best_eval_backlog:
                    best_eval_backlog = eval_log_info["eval/lta_backlog_training_envs"]
                    torch.save(q_module.state_dict(), os.path.join(logger.experiment.dir, f"trained_q_module.pt"))
                    agent_artifact = wandb.Artifact(f"trained_q_module_{artifact_name}", type="model")
                    agent_artifact.add_file(os.path.join(logger.experiment.dir, f"trained_q_module.pt"))
                    agent_artifact.add_file(os.path.join(logger.experiment.dir, f"config.yaml"))
                    wandb.log_artifact(agent_artifact, aliases=["best", "latest"])
                else:
                    torch.save(q_module.state_dict(), os.path.join(logger.experiment.dir, f"trained_q_module.pt"))
                    agent_artifact = wandb.Artifact(f"trained_q_module.pt_{artifact_name}", type="model")
                    agent_artifact.add_file(os.path.join(logger.experiment.dir, f"trained_q_module.pt"))
                    agent_artifact.add_file(os.path.join(logger.experiment.dir, f"config.yaml"))
                    wandb.log_artifact(agent_artifact, aliases=["latest"])
                q_module.train()
        try:
            #for key, value in log_info.items():
                #logger.log(key, value, collected_frames)
            log_info["trainer/step"] = collected_frames
            wandb.log(log_info, step=collected_frames)
        except Exception as e:
            print(e)
        collector.update_policy_weights_()
        sampling_start = time.time()


