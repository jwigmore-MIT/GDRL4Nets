from tqdm import tqdm
import time
from tensordict import TensorDict
import numpy as np
import torch
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
import yaml
import wandb
import os
import json
from torchrl_development.MultiEnvSyncDataCollector import MultiEnvSyncDataCollector
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
from torchrl.record.loggers import get_logger
from torchrl_development.utils.configuration import make_serializable
from torchrl_development.utils.metrics import compute_lta

from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type

def evaluate_ppo_agent(actor,
                       eval_env_generator,
                       training_envs_ind,
                       pbar,
                       cfg,
                       device="cpu"):
    log_info = {}
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):

            actor.eval()
            test_envs_ind = eval_env_generator.context_dicts.keys()
            eval_start = time.time()
            # update pbar to say that we are evaluating
            pbar.set_description("Evaluating")
            # Want to evaluate the policy on all environments from gen_env_generator
            lta_backlogs = {}
            final_mean_lta_backlogs = {}
            normalized_final_mean_lta_backlogs = {}
            num_evals = 0
            num_eval_envs = eval_env_generator.num_envs  # gen_env_generator.num_envs
            for i in eval_env_generator.context_dicts.keys():  # i =1,2 are scaled, 3-6 are general, and 0 is the same as training
                lta_backlogs[i] = []
                eval_env_generator.reseed()
                for n in range(cfg.eval.num_eval_envs):
                    # reset eval_env_generator
                    num_evals += 1
                    # update pbar to say that we are evaluating num_evals/gen_env_generator.num_envs*cfg.eval.num_eval_envs
                    pbar.set_description(
                        f"Evaluating {num_evals}/{eval_env_generator.num_envs * cfg.eval.num_eval_envs} eval environment")
                    eval_env = eval_env_generator.sample(true_ind=i)
                    eval_td = eval_env.rollout(cfg.eval.traj_steps, actor)
                    eval_backlog = eval_td["next", "backlog"].numpy()
                    eval_lta_backlog = compute_lta(eval_backlog)
                    lta_backlogs[i].append(eval_lta_backlog)
                final_mean_lta_backlogs[i] = np.mean([t[-1] for t in lta_backlogs[i]])
                # get MaxWeight LTA from gen_env_generator.context_dicts[i]["lta]
                max_weight_lta = eval_env_generator.context_dicts[i]["lta"]
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

    return log_info

def train_ppo_agent(agent,
                    env_generator,
                    eval_env_generator,
                    cfg,
                    device = "cpu"):

    # Create PPO Modules
    ## First get actor and critic from agent
    actor = agent.get_policy_operator().to(device)
    critic = agent.get_value_operator().to(device)
    ## Create Generalized Advantage Estimation module
    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
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

    collector = MultiEnvSyncDataCollector(
        policy=actor,
        create_env_fn=env_generator.sample(),
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_training_frames,
        device=device,
        storing_device=device,
        reset_at_each_iter=cfg.collector.reset_at_each_iter,
        env_generator=env_generator.sample,

    )
    #
    # # create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.collector.frames_per_batch),
        sampler=sampler,
        batch_size=cfg.loss.mini_batch_size,  # amount of samples to be sampled when sample is called
    )
    #
    # Create optimizer
    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.eps,
    )

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
    with open(os.path.join(logger.experiment.dir, cfg.context_set), "w") as file:
        json.dump(make_serializable(env_generator.context_dicts), file)
    wandb.save(cfg.context_set)  # this returns a WinError 1314 when running on windows

    # Initialize counters and limits
    total_training_frames = cfg.collector.total_training_frames
    collected_frames = 0
    num_network_updates = 0
    sampling_start = time.time()
    num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
    total_network_updates = (
            (total_training_frames // cfg.collector.frames_per_batch) * cfg.loss.ppo_epochs * num_mini_batches
    )

    pbar = tqdm(total=total_training_frames)

    for i, data in enumerate(collector):
        # actor.train()
        training_env_id = env_generator.history[-1]
        log_info = {}
        if i == 0 and cfg.eval.evaluate_before_training:
            pbar.set_description("Evaluating Agent before Training")
            eval_log_info = (evaluate_ppo_agent(agent,
                                                eval_env_generator,
                                                [training_env_id],
                                                pbar,
                                                cfg,
                                                device=device))
            log_info.update(eval_log_info)
        pbar.set_description(f"Finished Training rollout out env id {training_env_id}")
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch * cfg.collector.frame_skip
        pbar.update(data.numel())
        # Training performance logging
        mean_episode_reward = data["next", "reward"].mean()
        mean_backlog = data["next", "backlog"].float().mean()
        num_trajectories = data["done"].sum()
        normalized_backlog = mean_backlog / env_generator.context_dicts[training_env_id]["lta"]
        log_info.update(
            {
                "train/mean_episode_reward": mean_episode_reward.item(),
                "train/mean_backlog": mean_backlog.item(),
                "train/mean_normalized_backlog": normalized_backlog.item(),
                "train/num_trajectories": num_trajectories.item(),
                "train/training_env_id": training_env_id,
            }
        )
        # Save the policy if it has the best mean_episode_reward

        training_start = time.time()
        losses = TensorDict({}, batch_size=[cfg.loss.ppo_epochs, num_mini_batches])

        for j in range(cfg.loss.ppo_epochs):

            # Compute GAE
            with torch.no_grad():
                data = adv_module(data.to(device, non_blocking=True))
            data_reshape = data.reshape(-1)
            # Update the data buffer
            data_buffer.extend(data_reshape)

            for k, batch in enumerate(data_buffer):
                # Linearly decrease the learning rate and clip epsilon
                alpha = 1.0
                if cfg.optim.anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in optim.param_groups:
                        group["lr"] = cfg.optim.lr * alpha

                num_network_updates += 1

                batch["sample_log_prob"] = batch["sample_log_prob"].squeeze()

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
        # losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
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
            }
        )

        # get the weights from the actor
        if actor.module[0].module.__str__() == "MaxWeightNetwork()":
            actor_weights = actor.module[0].module.get_weights()
            for e, value in enumerate(actor_weights):
                log_info.update({f"actor_weights/{e}": value.item()})

        prev_frames_processed = (i - 1) * frames_in_batch * cfg.collector.frame_skip
        curr_frames_processed = i * frames_in_batch * cfg.collector.frame_skip
        if i > 0 and (prev_frames_processed // cfg.eval.eval_interval) < (curr_frames_processed // cfg.eval.eval_interval):
            eval_log_info = (evaluate_ppo_agent(agent,
                                                eval_env_generator,
                                                [training_env_id],
                                                pbar,
                                                cfg,
                                                device=device))
            log_info.update(eval_log_info)

            # Save the current agent, as model_{training_steps}
            torch.save(agent.state_dict(), os.path.join(logger.experiment.dir, f"trained_agent.pt"))
            agent.training_steps = collected_frames
            wandb.save(f"trained_agent.pt")
            actor.train()
            # update pbar to say that we are training
            pbar.set_description("Training")

        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)

        collector.update_policy_weights_()
        sampling_start = time.time()
    return agent