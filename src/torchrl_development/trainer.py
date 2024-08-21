import matplotlib.pyplot as plt
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
import os
import yaml
from torchrl_development.utils.configuration import load_config, create_config_from_dict
import numpy as np
from torchrl_development.utils.metrics import compute_lta
import torch
from torchrl.objectives.value import GAE
from torchrl.record.loggers import get_logger
import time
from tqdm import tqdm
from tensordict import TensorDict
import wandb
from copy import deepcopy
from torchrl_development.actors import create_actor_critic
from torchrl_development.maxweight import MaxWeightActor



def eval_model_multi_env(actor, test_envs, test_steps, plot_backlog=False):
    test_rewards = []
    test_final_backlogs = []
    test_backlogs = []
    for n, env_dict in test_envs.items():
        test_env = env_dict["env"]
        td_test = test_env.rollout(
            policy=actor,
            max_steps= test_steps,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        # get backlog as a torch.float32 tensor
        backlog = td_test["next", "backlog"][td_test["next", "done"]].float()
        test_rewards.append(reward.cpu())
        test_final_backlogs.append(backlog.cpu())
        test_backlogs.append(td_test["next", "backlog"].float().cpu())
    # TODO: Log actual backlog vector as plot... should probably not do it too frequently
    # if plot_backlog:
    #     # convert test_backlogs to a single tensor
    #     backlog_tf = torch.cat(test_backlogs, 1)
    #     # plot mean of test backlogs
    #     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    #     ax.plot(backlog_tf.mean(1))
    #     ax.set_ylabel("Test Backlog")
    #     ax.set_xlabel("Time")
    # else:
    #     fig = None


    del td_test
    return torch.cat(test_rewards, 0), torch.cat(test_final_backlogs, 0), test_backlogs









def get_stable_rollout(env, policy, max_steps):
    td = env.rollout(policy=policy, max_steps=max_steps)
    return td


def train(agent,
          training_generator,
          test_generator,
          cfg):

    # Set device
    device = cfg.device


    # Get MaxWeight Baseline
    max_weight_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
    print("Generating MaxWeight Baseline")
    max_weight_ltas = []
    fig, ax = plt.subplots()
    for i in range(3):
        test_env = test_generator.sample()
        td = test_env.rollout(policy=max_weight_actor, max_steps=cfg.eval.traj_steps)
        lta = compute_lta(td["backlog"])
        max_weight_ltas.append(lta)

    # take average of all max_weight_ltas
    ltas = np.array(max_weight_ltas)
    mean_lta = np.mean(ltas, axis=0)
    std_lta = np.std(ltas, axis=0)
    ci = 1.96 * std_lta / np.sqrt(len(max_weight_ltas))
    ax.plot(mean_lta, label="MaxWeight LTA")
    ax.fill_between(np.arange(len(mean_lta)), mean_lta - ci, mean_lta + ci, alpha=0.2)
    ax.set(xlabel='time', ylabel='backlog', title=f'MaxWeight Mean LTA performance ({len(max_weight_ltas)} trials)')
    #fig.show()
    final_lta = mean_lta[-1]
    print(f"MaxWeight LTA: {mean_lta[-1]} +- {std_lta[-1]}")
    test_generator.baseline_lta = final_lta
    training_generator.baseline_lta = final_lta

    # Create Actor Critic Agent
    base_env = training_generator.sample()

    # Specify actor and critic
    actor = agent.get_policy_operator().to(device)
    critic = agent.get_value_operator().to(device)

    # Create Generalized Advantage Estimation module
    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
    )

    # Create PPO loss module
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coef=cfg.loss.entropy_coef,
        normalize_advantage=cfg.loss.norm_advantage,
        loss_critic_type="l2"
    )

    collector = SyncDataCollector(
        policy=actor,
        create_env_fn=training_generator.sample(),
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_training_frames,
        device=device,
        storing_device=device,
        reset_at_each_iter=False,
    )

    # create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.collector.frames_per_batch),
        sampler=sampler,
        batch_size=cfg.loss.mini_batch_size, # amount of samples to be sampled when sample is called
    )

    # Create optimizer
    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.eps,
    )

    # Create logger

    logger = get_logger(
        "wandb",
        logger_name="..\\logs",
        experiment_name=getattr(cfg, "exp_name", None),
        wandb_kwargs={
            "config": cfg.as_dict(),
            "project": cfg.logger.project_name,
        },
    )
    # Save the cfg as a yaml file and upload to wandb
    cfg_dict = cfg.as_dict()
    with open(os.path.join(logger.experiment.dir, "config.yaml"), "w") as file:
        yaml.dump(cfg_dict, file)
    wandb.save("config.yaml")

    # log the maxweight baseline plot
    wandb.log({"maxweight/MaxWeight Baseline": wandb.Image(fig)})

    wandb.watch(actor.module[0], log = "all", log_freq=10)

    # Main Loop
    collected_frames = 0
    num_network_updates = 0
    sampling_start = time.time()

    total_training_frames = cfg.collector.total_training_frames
    pbar = tqdm(total=total_training_frames)


    num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
    total_network_updates = (
            (total_training_frames // cfg.collector.frames_per_batch) * cfg.loss.ppo_epochs * num_mini_batches
    )
    best_train_policy_best_score = (None, None)
    best_eval_policy_best_score = (None, None)
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
        if best_train_policy_best_score[0] is None or mean_episode_reward > best_train_policy_best_score[1]:
            best_train_policy_best_score = (deepcopy(agent.state_dict()), mean_episode_reward)
            # save the state_dict of the best policy to wandb
            torch.save(best_train_policy_best_score[0], os.path.join(logger.experiment.dir, "best_model.pt"))
            wandb.save("best_on_train_model.pt")
            pbar.set_postfix({"Best Train Mean Episode Reward": mean_episode_reward})


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
                test_lta_backlogs = []
                test_baseline_ltas = []
                for i in range(cfg.eval.num_eval_envs):
                    test_env = test_generator.sample()
                    test_baseline_ltas.append(test_env.baseline_lta)
                    test_td = get_stable_rollout(test_env, actor, cfg.eval.traj_steps)
                    test_backlog = test_td["next", "backlog"].numpy()
                    test_action_prob = test_td["sample_log_prob"].exp().numpy().mean()
                    test_lta_backlogs.append(compute_lta(test_backlog))
                # save the logits to a temp variable
                eval_time = time.time() - eval_start

                lta_test_backlogs_mean = np.mean([t[-1] for t in test_lta_backlogs])
                log_info.update(
                    {
                        "eval/lta_backlog": lta_test_backlogs_mean,
                        "eval/time": eval_time,
                        "eval/action_prob": test_action_prob,
                    }
                )
                if best_eval_policy_best_score[0] is None or lta_test_backlogs_mean < best_eval_policy_best_score[1]:
                    # update pbar
                    # pbar.set_postfix({"Best Eval LTA Backlog": lta_test_backlogs[-1]})
                    best_eval_policy_best_score = (deepcopy(agent.state_dict()), lta_test_backlogs_mean)
                    # save the state_dict of the best policy to wandb
                    torch.save(best_eval_policy_best_score[0], os.path.join(logger.experiment.dir, "best_model.pt"))
                    wandb.save("best_on_eval_model.pt")
                    pbar.set_postfix({"Best Eval LTA Backlog": lta_test_backlogs_mean})

                # check if all test_lta_backlogs are None
                if all([t is None for t in test_lta_backlogs]):
                    pass
                else: # take average of ratio between test and baseline lta
                    relative_lta_backlog = np.array([test[-1]/baseline for test, baseline in zip(test_lta_backlogs, test_baseline_ltas)]).mean()
                    log_info.update({
                        "eval/relative_lta_backlog": relative_lta_backlog,
                    })


                # Plot all lta backlogs
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                for lta_backlog in test_lta_backlogs:
                    ax.plot(lta_backlog)
                # plot a horizontal line at the mean of the the test_baseline_ltas
                ax.axhline(y=np.mean(test_baseline_ltas), color='r', linestyle='--')
                ax.set_ylabel("Test Backlog")
                ax.set_xlabel("Time")
                ax.set(title = f"Eval on training step {collected_frames}")
                #ax.legend()
                wandb.log({f"eval_plots/training_step {collected_frames}": wandb.Image(fig)})
                # plt.show()
                plt.close(fig)
            actor.train()
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)

        collector.update_policy_weights_()
        sampling_start = time.time()





if __name__ == "__main__":
    import json
    from torchrl_development.envs.env_generator import EnvGenerator, make_env, create_training_test_generators
    ## can either use a single environment
    #env_name = "SH2"
    #env_params = parse_env_json(f"{env_name}.json")
    ## or multiple environments
    env_params = json.load(open("C:\\Users\\Jerrod\\PycharmProjects\\IA-DRL_4SQN\\torchrl_development\\envs\\sampling\\SH2_generated_params.json", 'rb'))

    make_env_parameters = {"observe_lambda": True,
                           "seed": 0,
                           "device": "cpu",
                           "terminal_backlog": 100,
                           }

    training_generator, test_generator = create_training_test_generators(env_params["key_params"],
                                                                         make_env_parameters,
                                                                         seed = 0,
                                                                         test_size = 0.2)
    # Get training configuration
    cfg = load_config("multi_env_ppo1.yaml")

    # Create base env for agent generation
    base_env= training_generator.sample()
    check_env_specs(base_env)

    # Create agent
    input_shape = base_env.observation_spec["observation"].shape
    output_shape = base_env.action_spec.space.n

    agent = create_actor_critic(
        input_shape,
        output_shape,
        in_keys=["observation"],
        action_spec=base_env.action_spec,
    )

    train(agent, training_generator, test_generator, cfg)







