
import matplotlib.pyplot as plt
from torchrl.envs.transforms import CatTensors, TransformedEnv, SymLogTransform, Compose, RewardSum, RewardScaling, StepCounter, ActionMask

from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type

from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
from torchrl.collectors import SyncDataCollector
import os
from torchrl_development.envs.SingleHop import SingleHop
from torchrl_development.envs.env_generator import parse_env_json
from copy import deepcopy
from torchrl_development.utils.configuration import load_config
from torchrl_development.maxweight import MaxWeightActor
import numpy as np
from torchrl_development.utils.metrics import compute_lta
from torchrl.modules import Actor, MLP, ProbabilisticActor, ValueOperator, MaskedOneHotCategorical, ActorCriticWrapper
from tensordict.nn import TensorDictModule
from torchrl.data import CompositeSpec
import torch
from torchrl.objectives.iappo import IAPPOLoss, ClipIAPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record.loggers import generate_exp_name, get_logger
import time
from tqdm import tqdm
from tensordict import TensorDict
import wandb
from torchrl._utils import logger as torchrl_logger
from copy import deepcopy
from torchrl_development.actors import create_ia_actor, create_ia_actor_critic



def eval_model(actor, test_env, test_steps, num_episodes=3, plot_backlog=False):
    test_rewards = []
    test_final_backlogs = []
    test_backlogs = []
    for _ in range(num_episodes):
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
    return torch.cat(test_rewards, 0).mean(), torch.cat(test_final_backlogs, 0).mean(), torch.cat(test_backlogs, 1).mean(1)






env_name = "SH2"
env_params = parse_env_json(f"{env_name}.json")
env_params["obs_lambda"] = True
# load config as a namespace
cfg = load_config("iappo1.yaml")
def make_env(env_params = env_params,
             max_steps = 1000000,
             seed = 0,
             device = "cpu",
             terminal_backlog = None,
             observation_keys = ["Q", "Y"]):
    env_params = deepcopy(env_params)
    if terminal_backlog is not None:
        env_params["terminal_backlog"] = terminal_backlog

    base_env = SingleHop(env_params, seed, device)
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ActionMask(action_key="action", mask_key="mask"),
            CatTensors(in_keys=observation_keys, out_key="observation", del_keys=False),
            SymLogTransform(in_keys=["observation"], out_keys=["observation"]),
            #InverseReward(),
            RewardScaling(loc = 0, scale=0.01),
            RewardSum(),
            StepCounter(max_steps = max_steps)
        )
    )
    return env

def get_stable_rollout(env, policy, max_steps):
    td = env.rollout(policy=policy, max_steps=max_steps)
    return td




device = cfg.device

# Define the environment
env = make_env(env_params)

# Run check env
check_env_specs(env)

# Create actor network
input_shape = env.observation_spec["observation"].shape
output_shape = env.action_spec.space.n
# distribution = MaskedOneHotCategorical
max_weight_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])

mw_rollout = get_stable_rollout(env, max_weight_actor, max_steps=cfg.intervention_policy.stable_rollout_length)
mw_lta_backlogs = compute_lta(mw_rollout["backlog"].numpy())
# Create actor network
input_shape = env.observation_spec["observation"].shape
output_shape = env.action_spec.space.n
# distribution = MaskedOneHotCategorical
in_keys= ["observation"]
action_spec = env.action_spec

ia_actor_critic = create_ia_actor_critic(input_shape,
                                         output_shape,
                            in_keys,
                            action_spec,
                            threshold = cfg.intervention_policy.threshold)
#
# actor_mlp = MLP(in_features=input_shape[-1],
#           activation_class = torch.nn.ReLU,
#           activate_last_layer = True,
#           out_features=output_shape,
#           )
# actor_mlp_output = actor_mlp(torch.ones(input_shape))
#
# critic_mlp = MLP(in_features=input_shape[-1],
#           activation_class = torch.nn.ReLU,
#           activate_last_layer = True,
#           out_features=1,
#           )
# critic_mlp_output = critic_mlp(torch.ones(input_shape))
#
# actor_module = TensorDictModule(
#     module = actor_mlp,
#     in_keys = in_keys,
#     out_keys = ["logits"],
# )
# # Add probabilistic sampling to the actor
# # prob_module = ProbabilisticTensorDictModule(
# #     in_keys = ["logits", "mask"],
# #     out_keys = ["action", "log_prob"],
# #     distribution_class= MaskedOneHotCategorical,
# #     default_interaction_type=ExplorationType.RANDOM,
# #     return_log_prob=True,
# # )
#
# # actor_module = ProbabilisticTensorDictSequential(actor_module, prob_module)
#
#
# actor_module = ProbabilisticActor(
#     actor_module,
#     distribution_class= MaskedOneHotCategorical,
#     #distribution_kwargs = {"mask_key": "mask"},
#     in_keys = ["logits", "mask"],
#     spec=CompositeSpec(action=env.action_spec),
#     return_log_prob=True,
#     default_interaction_type = ExplorationType.RANDOM,
# ) # Need to do sequence maxweight -> actor -> output
# from torchrl_development.maxweight import MaxWeightActor
#
# # maxweight_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
#
# # ia_actor = InterventionActor(maxweight_actor, actor_module, threshold = 30)
#
# #ia_actor_module = InterventionActor(actor_module, intervention_type="maxweight", threshold = 30)
# ia_actor = InterventionActorWrapper(actor_module, threshold = 30)
#
# test = deepcopy(ia_actor)
# # ia_actor_module = InterventionActor(actor_module, intervention_type="maxweight", threshold = 30)
# # Create Value Module
# value_module = ValueOperator(
#     module = critic_mlp,
#     in_keys = in_keys,
# )
#
# #
# actor_critic = InterventionActorCriticWrapper(ia_actor, value_module)
actor = ia_actor_critic.get_policy_operator()
critic = ia_actor_critic.get_value_operator()



adv_module = GAE(
    gamma=cfg.loss.gamma,
    lmbda=cfg.loss.gae_lambda,
    value_network=critic,
    average_gae=False,
)

loss_module = ClipIAPPOLoss(
    actor_network=actor,
    critic_network=critic,
    clip_epsilon=cfg.loss.clip_epsilon,
    entropy_coef=cfg.loss.entropy_coef,
    normalize_advantage=False,
    loss_critic_type="l2"
)

# Create Collector
collector = SyncDataCollector(
    create_env_fn=make_env,
    create_env_kwargs={"env_params": env_params,
                       "max_steps": cfg.collector.max_steps_per_eps,
                       "seed": cfg.seed,
                       "terminal_backlog": cfg.env.terminal_backlog},
    policy=ia_actor_critic.ia_actor,
    frames_per_batch=cfg.collector.frames_per_batch,
    total_frames=cfg.collector.total_training_frames,
    device=device,
    storing_device=device,
    reset_at_each_iter=True,
)


test_env = make_env(env_params = env_params,
                    max_steps = cfg.eval.traj_steps,
                    seed = cfg.seed,
                    terminal_backlog = None)



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
exp_name = generate_exp_name("IA-PPO", env_name)
logger = get_logger(
    "wandb",
    logger_name="..\\logs",
    experiment_name=exp_name,
    wandb_kwargs={
        "config": cfg,
        "project": "torchrl_testing",
    },
)

# Main Loop
collected_frames = 0
# Main loop
collected_frames = 0
num_network_updates = 0
start_time = time.time()
pbar = tqdm(total=cfg.collector.total_training_frames)
num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
total_network_updates = (
        (cfg.collector.total_training_frames // cfg.collector.frames_per_batch) * cfg.loss.ppo_epochs * num_mini_batches
)

sampling_start = time.time()
# make x a list of lists
eval_backlog_buffer = []
eval_backlog_buffer_labels = []
eval_buffer_max_size = cfg.eval.plot_freq
best_policy_best_score = (None, None)
for i, data in enumerate(collector):

    log_info = {}
    sampling_time = time.time() - sampling_start
    frames_in_batch = data.numel()
    collected_frames += frames_in_batch * cfg.collector.frame_skip
    pbar.update(data.numel())

    # Get training rewards and episode lengths
    episode_rewards = data["next", "episode_reward"].mean()
    episode_backlog = data["next", "backlog"].float().mean()
    intervention_rate = data["intervene"].float().mean()

    episode_length = data["next", "step_count"][-1]
    log_info.update(
        {
            "train/episode_backlog": episode_backlog.mean().item(),
            "train/intervention_rate": intervention_rate.mean().item(),
            "train/reward": episode_rewards.mean().item(),
            "train/episode_length": episode_length.sum().item()
                                    / len(episode_length),
        }
    )

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

            num_network_updates += 1
            # Get a data batch
            batch = batch.to(device, non_blocking=True)

            # Forward pass PPO loss
            loss = loss_module(batch)
            losses[j, k] = loss.select(
                "loss_critic", "loss_entropy", "loss_objective", "entropy", "ESS"
            ).detach()
            loss_sum = (
                    loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
            )
            # Backward pass
            loss_sum.backward()

            # for name, param in loss_module.named_parameters():
            #     if param.grad is None:
            #         print(f"None gradient in actor {name}")
            #     else:
            #         print(f"Actor {name} gradient: {param.grad.max()}")
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
            "train/lr": alpha ,
            "train/sampling_time": sampling_time,
            "train/training_time": training_time,
            "train/clip_epsilon": alpha * cfg.loss.clip_epsilon,
        }
    )

    # Get eval rewards
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        if ((i - 1) * frames_in_batch * cfg.collector.frame_skip) // cfg.eval.eval_freq*cfg.collector.frames_per_batch < (
                i * frames_in_batch * cfg.collector.frame_skip
        ) // cfg.eval.eval_freq*cfg.collector.frames_per_batch:


            actor.eval()
            eval_start = time.time()
            test_rewards, test_final_backlogs, test_backlogs = eval_model(ia_actor_critic.ia_actor, test_env,
                                                     test_steps=cfg.eval.traj_steps,
                                                     num_episodes=cfg.eval.num_eval_trajs)

            eval_time = time.time() - eval_start
            lta_test_backlogs = compute_lta(test_backlogs)

            log_info.update(
                {
                    "eval/reward": test_rewards.mean(),
                    "eval/backlog": lta_test_backlogs[-1],
                    "eval/time": eval_time,
                    "eval/mw_backlog": mw_lta_backlogs[-1]
                }
            )
            # update best policy
            if best_policy_best_score[0] is None or lta_test_backlogs[-1] > best_policy_best_score[1]:
                # update pbar
                #pbar.set_postfix({"Best Eval LTA Backlog": lta_test_backlogs[-1]})
                best_policy_best_score = (deepcopy(ia_actor_critic.state_dict()), lta_test_backlogs[-1])
                # save the state_dict of the best policy to wandb
                torch.save(best_policy_best_score[0], os.path.join(logger.experiment.dir,"best_model.pt"))
                wandb.save("best_model.pt")


            # add the test_backlogs to the eval_backlog_buffer
            eval_backlog_buffer.append(compute_lta(test_backlogs))
            eval_backlog_buffer_labels.append(collected_frames)
            if True and len(eval_backlog_buffer) >= eval_buffer_max_size:
                # plot the past 6 backlogs from the eval_backlog_buffer
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                for plot_data, label in zip(eval_backlog_buffer, eval_backlog_buffer_labels):
                    ax.plot(plot_data, label=f"step {label}")
                ax.plot(mw_lta_backlogs[:cfg.eval.traj_steps], label="MW LTA Backlog", color = "black", linestyle = "dashed")
                ax.set_ylabel("Test Backlog")
                ax.set_xlabel("Time")
                # get labels and put them in the legend
                ax.legend()
                wandb.log({f"eval_plots/training_step {collected_frames}": wandb.Image(fig)})
                plt.close(fig)
                # pop the first 5 elements from the eval_backlog_buffer and eval_backlog_buffer_labels
                eval_backlog_buffer = [eval_backlog_buffer[-1]]
                eval_backlog_buffer_labels = [eval_backlog_buffer_labels[-1]]

            actor.train()

    if logger:
        for key, value in log_info.items():
            logger.log_scalar(key, value, collected_frames)

    collector.update_policy_weights_()
    sampling_start = time.time()

collector.shutdown()
end_time = time.time()
execution_time = end_time - start_time
torchrl_logger.info(f"Training took {execution_time:.2f} seconds to finish")


# #test ia + maxweight policy
# env = make_env(env_params = env_params,
#                  max_steps = 2048,
#                  seed = 0,
#                 terminal_backlog = None)
#
# td = env.rollout(policy=ia_actor_module, max_steps = 2048)
#
# backlogs = td["backlog"]
# interventions = td["intervene"]
# lta_backlogs = compute_lta(backlogs.numpy())
#     # np.divide(backlogs.numpy().sum(),np.arange(1, len(backlogs)+1).reshape(-1,1)))
# # plot backlogs
# fig, ax = plt.subplots(2,1)
# ax[0].plot(lta_backlogs)
# ax[0].set(xlabel='time', ylabel='backlog',
#        title='LTA Backlog over time')
# ax[0].grid()
# # plot interventions
# ax[1].plot(interventions)
# ax[1].set(xlabel='time', ylabel='intervene',
#        title='Interventions over time')
# ax[1].grid()
#
#
# plt.show()
# #
# #
# #
