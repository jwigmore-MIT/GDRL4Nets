import tensordict.nn
import wandb
import matplotlib.pyplot as plt
import torch
import numpy as np
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictModule, ProbabilisticTensorDictSequential
from tensordict import TensorDict

from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
from torchrl.envs.transforms import CatTensors, TransformedEnv, SymLogTransform, Compose, RewardSum, RewardScaling, StepCounter, ActionMask

from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, OneHotCategorical, ValueOperator, MLP, ActorCriticWrapper, MaskedOneHotCategorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from torchrl.data import CompositeSpec
from torchrl._utils import logger as torchrl_logger

from torchrl_development.envs.SingleHop import SingleHop
from torchrl_development.envs.env_generator import parse_env_json

from torchrl.record.loggers import generate_exp_name, get_logger
import time
import yaml
from copy import deepcopy
from torchrl_development.utils.configuration import load_config



env_name = "SH1"
env_params = parse_env_json(f"{env_name}.json")
# load config as a namespace
cfg = load_config("ppo1.yaml")

def make_env(env_params = env_params,
             max_steps = 2048,
             seed = 0,
             device = "cpu",
             terminal_backlog = None):
    env_params = deepcopy(env_params)
    if terminal_backlog is not None:
        env_params["terminal_backlog"] = terminal_backlog

    base_env = SingleHop(env_params, seed, device)
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ActionMask(action_key="action", mask_key="mask"),
            CatTensors(in_keys=["Q", "Y"], out_key="observation"),
            SymLogTransform(in_keys=["observation"], out_keys=["observation"]),
            #InverseReward(),
            RewardScaling(loc = 0, scale=0.01),
            RewardSum(),
            StepCounter(max_steps = max_steps)
        )
    )
    return env

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

def make_plots(data, key2):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(data["backlog"])
    ax[0].set_ylabel("Backlog")
    ax[1].plot(data[key2])
    ax[1].set_ylabel(key2)
    ax[1].legend()
    plt.show()
# check version of wandb
# global params
device = cfg.device

# Define the environment
env = make_env(env_params)

# Run check env
check_env_specs(env)

# Create actor network
input_shape = env.observation_spec["observation"].shape
output_shape = env.action_spec.space.n
# distribution = MaskedOneHotCategorical
in_keys= ["observation"]
actor_mlp = MLP(in_features=input_shape[-1],
          activation_class = torch.nn.ReLU,
          activate_last_layer = True,
          out_features=output_shape,
          )
actor_mlp_output = actor_mlp(torch.ones(input_shape))

critic_mlp = MLP(in_features=input_shape[-1],
          activation_class = torch.nn.ReLU,
          activate_last_layer = True,
          out_features=1,
          )
critic_mlp_output = critic_mlp(torch.ones(input_shape))

actor_module = TensorDictModule(
    module = actor_mlp,
    in_keys = in_keys,
    out_keys = ["logits"],
)
# Add probabilistic sampling to the actor
# prob_module = ProbabilisticTensorDictModule(
#     in_keys = ["logits", "mask"],
#     out_keys = ["action", "log_prob"],
#     distribution_class= MaskedOneHotCategorical,
#     default_interaction_type=ExplorationType.RANDOM,
#     return_log_prob=True,
# )

# actor_module = ProbabilisticTensorDictSequential(actor_module, prob_module)


actor_module = ProbabilisticActor(
    actor_module,
    distribution_class= MaskedOneHotCategorical,
    #distribution_kwargs = {"mask_key": "mask"},
    in_keys = ["logits", "mask"],
    spec=CompositeSpec(action=env.action_spec),
    return_log_prob=True,
    default_interaction_type = ExplorationType.RANDOM,
)

# Create Value Module
value_module = ValueOperator(
    module = critic_mlp,
    in_keys = in_keys,
)

#
actor_critic = ActorCriticWrapper(actor_module, value_module)

with torch.no_grad():
    td = env.rollout(max_steps=100, policy=actor_critic)
    td = actor_critic(td)

actor = actor_critic.get_policy_operator()
critic = actor_critic.get_value_operator()
actor, critic = actor.to(device), critic.to(device)

# Need an environment generator function

# Create Collector
collector = SyncDataCollector(
    create_env_fn=make_env,
    create_env_kwargs={"env_params": env_params,
                       "max_steps": cfg.collector.max_steps_per_eps,
                       "seed": cfg.seed,
                       "terminal_backlog": cfg.env.terminal_backlog},
    policy=actor,
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

adv_module = GAE(
    gamma=cfg.loss.gamma,
    lmbda=cfg.loss.gae_lambda,
    value_network=critic,
    average_gae=False,
)

loss_module = ClipPPOLoss(
    actor_network=actor,
    critic_network=critic,
    clip_epsilon=cfg.loss.clip_epsilon,
    entropy_coef=cfg.loss.entropy_coef,
    normalize_advantage=False,
    loss_critic_type="l2"
)

# Create optimizer
optim = torch.optim.Adam(
    loss_module.parameters(),
    lr=cfg.optim.lr,
    weight_decay=cfg.optim.weight_decay,
    eps=cfg.optim.eps,
)

# Create logger
exp_name = generate_exp_name("PPO", env_name)
logger = get_logger(
    "wandb",
    logger_name="logs",
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
for i, data in enumerate(collector):

    log_info = {}
    sampling_time = time.time() - sampling_start
    frames_in_batch = data.numel()
    collected_frames += frames_in_batch * cfg.collector.frame_skip
    pbar.update(data.numel())

    # Get training rewards and episode lengths
    episode_rewards = data["next", "episode_reward"][data["next", "done"]]
    episode_backlog = data["next", "backlog"][data["next", "done"]].float()
    if len(episode_rewards) > 0:
        episode_length = data["next", "step_count"][data["next", "done"]]
        log_info.update(
            {
                "train/episode_backlog": episode_backlog.mean().item(),
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
        if ((i - 1) * frames_in_batch * cfg.collector.frame_skip) // cfg.eval.eval_interval < (
                i * frames_in_batch * cfg.collector.frame_skip
        ) // cfg.eval.eval_interval:


            actor.eval()
            eval_start = time.time()
            test_rewards, test_final_backlogs, test_backlogs = eval_model(actor, test_env,
                                                     test_steps=cfg.eval.traj_steps,
                                                     num_episodes=cfg.eval.num_eval_trajs)

            eval_time = time.time() - eval_start
            log_info.update(
                {
                    "eval/reward": test_rewards.mean(),
                    "eval/backlog": test_final_backlogs.mean(),
                    "eval/time": eval_time,
                }
            )
            eval_backlog_buffer.append(test_backlogs)
            eval_backlog_buffer_labels.append(collected_frames)
            if True and len(eval_backlog_buffer) >= eval_buffer_max_size:
                # plot the past 6 backlogs from the eval_backlog_buffer
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                for plot_data, label in zip(eval_backlog_buffer, eval_backlog_buffer_labels):
                    ax.plot(plot_data, label=f"step {label}")
                ax.set_ylabel("Test Backlog")
                ax.set_xlabel("Time")
                # get labels and put them in the legend
                ax.legend()
                wandb.log({f"eval_plots/training_step {collected_frames}": wandb.Image(fig)})
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
