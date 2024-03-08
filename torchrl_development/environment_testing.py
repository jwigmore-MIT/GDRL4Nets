import torch
from matplotlib import pyplot as plt
from tensordict import TensorDict
from torchrl.envs.libs.gym import GymEnv, GymWrapper
from config_parsing import parse_config
import os
from torchrl_development.envs.SingleHop import SingleHop
from torchrl_development.envs.env_generator import parse_env_json
from torchrl.envs.transforms import ObservationTransform
from typing import Sequence, Union
from torchrl.envs.utils import check_env_specs, step_mdp


from main_utils import generate_env





## ------------------------------------------------------------#
"""
Below is for a standard gym environment
"""
# Standard gym environment
pend_env = GymEnv("Pendulum-v1")

## TensorSpec is the parent class which is a specification of characteristics of a tensor
# eg. shape, space, dtype etc.
#print("Pendulum-v1 observation spec: ", pend_env.observation_spec)
## Shows that the observation space is a Box space with shape (3,) and dtype float32
#print("Pendulum-v1 action spec: ", pend_env.action_spec)

## ------------------------------------------------------------#
## ------------------------------------------------------------#





env_config_path = "config/environments/SH1.json"
env_para = parse_env_json(env_config_path)
env = SingleHop(env_para)
#env.rand_step()

# print("env observation_spec: ", env.observation_spec)
# print("env action_spec: ", env.action_spec)
# print("env reward_spec: ", env.reward_spec)
# print("env done_spec: ", env.done_spec)

td = env.reset()

def policy(td: TensorDict, env = env):
    td.set("action", env.action_spec.rand())
    return td

policy(td)
td_out = env.step(td)

td_rollout = env.rollout(max_steps = 10, policy = policy)

q_check = (
    td_rollout.get("Q")[1:]
    == td_rollout.get(("next", "Q"))[:-1]
).all()

print("Queue check: ", q_check)

y_check = (
    td_rollout.get("Y")[1:]
    == td_rollout.get(("next", "Y"))[:-1]
).all()

print("Y check: ", y_check)


from torchrl.envs.transforms import CatTensors, TransformedEnv, SymLogTransform, Compose


#cat_transform = CatTensors(in_keys = ["Q", "Y"], out_key = "observation")
#env = TransformedEnv(env, cat_transform)
#env = TransformedEnv(env, SymLogTransform(in_keys = ["observation"], out_keys = ["observation"]))

env = TransformedEnv(env, Compose(
    CatTensors(in_keys = ["Q", "Y"], out_key = "observation", in_keys_inv = ["observation"], out_keys_inv = ["Q", "Y"]),
    SymLogTransform(in_keys = ["observation"], out_keys = ["observation"], in_keys_inv = ["observation"], out_keys_inv = ["observation"])
))
env.reset()

print(check_env_specs(env))

td_rollout = env.rollout(max_steps = 10, policy = policy)


#inv_transform_rollout = e(td_rollout)

# class SymLogTransform(ObservationTransform):
#     def __init__(self,
#         in_keys = None,
#         out_keys = None,
#         in_keys_inv = None,
#         out_keys_inv = None,):
#         if in_keys is None:
#             raise RuntimeError(
#                 "Not passing in_keys to ObservationNorm is a deprecated behaviour."
#             )
#
#         if out_keys is None:
#             out_keys = copy(in_keys)
#         if in_keys_inv is None:
#             in_keys_inv = []
#         if out_keys_inv is None:
#             out_keys_inv = copy(in_keys_inv)
#
#         super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)
#
#     def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
#         return obs.sign()*torch.log(1+obs.abs())
#
#     def _apply_inverse_transform(self, obs: torch.Tensor) -> torch.Tensor:
#         return obs.sign()*(torch.exp(obs.abs())-1)
#
#     @_apply_to_composite
#     def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
#         space = observation_spec.space
#         if isinstance(space, ContinuousBox):
#             space.low = self._apply_transform(space.low)
#             space.high = self._apply_transform(space.high)
#         return observation_spec
#
#     @_apply_to_composite_inv
#     def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
#         space = input_spec.space
#         if isinstance(space, ContinuousBox):
#             space.low = self._apply_transform(space.low)
#             space.high = self._apply_transform(space.high)
#         return input_spec




