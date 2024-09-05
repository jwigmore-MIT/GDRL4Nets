from __future__ import annotations

from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Tuple, Union
from copy import copy
import torch
from torchrl.envs.transforms.transforms import Transform, ObservationTransform, _apply_to_composite, _apply_to_composite_inv,_set_missing_tolerance
from tensordict.utils import  NestedKey
from torchrl.data.tensor_specs import (
    ContinuousBox,
    TensorSpec,
    UnboundedContinuousTensorSpec,
    Unbounded
)

from tensordict import (
    TensorDictBase,

)

from modules.torchrl_development.envs.utils import TimeAverageStatsCalculator

class SymLogTransform(ObservationTransform):
    def __init__(self,
        in_keys = None,
        out_keys = None,
        in_keys_inv = None,
        out_keys_inv = None,):
        if in_keys is None:
            raise RuntimeError(
                "Not passing in_keys to ObservationNorm is a deprecated behaviour."
            )

        if out_keys is None:
            out_keys = copy(in_keys)
        if in_keys_inv is None:
            in_keys_inv = []
        if out_keys_inv is None:
            out_keys_inv = copy(in_keys_inv)

        super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.sign()*torch.log(1+obs.abs())


    def _apply_inverse_transform(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.sign()*(torch.exp(obs.abs())-1)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
        observation_spec.dtype = torch.float32
        return observation_spec

    @_apply_to_composite_inv
    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        space = input_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
        #input_spec.dtype = torch.float32 #TODO: check if this is necessary
        return input_spec

    def _reset(
            self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


class InverseReward(Transform):
    """Applies the transformation r = 1/(1-r) to the rewards.
        For unbounded rewards in unbounded environments

    Args:
        in_keys (List[NestedKey]): input keys
        out_keys (List[NestedKey], optional): output keys. Defaults to value
            of ``in_keys``.
        dtype (torch.dtype, optional): the dtype of the output reward.
            Defaults to ``torch.float``.
    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = ["reward"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def _apply_transform(self, reward: torch.Tensor) -> torch.Tensor:
        # if not reward.shape or reward.shape[-1] != 1:
        #     raise RuntimeError(
        #         f"Reward shape last dimension must be singleton, got reward of shape {reward.shape}"
        #     )
        return 1/(1-reward)

    @_apply_to_composite
    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        return UnboundedContinuousTensorSpec(
            dtype=torch.float,
            device=reward_spec.device,
            shape=reward_spec.shape,
        )

class ReverseSignTransform(Transform):
    """Applies the transformation r = -r to the rewards.
        For unbounded rewards in unbounded environments

    Args:
        in_keys (List[NestedKey]): input keys
        out_keys (List[NestedKey], optional): output keys. Defaults to value
            of ``in_keys``.
        dtype (torch.dtype, optional): the dtype of the output reward.
            Defaults to ``torch.float``.
    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = ["reward"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def _apply_transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # if not reward.shape or reward.shape[-1] != 1:
        #     raise RuntimeError(
        #         f"Reward shape last dimension must be singleton, got reward of shape {reward.shape}"
        #     )
        return -input_tensor

    @_apply_to_composite
    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        return UnboundedContinuousTensorSpec(
            dtype=torch.float,
            device=reward_spec.device,
            shape=reward_spec.shape,
        )


class InverseTransform(Transform):
    """Applies the transformation s = 1/s to provided keys.
        For unbounded rewards in unbounded environments

    Args:
        in_keys (List[NestedKey]): input keys
        out_keys (List[NestedKey], optional): output keys. Defaults to value
            of ``in_keys``.
        dtype (torch.dtype, optional): the dtype of the output reward.
            Defaults to ``torch.float``.
    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = ["state"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def _apply_transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # if not reward.shape or reward.shape[-1] != 1:
        #     raise RuntimeError(
        #         f"Reward shape last dimension must be singleton, got reward of shape {reward.shape}"
        #     )
        return 1/input_tensor

    @_apply_to_composite
    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        return UnboundedContinuousTensorSpec(
            dtype=torch.float,
            device=reward_spec.device,
            shape=reward_spec.shape,
        )

class RunningAverageTransform(Transform):
    """
    Computes a running average of the corresponding keys
    UNUSED - DOING COMPUTATION IN THE TRAINING LOOP
    """

    def __init__(
        self,
        in_key: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        window_size: int = 10000,
    ):
        self.time_avg_stats = TimeAverageStatsCalculator(window_size)
        if in_key is None:
            in_key = ["reward"]
        if in_key.__len__() >1:
            raise ValueError("Only one input key is allowed")
        if out_keys is None:
            out_keys = ["ta_mean_" + copy(in_key[0]), "ta_stdev_" + copy(in_key[0])]
        super().__init__(in_keys=in_key, out_keys=out_keys)
        self.window_size = window_size


    def _apply_transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.time_avg_stats.update(input_tensor)
        return self.time_avg_stats.mean, self.time_avg_stats.sampleStdev
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        observation_spec["ta_mean_" + self.in_keys[0]] = Unbounded(
            dtype=torch.float,
            device=observation_spec.device,
            shape=observation_spec.shape,
        )
        observation_spec["ta_stdev_" + self.in_keys[0]] = Unbounded(
            dtype=torch.float,
            device=observation_spec.device,
            shape=observation_spec.shape,
        )
        return observation_spec