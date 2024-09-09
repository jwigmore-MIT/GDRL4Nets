from __future__ import annotations

from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Tuple, Union
from copy import copy
import torch
from torch_geometric.data import Data
from torchrl.envs.transforms.transforms import Transform, ObservationTransform, _apply_to_composite, _apply_to_composite_inv,_set_missing_tolerance
from torchrl.envs.transforms.transforms import _sort_keys
from tensordict.utils import  NestedKey
from torchrl.data.tensor_specs import (
    ContinuousBox,
    TensorSpec,
    UnboundedContinuousTensorSpec,
    Unbounded,
    Composite,

)

from tensordict import (
    TensorDictBase,

)

from modules.torchrl_development.envs.utils import TimeAverageStatsCalculator

# import torch_geometric as tg

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


class CatStackTensors(Transform):
    """Concatenates several keys in a single tensor.

    This is especially useful if multiple keys describe a single state (e.g.
    "observation_position" and
    "observation_velocity")

    Args:
        in_keys (sequence of NestedKey): keys to be concatenated. If `None` (or not provided)
            the keys will be retrieved from the parent environment the first time
            the transform is used. This behavior will only work if a parent is set.
        out_key (NestedKey): key of the resulting tensor.
        dim (int, optional): dimension along which the concatenation will occur.
            Default is ``-1``.

    Keyword Args:
        del_keys (bool, optional): if ``True``, the input values will be deleted after
            concatenation. Default is ``True``.
        unsqueeze_if_oor (bool, optional): if ``True``, CatTensor will check that
            the dimension indicated exist for the tensors to concatenate. If not,
            the tensors will be unsqueezed along that dimension.
            Default is ``False``.
        sort (bool, optional): if ``True``, the keys will be sorted in the
            transform. Otherwise, the order provided by the user will prevail.
            Defaults to ``True``.

    Examples:
        >>> transform = CatTensors(in_keys=["key1", "key2"])
        >>> td = TensorDict({"key1": torch.zeros(1, 1),
        ...     "key2": torch.ones(1, 1)}, [1])
        >>> _ = transform(td)
        >>> print(td.get("observation_vector"))
        tensor([[0., 1.]])
        >>> transform = CatTensors(in_keys=["key1", "key2"], dim=-2, unsqueeze_if_oor=True)
        >>> td = TensorDict({"key1": torch.zeros(1),
        ...     "key2": torch.ones(1)}, [])
        >>> _ = transform(td)
        >>> print(td.get("observation_vector").shape)
        torch.Size([2, 1])

    """

    invertible = False

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_key: NestedKey = "observation_vector",
        dim: int = -1,
        *,
        del_keys: bool = True,
        unsqueeze_if_oor: bool = False,
        sort: bool = True,
    ):
        self._initialized = in_keys is not None
        if not self._initialized:
            if dim != -1:
                raise ValueError(
                    "Lazy call to CatTensors is only supported when `dim=-1`."
                )
        elif sort:
            in_keys = sorted(in_keys, key=_sort_keys)
        if not isinstance(out_key, (str, tuple)):
            raise Exception("CatTensors requires out_key to be of type NestedKey")
        super(CatStackTensors, self).__init__(in_keys=in_keys, out_keys=[out_key])
        self.dim = dim
        self._del_keys = del_keys
        self._keys_to_exclude = None
        self.unsqueeze_if_oor = unsqueeze_if_oor

    @property
    def keys_to_exclude(self):
        if self._keys_to_exclude is None:
            self._keys_to_exclude = [
                key for key in self.in_keys if key != self.out_keys[0]
            ]
        return self._keys_to_exclude

    def _find_in_keys(self):
        """Gathers all the entries from observation spec which shape is 1d."""
        parent = self.parent
        obs_spec = parent.observation_spec
        in_keys = []
        for key, value in obs_spec.items(True, True):
            if len(value.shape) == 1:
                in_keys.append(key)
        return sorted(in_keys, key=_sort_keys)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self._initialized:
            self.in_keys = self._find_in_keys()
            self._initialized = True

        values = [tensordict.get(key, None) for key in self.in_keys]
        if any(value is None for value in values):
            raise Exception(
                f"CatTensor failed, as it expected input keys ="
                f" {sorted(self.in_keys, key=_sort_keys)} but got a TensorDict with keys"
                f" {sorted(tensordict.keys(include_nested=True), key=_sort_keys)}"
            )
        if self.unsqueeze_if_oor:
            pos_idx = self.dim > 0
            abs_idx = self.dim if pos_idx else -self.dim - 1
            values = [
                v
                if abs_idx < v.ndimension()
                else v.unsqueeze(0)
                if not pos_idx
                else v.unsqueeze(-1)
                for v in values
            ]
        """
        Assume values is a list of tensor, each tensor has shape [batch_size, N], and values has length M
        We want to return a tensor of shape [batch_size, N*M], but we want it such that each of t
        """
        # stacked_values = torch.stack(values, dim=-1)
        out_tensor = torch.stack(values, dim=-1).reshape(values[0].shape[0]*len(values))
        tensordict.set(self.out_keys[0], out_tensor)
        if self._del_keys:
            tensordict.exclude(*self.keys_to_exclude, inplace=True)
        return tensordict

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if not self._initialized:
            self.in_keys = self._find_in_keys()
            self._initialized = True

        # check that all keys are in observation_spec
        if len(self.in_keys) > 1 and not isinstance(observation_spec, Composite):
            raise ValueError(
                "CatTensor cannot infer the output observation spec as there are multiple input keys but "
                "only one observation_spec."
            )

        if isinstance(observation_spec, Composite) and len(
            [key for key in self.in_keys if key not in observation_spec.keys(True)]
        ):
            raise ValueError(
                "CatTensor got a list of keys that does not match the keys in observation_spec. "
                "Make sure the environment has an observation_spec attribute that includes all the specs needed for CatTensor."
            )

        if not isinstance(observation_spec, Composite):
            # by def, there must be only one key
            return observation_spec

        keys = [key for key in observation_spec.keys(True, True) if key in self.in_keys]

        sum_shape = sum(
            [
                observation_spec[key].shape[self.dim]
                if observation_spec[key].shape
                else 1
                for key in keys
            ]
        )
        spec0 = observation_spec[keys[0]]
        out_key = self.out_keys[0]
        shape = list(spec0.shape)
        device = spec0.device
        shape[self.dim] = sum_shape
        shape = torch.Size(shape)
        observation_spec[out_key] = Unbounded(
            shape=shape,
            dtype=spec0.dtype,
            device=device,
        )
        if self._del_keys:
            for key in self.keys_to_exclude:
                if key in observation_spec.keys(True):
                    del observation_spec[key]
        return observation_spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_keys={self.in_keys}, out_key"
            f"={self.out_keys[0]})"
        )


class StackTensors(Transform):
    """Stacks several keys in a single tensor observation tensor of shape [B, N, K]. where B is the batch size,
    N is the number of nodes, and K is the number of keys

    This is especially useful if multiple keys describe a single state (e.g.
    "observation_position" and
    "observation_velocity")

    Args:
        in_keys (sequence of NestedKey): keys to be concatenated. If `None` (or not provided)
            the keys will be retrieved from the parent environment the first time
            the transform is used. This behavior will only work if a parent is set.
        out_key (NestedKey): key of the resulting tensor.
        dim (int, optional): dimension along which the concatenation will occur.
            Default is ``-1``.

    Keyword Args:
        del_keys (bool, optional): if ``True``, the input values will be deleted after
            concatenation. Default is ``True``.
        unsqueeze_if_oor (bool, optional): if ``True``, CatTensor will check that
            the dimension indicated exist for the tensors to concatenate. If not,
            the tensors will be unsqueezed along that dimension.
            Default is ``False``.
        sort (bool, optional): if ``True``, the keys will be sorted in the
            transform. Otherwise, the order provided by the user will prevail.
            Defaults to ``True``.

    Examples:
        >>> transform = CatTensors(in_keys=["key1", "key2"])
        >>> td = TensorDict({"key1": torch.zeros(1, 1),
        ...     "key2": torch.ones(1, 1)}, [1])
        >>> _ = transform(td)
        >>> print(td.get("observation_vector"))
        tensor([[0., 1.]])
        >>> transform = CatTensors(in_keys=["key1", "key2"], dim=-2, unsqueeze_if_oor=True)
        >>> td = TensorDict({"key1": torch.zeros(1),
        ...     "key2": torch.ones(1)}, [])
        >>> _ = transform(td)
        >>> print(td.get("observation_vector").shape)
        torch.Size([2, 1])

    """

    invertible = False

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_key: NestedKey = "observation_vector",
        dim: int = -1,
        *,
        del_keys: bool = True,
        unsqueeze_if_oor: bool = False,
        sort: bool = True,
    ):
        self._initialized = in_keys is not None
        if not self._initialized:
            if dim != -1:
                raise ValueError(
                    "Lazy call to CatTensors is only supported when `dim=-1`."
                )
        elif sort:
            in_keys = sorted(in_keys, key=_sort_keys)
        if not isinstance(out_key, (str, tuple)):
            raise Exception("CatTensors requires out_key to be of type NestedKey")
        super(StackTensors, self).__init__(in_keys=in_keys, out_keys=[out_key])
        self.dim = dim
        self._del_keys = del_keys
        self._keys_to_exclude = None
        self.unsqueeze_if_oor = unsqueeze_if_oor

    @property
    def keys_to_exclude(self):
        if self._keys_to_exclude is None:
            self._keys_to_exclude = [
                key for key in self.in_keys if key != self.out_keys[0]
            ]
        return self._keys_to_exclude

    def _find_in_keys(self):
        """Gathers all the entries from observation spec which shape is 1d."""
        parent = self.parent
        obs_spec = parent.observation_spec
        in_keys = []
        for key, value in obs_spec.items(True, True):
            if len(value.shape) == 1:
                in_keys.append(key)
        return sorted(in_keys, key=_sort_keys)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self._initialized:
            self.in_keys = self._find_in_keys()
            self._initialized = True

        values = [tensordict.get(key, None) for key in self.in_keys]
        if any(value is None for value in values):
            raise Exception(
                f"CatTensor failed, as it expected input keys ="
                f" {sorted(self.in_keys, key=_sort_keys)} but got a TensorDict with keys"
                f" {sorted(tensordict.keys(include_nested=True), key=_sort_keys)}"
            )
        if self.unsqueeze_if_oor:
            pos_idx = self.dim > 0
            abs_idx = self.dim if pos_idx else -self.dim - 1
            values = [
                v
                if abs_idx < v.ndimension()
                else v.unsqueeze(0)
                if not pos_idx
                else v.unsqueeze(-1)
                for v in values
            ]
        """
        Assume values is a list of tensor, each tensor has shape [batch_size, N], and values has length M
        We want to return a tensor of shape [batch_size, N*M], but we want it such that each of t
        """
        # stacked_values = torch.stack(values, dim=-1)
        out_tensor = torch.stack(values, dim=-1)
        tensordict.set(self.out_keys[0], out_tensor)
        if self._del_keys:
            tensordict.exclude(*self.keys_to_exclude, inplace=True)
        return tensordict

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if not self._initialized:
            self.in_keys = self._find_in_keys()
            self._initialized = True

        # check that all keys are in observation_spec
        if len(self.in_keys) > 1 and not isinstance(observation_spec, Composite):
            raise ValueError(
                "CatTensor cannot infer the output observation spec as there are multiple input keys but "
                "only one observation_spec."
            )

        if isinstance(observation_spec, Composite) and len(
            [key for key in self.in_keys if key not in observation_spec.keys(True)]
        ):
            raise ValueError(
                "CatTensor got a list of keys that does not match the keys in observation_spec. "
                "Make sure the environment has an observation_spec attribute that includes all the specs needed for CatTensor."
            )

        if not isinstance(observation_spec, Composite):
            # by def, there must be only one key
            return observation_spec

        keys = [key for key in observation_spec.keys(True, True) if key in self.in_keys]

        # sum_shape = sum(
        #     [
        #         observation_spec[key].shape[self.dim]
        #         if observation_spec[key].shape
        #         else 1
        #         for key in keys
        #     ]
        # )
        spec0 = observation_spec[keys[0]]
        out_key = self.out_keys[0]
        shape = list(spec0.shape)
        shape.append(len(keys))
        device = spec0.device
        # shape[self.dim] = sum_shape
        shape = torch.Size(shape)
        observation_spec[out_key] = Unbounded(
            shape=shape,
            dtype=spec0.dtype,
            device=device,
        )
        if self._del_keys:
            for key in self.keys_to_exclude:
                if key in observation_spec.keys(True):
                    del observation_spec[key]
        return observation_spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_keys={self.in_keys}, out_key"
            f"={self.out_keys[0]})"
        )


class PyGObservationTransform(ObservationTransform):
    """
    Transforms a td object into a PyG Data object
    What is in td:
        q: torch.Tensor of shape [B, N]
        s: torch.Tensor of shape [B, N]
        arrival_rate: torch.Tensor of shape [B, N]
        service_rate: torch.Tensor of shape [B, N]
        context_id: torch.Tensor of shape [B, 1]
        reward: torch.Tensor of shape [B, 1]
        valid_action: torch.Tensor of shape [B, N]
        done: torch.Tensor of shape [B, 1]
        adj: torch.Tensor of shape [N, N]
        observation: torch.Tensor of shape [B, N*K] where K is the number of keys in the observation

        We only want to convert observation and adj into a PyG Data object, where each node

    """


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