import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import (
    dispatch,
    TensorDictModule,
    TensorDictModuleBase,
)
import warnings
from typing import Optional, Sequence, Tuple, Union
from tensordict.utils import NestedKey
from torchrl.data.tensor_specs import CompositeSpec, TensorSpec
from torchrl.modules.tensordict_module.common import SafeModule
from tensordict import TensorDictBase
from torchrl.data.utils import _process_action_space_spec
from torchrl.modules.tensordict_module.sequence import SafeSequential


"""
For Value based methods, creates actors that can interact with TorchRL environments
"""

class MinQValueModule(TensorDictModuleBase):
    """Q-Value TensorDictModule for Q-value policies.

    This module processes a tensor containing action value into is argmax
    component (i.e. the resulting greedy action), following a given
    action space (one-hot, binary or categorical).
    It works with both tensordict and regular tensors.

    Args:
        action_space (str, optional): Action space. Must be one of
            ``"one-hot"``, ``"mult-one-hot"``, ``"binary"`` or ``"categorical"``.
            This argument is exclusive with ``spec``, since ``spec``
            conditions the action_space.
        action_value_key (str or tuple of str, optional): The input key
            representing the action value. Defaults to ``"action_value"``.
        action_mask_key (str or tuple of str, optional): The input key
            representing the action mask. Defaults to ``"None"`` (equivalent to no masking).
        out_keys (list of str or tuple of str, optional): The output keys
            representing the actions, action values and chosen action value.
            Defaults to ``["action", "action_value", "chosen_action_value"]``.
        var_nums (int, optional): if ``action_space = "mult-one-hot"``,
            this value represents the cardinality of each
            action component.
        spec (TensorSpec, optional): if provided, the specs of the action (and/or
            other outputs). This is exclusive with ``action_space``, as the spec
            conditions the action space.
        safe (bool): if ``True``, the value of the output is checked against the
            input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow issues.
            If this value is out of bounds, it is projected back onto the
            desired space using the :obj:`TensorSpec.project`
            method. Default is ``False``.

    Returns:
        if the input is a single tensor, a triplet containing the chosen action,
        the values and the value of the chose action is returned. If a tensordict
        is provided, it is updated with these entries at the keys indicated by the
        ``out_keys`` field.

    Examples:
        >>> from tensordict import TensorDict
        >>> action_space = "categorical"
        >>> action_value_key = "my_action_value"
        >>> actor = QValueModule(action_space, action_value_key=action_value_key)
        >>> # This module works with both tensordict and regular tensors:
        >>> value = torch.zeros(4)
        >>> value[-1] = 1
        >>> actor(my_action_value=value)
        (tensor(3), tensor([0., 0., 0., 1.]), tensor([1.]))
        >>> actor(value)
        (tensor(3), tensor([0., 0., 0., 1.]), tensor([1.]))
        >>> actor(TensorDict({action_value_key: value}, []))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                action_value: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                my_action_value: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        action_space: Optional[str],
        action_value_key: Optional[NestedKey] = None,
        action_mask_key: Optional[NestedKey] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
        var_nums: Optional[int] = None,
        spec: Optional[TensorSpec] = None,
        safe: bool = False,
    ):
        if isinstance(action_space, TensorSpec):
            warnings.warn(
                "Using specs in action_space will be deprecated in v0.4.0,"
                " please use the 'spec' argument if you want to provide an action spec",
                category=DeprecationWarning,
            )
        action_space, spec = _process_action_space_spec(action_space, spec)
        self.action_space = action_space
        self.var_nums = var_nums
        self.action_func_mapping = {
            "one_hot": self._one_hot,
            "mult_one_hot": self._mult_one_hot,
            "binary": self._binary,
            "categorical": self._categorical,
        }
        self.action_value_func_mapping = {
            "categorical": self._categorical_action_value,
        }
        if action_space not in self.action_func_mapping:
            raise ValueError(
                f"action_space must be one of {list(self.action_func_mapping.keys())}, got {action_space}"
            )
        if action_value_key is None:
            action_value_key = "action_value"
        self.action_mask_key = action_mask_key
        in_keys = [action_value_key]
        if self.action_mask_key is not None:
            in_keys.append(self.action_mask_key)
        self.in_keys = in_keys
        if out_keys is None:
            out_keys = ["action", action_value_key, "chosen_action_value"]
        elif action_value_key not in out_keys:
            raise RuntimeError(
                f"Expected the action-value key to be '{action_value_key}' but got {out_keys[1]} instead."
            )
        self.out_keys = out_keys
        action_key = out_keys[0]
        if not isinstance(spec, CompositeSpec):
            spec = CompositeSpec({action_key: spec})
        super().__init__()
        self.register_spec(safe=safe, spec=spec)

    register_spec = SafeModule.register_spec

    @property
    def spec(self) -> CompositeSpec:
        return self._spec

    @spec.setter
    def spec(self, spec: CompositeSpec) -> None:
        if not isinstance(spec, CompositeSpec):
            raise RuntimeError(
                f"Trying to set an object of type {type(spec)} as a tensorspec but expected a CompositeSpec instance."
            )
        self._spec = spec

    @property
    def action_value_key(self):
        return self.in_keys[0]

    @dispatch(auto_batch_size=False)
    def forward(self, tensordict: torch.Tensor) -> TensorDictBase:
        action_values = tensordict.get(self.action_value_key, None)
        if action_values is None:
            raise KeyError(
                f"Action value key {self.action_value_key} not found in {tensordict}."
            )
        if self.action_mask_key is not None:
            action_mask = tensordict.get(self.action_mask_key, None)
            if action_mask is None:
                raise KeyError(
                    f"Action mask key {self.action_mask_key} not found in {tensordict}."
                )
            action_values = torch.where(
                action_mask, action_values, torch.finfo(action_values.dtype).max
            )

        action = self.action_func_mapping[self.action_space](action_values)

        action_value_func = self.action_value_func_mapping.get(
            self.action_space, self._default_action_value
        )
        chosen_action_value = action_value_func(action_values, action)
        tensordict.update(
            dict(zip(self.out_keys, (action, action_values, chosen_action_value)))
        )
        return tensordict

    @staticmethod
    def _one_hot(value: torch.Tensor) -> torch.Tensor:
        out = (value == value.min(dim=-1, keepdim=True)[0]).to(torch.long)
        return out

    @staticmethod
    def _categorical(value: torch.Tensor) -> torch.Tensor:
        return torch.argmin(value, dim=-1).to(torch.long)

    def _mult_one_hot(
        self, value: torch.Tensor, support: torch.Tensor = None
    ) -> torch.Tensor:
        if self.var_nums is None:
            raise ValueError(
                "var_nums must be provided to the constructor for multi one-hot action spaces."
            )
        values = value.split(self.var_nums, dim=-1)
        return torch.cat(
            [
                self._one_hot(
                    _value,
                )
                for _value in values
            ],
            -1,
        )

    @staticmethod
    def _binary(value: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _default_action_value(
        values: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return (action * values).sum(-1, True)

    @staticmethod
    def _categorical_action_value(
        values: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return values.gather(-1, action.unsqueeze(-1))
        # if values.ndim == 1:
        #     return values[action].unsqueeze(-1)
        # batch_size = values.size(0)
        # return values[range(batch_size), action].unsqueeze(-1)


class MinQValueActor(SafeSequential):
    """A Q-Value actor class.

    This class appends a :class:`~.QValueModule` after the input module
    such that the action values are used to select an action.

    Args:
        module (nn.Module): a :class:`torch.nn.Module` used to map the input to
            the output parameter space. If the class provided is not compatible
            with :class:`tensordict.nn.TensorDictModuleBase`, it will be
            wrapped in a :class:`tensordict.nn.TensorDictModule` with
            ``in_keys`` indicated by the following keyword argument.

    Keyword Args:
        in_keys (iterable of str, optional): If the class provided is not
            compatible with :class:`tensordict.nn.TensorDictModuleBase`, this
            list of keys indicates what observations need to be passed to the
            wrapped module to get the action values.
            Defaults to ``["observation"]``.
        spec (TensorSpec, optional): Keyword-only argument.
            Specs of the output tensor. If the module
            outputs multiple output tensors,
            spec characterize the space of the first output tensor.
        safe (bool): Keyword-only argument.
            If ``True``, the value of the output is checked against the
            input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow
            issues. If this value is out of bounds, it is projected back onto the
            desired space using the :obj:`TensorSpec.project`
            method. Default is ``False``.
        action_space (str, optional): Action space. Must be one of
            ``"one-hot"``, ``"mult-one-hot"``, ``"binary"`` or ``"categorical"``.
            This argument is exclusive with ``spec``, since ``spec``
            conditions the action_space.
        action_value_key (str or tuple of str, optional): if the input module
            is a :class:`tensordict.nn.TensorDictModuleBase` instance, it must
            match one of its output keys. Otherwise, this string represents
            the name of the action-value entry in the output tensordict.
        action_mask_key (str or tuple of str, optional): The input key
            representing the action mask. Defaults to ``"None"`` (equivalent to no masking).

    .. note::
        ``out_keys`` cannot be passed. If the module is a :class:`tensordict.nn.TensorDictModule`
        instance, the out_keys will be updated accordingly. For regular
        :class:`torch.nn.Module` instance, the triplet ``["action", action_value_key, "chosen_action_value"]``
        will be used.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torch import nn
        >>> from torchrl.data import OneHotDiscreteTensorSpec
        >>> from torchrl.modules.tensordict_module.actors import QValueActor
        >>> td = TensorDict({'observation': torch.randn(5, 4)}, [5])
        >>> # with a regular nn.Module
        >>> module = nn.Linear(4, 4)
        >>> action_spec = OneHotDiscreteTensorSpec(4)
        >>> qvalue_actor = QValueActor(module=module, spec=action_spec)
        >>> td = qvalue_actor(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.int64, is_shared=False),
                action_value: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([5]),
            device=None,
            is_shared=False)
        >>> # with a TensorDictModule
        >>> td = TensorDict({'obs': torch.randn(5, 4)}, [5])
        >>> module = TensorDictModule(lambda x: x, in_keys=["obs"], out_keys=["action_value"])
        >>> action_spec = OneHotDiscreteTensorSpec(4)
        >>> qvalue_actor = QValueActor(module=module, spec=action_spec)
        >>> td = qvalue_actor(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.int64, is_shared=False),
                action_value: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                obs: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([5]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        module,
        *,
        in_keys=None,
        spec=None,
        safe=False,
        action_space: Optional[str] = None,
        action_value_key=None,
        action_mask_key: Optional[NestedKey] = None,
    ):
        if isinstance(action_space, TensorSpec):
            warnings.warn(
                "Using specs in action_space will be deprecated v0.4.0,"
                " please use the 'spec' argument if you want to provide an action spec",
                category=DeprecationWarning,
            )
        action_space, spec = _process_action_space_spec(action_space, spec)

        self.action_space = action_space
        self.action_value_key = action_value_key
        if action_value_key is None:
            action_value_key = "action_value"
        out_keys = [
            "action",
            action_value_key,
            "chosen_action_value",
        ]
        if isinstance(module, TensorDictModuleBase):
            if action_value_key not in module.out_keys:
                raise KeyError(
                    f"The key '{action_value_key}' is not part of the module out-keys."
                )
        else:
            if in_keys is None:
                in_keys = ["observation"]
            module = TensorDictModule(
                module, in_keys=in_keys, out_keys=[action_value_key]
            )
        if spec is None:
            spec = CompositeSpec()
        if isinstance(spec, CompositeSpec):
            spec = spec.clone()
            if "action" not in spec.keys():
                spec["action"] = None
        else:
            spec = CompositeSpec(action=spec, shape=spec.shape[:-1])
        spec[action_value_key] = None
        spec["chosen_action_value"] = None
        qvalue = MinQValueModule(
            action_value_key=action_value_key,
            out_keys=out_keys,
            spec=spec,
            safe=safe,
            action_space=action_space,
            action_mask_key=action_mask_key,
        )

        super().__init__(module, qvalue)

class ProbabilisticMinQValueModule(TensorDictModuleBase):
    """Q-Value TensorDictModule for Q-value policies.

    This module processes a tensor containing action value into is argmax
    component (i.e. the resulting greedy action), following a given
    action space (one-hot, binary or categorical).
    It works with both tensordict and regular tensors.

    Args:
        action_space (str, optional): Action space. Must be one of
            ``"one-hot"``, ``"mult-one-hot"``, ``"binary"`` or ``"categorical"``.
            This argument is exclusive with ``spec``, since ``spec``
            conditions the action_space.
        action_value_key (str or tuple of str, optional): The input key
            representing the action value. Defaults to ``"action_value"``.
        action_mask_key (str or tuple of str, optional): The input key
            representing the action mask. Defaults to ``"None"`` (equivalent to no masking).
        out_keys (list of str or tuple of str, optional): The output keys
            representing the actions, action values and chosen action value.
            Defaults to ``["action", "action_value", "chosen_action_value"]``.
        var_nums (int, optional): if ``action_space = "mult-one-hot"``,
            this value represents the cardinality of each
            action component.
        spec (TensorSpec, optional): if provided, the specs of the action (and/or
            other outputs). This is exclusive with ``action_space``, as the spec
            conditions the action space.
        safe (bool): if ``True``, the value of the output is checked against the
            input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow issues.
            If this value is out of bounds, it is projected back onto the
            desired space using the :obj:`TensorSpec.project`
            method. Default is ``False``.

    Returns:
        if the input is a single tensor, a triplet containing the chosen action,
        the values and the value of the chose action is returned. If a tensordict
        is provided, it is updated with these entries at the keys indicated by the
        ``out_keys`` field.

    Examples:
        >>> from tensordict import TensorDict
        >>> action_space = "categorical"
        >>> action_value_key = "my_action_value"
        >>> actor = QValueModule(action_space, action_value_key=action_value_key)
        >>> # This module works with both tensordict and regular tensors:
        >>> value = torch.zeros(4)
        >>> value[-1] = 1
        >>> actor(my_action_value=value)
        (tensor(3), tensor([0., 0., 0., 1.]), tensor([1.]))
        >>> actor(value)
        (tensor(3), tensor([0., 0., 0., 1.]), tensor([1.]))
        >>> actor(TensorDict({action_value_key: value}, []))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                action_value: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                my_action_value: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        action_space: Optional[str],
        action_value_key: Optional[NestedKey] = None,
        action_mask_key: Optional[NestedKey] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
        var_nums: Optional[int] = None,
        spec: Optional[TensorSpec] = None,
        safe: bool = False,
    ):
        if isinstance(action_space, TensorSpec):
            warnings.warn(
                "Using specs in action_space will be deprecated in v0.4.0,"
                " please use the 'spec' argument if you want to provide an action spec",
                category=DeprecationWarning,
            )
        action_space, spec = _process_action_space_spec(action_space, spec)
        self.action_space = action_space
        self.var_nums = var_nums
        self.action_func_mapping = {
            "one_hot": self._one_hot,
            "mult_one_hot": self._mult_one_hot,
            "binary": self._binary,
            "categorical": self._categorical,
        }
        self.action_value_func_mapping = {
            "categorical": self._categorical_action_value,
        }
        if action_space not in self.action_func_mapping:
            raise ValueError(
                f"action_space must be one of {list(self.action_func_mapping.keys())}, got {action_space}"
            )
        if action_value_key is None:
            action_value_key = "action_value"
        self.action_mask_key = action_mask_key
        in_keys = [action_value_key]
        if self.action_mask_key is not None:
            in_keys.append(self.action_mask_key)
        self.in_keys = in_keys
        if out_keys is None:
            out_keys = ["action", action_value_key, "chosen_action_value"]
        elif action_value_key not in out_keys:
            raise RuntimeError(
                f"Expected the action-value key to be '{action_value_key}' but got {out_keys[1]} instead."
            )
        self.out_keys = out_keys
        action_key = out_keys[0]
        if not isinstance(spec, CompositeSpec):
            spec = CompositeSpec({action_key: spec})
        super().__init__()
        self.register_spec(safe=safe, spec=spec)

    register_spec = SafeModule.register_spec

    @property
    def spec(self) -> CompositeSpec:
        return self._spec

    @spec.setter
    def spec(self, spec: CompositeSpec) -> None:
        if not isinstance(spec, CompositeSpec):
            raise RuntimeError(
                f"Trying to set an object of type {type(spec)} as a tensorspec but expected a CompositeSpec instance."
            )
        self._spec = spec

    @property
    def action_value_key(self):
        return self.in_keys[0]

    @dispatch(auto_batch_size=False)
    def forward(self, tensordict: torch.Tensor) -> TensorDictBase:
        action_values = tensordict.get(self.action_value_key, None)
        if action_values is None:
            raise KeyError(
                f"Action value key {self.action_value_key} not found in {tensordict}."
            )
        if self.action_mask_key is not None:
            action_mask = tensordict.get(self.action_mask_key, None)
            if action_mask is None:
                raise KeyError(
                    f"Action mask key {self.action_mask_key} not found in {tensordict}."
                )
            action_values = torch.where(
                action_mask, action_values, torch.finfo(action_values.dtype).max
            )

        action = self.action_func_mapping[self.action_space](action_values)

        action_value_func = self.action_value_func_mapping.get(
            self.action_space, self._default_action_value
        )
        chosen_action_value = action_value_func(action_values, action)
        tensordict.update(
            dict(zip(self.out_keys, (action, action_values, chosen_action_value)))
        )
        return tensordict

    @staticmethod
    def _one_hot(value: torch.Tensor) -> torch.Tensor:
        out = (value == value.min(dim=-1, keepdim=True)[0]).to(torch.long)
        return out

    @staticmethod
    def _categorical(value: torch.Tensor) -> torch.Tensor:
        return torch.argmin(value, dim=-1).to(torch.long)

    def _mult_one_hot(
        self, value: torch.Tensor, support: torch.Tensor = None
    ) -> torch.Tensor:
        if self.var_nums is None:
            raise ValueError(
                "var_nums must be provided to the constructor for multi one-hot action spaces."
            )
        values = value.split(self.var_nums, dim=-1)
        return torch.cat(
            [
                self._one_hot(
                    _value,
                )
                for _value in values
            ],
            -1,
        )

    @staticmethod
    def _binary(value: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _default_action_value(
        values: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return (action * values).sum(-1, True)

    @staticmethod
    def _categorical_action_value(
        values: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return values.gather(-1, action.unsqueeze(-1))
        # if values.ndim == 1:
        #     return values[action].unsqueeze(-1)
        # batch_size = values.size(0)
        # return values[range(batch_size), action].unsqueeze(-1)
