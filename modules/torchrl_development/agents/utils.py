from torchrl.modules import ReparamGradientStrategy
from torchrl.modules.distributions.discrete import _one_hot_wrapper
import torch.distributions as D

from torchrl.modules import  MaskedOneHotCategorical,MaskedCategorical
import torch

from typing import Optional, Sequence, Union



class MaskedOneHotCategorical(MaskedCategorical):
    """MaskedOneHotCategorical distribution.
    Like MaskedCategorical, but samples are one-hot encoded.

    Reference:
    https://www.tensorflow.org/agents/api_docs/python/tf_agents/distributions/masked/MaskedCategorical

    Args:
        logits (torch.Tensor): event log probabilities (unnormalized)
        probs (torch.Tensor): event probabilities. If provided, the probabilities
            corresponding to to masked items will be zeroed and the probability
            re-normalized along its last dimension.

    Keyword Args:
        mask (torch.Tensor): A boolean mask of the same shape as ``logits``/``probs``
            where ``False`` entries are the ones to be masked. Alternatively,
            if ``sparse_mask`` is True, it represents the list of valid indices
            in the distribution. Exclusive with ``indices``.
        indices (torch.Tensor): A dense index tensor representing which actions
            must be taken into account. Exclusive with ``mask``.
        neg_inf (float, optional): The log-probability value allocated to
            invalid (out-of-mask) indices. Defaults to -inf.
        padding_value: The padding value in then mask tensor when
            sparse_mask == True, the padding_value will be ignored.
        grad_method (ReparamGradientStrategy, optional): strategy to gather
            reparameterized samples.
            ``ReparamGradientStrategy.PassThrough`` will compute the sample gradients
             by using the softmax valued log-probability as a proxy to the
             samples gradients.
            ``ReparamGradientStrategy.RelaxedOneHot`` will use
            :class:`torch.distributions.RelaxedOneHot` to sample from the distribution.

        >>> torch.manual_seed(0)
        >>> logits = torch.randn(4) / 100  # almost equal probabilities
        >>> mask = torch.tensor([True, False, True, True])
        >>> dist = MaskedOneHotCategorical(logits=logits, mask=mask)
        >>> sample = dist.sample((10,))
        >>> print(sample)  # no `1` in the sample
        tensor([[0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0]])
        >>> print(dist.log_prob(sample))
        tensor([-1.1203, -1.0928, -1.0831, -1.1203, -1.1203, -1.0831, -1.1203, -1.0831,
                -1.1203, -1.1203])
        >>> sample_non_valid = torch.zeros_like(sample)
        >>> sample_non_valid[..., 1] = 1
        >>> print(dist.log_prob(sample_non_valid))
        tensor([-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf])
        >>> # with probabilities
        >>> prob = torch.ones(10)
        >>> prob = prob / prob.sum()
        >>> mask = torch.tensor([False] + 9 * [True])  # first outcome is masked
        >>> dist = MaskedOneHotCategorical(probs=prob, mask=mask)
        >>> s = torch.arange(10)
        >>> s = torch.nn.functional.one_hot(s, 10)
        >>> print(dist.log_prob(s))
        tensor([   -inf, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972,
                -2.1972, -2.1972])
    """

    def __init__(
            self,
            logits: Optional[torch.Tensor] = None,
            probs: Optional[torch.Tensor] = None,
            mask: torch.Tensor = None,
            indices: torch.Tensor = None,
            neg_inf: float = float("-inf"),
            padding_value: Optional[int] = None,
            grad_method: ReparamGradientStrategy = ReparamGradientStrategy.PassThrough,
            temperature: float = 0.1,
    ) -> None:
        self.grad_method = grad_method
        self.has_rsample = True
        self.temperature = temperature
        if logits is not None:
            logits = logits / temperature


        super().__init__(
            logits=logits,
            probs=probs,
            mask=mask,
            indices=indices,
            neg_inf=neg_inf,
            padding_value=padding_value,
        )

    @_one_hot_wrapper(MaskedCategorical)
    def sample(
            self, sample_shape: Optional[Union[torch.Size, Sequence[int]]] = None
    ) -> torch.Tensor:
        ...

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # this calls the log_prob method of the MaskedCategorical class
        log_prob = super().log_prob(value.argmax(dim=-1, keepdim=False))
        # if value.shape[0] > 1:
        #     log_prob = log_prob.unsqueeze(-1)
        return log_prob

    @property
    def mode(self) -> torch.Tensor:
        if hasattr(self, "logits"):
            # get the argmax of the logits
            """
            logits is either (W,N) or (N,)
            logits.argmax() is (W,)
            Want to return (W,N) or (N,) tensor of one-hot vectors
            How do I do this?

            """
            argmax = self.logits.argmax(dim=-1, keepdim=True)
            argmax_one_hot = torch.zeros_like(self.logits)
            argmax_one_hot.scatter_(-1, argmax, 1)
            return argmax_one_hot.to(torch.long)

        else:
            return (self.probs == self.probs.max(-1, True)[0]).to(torch.long)

    def rsample(self, sample_shape: Union[torch.Size, Sequence] = None) -> torch.Tensor:
        if sample_shape is None:
            sample_shape = torch.Size([])
        if hasattr(self, "logits") and self.logits is not None:
            logits = self.logits
            probs = None
        else:
            logits = None
            probs = self.probs
        if self.grad_method == ReparamGradientStrategy.RelaxedOneHot:
            if self._sparse_mask:
                if probs is not None:
                    probs_extended = torch.full(
                        (*probs.shape[:-1], self.num_samples),
                        0,
                        device=probs.device,
                        dtype=probs.dtype,
                    )
                    probs_extended = torch.scatter(
                        probs_extended, -1, self._mask, probs
                    )
                    logits_extended = None
                else:
                    probs_extended = torch.full(
                        (*logits.shape[:-1], self.num_samples),
                        self.neg_inf,
                        device=logits.device,
                        dtype=logits.dtype,
                    )
                    logits_extended = torch.scatter(
                        probs_extended, -1, self._mask, logits
                    )
                    probs_extended = None
            else:
                probs_extended = probs
                logits_extended = logits

            d = D.relaxed_categorical.RelaxedOneHotCategorical(
                1, probs=probs_extended, logits=logits_extended
            )
            out = d.rsample(sample_shape)
            out.data.copy_((out == out.max(-1)[0].unsqueeze(-1)).to(out.dtype))
            return out
        elif self.grad_method == ReparamGradientStrategy.PassThrough:
            if logits is not None:
                probs = self.probs
            else:
                probs = torch.softmax(self.logits, dim=-1)
            if self._sparse_mask:
                probs_extended = torch.full(
                    (*probs.shape[:-1], self.num_samples),
                    0,
                    device=probs.device,
                    dtype=probs.dtype,
                )
                probs_extended = torch.scatter(probs_extended, -1, self._mask, probs)
            else:
                probs_extended = probs

            out = self.sample(sample_shape)
            out = out + probs_extended - probs_extended.detach()
            return out
        else:
            raise ValueError(
                f"Unknown reparametrization strategy {self.reparam_strategy}."
            )