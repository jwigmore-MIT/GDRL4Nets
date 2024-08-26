from torchrl.collectors import SyncDataCollector
from torchrl.collectors.utils import split_trajectories
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, Union
from tensordict import TensorDictBase
from torchrl.envs.utils import ExplorationType
DEFAULT_EXPLORATION_TYPE  = ExplorationType.RANDOM




class MultiEnvSyncDataCollector(SyncDataCollector):
    """
    A SyncDataCollector that samples from the environment generator on each iteration

    """

    def __init__(
            self,
            create_env_fn,
            policy,
            *,
            frames_per_batch,
            total_frames=-1,
            device=None,
            storing_device=None,
            policy_device=None,
            env_device=None,
            create_env_kwargs=None,
            max_frames_per_traj=None,
            init_random_frames=None,
            reset_at_each_iter=False,
            postproc=None,
            split_trajs=None,
            exploration_type=DEFAULT_EXPLORATION_TYPE,
            exploration_mode=None,
            return_same_td=False,
            reset_when_done=True,
            interruptor=None,
            env_generator=None,
    ):
        super().__init__(
            create_env_fn,
            policy,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            device=device,
            storing_device=storing_device,
            policy_device=policy_device,
            env_device=env_device,
            create_env_kwargs=create_env_kwargs,
            max_frames_per_traj=max_frames_per_traj,
            init_random_frames=init_random_frames,
            reset_at_each_iter=reset_at_each_iter,
            postproc=postproc,
            split_trajs=split_trajs,
            exploration_type=exploration_type,
            exploration_mode=exploration_mode,
            return_same_td=return_same_td,
            reset_when_done=reset_when_done,
            interruptor=interruptor
        )

        self.env_generator = env_generator
        self.env = self.env_generator()

    # def iterator(self) -> Iterator[TensorDictBase]:
    #     """
    #     modifies the parent iterator to sample from the environment generator
    #     """
    #     self.env = self.env_generator()
    #     return super().iterator()
    #
    def rollout(self):
        if self.reset_at_each_iter:
            self.env = self.env_generator()
        return super().rollout()
