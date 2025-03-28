"""Common aliases for type hints"""

from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Protocol, SupportsFloat, Union

import gymnasium as gym
import numpy as np
import torch as th

# Avoid circular imports, we use type hint as string to avoid it too

if TYPE_CHECKING:
    from stable_baselines.common.callbacks import BaseCallback
    from stable_baselines.common.vec_env import VecEnv

GymEnv = Union[gym.Env, "VecEnv"]
GymObs = Union[tuple, dict[str, Any], np.ndarray, int]
GymResetReturn = tuple[GymObs, dict]
AtariResetReturn = tuple[np.ndarray, dict[str, Any]]
GymStepReturn = tuple[GymObs, float, bool, bool, dict]
AtariStepReturn = tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]
TensorDict = dict[str, th.Tensor]
OptimizerStateDict = dict[str, Any]
MaybeCallback = Union[None, Callable, list["BaseCallback"], "BaseCallback"]
PyTorchObs = Union[th.Tensor, TensorDict]

# A schedule takes the remaining progress as input
# and outputs a scalar (e.g. learning rate, clip range, ...)
Schedule = Callable[[float], float]

# A schedule takes the remaining progress as input
# and ouputs a scalar (e.g. learning rate, clip range, ...)
Schedule = Callable[[float], float]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class DictRolloutBufferSamples(RolloutBufferSamples):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class DictReplayBufferSamples(ReplayBufferSamples):
    observations: TensorDict
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class RolloutReturn(NamedTuple):
    episode_reward: float
    episode_timesteps: int
    n_episodes: int
    continue_training: bool


class TrainFrequencyUnit(Enum):
    STEP = "step"
    EPISODE = "episode"


class TrainFreq(NamedTuple):
    frequency: int
    unit: TrainFrequencyUnit  # either "step" or "episode"