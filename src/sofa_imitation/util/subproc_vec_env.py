from typing import List, Callable, Optional

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs

import gymnasium as gym
import numpy as np

class CustomSubprocVecEnv(SubprocVecEnv):
    # instantly returns obs without flattening

    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        assert len(env_fns) == 1, print('Only handles 1 process')
        super().__init__(env_fns, start_method)

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)  # type: ignore[assignment]
        return obs, np.stack(rews), np.stack(dones), infos  # type: ignore[return-value]

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        obs, self.reset_infos = zip(*results)  # type: ignore[assignment]
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return obs
