from sofa_env.base import SofaEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from .wrappers import RolloutInfoWrapper, SofaEnvPointCloudObservations


def make_env(sofa_env: SofaEnv, use_color: bool = False):
    env = SofaEnvPointCloudObservations(sofa_env, 220, max_expected_num_points=256 * 256, color=use_color)
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps=500)
    env = RolloutInfoWrapper(env)
    return env


def make_env_func(sofa_env: SofaEnv, use_color: bool = False):
    """Helper function to create a single environment. Put any logic here, but make sure to return a
    RolloutInfoWrapper."""

    def make_env():
        env = SofaEnvPointCloudObservations(sofa_env, 220, max_expected_num_points=256 * 256, color=use_color)
        env = Monitor(env)
        env = TimeLimit(env, max_episode_steps=500)
        env = RolloutInfoWrapper(env)
        return env

    return make_env


def make_vec_sofa_env(sofa_env: SofaEnv, use_color: bool = False):
    env_kwargs = {'sofa_env': sofa_env, 'use_color': use_color}
    return make_vec_env(make_env, n_envs=1, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv)
