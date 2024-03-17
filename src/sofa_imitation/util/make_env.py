from sofa_env.base import SofaEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from .wrappers import RolloutInfoWrapper, SofaEnvPointCloudObservations, WatchdogVecEnv, PygDataObservationWrapper


def make_env(sofa_env: SofaEnv, use_color: bool = False, max_episode_steps=500, grid_size=1, depth_cutoff: int = None):
    env = SofaEnvPointCloudObservations(sofa_env, depth_cutoff, max_expected_num_points=256 * 256, color=use_color)
   # env = PygDataObservationWrapper(env, grid_size)
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = RolloutInfoWrapper(env)
    return env


# def make_env_func(sofa_env: SofaEnv, use_color: bool = False):
#     """Helper function to create a single environment. Put any logic here, but make sure to return a
#     RolloutInfoWrapper."""
#
#     def make_env():
#         env = SofaEnvPointCloudObservations(sofa_env, 220, max_expected_num_points=256 * 256, color=use_color)
#         env = PygDataObservationWrapper(env, )
#         env = Monitor(env)
#         env = TimeLimit(env, max_episode_steps=500)
#         env = RolloutInfoWrapper(env)
#         return env
#
#     return make_env
#
#
# def make_vec_sofa_env(sofa_env: SofaEnv, use_color: bool = False):
#     return WatchdogVecEnv([lambda : make_env(sofa_env, use_color, 500, 220)], step_timeout_sec=45)
