import logging

import datasets
from datetime import datetime
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import sqil
from imitation.data import huggingface_utils

from util.env_from_string import get_env, get_grid_size_from_string, action_dim_from_string
from util.data import npz_to_transitions, npz_to_state_transitions
from util.evaluate_policy import evaluate_policy
from policy.PointNetActorCritic import PointNetFeaturesExtractor, PointNetActorCriticPolicy

log = logging.getLogger(__name__)
def run_AIRL(env_name: str, env_prefix: str, train_steps: int, demo_batch_size: int, batch_size: int = 256, learning_rate: float = 0.0001,
             num_traj: int = 2, n_eval: int = 0, use_state: bool = False, use_color: bool = False):
    # path = f'../../../sofa_env_demonstrations/{env_name}'
    path = f'/media/erik/Volume/sofa_env_demonstrations/{env_name}'
    start_time = datetime.now().strftime('%Y-%m-%d_%H:%M')
    grid_size = get_grid_size_from_string(env_name)

    assert isinstance(learning_rate, float) or isinstance(learning_rate, int)

    env = get_env(env_name, use_state, should_render=False)

    if use_state:
        demos = npz_to_state_transitions(path, env_prefix, num_traj)
        policy = MlpPolicy
        policy_kwargs = {}


    else:
        demos = npz_to_transitions(path, env_prefix, num_traj, use_color, grid_size['Demo'])
        policy = PointNetActorCriticPolicy
        policy_kwargs = {
            'net_arch': [256, 128, 64, 32],
            'grid_size': grid_size['FeatureExtractor'],
            'inp_features_dim': 3 if use_color else 0
        }

    log.info('Finished loading train data')
    sqil_trainer = sqil.SQIL(
        venv=env,
        demonstrations=demos,
        policy=policy,
        policy_kwargs=policy_kwargs,
    )
    # Hint: set to 1_000_000 to match the expert performance.
    sqil_trainer.train(total_timesteps=1_000_000)
    reward, _ = evaluate_policy(sqil_trainer.policy, sqil_trainer.venv, 10)
    print("Reward:", reward)