import os
from datetime import datetime
from pathlib import Path

import numpy as np
from gymnasium.spaces import Box
from imitation.algorithms import bc
from imitation.data.rollout import flatten_trajectories
from imitation.data.types import Trajectory, TransitionsMinimal, Transitions
from imitation.policies.serialize import save_stable_model
from sofa_env.base import RenderMode
from sofa_env.scenes.ligating_loop.ligating_loop_env import LigatingLoopEnv, ObservationType, ActionType
from gymnasium.wrappers import TimeLimit
from sofa_env.wrappers.point_cloud import PointCloudFromDepthImageObservationWrapper
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Data, Batch

from policy.PointNetActorCritic import PointNetFeaturesExtractor, PointNetActorCriticPolicy
from util.data import make_transitions
from util.evaluate_policy import evaluate_policy
from util.wrappers import RolloutInfoWrapper



def _make_env():
    """Helper function to create a single environment. Put any logic here, but make sure to return a RolloutInfoWrapper."""
    _env = LigatingLoopEnv(
        observation_type=ObservationType.RGBD,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.CONTINUOUS,
        image_shape=(256, 256),
        frame_skip=1,
        time_step=0.1,
        settle_steps=50,
    )
    _env = PointCloudFromDepthImageObservationWrapper(_env)
    _env = TimeLimit(_env, max_episode_steps=1)
    _env = RolloutInfoWrapper(_env)
    return _env


num_traj = 10
env = DummyVecEnv([_make_env for _ in range(1)])
#demos = _npz_to_traj(1)

#transitions = flatten_trajectories(demos)
rng = np.random.default_rng()
obs_array_shape = (65536, 3)
observation_space = Box(low=float('-inf'), high=float('inf'), shape=obs_array_shape, dtype='float32')
#policy = ActorCriticPolicy(observation_space, env.action_space, lambda epoch: 1e-3 * 0.99 ** epoch, [256, 128], features_extractor_class=PointNetFeaturesExtractor)
policy = PointNetActorCriticPolicy(observation_space, env.action_space, lambda epoch: 1e-3 * 0.99 ** epoch, [256, 128])
demos = make_transitions(n_traj=1)

bc_trainer = bc.BC_Pyg(
    observation_space=observation_space,
    action_space=env.action_space,
    demonstrations=demos,
    policy=policy,
    rng=rng,
    device='cuda',
    batch_size=8,
    minibatch_size=2,
)
#reward_before_training, _ = evaluate_policy(bc_trainer.policy, _make_env(), 1)
#print(f"Reward before training: {reward_before_training}")
# for i in range(1, num_traj):
#     demos = make_transitions(1, i)
#     bc_trainenr.set
#     bc_trainer.train(n_epochs=1)

bc_trainer.train(n_epochs=1)
saved_time = datetime.now().strftime('%Y-%m-%d_%H:%M')
save_stable_model(Path(f'./model/ligating_loop/{saved_time}'), bc_trainer.policy)
print('Saved_model')

reward_after_training, _ = evaluate_policy(bc_trainer.policy, _make_env(), 10)
print(f"Reward after training: {reward_after_training}")

print('done')
