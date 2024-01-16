import numpy as np
from imitation.algorithms import bc
from imitation.data.rollout import flatten_trajectories
from imitation.data.types import Trajectory
from imitation.data.wrappers import RolloutInfoWrapper
from sofa_env.base import RenderMode
from sofa_env.scenes.ligating_loop.ligating_loop_env import LigatingLoopEnv, ObservationType, ActionType
from gymnasium.wrappers import TimeLimit
from sofa_env.wrappers.point_cloud import PointCloudFromDepthImageObservationWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv


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
    _env = TimeLimit(_env, max_episode_steps=500)
    _env = RolloutInfoWrapper(_env)
    return _env


def _npz_to_traj(n_traj: 500):
    ret = []
    for i in range(n_traj):
        path = f'/home/erik/sofa_env_demonstrations/ligating_loop/LigatingLoopEnv_{i}.npz'
        pcds = np.load(f'/home/erik/sofa_env_demonstrations/Pointclouds/LigatingLoopEnv_{i}.npy')
        npz_data = np.load(path)
        print(f'{i + 1}/{n_traj}')
        ret.append(Trajectory(pcds,
                              npz_data['action'], None, True))

    return ret


env = DummyVecEnv([_make_env for _ in range(1)])
demos = _npz_to_traj(1)
transitions = flatten_trajectories(demos)
rng = np.random.default_rng()
policy = ActorCriticPolicy(env.observation_space, env.action_space, lambda epoch: 1e-3 * 0.99 ** epoch, [256, 128])

print(env.observation_space)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    policy=policy,
    rng=rng,
    device='cuda'
)
# bc_trainer.train(n_epochs=2)
# reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
# print(f"Reward after training: {reward_after_training}")
#
# print('done')
