import os
from pathlib import Path

import numpy as np
import open3d as o3d
from gymnasium.spaces import Box
from imitation.algorithms import bc
from imitation.data.rollout import flatten_trajectories
from imitation.data.types import Trajectory, TransitionsMinimal, Transitions
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import save_stable_model
from sofa_env.base import RenderMode
from sofa_env.scenes.ligating_loop.ligating_loop_env import LigatingLoopEnv, ObservationType, ActionType
from gymnasium.wrappers import TimeLimit
from sofa_env.wrappers.point_cloud import PointCloudFromDepthImageObservationWrapper
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import PointNetConv, fps, radius, global_max_pool, MLP


def evaluate_policy(
    model,
    env,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback = None,
    reward_threshold = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::oki
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False

    n_envs = 1
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations, infos = env.reset()
    states = None
    episode_starts = np.ones((1,), dtype=bool)

    while (episode_counts < episode_count_targets).any():
        actions, _ = model.predict(
            observations
        )
        new_observations, rewards, dones, truncated, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards
                done = dones
                info = infos
                episode_starts = done

                if callback is not None:
                    callback(locals(), globals())

                if dones:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.r = r
        self.ratio = ratio
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNetFeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, features_dim], norm=None)

    def forward(self, observations: Data) -> torch.Tensor:
        if isinstance(observations, torch.Tensor):
            # num_points = torch.full((observations.shape[0],), 1)
            # batch = torch.repeat_interleave(
            #     torch.arange(len(num_points), device=num_points.device),
            #     repeats=num_points,
            # )
            # flattened_observations = torch.flatten(observations, end_dim=1)
            observations = Data(pos=observations, batch=torch.full((len(observations),), 0)).to(observations.device)

        sa0_out = (None, observations.pos.to(torch.float32), observations.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        return self.mlp(x).log_softmax(dim=-1) # returns 2 stacked vectors



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

def load_point_clouds_from_directory(directory):
    pcds = []
    file_list = [filename for filename in os.listdir(directory) if filename.endswith('.ply')]
    file_list.sort()

    # Iterate over the sorted file list
    for filename in file_list:
        if filename.endswith('.ply'):  # Adjust the file extension if needed
            file_path = os.path.join(directory, filename)

            # Read the PointCloud from the file
            pcd = o3d.io.read_point_cloud(file_path)
            pcds.append(Data(pos=torch.from_numpy(np.asarray(pcd.points)), num_nodes=len(pcd.points)))
    return pcds

def make_trajectories(n_traj = 10):
    ret = []
    for i in range(n_traj):
        pcds = load_point_clouds_from_directory(f'/home/erik/sofa_env_demonstrations/pointclouds/ligating_loop'
                                                f'/LigatingLoopEnv_{i}')
        npz_data = np.load(f'/home/erik/sofa_env_demonstrations/ligating_loop/LigatingLoopEnv_{i}.npz')
        print(f'{i + 1}/{n_traj}')
        ret.append(Trajectory(pcds,
                              npz_data['action'], None, True))

    return ret

def make_transitions(n_traj = 10, start=0):
    obs = []
    next_obs = []
    actions = []
    dones = []
    for i in range(start, start + n_traj):
        pcds = load_point_clouds_from_directory(f'/home/erik/sofa_env_demonstrations/pointclouds/ligating_loop'
                                                f'/LigatingLoopEnv_{i}')
        npz_data = np.load(f'/home/erik/sofa_env_demonstrations/ligating_loop/LigatingLoopEnv_{i}.npz')
        action = npz_data['action']
        done = np.zeros(len(action), dtype=bool)
        done[-1] = True

        obs.extend(pcds[:-1])
        next_obs.extend(pcds[1:])
        actions.extend(action)
        dones.extend(done)
        print(f'{i + 1}/{n_traj}: actions:{len(actions)}  observations:{len(obs)}')


    return Transitions(obs, np.asarray(actions), np.array([{}] * len(actions)), next_obs,  np.asarray(dones))




num_traj = 10
env = DummyVecEnv([_make_env for _ in range(1)])
#demos = _npz_to_traj(1)

#transitions = flatten_trajectories(demos)
rng = np.random.default_rng()
obs_array_shape = (65536, 3)
observation_space = Box(low=float('-inf'), high=float('inf'), shape=obs_array_shape, dtype='float32')
policy = ActorCriticPolicy(observation_space, env.action_space, lambda epoch: 1e-3 * 0.99 ** epoch, [256, 128], features_extractor_class=PointNetFeaturesExtractor)

demos = make_transitions(n_traj=10)

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
#reward_before_training, _ = evaluate_policy(bc_trainer.policy, _make_env(), 10)
#print(f"Reward before training: {reward_before_training}")
# for i in range(1, num_traj):
#     demos = make_transitions(1, i)
#     bc_trainenr.set
#     bc_trainer.train(n_epochs=1)
bc_trainer.train(n_epochs=1)
save_stable_model(Path('./model'), bc_trainer.policy)

reward_after_training, _ = evaluate_policy(bc_trainer.policy, _make_env(), 10)
print(f"Reward after training: {reward_after_training}")

print('done')
