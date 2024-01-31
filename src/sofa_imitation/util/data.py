import os

import numpy as np
import open3d as o3d
import torch
from imitation.data.types import Trajectory, Transitions
from torch_geometric.data import Data


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

