import numpy as np
import open3d as o3d

pcd = o3d.geometry.PointCloud()
npz = np.load('/home/erik/sofa_env_demonstrations/Pointclouds/LigatingLoopEnv_0.npy')

pcd.points = o3d.utility.Vector3dVector(npz[0])

o3d.visualization.draw_geometries([pcd])