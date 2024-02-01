import numpy as np
import open3d as o3d
import os

def load_point_clouds_from_directory(directory):
    file_list = [filename for filename in os.listdir(directory) if filename.endswith('.ply')]
    file_list.sort()

    # Iterate over the sorted file list
    for filename in file_list:
        if filename.endswith('.ply'):  # Adjust the file extension if needed
            file_path = os.path.join(directory, filename)

            # Read the PointCloud from the file
            pcd = o3d.io.read_point_cloud(file_path)

            o3d.visualization.draw_geometries([pcd])



# Specify the directory containing your PLY files
input_directory = '/home/erik/RL_Praktikum/Pointclouds'

# Load point clouds from the directory
loaded_point_clouds = load_point_clouds_from_directory(input_directory)
