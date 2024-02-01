import os
import open3d as o3d
from tqdm import tqdm
import numpy as np


def log(file: int):
    f = open("validate_log.txt", "a")
    f.write(f'{file}, ')
    f.close()


def validatePointclouds(file_directory, ground_truth_dir, visualize: bool = False):
    outer_progress_bar = tqdm(range(500), desc=f'File :')
    for i in outer_progress_bar:
        outer_progress_bar.set_description(f'File {i}')
        gd_path = f'{ground_truth_dir}/LigatingLoopEnv_{i}.npz'
        file_path = f'{file_directory}/LigatingLoopEnv_{i}'

        try:
            npz_data = np.load(gd_path)

            z_far = npz_data['metadata.camera.z_far']
            depths = npz_data['depth']

            file_list = [filename for filename in os.listdir(file_path) if filename.endswith('.npz')]
            file_list.sort()

            if len(file_list) != len(depths):
                print(f'File {i} has {len(file_list)} files, but {len(depths)} are needed')
                log(i)
                continue

            # Iterate over the sorted file list
            for filename in file_list:
                if filename.endswith('.npz'):  # Adjust the file extension if needed
                    file_p = os.path.join(file_path, filename)
                    pcd_data = np.load(file_p)
                    #print(pcd_data.files)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pcd_data['pcds'])
                    pcd.colors = o3d.utility.Vector3dVector(pcd_data['colors'])

                    if pcd_data['pcds'].shape != pcd_data['colors'].shape:
                        log(i)
                        print(
                            f'File {i} with {filename} has points shape {pcd_data["pcds"].shape} and color shape {pcd_data["colors"].shape}')
                        continue

                    if visualize:
                        o3d.visualization.draw_geometries([pcd])

        except Exception as e:
            log(i)
            print(f'File:{i} erroneous, because of {e}')


if __name__ == '__main__':
    # Specify the file_directory containing your PLY files
    input_directory = '/home/erik/RL_Praktikum/Pointclouds/ligating_loop'

    # Load point clouds from the file_directory
    loaded_point_clouds = validatePointclouds(input_directory, '/home/erik/sofa_env_demonstrations/ligating_loop', False)
