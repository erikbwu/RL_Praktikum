from multiprocessing import Pool
from pathlib import Path

import open3d as o3d
import numpy as np
from tqdm import tqdm

output_directory = "/home/erik/RL_Praktikum/Pointclouds/ligating_loop"


def convertPointcloudNumpy(startidx: int, endidx: int, step: int = 1):
    outer_progress_bar = tqdm(range(startidx, endidx, step), desc=f'File :')

    for i in outer_progress_bar:
        outer_progress_bar.set_description(f'File {i}')
        path = f'/home/erik/sofa_env_demonstrations/ligating_loop/LigatingLoopEnv_{i}.npz'
        try:
            npz_data = np.load(path)
            # print(len(npz_data['rgb']))

            width = npz_data['metadata.camera.width']
            height = npz_data['metadata.camera.height']
            focal_x = npz_data['metadata.camera.focal_x']
            focal_y = npz_data['metadata.camera.focal_y']
            z_near = npz_data['metadata.camera.z_near']
            z_far = npz_data['metadata.camera.z_far']

            rgbs = npz_data['rgb']
            depths = npz_data['depth']

            for timestep in tqdm(range(len(rgbs)), leave=False):
                rgb = rgbs[timestep]
                depth = depths[timestep]

                depth = np.where(depth >= z_far, 0, depth)

                color_image = o3d.geometry.Image(rgb)
                depth_image = o3d.geometry.Image(depth)

                intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, focal_x, focal_y, width / 2, height / 2)

                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image,
                                                                                convert_rgb_to_intensity=False)
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image,
                    intrinsics)

                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                Path(f'{output_directory}/LigatingLoopEnv_{i}').mkdir(parents=True, exist_ok=True)
                file_name = f'{output_directory}/LigatingLoopEnv_{i}/t_{timestep}.npz'

                pcds = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                np.savez_compressed(file_name, pcds=pcds, colors=colors)
                # print(f'Saved timestep: {timestep} with {pcds.shape[0]} points')

                # print('----------------------------------------------------------------------')
        except Exception as e:
            print(f'Could not save File:{i} because of {e}')
            f = open("log.txt", "a")
            f.write(f'{i}, ')
            f.close()


def parallel():
    num_process = 5
    # max_traj = 250
    with Pool(num_process) as pool:
        starts = range(0, 500, 100)
        ends = range(101, 501, 100)
        zipped = zip(starts, ends, [1] * num_process)
        results = pool.starmap(convertPointcloudNumpy, zipped)


def convert_list(arr: list[int]):
    for i in arr:
        convertPointcloudNumpy(i, i + 1)


if __name__ == "__main__":
    convertPointcloudNumpy(46,47)
