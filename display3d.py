import open3d as o3d
import numpy as np


def display3d():
    path = '/home/erik/RL_P/LigatingLoopEnv_0.npz'
    npz_data = np.load(path)
    print(len(npz_data['rgb']))

    width = npz_data['metadata.camera.width']
    height = npz_data['metadata.camera.height']
    focal_x = npz_data['metadata.camera.focal_x']
    focal_y = npz_data['metadata.camera.focal_y']
    z_near = npz_data['metadata.camera.z_near']
    z_far = npz_data['metadata.camera.z_far']


    for timestep in range(5, len(npz_data['rgb']),20):
        print('timestep: ', timestep)
        rgb = npz_data['rgb'][timestep]
        depth = npz_data['depth'][timestep]

        color_image = o3d.geometry.Image(rgb)
        depth_image = o3d.geometry.Image(depth)

        intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, focal_x, focal_y, width / 2, height / 2)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image,
                                                                        convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsics)
        # o3d.camera.PinholeCameraIntrinsic(
        #     o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    display3d()
