import open3d as o3d
import numpy as np
import torch

ligating_loop_path = '/home/erik/sofa_env_demonstrations/ligating_loop'
pick_and_place = f'/media/erik/Volume/sofa_env_demonstrations/pick_and_place'

def display3d(path):
    npz_data = np.load(path)
    print(len(npz_data['rgb']))

    width = npz_data['metadata.camera.width']
    height = npz_data['metadata.camera.height']
    focal_x = npz_data['metadata.camera.focal_x']
    focal_y = npz_data['metadata.camera.focal_y']
    z_near = npz_data['metadata.camera.z_near']
    z_far = npz_data['metadata.camera.z_far']

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # geometry is the point cloud used in your animaiton
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)


    for timestep in range(0, len(npz_data['rgb']), 5):
        print('timestep: ', timestep)
        rgb = npz_data['rgb'][timestep]
        depth = npz_data['depth'][timestep]

        color_image = o3d.geometry.Image(rgb)
        depth_image = o3d.geometry.Image(depth)

        intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, focal_x, focal_y, width / 2, height / 2)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image,
                                                                        convert_rgb_to_intensity=False)

        vis.remove_geometry(pcd, False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsics)
        # o3d.camera.PinholeCameraIntrinsic(
        #     o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        #o3d.visualization.draw_geometries([pcd])

        #geometry.points = pcd.points
        #geometry.colors = pcd.colors
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()



def display_array(pcd_array, colors = None):
    pcd = o3d.geometry.PointCloud()
    pcd_array = np.asarray(pcd_array)
    pcd.points = o3d.utility.Vector3dVector(pcd_array)
    print(len(pcd_array))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    o3d.visualization.draw_geometries([pcd])


def get_z_fars(num=100):
    for i in range(num):
        path = f'/media/erik/Volume/sofa_env_demonstrations/rope_cutting/RopeCuttingEnv_{i}.npz'
        npz_data = np.load(path)
        print(npz_data['metadata.camera.z_far'])


if __name__ == '__main__':
    path = f'{pick_and_place}/PickAndPlaceEnv_4.npz'
    display3d(path)
   #get_z_fars(100)
