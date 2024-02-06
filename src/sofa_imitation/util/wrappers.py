from __future__ import annotations

from imitation.data import types

from typing import Callable

import gymnasium as gym
import numpy as np
import open3d as o3d
from open3d.geometry import PointCloud
from sofa_env.base import RenderMode, SofaEnv
from sofa_env.utils.camera import get_focal_length

from .pointcloud_space import PointCloudSpace

class RolloutInfoWrapper(gym.Wrapper):
    """Add the entire episode's rewards and observations to `info` at episode end.

    Whenever done=True, `info["rollouts"]` is a dict with keys "obs" and "rews", whose
    corresponding values hold the NumPy arrays containing the raw observations and
    rewards seen during this episode.
    """

    def __init__(self, env: gym.Env):
        """Builds RolloutInfoWrapper.

        Args:
            env: Environment to wrap.
        """
        super().__init__(env)
        self._obs = None
        self._rews = None

    def reset(self, **kwargs):
        new_obs, info = super().reset(**kwargs)
        self._obs = [types.maybe_wrap_in_dictobs(new_obs)]
        self._rews = []
        return new_obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self._obs.append(types.maybe_wrap_in_dictobs(obs))
        self._rews.append(rew)

        if done:
            assert "rollout" not in info
            info["rollout"] = {
                "obs": self._obs,
                "rews": np.stack(self._rews),
            }
        return obs, rew, terminated, truncated, info



STATE_KEY = "state"


class SofaEnvPointCloudObservations(gym.ObservationWrapper):
    def __init__(
        self,
        env: SofaEnv,
        depth_cutoff: float | None = None,
        max_expected_num_points: int | None = None,
        color: bool = False,
        transform_to_world_coordinates: bool = False,
        points_only: bool = True,
        points_key: str = "points",
        post_processing: list[Callable[[PointCloud], PointCloud]] | None = None,
        debug_o3d_pcd: bool = False,
    ):
        super().__init__(env)

        self.env = env
        self.depth_cutoff = depth_cutoff
        self.color = color
        self.transform_to_world_coordinates = transform_to_world_coordinates
        self.points_only = points_only
        self.points_key = points_key
        self.post_processing = post_processing
        self.debug_o3d_pcd = debug_o3d_pcd

        self._initialized = False

        if self.env.render_mode == RenderMode.NONE:
            raise ValueError(
                "RenderMode of environment cannot be RenderMode.NONE, if point clouds are to be created from OpenGL depth images."
            )

        if max_expected_num_points is None:
            max_expected_num_points = int(np.prod(self.env.observation_space.shape[:2]))  # type: ignore

        self.observation_space = PointCloudSpace(
            max_expected_num_points=max_expected_num_points,
            low=-np.float32("inf"),
            high=np.float32("inf"),
            feature_shape=(6,) if self.color else (3,),
        )

    def reset(self, **kwargs):
        """Reads the data for the point clouds from the sofa_env after it is resetted."""

        # First reset calls _init_sim to setup the scene
        observation, reset_info = self.env.reset(**kwargs)

        if not self._initialized:
            env = self.env.unwrapped
            scene_creation_result = env.scene_creation_result
            if (
                not isinstance(scene_creation_result, dict)
                and "camera" in scene_creation_result
                and isinstance(
                    scene_creation_result["camera"],
                    (env.sofa_core.Object, env.camera_templates.Camera),
                )
            ):
                raise AttributeError(
                    "No camera was found to create a raycasting scene. Please make sure createScene() returns a dictionary with key 'camera' or specify the cameras for point cloud creation in camera_configs."
                )

            if isinstance(
                scene_creation_result["camera"],
                env.camera_templates.Camera,
            ):
                self.camera_object = scene_creation_result["camera"].sofa_object
            else:
                self.camera_object = scene_creation_result["camera"]

            # Read camera parameters from SOFA camera
            width = int(self.camera_object.widthViewport.array())
            height = int(self.camera_object.heightViewport.array())
            fx, fy = get_focal_length(self.camera_object, width, height)
            self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=width / 2,
                cy=height / 2,
            )

            self._initialized = True

        return self.observation(observation), reset_info

    def observation(self, observation) -> PointCloud | dict:
        """Replaces the observation of a step in a sofa_env scene with a point cloud."""

        pcd = self.pointcloud(observation)

        if self.post_processing is not None:
            for func in self.post_processing:
                pcd = func(pcd)

        if not self.debug_o3d_pcd:
            pos = np.asarray(pcd.points, dtype=self.observation_space.dtype)
            if self.color:
                colors = np.asarray(pcd.colors, dtype=self.observation_space.dtype)
                pcd = np.concatenate((pos, colors), axis=-1)
            else:
                pcd = pos

        if self.points_only:
            return pcd
        else:
            return {
                STATE_KEY: observation,
                self.points_key: pcd,
            }

    def pointcloud(self, observation) -> PointCloud:
        """Returns a point cloud calculated from the depth image of the sofa scene"""
        # Get the depth image from the SOFA scene
        depth = self.env.unwrapped.get_depth_from_open_gl()

        if (depth_cutoff := self.depth_cutoff) is None:
            depth_cutoff = 0.99 * depth.max()

        if self.transform_to_world_coordinates:
            # Get the model view matrix of the camera
            model_view_matrix = self.camera_object.getOpenGLModelViewMatrix()
            # Reshape from list into 4x4 matrix
            model_view_matrix = np.asarray(model_view_matrix).reshape((4, 4), order="F")
            cam_rotation = model_view_matrix[:3, :3]
            cam_position = model_view_matrix[:3, 3]

            extrinsic = compute_camera_extrinics(cam_rotation, cam_position)
        else:
            extrinsic = np.identity(4)

        if self.color:
            rgb = observation

            # Calculate point cloud
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=o3d.geometry.Image(np.ascontiguousarray(rgb)),
                depth=o3d.geometry.Image(np.ascontiguousarray(depth)),
                # depth is in meters, no need to rescale
                depth_scale=1.0,
                depth_trunc=depth_cutoff,
                convert_rgb_to_intensity=False,
            )

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                self.intrinsic,
                extrinsic=extrinsic,
            )
        else:
            # create_from_depth_image is supposed to do this cutoff, but it doesn't
            depth = np.where(depth > depth_cutoff, 0, depth)

            depth = o3d.geometry.Image(np.ascontiguousarray(depth))

            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth,
                self.intrinsic,
                extrinsic=extrinsic,
                depth_scale=1.0,
                # depth_trunc=depth_cutoff,
            )

        return pcd


def compute_camera_extrinics(
    cam_rotation: np.ndarray,
    cam_position: np.ndarray,
) -> np.ndarray:
    # Flip around the z axis.
    # This is necessary because the camera looks in the negative z direction in SOFA,
    # but we invert z in get_depth_from_open_gl().
    rotate_about_x_axis = o3d.geometry.get_rotation_matrix_from_quaternion([0, 1, 0, 0])
    cam_rotation = np.matmul(rotate_about_x_axis, cam_rotation)

    # the extrinsic matrix is that it describes how the world is transformed relative to the camera
    # this is described with an affine transform, which is a rotation followed by a translation
    # https://ksimek.github.io/2012/08/22/extrinsic/
    assert cam_rotation.shape == (3, 3)
    assert cam_position.shape == (3,)
    inverse_rotation = cam_rotation.T
    extrinsic_matrix = np.identity(4)
    extrinsic_matrix[:3, :3] = inverse_rotation
    extrinsic_matrix[:3, 3] = -inverse_rotation @ cam_position
    return extrinsic_matrix
