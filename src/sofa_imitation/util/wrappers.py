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


# https://github.com/balazsgyenes/pc_rl/blob/main/pc_rl/
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
        # new: orient right way
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
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



import selectors
import time
import gymnasium as gym
import numpy as np
import multiprocessing as mp

from datetime import datetime
from typing import Callable, List, Optional
from collections import defaultdict

from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv, _flatten_obs, _worker

# https://github.com/ScheiklP/sofa_zoo/blob/main/sofa_zoo/common/reset_process_vec_env.py
class WatchdogVecEnv(SubprocVecEnv):
    """Variant of Stable Baseline 3's SubprocVecEnv that closes and restarts environment processes that hang during a step.

    This VecEnv features a watchdog in its asynchronous step function to reset environments that take longer than
    ``step_timeout_sec`` for one step. This might happen when unstable deformations cause SOFA to hang.

    Resetting a SOFA scene that features topological changes such as removing/cutting tetrahedral elements does not
    restore the initial number of elements in the meshes. Manually removing and adding elements to SOFA's simulation
    tree technically works, but is sometimes quite unreliable and prone to memory leaks. This VecEnv avoids this
    problem by creating a completely new environment with simulation, if a reset signal is sent to the environment.

    Notes:
        If an environment is reset by the step watchdog, the returned values for this environment will be:
        ``reset_obs, 0.0, True, defaultdict(float), reset_info``. Meaning it returs the reset observation, a reward of 0.0, a done signal,
        an empty info dict that defaults to returning 0.0, if a key is accessed and the reset_info dict. The ``defaultdict`` is used to prevent
        crashing the ``VecMonitor`` when accessing the info dict.

    Args:
        env_fns (List[Callable[[], gymnasium.Env]]): List of environment constructors.
        step_timeout_sec (Optional[float]): Timeout in seconds for a single step. If a step takes longer than this
            timeout, the environment will be reset. If ``None``, no timeout is used.
        reset_process_on_env_reset (bool): Additionally to hanging envs, close and restart the process of envs at every reset.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        step_timeout_sec: Optional[float] = None,
        reset_process_on_env_reset: bool = False,
    ) -> None:
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        # forkserver is way too slow since we need to start a new process on
        # every reset
        ctx = mp.get_context("fork")

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            # do not close work_remote to prevent it being garbage collected

        self.ctx = ctx
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

        self.env_fns = env_fns
        self.step_timeout = step_timeout_sec
        self.reset_process_on_env_reset = reset_process_on_env_reset

    def step_wait(self) -> VecEnvStepReturn:
        hanging_envs = []
        if self.step_timeout is not None:
            # wait for all remotes to finish
            successes = wait_all(self.remotes, timeout=self.step_timeout)
            if len(successes) != len(self.remotes):
                hanging_envs = [i for i, remote in enumerate(self.remotes) if remote not in successes]
                for i in hanging_envs:
                    print(f"Environment {i} is hanging and will be terminated and restarted " f"({datetime.now().strftime('%H:%M:%S')})")
                    self.processes[i].terminate()  # terminate worker
                    # clear any data in the pipe
                    while self.remotes[i].poll():
                        self.remotes[i].recv()
                    # start new worker, seed, and reset it
                    self._restart_process(i)

        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        # any environments that were just reset will send an extra message that
        # must be consumed.
        # in addition, the observation and done state and reset info must be updated
        for i in hanging_envs:
            # Return of reset is (obs, reset_info)
            reset_obs, reset_info = results[i]
            # Result order: obs, reward, done, info, reset_info: See class SubProcVecEnv
            results[i] = (reset_obs, 0.0, True, defaultdict(float), reset_info)

        obs, rews, dones, infos, self.reset_infos = zip(*results)
        obs = list(obs)  # convert to list to allow modification

        if self.reset_process_on_env_reset:
            for i, (done, remote, process) in enumerate(zip(dones, self.remotes, self.processes)):
                if done and i not in hanging_envs:  # do not double-reset environments that were hanging
                    remote.send(("close", None))  # command worker to stop
                    process.join()  # wait for worker to stop
                    # start new worker, seed, and reset it
                    self._restart_process(i)
                    obs[i], self.reset_infos[i] = remote.recv()  # collect reset observation

        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def _restart_process(self, i: int) -> None:
        """Restarts the worker process ``i`` with its original ``env_fn``. The
        original pipe is reused. The new environment is seeded and reset, but
        the reset observation is *not* yet collected from the pipe.
        """
        work_remote, remote = self.work_remotes[i], self.remotes[i]
        # start new worker
        args = (work_remote, remote, CloudpickleWrapper(self.env_fns[i]))
        process = self.ctx.Process(target=_worker, args=args, daemon=True)
        process.start()

        # reseed and reset new env
        remote.send(("reset", (self._seeds[i], self._options[i])))

        self.processes[i] = process

    def reset(self) -> VecEnvObs:
        if self.reset_process_on_env_reset:
            # command environments to shut down
            for remote in self.remotes:
                remote.send(("close", None))  # command worker to stop
            for process in self.processes:
                process.join()  # wait for worker to stop
            # start new workers, seed, and reset them
            for i in range(len(self.processes)):
                self._restart_process(i)
            results = [remote.recv() for remote in self.remotes]
            obs, self.reset_infos = zip(*results)
            # Seeds and options are only used once
            self._reset_seeds()
            self._reset_options()
            return _flatten_obs(obs, self.observation_space)
        else:
            return super().reset()

    def close(self) -> None:
        super().close()
        for remote in self.remotes:
            remote.close()  # close pipe


# poll/select have the advantage of not requiring any extra file
# descriptor, contrarily to epoll/kqueue (also, they require a single
# syscall).
if hasattr(selectors, "PollSelector"):
    _WaitSelector = selectors.PollSelector
else:
    _WaitSelector = selectors.SelectSelector


def wait_all(object_list, timeout: Optional[float] = None):
    """
    Wait till all objects in ``object_list`` are ready/readable, or the timeout expires.

    Adapted from ``multiprocessing.connection.wait`` in the standard library.

    Args:
        object_list (list): list of objects to wait for. E.g. a list of pipes.
        timeout (float): timeout in seconds. If ``None``, wait forever.

    Returns:
        list: list of objects in ``object_list`` which are ready/readable.
    """
    with _WaitSelector() as selector:
        for obj in object_list:
            selector.register(obj, selectors.EVENT_READ)

        if timeout is not None:
            deadline = time.monotonic() + timeout

        all_ready = []
        while True:
            ready = selector.select(timeout)
            ready = [key.fileobj for (key, events) in ready]

            all_ready.extend(ready)
            for obj in ready:
                selector.unregister(obj)

            if len(all_ready) == len(object_list):
                return all_ready
            else:
                if timeout is not None:
                    timeout = deadline - time.monotonic()
                    if timeout < 0:
                        return all_ready
