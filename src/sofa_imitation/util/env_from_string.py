from gymnasium import Env
from sofa_env.base import RenderMode
import numpy as np

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from .make_env import make_env
from .wrappers import WatchdogVecEnv


def get_env(env_name: str, should_render: bool = False, use_color: bool = True):
    render_mode = RenderMode.HUMAN if should_render else RenderMode.HEADLESS
    image_shape = (256,256)

    if env_name == 'ligating_loop':
        from sofa_env.scenes.ligating_loop.ligating_loop_env import LigatingLoopEnv, ObservationType, ActionType

        env = LigatingLoopEnv(
            observation_type=ObservationType.RGB,
            render_mode=render_mode,
            action_type=ActionType.CONTINUOUS,
            image_shape=image_shape,
            frame_skip=2,
            time_step=0.1,
            settle_steps=50,
            reward_amount_dict={
                "distance_loop_to_marking_center": -0.05,
                "delta_distance_loop_to_marking_center": -100.0,
                "loop_center_in_cavity": 0.01,
                "instrument_not_in_cavity": -0.0,
                "instrument_shaft_collisions": -0.0,
                "loop_marking_overlap": 0.8,
                "loop_closed_around_marking": 0.5,
                "loop_closed_in_thin_air": -0.1,
                "successful_task": 100.0,
            }
        )
        env = make_env(env, use_color, 500, 220)
        return WatchdogVecEnv([lambda : env], step_timeout_sec=45)

    elif env_name == 'rope_cutting':
        from sofa_env.scenes.rope_cutting.rope_cutting_env import RopeCuttingEnv, ObservationType, ActionType

        env = RopeCuttingEnv(
            observation_type=ObservationType.RGB,
            render_mode=render_mode,
            action_type=ActionType.CONTINUOUS,
            image_shape=image_shape,
            frame_skip=1,
            time_step=0.1,
            settle_steps=10,
            settle_step_dt=0.01,
            # reward_amount_dict={
            #     "distance_cauter_active_rope": -0.0,
            #     "delta_distance_cauter_active_rope": -5.0,
            #     "cut_active_rope": 5.0,
            #     "cut_inactive_rope": -5.0,
            #     "workspace_violation": -0.0,
            #     "state_limits_violation": -0.0,
            #     "successful_task": 10.0,
            #     "failed_task": -20.0,
            # },
        )
        env = make_env(env, use_color, 400, depth_cutoff=245)
        return env
        # return WatchdogVecEnv([lambda: env], step_timeout_sec=45)
        return make_vec_env(lambda : env, n_envs=1, vec_env_cls=SubprocVecEnv)

    elif env_name == 'pick_and_place':
        from sofa_env.scenes.pick_and_place.pick_and_place_env import PickAndPlaceEnv, Phase, ObservationType, ActionType

        env = PickAndPlaceEnv(
            observation_type=ObservationType.RGB,
            render_mode=render_mode,
            action_type=ActionType.CONTINUOUS,
            image_shape=image_shape,
            frame_skip=2,
            time_step=0.01,
            settle_steps=50,
            create_scene_kwargs={
                "gripper_randomization": {
                    "angle_reset_noise": 0.0,
                    "ptsd_reset_noise": np.array([10.0, 10.0, 40.0, 5.0]),
                    "rcm_reset_noise": np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0]),
                },
            },
            num_active_pegs=3,
            num_torus_tracking_points=5,
            start_grasped=False,
            only_learn_pick=False,
            minimum_lift_height=0.0,
            randomize_torus_position=False,
            randomize_color=True,
            reward_amount_dict={
                Phase.ANY: {
                    "lost_grasp": -30.0,
                    "grasped_torus": 0.0,
                    "gripper_jaw_peg_collisions": -0.01,
                    "gripper_jaw_floor_collisions": -0.01,
                    "unstable_deformation": -0.01,
                    "torus_velocity": -0.0,
                    "gripper_velocity": -0.0,
                    "torus_dropped_off_board": -0.0,
                    "action_violated_state_limits": -0.0,
                    "action_violated_cartesian_workspace": -0.0,
                    "successful_task": 50.0,
                },
                Phase.PICK: {
                    "established_grasp": 30.0,
                    "gripper_distance_to_torus_center": -0.0,
                    "delta_gripper_distance_to_torus_center": -0.0,
                    "gripper_distance_to_torus_tracking_points": -0.0,
                    "delta_gripper_distance_to_torus_tracking_points": -10.0,
                    "distance_to_minimum_pick_height": -0.0,
                    "delta_distance_to_minimum_pick_height": -50.0,
                },
                Phase.PLACE: {
                    "torus_distance_to_active_pegs": -0.0,
                    "delta_torus_distance_to_active_pegs": -100.0,
                },
            },
        )
        env = make_env(env, use_color, 600, depth_cutoff=350)

        #return WatchdogVecEnv([lambda: env], step_timeout_sec=45)
        return make_vec_env(lambda : env, n_envs=1, vec_env_cls=SubprocVecEnv)



