from gymnasium import Env
from sofa_env.base import RenderMode
import numpy as np


from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from .make_env import make_env
from .wrappers import WatchdogVecEnv, FloatObservationWrapper

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from .wrappers import RolloutInfoWrapper


def get_env(env_name: str, use_state:bool = False, should_render: bool = False, use_color: bool = True):
    render_mode = RenderMode.HUMAN if should_render else RenderMode.HEADLESS
    image_shape = (256,256)

    if env_name == 'ligating_loop':
        from sofa_env.scenes.ligating_loop.ligating_loop_env import LigatingLoopEnv, ObservationType, ActionType

        env = LigatingLoopEnv(
            observation_type=ObservationType.STATE if use_state else ObservationType.RGB,
            render_mode=render_mode,
            action_type=ActionType.CONTINUOUS,
            image_shape=image_shape,
            frame_skip=2,
            time_step=0.1,
            settle_steps=50,
            band_width=10,
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
        if use_state:
            env = Monitor(env)
            env = TimeLimit(env, max_episode_steps=500)
            env = RolloutInfoWrapper(env)
            env = FloatObservationWrapper(env)
        else:
            env = make_env(env, use_color, 500, 220)
        return WatchdogVecEnv([lambda : env], step_timeout_sec=45)

    elif env_name == 'rope_cutting':
        from sofa_env.scenes.rope_cutting.rope_cutting_env import RopeCuttingEnv, ObservationType, ActionType

        env = RopeCuttingEnv(
            observation_type=ObservationType.STATE if use_state else ObservationType.RGB,
            render_mode=render_mode,
            action_type=ActionType.CONTINUOUS,
            image_shape=image_shape,
            frame_skip=1,
            time_step=0.01,
            settle_steps=10,
            settle_step_dt=0.01,
            reward_amount_dict={
                "distance_cauter_active_rope": -0.0,
                "delta_distance_cauter_active_rope": -5.0,
                "cut_active_rope": 5.0,
                "cut_inactive_rope": -5.0,
                "workspace_violation": -0.0,
                "state_limits_violation": -0.0,
                "successful_task": 10.0,
                "failed_task": -20.0,
            },
        )
        if use_state:
            env = Monitor(env)
            env = TimeLimit(env, max_episode_steps=1000)
            env = RolloutInfoWrapper(env)
            env = FloatObservationWrapper(env)
        else:
            env = make_env(env, use_color, 1000, depth_cutoff=245)
        return make_vec_env(lambda : env, n_envs=1, vec_env_cls=SubprocVecEnv)

    elif env_name == 'pick_and_place':
        from sofa_env.scenes.pick_and_place.pick_and_place_env import PickAndPlaceEnv, Phase, ObservationType, ActionType

        env = PickAndPlaceEnv(
            observation_type=ObservationType.STATE if use_state else ObservationType.RGB,
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
        if use_state:
            env = Monitor(env)
            env = TimeLimit(env, max_episode_steps=600)
            env = RolloutInfoWrapper(env)
            env = FloatObservationWrapper(env)
        else:
            env = make_env(env, use_color, 600, depth_cutoff=350)

        #return WatchdogVecEnv([lambda: env], step_timeout_sec=45)
        return make_vec_env(lambda : env, n_envs=1, vec_env_cls=SubprocVecEnv)

    elif env_name == 'grasp_lift_touch':
        from sofa_env.scenes.grasp_lift_touch.grasp_lift_touch_env import GraspLiftTouchEnv, Phase, ObservationType

        env = GraspLiftTouchEnv(
            observation_type=ObservationType.STATE if use_state else ObservationType.RGB,
            render_mode=render_mode,
            start_in_phase=Phase.GRASP,
            end_in_phase=Phase.DONE,
            image_shape=image_shape,
            reward_amount_dict={
                Phase.ANY: {
                    "collision_cauter_gripper": -0.1,
                    "collision_cauter_gallbladder": -0.1,
                    "collision_cauter_liver": -0.1,
                    "collision_gripper_liver": -0.1,
                    "distance_cauter_target": -0.5,
                    "delta_distance_cauter_target": -1.0,
                    "target_visible": 0.0,
                    "gallbladder_is_grasped": 20.0,
                    "new_grasp_on_gallbladder": 10.0,
                    "lost_grasp_on_gallbladder": -10.0,
                    "active_grasping_springs": 0.0,
                    "delta_active_grasping_springs": 0.0,
                    "gripper_pulls_gallbladder_out": 0.005,
                    "overlap_gallbladder_liver": -0.1,
                    "dynamic_force_on_gallbladder": -0.003,
                    "delta_overlap_gallbladder_liver": -0.01,
                    "successful_task": 200.0,
                    "failed_task": -0.0,
                    "cauter_action_violated_state_limits": -0.0,
                    "cauter_action_violated_cartesian_workspace": -0.0,
                    "gripper_action_violated_state_limits": -0.0,
                    "gripper_action_violated_cartesian_workspace": -0.0,
                    "phase_change": 0.0,
                },
                Phase.GRASP: {
                    "distance_gripper_graspable_region": -0.2,
                    "delta_distance_gripper_graspable_region": -10.0,
                },
                Phase.TOUCH: {
                    "cauter_activation_in_target": 0.0,
                    "cauter_delta_activation_in_target": 1.0,
                    "cauter_touches_target": 0.0,
                },
            },
        )

        if use_state:
            env = Monitor(env)
            env = TimeLimit(env, max_episode_steps=600)
            env = RolloutInfoWrapper(env)
            env = FloatObservationWrapper(env)
        else:
            env = make_env(env, use_color, 600)

        #return WatchdogVecEnv([lambda: env], step_timeout_sec=45)
        return make_vec_env(lambda : env, n_envs=1, vec_env_cls=SubprocVecEnv)


def get_grid_size_from_string(env: str):
    grids = {
        'ligating_loop': {
            'FeatureExtractor': 1,
            'Demo': 0.001
        },
        'pick_and_place': {
            'FeatureExtractor': 2,
            'Demo': 0.002
        },
        'rope_cutting': {
            'FeatureExtractor': 1,
            'Demo': 0.001
        },
        'grasp_lift_touch': {
            'FeatureExtractor': 1,
            'Demo': 0.001
        }
    }
    return grids[env]

def action_dim_from_string(env: str):
    dims = {
        'ligating_loop': 5,
        'pick_and_place': 5,
        'rope_cutting': 5,
        'grasp_lift_touch': 10,
    }
    return dims[env]


