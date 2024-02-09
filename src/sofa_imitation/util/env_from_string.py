from gymnasium import Env
from sofa_env.base import RenderMode
import numpy as np

def get_env(env_name: str, should_render: bool = False) -> Env:
    render_mode = RenderMode.HUMAN if should_render else RenderMode.HEADLESS
    image_shape = (256,256)

    if env_name == 'ligating_loop':
        from sofa_env.scenes.ligating_loop.ligating_loop_env import LigatingLoopEnv, ObservationType, ActionType

        env = LigatingLoopEnv(
            observation_type=ObservationType.RGBD,
            render_mode=render_mode,
            action_type=ActionType.CONTINUOUS,
            image_shape=image_shape,
            frame_skip=2,
            time_step=0.1,
            settle_steps=50,
        )

    elif env_name == 'pick_and_place':
        from sofa_env.scenes.pick_and_place.pick_and_place_env import PickAndPlaceEnv, ObservationType, ActionType

        env = PickAndPlaceEnv(
            observation_type=ObservationType.RGBD,
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
            start_grasped=True,
            only_learn_pick=False,
            minimum_lift_height=50.0,
            randomize_torus_position=False,
        )

    elif env_name == 'pick_and_place':
        from sofa_env.scenes.rope_cutting import RopeCuttingEnv, ObservationType, ActionType

        env = RopeCuttingEnv(
            observation_type=ObservationType.RGBD,
            render_mode=render_mode,
            action_type=ActionType.CONTINUOUS,
            image_shape=image_shape,
            frame_skip=1,
            time_step=0.1,
            settle_steps=10,
            settle_step_dt=0.01,
        )



