from sofa_env.base import RenderMode

def render_model(model_path: str, env_name: str):
    image_shape = (256,256)

    if env_name == "ligating_loop":
        from sofa_env.scenes.ligating_loop.ligating_loop_env import LigatingLoopEnv, ObservationType, ActionType

        env = LigatingLoopEnv(
            observation_type=ObservationType.RGBD,
            render_mode=RenderMode.HUMAN,
            action_type=ActionType.CONTINUOUS,
            image_shape=image_shape,
            frame_skip=2,
            time_step=0.1,
            settle_steps=50,
        )
