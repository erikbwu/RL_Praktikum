from sofa_env.base import RenderMode
from sofa_env.scenes.ligating_loop.ligating_loop_env import LigatingLoopEnv, ObservationType, ActionType

env = LigatingLoopEnv(
    observation_type=ObservationType.RGBD,
    render_mode=RenderMode.HUMAN,
    action_type=ActionType.CONTINUOUS,
    image_shape=(256, 256),
    frame_skip=1,
    time_step=0.1,
    settle_steps=50,
)