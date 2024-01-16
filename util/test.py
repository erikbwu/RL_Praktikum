from sofa_env.base import RenderMode
from sofa_env.scenes.ligating_loop.ligating_loop_env import LigatingLoopEnv, ObservationType, ActionType

import pprint
import time

from sofa_env.wrappers.point_cloud import PointCloudFromDepthImageObservationWrapper

pp = pprint.PrettyPrinter()

env = LigatingLoopEnv(
    observation_type=ObservationType.RGBD,
    render_mode=RenderMode.HUMAN,
    action_type=ActionType.CONTINUOUS,
    image_shape=(800, 800),
    frame_skip=1,
    time_step=0.1,
    settle_steps=50,
)
env = PointCloudFromDepthImageObservationWrapper(env)
env.reset()
done = False
while not done:
    for _ in range(1):
        start = time.perf_counter()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print('obs',obs.shape)
        end = time.perf_counter()
        done = terminated or truncated
        fps = 1 / (end - start)

    env.reset()