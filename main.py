from collections import deque

import numpy as np
from sofa_env.base import RenderMode
from sofa_env.scenes.ligating_loop.ligating_loop_env import LigatingLoopEnv, ObservationType, ActionType
import pprint
import time


def run_env():
    pp = pprint.PrettyPrinter()

    env = LigatingLoopEnv(
        observation_type=ObservationType.STATE,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.CONTINUOUS,
        image_shape=(800, 800),
        frame_skip=1,
        time_step=0.1,
        settle_steps=50,
    )

    env.reset()
    done = False

    fps_list = deque(maxlen=100)
    while not done:
        for _ in range(100):
            start = time.perf_counter()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            end = time.perf_counter()
            done = terminated or truncated
            fps = 1 / (end - start)
            fps_list.append(fps)
            # pp.pprint(info)
            print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")

        env.reset()





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_env()

