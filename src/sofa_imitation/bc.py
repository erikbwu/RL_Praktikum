import logging
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from gymnasium.spaces import Box
from imitation.algorithms import bc
from imitation.policies.serialize import save_stable_model
from sofa_env.base import RenderMode
from sofa_env.scenes.ligating_loop.ligating_loop_env import LigatingLoopEnv, ObservationType, ActionType

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from policy.PointNetActorCritic import PointNetActorCriticPolicy
from util.data import npz_to_transitions
from util.evaluate_policy import evaluate_policy
from util.make_env import make_env_func, make_vec_sofa_env

log = logging.getLogger(__name__)


def run_bc(batch_size: int = 2, learning_rate=lambda epoch: 1e-3 * 0.99 ** epoch, num_epoch: int = 1,
           num_traj: int = 5, use_color: bool = False, n_eval: int = 0):

    path = '../../../sofa_env_demonstrations/ligating_loop'

    if isinstance(learning_rate, float) or isinstance(learning_rate, int):
        lr = learning_rate
        learning_rate = lambda a: lr

    ligating_loop_env = LigatingLoopEnv(
            observation_type=ObservationType.RGBD,
            render_mode=RenderMode.HEADLESS,
            action_type=ActionType.CONTINUOUS,
            image_shape=(256, 256),
            frame_skip=1,
            time_step=0.1,
            settle_steps=50,
        )
    env = make_vec_sofa_env(ligating_loop_env, use_color)

    rng = np.random.default_rng()
    obs_array_shape = (65536, 3)
    observation_space = Box(low=float('-inf'), high=float('inf'), shape=obs_array_shape, dtype='float32')
    policy = PointNetActorCriticPolicy(observation_space, env.action_space, learning_rate, [256, 128])

    demos = npz_to_transitions(path, 'LigatingLoopEnv_', num_traj, use_color)

    bc_trainer = bc.BC_Pyg(
        observation_space=observation_space,
        action_space=env.action_space,
        demonstrations=demos,
        policy=policy,
        rng=rng,
        device='cuda',
        batch_size=batch_size,
    )

    reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, n_eval)
    log.info(f"Reward before training: {reward_before_training}")

    bc_trainer.train(n_epochs=num_epoch, progress_bar=True)
    saved_time = datetime.now().strftime('%Y-%m-%d_%H:%M')
    save_stable_model(Path(f'./model/ligating_loop'), bc_trainer.policy, saved_time)
    log.info('Finished training and saved model')


    reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, n_eval)
    log.info(f"Reward after training: {reward_after_training}")

    log.info('done')


@hydra.main(version_base=None, config_path="conf", config_name="local")
def hydra_run(cfg: DictConfig):
    # env = submitit.JobEnvironment()
    # log.info(f"Process ID {os.getpid()} executing task {cfg.task}, with {env}")
    log.info(OmegaConf.to_yaml(cfg))
    lr = cfg.hyperparameter.learning_rate
    n_epochs = cfg.hyperparameter.number_epochs
    if isinstance(lr, str):
        lr = eval(lr)
    bs = cfg.hyperparameter.batch_size
    num_traj = cfg.hyperparameter.number_trajectories
    use_color = cfg.hyperparameter.use_color
    n_eval = cfg.hyperparameter.number_evaluations

    run_bc(bs, lr, n_epochs, num_traj, use_color, n_eval)


if __name__ == "__main__":
    hydra_run()
