import logging
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from imitation.algorithms import bc

from policy.PointNetActorCritic import PointNetActorCriticPolicy
from util.env_from_string import get_env, get_grid_size_from_string
from util.data import npz_to_transitions, npz_to_state_transitions
from util.evaluate_policy import evaluate_policy
import wandb

log = logging.getLogger(__name__)


def run_bc(env_name, env_prefix, batch_size: int = 2, learning_rate=lambda epoch: 1e-3 * 0.99 ** epoch, num_epoch: int = 1,
           num_traj: int = 5, use_color: bool = False, use_state = False, n_eval: int = 0):
    n_run = 28
    start_time = '2024-03-17_15:53'
    path = '../../../sofa_env_demonstrations/ligating_loop'
    model_path = f'./model/BC/rope_cutting/{start_time}/run_{n_run}'
    grid_size = get_grid_size_from_string(env_name)

    if isinstance(learning_rate, float) or isinstance(learning_rate, int):
        lr = learning_rate
        learning_rate = lambda _: lr

    env = get_env(env_name, use_state, should_render=True)

    rng = np.random.default_rng()

    if use_state:
        pass

    else:
        demos = npz_to_transitions(path, env_prefix, num_traj, use_color, grid_size['Demo'])
        policy = PointNetActorCriticPolicy(env.observation_space, env.action_space, learning_rate, [256, 128, 64, 32],
                                           grid_size['FeatureExtractor'], 3 if use_color else 0)
        policy.load(model_path, device='cuda')

    log.info('Finished loading train data')

    bc_trainer = bc.BC_Pyg( 
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=demos,
        policy=policy,
        rng=rng,
        device='cuda',
        batch_size=batch_size,
        optimizer_kwargs={'lr': learning_rate(0)}
    )

    Path(f'./model/BC/{env_name}/{start_time}/').mkdir(parents=True, exist_ok=True)
    while True:
        bc_trainer.train(n_epochs=num_epoch, progress_bar=True, log_interval=50, )
        bc_trainer.policy.save(f'./model/BC/{env_name}/{start_time}/run_{n_run}')
        # save_stable_model(Path(f'./model/ligating_loop/{start_time}'), bc_trainer.policy, f'run_{n_run}')
        log.info('Finished run and saved model')

        reward_after_training, std_reward = evaluate_policy(bc_trainer.policy, env, n_eval)
        wandb.log({"reward": reward_after_training, "std_reward": std_reward, "epoch": n_run * num_epoch})
        log.info(f"Reward after training run {n_run}: {reward_after_training}")
        n_run += 1
    log.info('done')


@hydra.main(version_base=None, config_path="conf", config_name="local")
def hydra_run(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    lr = cfg.hyperparameter.learning_rate
    n_epochs = cfg.hyperparameter.number_epochs
    if isinstance(lr, str):
        lr = eval(lr)
    bs = cfg.hyperparameter.batch_size
    num_traj = cfg.hyperparameter.number_trajectories
    use_color = cfg.hyperparameter.use_color
    n_eval = cfg.hyperparameter.number_evaluations
    env_name = cfg.env.env_name
    env_prefix = cfg.env.env_prefix
    use_state = cfg.hyperparameter.use_state

    wandb.init(project="Imitation_Sofa", config=OmegaConf.to_container(cfg, resolve=True),
               settings=wandb.Settings(start_method="thread"), notes='',
               resume='jmwu3pmx', tags=['nn=[256,128,64,32]', f'lr={lr}', env_name, 'BC', f'use_state={use_state}'])

    run_bc(env_name, env_prefix, bs, lr, n_epochs, num_traj, use_color,use_state, n_eval)
    wandb.finish()


if __name__ == "__main__":
    hydra_run()
