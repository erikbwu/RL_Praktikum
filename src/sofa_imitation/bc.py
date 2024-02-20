import logging
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from imitation.algorithms import bc

from policy.PointNetActorCritic import PointNetActorCriticPolicy
from util.env_from_string import get_env
from util.data import npz_to_transitions
from util.evaluate_policy import evaluate_policy
import wandb

log = logging.getLogger(__name__)


def run_bc(batch_size: int = 2, learning_rate=lambda epoch: 1e-3 * 0.99 ** epoch, num_epoch: int = 1,
           num_traj: int = 5, use_color: bool = False, n_eval: int = 0):
    path = '../../../sofa_env_demonstrations/ligating_loop'
    start_time = datetime.now().strftime('%Y-%m-%d_%H:%M')
    Path(f'./model/ligating_loop/{start_time}/').mkdir(parents=True, exist_ok=True)

    if isinstance(learning_rate, float) or isinstance(learning_rate, int):
        lr = learning_rate
        learning_rate = lambda _: lr

    env = get_env('ligating_loop', False)

    rng = np.random.default_rng()
    policy = PointNetActorCriticPolicy(env.observation_space, env.action_space, learning_rate, [256, 128], 1)

    demos = npz_to_transitions(path, 'LigatingLoopEnv_', num_traj, use_color, 0.001)

    log.info('Finished loading train data')

    bc_trainer = bc.BC_Pyg( 
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=demos,
        policy=policy,
        rng=rng,
        device='cuda',
        batch_size=batch_size,
    )

    reward_before_training, std_reward = evaluate_policy(bc_trainer.policy, env, 1)
    log.info(f"Reward before training: {reward_before_training}")
    wandb.log({"reward": reward_before_training, "std_reward": std_reward, 'epoch': 0})

    n_run = 1
    while True:
        bc_trainer.train(n_epochs=num_epoch, progress_bar=True)
        bc_trainer.policy.save(f'./model/ligating_loop/{start_time}/run_{n_run}')
        #save_stable_model(Path(f'./model/ligating_loop/{start_time}'), bc_trainer.policy, f'run_{n_run}')
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

    wandb.init(project="Imitation_Sofa", config=OmegaConf.to_container(cfg, resolve=True), settings=wandb.Settings(start_method="thread"), notes='increased pointcloud size')

    run_bc(bs, lr, n_epochs, num_traj, use_color, n_eval)
    wandb.finish()


if __name__ == "__main__":
    hydra_run()
