import logging
from datetime import datetime
from pathlib import Path

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf


from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.airl import AIRL, AIRL_Pyg
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm

from util.env_from_string import get_env, get_grid_size_from_string
from util.data import npz_to_transitions, npz_to_state_transitions
from util.evaluate_policy import evaluate_policy
from policy.PointNetActorCritic import PointNetFeaturesExtractor, PointNetActorCriticPolicy
from util.wrappers import FloatObservationWrapper

log = logging.getLogger(__name__)
SEED = 42


# doenst work with using pointclouds yet
def run_AIRL(env_name: str, env_prefix: str, train_steps: int, demo_batch_size: int, batch_size: int = 256, learning_rate: float = 0.0001,
             num_traj: int = 2, n_eval: int = 0, use_state: bool = False, use_color: bool = False):

    path = f'../../../sofa_env_demonstrations/{env_name}'
    path = f'/media/erik/Volume/sofa_env_demonstrations/pick_and_place'
    start_time = datetime.now().strftime('%Y-%m-%d_%H:%M')
    grid_size = get_grid_size_from_string(env_name)

    assert isinstance(learning_rate, float) or isinstance(learning_rate, int)

    env = get_env(env_name, use_state)

    if use_state:
        demos = npz_to_state_transitions(path, env_prefix, num_traj)
    else:
        demos = npz_to_transitions(path, env_prefix, num_traj, use_color, grid_size['Demo'])
    log.info('Finished loading train data')

    policy_kwargs = {}
    policy = MlpPolicy
    if not use_state:
        policy = PointNetActorCriticPolicy
        policy_kwargs = {
            'net_arch': [256, 128, 64, 32],
            'grid_size': grid_size['FeatureExtractor'],
            'inp_features_dim': 3 if use_color else 0
        }

    learner = PPO(
        env=env,
        policy=policy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=learning_rate,
        gamma=0.95,
        clip_range=0.1,
        vf_coef=0.1,
        n_epochs=5,
        seed=SEED,
        policy_kwargs=policy_kwargs,
    )
    reward_net = BasicShapedRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm, #todo maybe normalize myself?
    )
    if use_state:
        airl_trainer = AIRL(
            demonstrations=demos,
            demo_batch_size=2048,
            gen_replay_buffer_capacity=512,
            n_disc_updates_per_round=16,
            venv=env,
            gen_algo=learner,
            reward_net=reward_net,
        )
    else:
        airl_trainer = AIRL_Pyg(
            demonstrations=demos,
            demo_batch_size=2048,
            gen_replay_buffer_capacity=512,
            n_disc_updates_per_round=16,
            venv=env,
            gen_algo=learner,
            reward_net=reward_net,
        )

    env.seed(SEED)
    # learner_rewards_before_training, _ = evaluate_policy(learner, env, 0)
    # print("mean reward before training:", learner_rewards_before_training)

    Path(f'./model/AIRL/{env_name}/{start_time}/').mkdir(parents=True, exist_ok=True)
    n_run = 0
    while True:
        n_run += 1
        airl_trainer.train(train_steps)  # Train for 2_000_000 steps to match expert.
        airl_trainer.policy.save(f'./model/AIRL/{env_name}/{start_time}/run_{n_run}')
        env.seed(SEED)
        learner_rewards_after_training, std = evaluate_policy(learner, env, n_eval)
        wandb.log({"reward": learner_rewards_after_training, "std_reward": std, "train_steps": n_run * train_steps})

    print("mean reward after training:", learner_rewards_after_training)


@hydra.main(version_base=None, config_path="conf", config_name="local_AIRL")
def hydra_run(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    lr = cfg.hyperparameter.learning_rate
    bs = cfg.hyperparameter.batch_size
    num_traj = cfg.hyperparameter.number_trajectories
    use_color = cfg.hyperparameter.use_color
    n_eval = cfg.hyperparameter.number_evaluations
    use_state = cfg.hyperparameter.use_state
    train_steps = cfg.hyperparameter.train_steps
    demo_batch_size = cfg.hyperparameter.demo_batch_size

    wandb.init(project="Imitation_Sofa", config=OmegaConf.to_container(cfg, resolve=True),
               settings=wandb.Settings(start_method="thread"),
               notes='', tags=['AIRL', f'lr={lr}'])

    #run_AIRL('ligating_loop', 'LigatingLoopEnv_', bs, lr, n_epochs, num_traj, n_eval, use_state, use_color)
    run_AIRL('pick_and_place', 'PickAndPlaceEnv_', train_steps, demo_batch_size,
             bs, lr, num_traj, n_eval, use_state, use_color)

    #wandb.finish()


if __name__ == "__main__":
    hydra_run()
