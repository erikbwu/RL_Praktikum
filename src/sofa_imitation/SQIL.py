import datasets
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms import sqil
from imitation.data import huggingface_utils

# Download some expert trajectories from the HuggingFace Datasets Hub.
dataset = datasets.load_dataset("HumanCompatibleAI/ppo-CartPole-v1")
rollouts = huggingface_utils.TrajectoryDatasetSequence(dataset["train"])

sqil_trainer = sqil.SQIL(
    venv=DummyVecEnv([lambda: gym.make("CartPole-v1")]),
    demonstrations=rollouts,
    policy="MlpPolicy",
)
# Hint: set to 1_000_000 to match the expert performance.
sqil_trainer.train(total_timesteps=1000000)
reward, _ = evaluate_policy(sqil_trainer.policy, sqil_trainer.venv, 10)
print("Reward:", reward)