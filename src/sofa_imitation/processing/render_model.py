from policy.PointNetActorCritic import PointNetActorCriticPolicy
from util.env_from_string import get_env
from util.evaluate_policy import evaluate_policy


def render_model(model_path: str, env_name: str):
    env = get_env(env_name, should_render=True)
    policy = PointNetActorCriticPolicy(env.observation_space, env.action_space, lambda _ :1e-4, [256, 128])
    policy = policy.load(model_path, device='cuda')
    evaluate_policy(policy, env, 5)



if __name__ == "__main__":
    model_path = '/home/erik/RL_Praktikum/src/sofa_imitation/model/ligating_loop/2024-02-24_01:47/run_15'
    render_model(model_path, 'ligating_loop')

