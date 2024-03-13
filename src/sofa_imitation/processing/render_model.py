from policy.PointNetActorCritic import PointNetActorCriticPolicy
from util.env_from_string import get_env
from util.evaluate_policy import evaluate_policy

from stable_baselines3.common.policies import ActorCriticPolicy


def render_model(model_path: str, env_name: str):
    env = get_env(env_name, use_state=False, should_render=True)
    policy = PointNetActorCriticPolicy(env.observation_space, env.action_space, lambda _ :1e-4, [256, 128])
    policy = policy.load(model_path, device='cuda')
    print(evaluate_policy(policy, env, 5))

def render_model_state(model_path: str, env_name:str ):
    env = get_env(env_name,use_state=True, should_render=True)
    policy = ActorCriticPolicy(env.observation_space, env.action_space, lambda _ :1e-4, [256, 128])
    policy = policy.load(model_path, device='cuda')
    print(evaluate_policy(policy, env, 5))


if __name__ == "__main__":
    model_path_state = '/home/erik/RL_Praktikum/src/sofa_imitation/model/ligating_loop/2024-03-12_14:40/run_2'
    mp_state_pap = '/home/erik/RL_Praktikum/src/sofa_imitation/model/AIRL/pick_and_place/2024-03-13_13:14/run_0'
    model_path = '/home/erik/RL_Praktikum/src/sofa_imitation/model/ligating_loop/2024-02-24_01:47/run_161' #ligating loop
    model_path = '/home/erik/RL_Praktikum/src/sofa_imitation/model/pick_and_place/2024-02-29_09:24/run_63'
    #model_path = '/home/erik/run_1000'
    #render_model(model_path, 'ligating_loop')
    render_model_state(mp_state_pap, 'pick_and_place')
    #render_model(model_path, 'pick_and_place')

