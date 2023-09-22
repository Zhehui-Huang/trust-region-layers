# from metaworld.envs.mujoco.env_dict import MT10_V2
from garage.envs import GymEnv
from metaworld.envs.mujoco.env_dict import MT10_V2


def gen_env(env: str):
    env_cls = MT10_V2[env]
    expert_env = env_cls()
    expert_env._partially_observable = False
    expert_env._set_task_called = True
    expert_env._freeze_rand_vec = False
    expert_env.reset()
    max_path_length = expert_env.max_path_length
    expert_env = GymEnv(expert_env, max_episode_length=max_path_length)
    assert max_path_length is not None
    return expert_env
