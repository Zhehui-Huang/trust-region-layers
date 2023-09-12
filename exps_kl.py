from doexp import cmd, In, Out, GLOBAL_CONTEXT
from socket import gethostname
import random
import json
from subprocess import run
import sys
from datetime import datetime

HOST = gethostname()


mujoco_envs = [
    # "InvertedDoublePendulum-v2",
    "HalfCheetah-v2",
    "Hopper-v2",
    "Walker2d-v2",
]
seeds = [
    1111,
    2222,
    3333,
    4444,
    5555,
    6666,
    7777,
    8888,
    1, 2, 3, 4, 5, 6, 7, 8
]


ppo_env_names_v3 = [
    "HalfCheetah-v3",
    "Walker2d-v3",
    "Hopper-v3",
    "Swimmer-v3",
    "InvertedPendulum-v2",
    "Reacher-v2",
]

total_steps_for_env = {
    "HalfCheetah-v2": 10_000_000,
    "Hopper-v2": 10_000_000,
    "Walker2d-v2": 10_000_000,
}


GLOBAL_CONTEXT.max_concurrent_jobs = 16


with open('configs/pg/mujoco_kl_config.json') as f:
    papi_conf = json.load(f)

for seed in seeds:
    for env in mujoco_envs:
        conf_name = f"data_tmp/trust-region-layers_papi_seed={seed}_env={env}.json"
        out_dir = f"trust-region-layers_papi_seed={seed}_env={env}/"
        papi_conf["n_envs"] = 1  # Should only affect sampling speed
        papi_conf["seed"] = seed
        papi_conf["game"] = env
        papi_conf["out_dir"] = f"data_tmp_kl/{out_dir}"
        papi_conf["exp_name"] = f'seed_{seed}_env_{env}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        with open(conf_name, 'w') as f:
            json.dump(papi_conf, f, indent=2)
        cmd("python", "main.py", conf_name, extra_outputs=[Out(out_dir)],
            cores=3, ram_gb=8)
