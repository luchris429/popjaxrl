import jax
import jax.numpy as jnp
import time
from envs import make
from envs.wrappers import AliasPrevActionV2
from algorithms.ppo_gru import make_train as make_train_gru
from algorithms.ppo_s5 import make_train as make_train_s5
import argparse

def run(num_runs, env_name, arch="gru", file_tag=""):
    print("*"*10)
    print(f"Running {num_runs} runs of {env_name} with arch {arch}")
    env, env_params = make(env_name)

    config = {
        "LR": 5e-5,
        "NUM_ENVS": 64,
        "NUM_STEPS": 1024,
        "TOTAL_TIMESTEPS": 15e6,
        "UPDATE_EPOCHS": 30,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 1.0,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 1.0,
        "MAX_GRAD_NORM": 0.5,
        "ENV": AliasPrevActionV2(env),
        "ENV_PARAMS": env_params,
        "ANNEAL_LR": False,
        "DEBUG": True,
        "S5_D_MODEL": 256,
        "S5_SSM_SIZE": 256,
        "S5_N_LAYERS": 4,
        "S5_BLOCKS": 1,
        "S5_ACTIVATION": "full_glu",
        "S5_DO_NORM": False,
        "S5_PRENORM": False,
        "S5_DO_GTRXL_NORM": False,
    }

    rng = jax.random.PRNGKey(42)
    train_vjit_rnn = jax.jit(jax.vmap(make_train_gru(config)))
    train_vjit_s5 = jax.jit(jax.vmap(make_train_s5(config)))
    rngs = jax.random.split(rng, num_runs)
    info_dict = {}

    if arch == "s5":
        t0 = time.time()
        compiled_s5 = train_vjit_s5.lower(rngs).compile()
        compile_s5_time = time.time() - t0
        print(f"s5 compile time: {compile_s5_time}")

        t0 = time.time()
        out_s5 = jax.block_until_ready(compiled_s5(rngs))
        run_s5_time = time.time() - t0
        print(f"s5 time: {run_s5_time}")
        info_dict["s5"] = {
            "compile_s5_time": compile_s5_time,
            "run_s5_time": run_s5_time,
            "out": out_s5[1],
        }

    elif arch == "gru":
        t0 = time.time()
        compiled_rnn = train_vjit_rnn.lower(rngs).compile()
        compile_rnn_time = time.time() - t0
        print(f"gru compile time: {compile_rnn_time}")

        t0 = time.time()
        out_rnn = jax.block_until_ready(compiled_rnn(rngs))
        run_rnn_time = time.time() - t0
        print(f"gru time: {run_rnn_time}")
        info_dict["gru"] = {
            "compile_rnn_time": compile_rnn_time,
            "run_rnn_time": run_rnn_time,
            "out": out_rnn[1],
        }

    else:
        raise NotImplementedError

    jnp.save(f"results/{num_runs}_{env_name}_{arch}_{file_tag}.npy", info_dict)

parser = argparse.ArgumentParser()
parser.add_argument("--num-runs", type=int, required=True)
parser.add_argument("--env", type=str, default="")
parser.add_argument("--arch", type=str, default="s5")
args = parser.parse_args()

if __name__ == "__main__":
    run(args.num_runs, args.env, args.arch)
