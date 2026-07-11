"""
Compare CnnPolicy against MlpPolicy on the same environment, seed, and
timestep budget.

This exists to produce actual evidence for the "MLP vs CNN" requirement,
rather than a claim without a chart behind it. Atari observations are
image frames, so CnnPolicy has a structural advantage: convolutional
layers exploit spatial locality, while MlpPolicy flattens the frame into
a single long vector and has to learn spatial structure from scratch with
far more parameters. This script runs both and lets the reward curves
make that case directly.

Example:
    python compare_policies.py --timesteps 200000
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import VecFrameStack

from config import ENV_ID, LOGS_DIR, PLOTS_DIR


def train_one(policy: str, env_id: str, timesteps: int, seed: int, n_envs: int) -> str:
    run_dir = os.path.join(LOGS_DIR, f"compare_{policy}")
    os.makedirs(run_dir, exist_ok=True)

    env = make_atari_env(env_id, n_envs=n_envs, seed=seed, monitor_dir=run_dir)
    env = VecFrameStack(env, n_stack=4)

    model = DQN(policy, env, seed=seed, verbose=0)
    model.learn(total_timesteps=timesteps, progress_bar=True)

    eval_env = make_atari_env(env_id, n_envs=1, seed=seed + 1000,
                               wrapper_kwargs={"clip_reward": False})
    eval_env = VecFrameStack(eval_env, n_stack=4)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10,
                                               deterministic=True)
    print(f"{policy}: mean_reward={mean_reward:.2f} std_reward={std_reward:.2f}")

    eval_env.close()
    env.close()
    return run_dir


def plot_comparison(run_dirs: dict, output_path: str) -> None:
    plt.figure(figsize=(9, 5))
    for policy, run_dir in run_dirs.items():
        try:
            x, y = ts2xy(load_results(run_dir), "timesteps")
        except Exception:
            continue
        if len(x) == 0:
            continue
        window = min(50, len(y))
        if window > 1:
            y = np.convolve(y, np.ones(window) / window, mode="valid")
            x = x[len(x) - len(y):]
        plt.plot(x, y, label=policy)

    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward (smoothed)")
    plt.title("CnnPolicy vs MlpPolicy on Freeway")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved comparison plot to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CnnPolicy vs MlpPolicy.")
    parser.add_argument("--env-id", type=str, default=ENV_ID)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-envs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = {}

    for policy in ["CnnPolicy", "MlpPolicy"]:
        run_dirs[policy] = train_one(policy, args.env_id, args.timesteps,
                                      args.seed, args.n_envs)

    plot_path = os.path.join(PLOTS_DIR, "mlp_vs_cnn.png")
    plot_comparison(run_dirs, plot_path)


if __name__ == "__main__":
    main()
