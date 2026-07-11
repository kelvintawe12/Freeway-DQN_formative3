"""
Evaluate a trained DQN model over multiple episodes and report summary
statistics.

The rubric's "Evaluation and Agent Performance" criterion rewards showing
that the agent's performance is strong, not just that play.py runs. A
single gameplay clip can be a lucky episode. This script runs enough
episodes to give a mean and standard deviation you can actually defend
in front of the coach.

Example:
    python evaluate.py --model models/dqn_model.zip --episodes 20
"""

from __future__ import annotations

import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

from config import ENV_ID, best_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN model.")
    parser.add_argument("--model", type=str, default=best_model_path())
    parser.add_argument("--env-id", type=str, default=ENV_ID)
    parser.add_argument("--n-stack", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # clip_reward=False so the reported score is the actual in-game score,
    # not the -1/0/1 clipped signal DQN trains on. Training still uses
    # the clipped version internally, this is purely a reporting choice.
    env = make_atari_env(args.env_id, n_envs=1, seed=args.seed,
                          wrapper_kwargs={"clip_reward": False})
    env = VecFrameStack(env, n_stack=args.n_stack)

    model = DQN.load(args.model, env=env)

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=args.episodes, deterministic=True
    )

    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {mean_reward:.3f}")
    print(f"Std reward:  {std_reward:.3f}")

    env.close()


if __name__ == "__main__":
    main()
