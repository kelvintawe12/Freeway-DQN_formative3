"""
Load a trained DQN model and run it greedily (deterministic=True is SB3's
equivalent of a GreedyQPolicy: always pick the action with the highest
Q-value, no exploration).

Two modes:
  --render  opens a live window. Only works where a display exists, i.e.
            your local machine, not Colab.
  --record  writes an .mp4 to videos/. Works anywhere, including headless
            Colab, since it captures rgb_array frames rather than opening
            a window.

You can pass both, but on Colab you must use --record only.

Example (local, live window):
    python play.py --model models/dqn_model.zip --render --episodes 3

Example (Colab, headless recording):
    python play.py --model models/dqn_model.zip --record --episodes 3
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecVideoRecorder

from config import ENV_ID, VIDEOS_DIR, best_model_path


def make_single_env(env_id: str, render_mode: str):
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        # clip_reward=False: the printed episode reward should be the real
        # game score you can quote in the presentation, not the -1/0/1
        # signal DQN actually trains on.
        env = AtariWrapper(env, clip_reward=False)
        return env
    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained DQN agent greedily.")
    parser.add_argument("--model", type=str, default=best_model_path())
    parser.add_argument("--env-id", type=str, default=ENV_ID)
    parser.add_argument("--n-stack", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", action="store_true",
                         help="Open a live window. Requires a local display.")
    parser.add_argument("--record", action="store_true",
                         help="Save an mp4 to the videos directory.")
    parser.add_argument("--video-name", type=str, default="freeway_after_training")
    parser.add_argument("--video-length", type=int, default=20_000,
                         help="Upper bound on recorded frames. Set generously; "
                              "recording stops once --episodes complete regardless.")
    return parser.parse_args()


def run_with_render(args: argparse.Namespace) -> None:
    env = DummyVecEnv([make_single_env(args.env_id, render_mode="human")])
    env = VecFrameStack(env, n_stack=args.n_stack)
    model = DQN.load(args.model, env=env)

    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            done = bool(done[0])
        print(f"Episode {episode}: reward={total_reward:.1f}")

    env.close()


def run_with_recording(args: argparse.Namespace) -> None:
    env = DummyVecEnv([make_single_env(args.env_id, render_mode="rgb_array")])
    env = VecFrameStack(env, n_stack=args.n_stack)

    os.makedirs(VIDEOS_DIR, exist_ok=True)
    env = VecVideoRecorder(
        env,
        video_folder=VIDEOS_DIR,
        record_video_trigger=lambda step: step == 0,
        video_length=args.video_length,
        name_prefix=args.video_name,
    )

    model = DQN.load(args.model, env=env)

    obs = env.reset()
    episode_rewards = []
    total_reward = 0.0
    episodes_done = 0

    while episodes_done < args.episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward[0])
        if bool(done[0]):
            episode_rewards.append(total_reward)
            print(f"Episode {episodes_done + 1}: reward={total_reward:.1f}")
            total_reward = 0.0
            episodes_done += 1

    print(f"Recorded {args.episodes} episodes to {VIDEOS_DIR}")
    print(f"Mean reward across recorded episodes: {np.mean(episode_rewards):.2f}")

    env.close()


def main() -> None:
    args = parse_args()

    if not args.render and not args.record:
        raise SystemExit("Pass --render, --record, or both.")

    if args.render:
        run_with_render(args)

    if args.record:
        run_with_recording(args)


if __name__ == "__main__":
    main()
