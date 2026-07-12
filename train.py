"""
Train a DQN agent on an Atari environment (default: ALE/Freeway-v5).

Every hyperparameter the assignment requires you to tune is exposed as a
CLI flag. This is deliberate: running ten experiments should mean running
ten commands with different flags and a different --run-id, not editing
the script ten times. Each run writes its own TensorBoard log directory
and appends one row to experiments/experiment_log.csv, so the log builds
itself as you work instead of being reconstructed from memory afterward.

Example (manual flags):
    python train.py --run-id exp01_lr --learning-rate 5e-4 --member kelvin

Example (named preset, see config.PRESETS for the full list):
    python train.py --preset m1_lr_02_verylow --notes "slow but stable"
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import fields

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

from config import DQNConfig, EXPERIMENT_LOG_CSV, MODELS_DIR, PRESETS, apply_preset


def build_train_env(cfg: DQNConfig):
    """
    Build the vectorized, frame-stacked environment used for training.

    make_atari_env applies the standard Atari preprocessing wrapper (frame
    skip, grayscale, resize to 84x84, terminal-on-life-loss, reward
    clipping to -1/0/1). Frame stacking is added on top so the agent can
    infer motion, since a single frame carries no velocity information.

    Reward clipping stays on here, matching the original DQN paper
    (Mnih et al., 2015). Clipping stabilizes Q-value targets across games
    with wildly different score scales, which is the point even though
    Freeway's own reward is already binary and unaffected by it either way.

    Monitor logs are written to the run's own directory so episode reward
    and length can be plotted from disk later, independent of TensorBoard.
    """
    env = make_atari_env(cfg.env_id, n_envs=cfg.n_envs, seed=cfg.seed,
                          monitor_dir=cfg.run_dir())
    env = VecFrameStack(env, n_stack=cfg.n_stack)
    return env


def build_eval_env(cfg: DQNConfig):
    """
    Build a single-env, frame-stacked environment used for evaluation.

    Reward clipping is turned off here on purpose. Training needs clipped
    rewards for stable targets, but a mean reward you are going to quote
    in a presentation should reflect the actual in-game score, not a
    clipped proxy for it.
    """
    env = make_atari_env(cfg.env_id, n_envs=1, seed=cfg.seed + 1000,
                          wrapper_kwargs={"clip_reward": False})
    env = VecFrameStack(env, n_stack=cfg.n_stack)
    return env


def curve_metrics(run_dir: str, threshold: float) -> dict:
    """
    Derive sample-efficiency and stability metrics from the learning curve
    EvalCallback already wrote to run_dir/evaluations.npz.

    On Freeway the final greedy eval reward barely separates configurations
    (a chicken that mostly holds "up" already scores ~20, and a fixed-seed
    deterministic eval produces near-identical trajectories regardless of
    learning rate). What *does* separate them is how the reward gets there:

      steps_to_threshold  first eval timestep whose mean reward >= threshold,
                          i.e. how fast the config becomes competent. Lower is
                          better; NaN means it never crossed the bar.
      auc_reward          mean of all eval means over training, a single number
                          for "area under the reward curve" / overall efficiency.
      late_reward_std     std of the last few eval means, i.e. how stable the
                          policy is once it has settled. High values flag the
                          oscillation the high-learning-rate runs are meant to
                          show.

    Returns NaNs if the file is missing (e.g. training never reached one
    eval), so a row is still written rather than crashing the run.
    """
    npz = os.path.join(run_dir, "evaluations.npz")
    if not os.path.isfile(npz):
        return {"steps_to_threshold": float("nan"),
                "auc_reward": float("nan"),
                "late_reward_std": float("nan")}

    data = np.load(npz)
    timesteps = data["timesteps"]
    per_eval_mean = data["results"].mean(axis=1)  # results: (n_evals, n_episodes)

    crossed = np.where(per_eval_mean >= threshold)[0]
    steps_to_threshold = float(timesteps[crossed[0]]) if len(crossed) else float("nan")

    tail = per_eval_mean[-min(5, len(per_eval_mean)):]
    return {
        "steps_to_threshold": steps_to_threshold,
        "auc_reward": float(per_eval_mean.mean()),
        "late_reward_std": float(tail.std()),
    }


def append_to_experiment_log(cfg: DQNConfig, mean_reward: float, std_reward: float,
                              final_mean_reward: float, final_std_reward: float,
                              stochastic_mean: float, stochastic_std: float,
                              metrics: dict, wall_clock_seconds: float,
                              notes: str) -> None:
    """
    Append one row to the shared experiment log.

    The log is intentionally a flat CSV rather than per-member files that
    get merged later, because merging always introduces silent mistakes
    right before a deadline. Every run, from every group member, lands in
    the same file.

    Reward columns, and why there are several:
      mean_reward/std_reward      best checkpoint, greedy (deterministic) eval
      final_mean/final_std        end-of-training weights, greedy eval
      stochastic_mean/std         best checkpoint, sampled (non-greedy) eval
    Greedy eval on Freeway is nearly degenerate across configs; the
    stochastic pass and the curve metrics (steps_to_threshold, auc_reward,
    late_reward_std) are what actually separate one hyperparameter setting
    from another. See curve_metrics() and EXPERIMENTS.md for how to read them.
    """
    row = cfg.to_dict()
    row["mean_reward"] = round(mean_reward, 3)
    row["std_reward"] = round(std_reward, 3)
    row["final_mean_reward"] = round(final_mean_reward, 3)
    row["final_std_reward"] = round(final_std_reward, 3)
    row["stochastic_mean_reward"] = round(stochastic_mean, 3)
    row["stochastic_std_reward"] = round(stochastic_std, 3)
    row["steps_to_threshold"] = round(metrics["steps_to_threshold"], 1)
    row["auc_reward"] = round(metrics["auc_reward"], 3)
    row["late_reward_std"] = round(metrics["late_reward_std"], 3)
    row["wall_clock_seconds"] = round(wall_clock_seconds, 1)
    row["notes"] = notes

    file_exists = os.path.isfile(EXPERIMENT_LOG_CSV)
    with open(EXPERIMENT_LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> DQNConfig:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--preset", type=str, default=None,
        help="Optional named preset from config.PRESETS (e.g. m1_lr_02_verylow). "
             "Any explicit flag below overrides the preset's value for that field. "
             f"Valid presets: {', '.join(sorted(PRESETS.keys()))}",
    )
    pre_args, _ = pre_parser.parse_known_args()

    defaults = DQNConfig()
    if pre_args.preset is not None:
        defaults = apply_preset(defaults, pre_args.preset)

    parser = argparse.ArgumentParser(
        parents=[pre_parser], description="Train a DQN agent on Atari Freeway."
    )

    parser.add_argument("--run-id", type=str, default=defaults.run_id)
    parser.add_argument("--member", type=str, default=defaults.member)
    parser.add_argument("--policy", type=str, default=defaults.policy,
                         choices=["CnnPolicy", "MlpPolicy"])
    parser.add_argument("--env-id", type=str, default=defaults.env_id)
    parser.add_argument("--n-envs", type=int, default=defaults.n_envs)
    parser.add_argument("--n-stack", type=int, default=defaults.n_stack)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--total-timesteps", type=int, default=defaults.total_timesteps)

    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--buffer-size", type=int, default=defaults.buffer_size)
    parser.add_argument("--learning-starts", type=int, default=defaults.learning_starts)
    parser.add_argument("--target-update-interval", type=int,
                         default=defaults.target_update_interval)
    parser.add_argument("--train-freq", type=int, default=defaults.train_freq)
    parser.add_argument("--gradient-steps", type=int, default=defaults.gradient_steps)

    parser.add_argument("--exploration-initial-eps", type=float,
                         default=defaults.exploration_initial_eps)
    parser.add_argument("--exploration-final-eps", type=float,
                         default=defaults.exploration_final_eps)
    parser.add_argument("--exploration-fraction", type=float,
                         default=defaults.exploration_fraction)

    parser.add_argument("--eval-freq", type=int, default=defaults.eval_freq)
    parser.add_argument("--n-eval-episodes", type=int, default=defaults.n_eval_episodes)
    parser.add_argument("--notes", type=str, default="",
                         help="Free-text note describing observed behavior for this run.")
    parser.add_argument("--promote", action="store_true",
                         help="Also save this model as models/dqn_model.zip, "
                              "the one play.py loads by default.")

    args = parser.parse_args()
    # argparse already turns --learning-rate into args.learning_rate, so the
    # keys here match DQNConfig field names directly.
    field_names = {f.name for f in fields(DQNConfig)}
    cfg_kwargs = {k: v for k, v in vars(args).items() if k in field_names}
    cfg = DQNConfig(**cfg_kwargs)
    return cfg, args.notes, args.promote


def main() -> None:
    cfg, notes, promote = parse_args()

    print(f"Starting run '{cfg.run_id}' | policy={cfg.policy} | "
          f"lr={cfg.learning_rate} gamma={cfg.gamma} batch_size={cfg.batch_size} "
          f"eps=({cfg.exploration_initial_eps} -> {cfg.exploration_final_eps}, "
          f"fraction={cfg.exploration_fraction})")

    train_env = build_train_env(cfg)
    eval_env = build_eval_env(cfg)

    model = DQN(
        policy=cfg.policy,
        env=train_env,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        batch_size=cfg.batch_size,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        target_update_interval=cfg.target_update_interval,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        exploration_initial_eps=cfg.exploration_initial_eps,
        exploration_final_eps=cfg.exploration_final_eps,
        exploration_fraction=cfg.exploration_fraction,
        seed=cfg.seed,
        tensorboard_log=cfg.run_dir(),
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=cfg.run_dir(),
        log_path=cfg.run_dir(),
        eval_freq=max(cfg.eval_freq // cfg.n_envs, 1),
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    start = time.time()
    model.learn(total_timesteps=cfg.total_timesteps, callback=eval_callback,
                tb_log_name=cfg.run_id, progress_bar=True)
    elapsed = time.time() - start

    # Evaluate the weights as they stand at the end of training. For an
    # unstable run this can be well below the best point ever reached, so
    # it is reported but never the one we save or promote.
    final_mean, final_std = evaluate_policy(
        model, eval_env, n_eval_episodes=cfg.n_eval_episodes, deterministic=True
    )

    # EvalCallback wrote the best checkpoint it saw to run_dir/best_model.zip.
    # Prefer that over the final weights: it is the model that actually earned
    # this run's headline number. Fall back to the final weights only if the
    # callback never triggered (e.g. total_timesteps < eval_freq, as in the
    # smoke test), so a tiny run still produces a usable model and log row.
    best_ckpt = os.path.join(cfg.run_dir(), "best_model.zip")
    if os.path.isfile(best_ckpt):
        best_model = DQN.load(best_ckpt, env=eval_env)
        best_mean, best_std = evaluate_policy(
            best_model, eval_env, n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,
        )
        model_to_save = best_model
        print(f"Using best checkpoint from {best_ckpt} "
              f"(best mean_reward={best_mean:.2f} vs final={final_mean:.2f})")
    else:
        best_mean, best_std = final_mean, final_std
        model_to_save = model
        print("No best_model.zip from EvalCallback (training shorter than "
              "eval_freq?); logging and saving the final weights instead.")

    model_to_save.save(cfg.model_path())
    print(f"Saved model to {cfg.model_path()}")

    if promote:
        promoted_path = os.path.join(MODELS_DIR, "dqn_model.zip")
        model_to_save.save(promoted_path)
        print(f"Promoted this run to {promoted_path} (used by play.py default)")

    # Stochastic (non-greedy) eval of the model we kept. Greedy eval on
    # Freeway is near-degenerate across configs (see curve_metrics), so this
    # sampled pass, where the learned action *distribution* matters, gives a
    # reward number that can actually differ between hyperparameter settings.
    stochastic_mean, stochastic_std = evaluate_policy(
        model_to_save, eval_env, n_eval_episodes=cfg.n_eval_episodes,
        deterministic=False,
    )

    # Sample-efficiency / stability metrics from the learning curve.
    metrics = curve_metrics(cfg.run_dir(), cfg.reward_threshold)

    append_to_experiment_log(cfg, best_mean, best_std, final_mean, final_std,
                             stochastic_mean, stochastic_std, metrics,
                             elapsed, notes)
    print(f"Run '{cfg.run_id}' finished: mean_reward={best_mean:.2f} "
          f"std_reward={best_std:.2f} (final_mean={final_mean:.2f}, "
          f"stochastic={stochastic_mean:.2f}) | "
          f"steps_to_{cfg.reward_threshold:g}={metrics['steps_to_threshold']} "
          f"auc={metrics['auc_reward']:.2f} "
          f"late_std={metrics['late_reward_std']:.3f} "
          f"wall_clock={elapsed:.1f}s")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
