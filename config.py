"""
Central configuration for the Freeway DQN project.

All scripts import from here instead of hardcoding paths or default
hyperparameters. Keeping this in one place means a run's configuration
can be reconstructed from a single object, which matters when you are
trying to reproduce or explain a specific experiment later.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

# Register the ALE/* environments with Gymnasium exactly once, here, in the
# module every script imports. ale-py >=0.9 ships the ROMs in the wheel but
# does not auto-register with Gymnasium, so make_atari_env / gym.make on
# "ALE/Freeway-v5" fails with a "namespace ALE not found" error unless this
# runs first. Putting it in config.py means train.py, evaluate.py, play.py,
# and compare_policies.py all get it for free via their `from config import`
# line — no per-file import, no runtime source patching in the notebook.
#
# Guarded so config.py stays importable on a machine without the training
# stack installed (local editing, check_presets.py): the preset table and
# paths do not need ALE, only the scripts that actually build an env do, and
# those run where ale-py is present.
try:
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)
except ImportError:
    pass


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
VIDEOS_DIR = os.path.join(PROJECT_ROOT, "videos")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
EXPERIMENT_LOG_CSV = os.path.join(EXPERIMENTS_DIR, "experiment_log.csv")

ENV_ID = "ALE/Freeway-v5"

for _dir in (MODELS_DIR, LOGS_DIR, PLOTS_DIR, VIDEOS_DIR, EXPERIMENTS_DIR):
    os.makedirs(_dir, exist_ok=True)


@dataclass
class DQNConfig:
    """
    One DQN training configuration.

    This maps directly onto the hyperparameters the assignment asks you
    to tune: learning rate, gamma, batch size, and the epsilon-greedy
    exploration schedule. Everything else here is either fixed
    infrastructure (env id, seed, frame stack) or run bookkeeping
    (run_id, member) needed to keep the experiment log honest.
    """

    run_id: str = "baseline"
    member: str = "unassigned"
    policy: str = "CnnPolicy"          # "CnnPolicy" or "MlpPolicy"
    env_id: str = ENV_ID
    n_envs: int = 4
    n_stack: int = 4
    seed: int = 42
    total_timesteps: int = 500_000

    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 32
    buffer_size: int = 100_000
    learning_starts: int = 10_000
    target_update_interval: int = 1_000
    train_freq: int = 4
    gradient_steps: int = 1

    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    exploration_fraction: float = 0.1   # fraction of training over which eps decays

    eval_freq: int = 25_000
    n_eval_episodes: int = 10
    # Reward a run must reach to count as "competent", used by the
    # sample-efficiency metric (steps_to_threshold). Set a little below the
    # ~22 plateau a trained Freeway agent reaches, so a config's speed to
    # near-competence is what varies, not whether it ever gets there.
    reward_threshold: float = 18.0

    def run_dir(self) -> str:
        return os.path.join(LOGS_DIR, self.run_id)

    def model_path(self) -> str:
        return os.path.join(MODELS_DIR, f"{self.run_id}.zip")

    def to_dict(self) -> dict:
        return asdict(self)


def best_model_path() -> str:
    """Path play.py loads by default: the model you promote as final."""
    return os.path.join(MODELS_DIR, "dqn_model.zip")


# ---------------------------------------------------------------------------
# Experiment presets
# ---------------------------------------------------------------------------
#
# Thirty experiments split across three members, one hyperparameter family
# each, so every member's ten runs tell one coherent story instead of ten
# disconnected data points:
#
#   Member 1: learning rate        (m1_lr_01 .. m1_lr_10)
#   Member 2: gamma and batch size (m2_gamma_01 .. m2_gamma_05,
#                                    m2_batch_01 .. m2_batch_05)
#   Member 3: exploration schedule (m3_eps_01 .. m3_eps_10)
#
# Every preset name and value pair here is mirrored in EXPERIMENTS.md, so
# that document and this dictionary must be kept in sync if either changes.
# Presets only set overrides; run_id and member are always included so the
# experiment log is self-describing without cross-referencing this file.

PRESETS: dict[str, dict] = {
    # --- Member 1: learning rate -------------------------------------
    "m1_lr_01_tiny":      {"run_id": "m1_lr_01_tiny",      "member": "member1", "learning_rate": 1e-6},
    "m1_lr_02_verylow":   {"run_id": "m1_lr_02_verylow",   "member": "member1", "learning_rate": 1e-5},
    "m1_lr_03_low":       {"run_id": "m1_lr_03_low",       "member": "member1", "learning_rate": 5e-5},
    "m1_lr_04_baseline":  {"run_id": "m1_lr_04_baseline",  "member": "member1", "learning_rate": 1e-4},
    "m1_lr_05_modhigh":   {"run_id": "m1_lr_05_modhigh",   "member": "member1", "learning_rate": 3e-4},
    "m1_lr_06_high":      {"run_id": "m1_lr_06_high",      "member": "member1", "learning_rate": 5e-4},
    "m1_lr_07_veryhigh":  {"run_id": "m1_lr_07_veryhigh",  "member": "member1", "learning_rate": 1e-3},
    "m1_lr_08_extreme":   {"run_id": "m1_lr_08_extreme",   "member": "member1", "learning_rate": 3e-3},
    "m1_lr_09_extreme2":  {"run_id": "m1_lr_09_extreme2",  "member": "member1", "learning_rate": 1e-2},
    # Fixed-schedule stand-in: SB3's DQN learning_rate accepts a float only
    # through this CLI, so the "schedule" experiment uses a low fixed rate
    # as a proxy for the late-training portion of a decayed schedule. Note
    # this honestly in the writeup rather than claiming a true schedule ran.
    "m1_lr_10_lowfixed":  {"run_id": "m1_lr_10_lowfixed",  "member": "member1", "learning_rate": 3e-5},

    # --- Member 2: gamma ------------------------------------------------
    "m2_gamma_01_short":     {"run_id": "m2_gamma_01_short",     "member": "member2", "gamma": 0.90},
    "m2_gamma_02_shortmed":  {"run_id": "m2_gamma_02_shortmed",  "member": "member2", "gamma": 0.95},
    "m2_gamma_03_baseline":  {"run_id": "m2_gamma_03_baseline",  "member": "member2", "gamma": 0.99},
    "m2_gamma_04_long":      {"run_id": "m2_gamma_04_long",      "member": "member2", "gamma": 0.995},
    "m2_gamma_05_verylong":  {"run_id": "m2_gamma_05_verylong",  "member": "member2", "gamma": 0.999},

    # --- Member 2: batch size --------------------------------------------
    "m2_batch_01_small":      {"run_id": "m2_batch_01_small",      "member": "member2", "batch_size": 8},
    "m2_batch_02_baseline":   {"run_id": "m2_batch_02_baseline",   "member": "member2", "batch_size": 32},
    "m2_batch_03_mod":        {"run_id": "m2_batch_03_mod",        "member": "member2", "batch_size": 64},
    "m2_batch_04_large":      {"run_id": "m2_batch_04_large",      "member": "member2", "batch_size": 128},
    "m2_batch_05_verylarge":  {"run_id": "m2_batch_05_verylarge",  "member": "member2", "batch_size": 256},

    # --- Member 3: exploration schedule ----------------------------------
    "m3_eps_01_fastdecay":     {"run_id": "m3_eps_01_fastdecay",     "member": "member3",
                                "exploration_final_eps": 0.05, "exploration_fraction": 0.02},
    "m3_eps_02_baseline":      {"run_id": "m3_eps_02_baseline",      "member": "member3",
                                "exploration_final_eps": 0.05, "exploration_fraction": 0.10},
    "m3_eps_03_slowdecay":     {"run_id": "m3_eps_03_slowdecay",     "member": "member3",
                                "exploration_final_eps": 0.05, "exploration_fraction": 0.30},
    "m3_eps_04_veryslow":      {"run_id": "m3_eps_04_veryslow",      "member": "member3",
                                "exploration_final_eps": 0.05, "exploration_fraction": 0.50},
    "m3_eps_05_highfloor":     {"run_id": "m3_eps_05_highfloor",     "member": "member3",
                                "exploration_final_eps": 0.20, "exploration_fraction": 0.10},
    "m3_eps_06_lowfloor":      {"run_id": "m3_eps_06_lowfloor",      "member": "member3",
                                "exploration_final_eps": 0.01, "exploration_fraction": 0.10},
    "m3_eps_07_zerofloor":     {"run_id": "m3_eps_07_zerofloor",     "member": "member3",
                                "exploration_final_eps": 0.0, "exploration_fraction": 0.10},
    "m3_eps_08_lowstart":      {"run_id": "m3_eps_08_lowstart",      "member": "member3",
                                "exploration_initial_eps": 0.5, "exploration_final_eps": 0.05,
                                "exploration_fraction": 0.10},
    "m3_eps_09_alwaysexplore": {"run_id": "m3_eps_09_alwaysexplore", "member": "member3",
                                "exploration_final_eps": 0.30, "exploration_fraction": 0.50},
    "m3_eps_10_aggressive":    {"run_id": "m3_eps_10_aggressive",    "member": "member3",
                                "exploration_final_eps": 0.02, "exploration_fraction": 0.05},
}


def apply_preset(base: "DQNConfig", preset_name: str) -> "DQNConfig":
    """
    Return a copy of base with a named preset's overrides applied.

    Raises KeyError with the list of valid names if preset_name is not
    recognized, rather than silently falling back to defaults.
    """
    from dataclasses import replace

    if preset_name not in PRESETS:
        valid = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(f"Unknown preset '{preset_name}'. Valid presets: {valid}")
    return replace(base, **PRESETS[preset_name])
