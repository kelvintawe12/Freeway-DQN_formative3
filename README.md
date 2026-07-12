# Freeway DQN

A Deep Q-Network agent trained to play Atari Freeway, built with Stable Baselines3 and Gymnasium.

The environment: a chicken has to cross a multi-lane highway full of moving cars. Every successful crossing gives a reward. Getting hit does not end the episode, it just resets the chicken's position, so the agent has to learn timing and patience rather than reflexes alone.

## Why Freeway

Freeway has a small action space (up, down, no-op) and trains faster than most Atari titles, which left more time for the part of this assignment that actually carries marks: running a real hyperparameter sweep, comparing policy architectures with evidence, and producing a working evaluation pipeline. The gameplay is also easy to judge visually. A random agent gets hit constantly. A trained agent waits for gaps and crosses in clean, deliberate movements.

## Project layout

```
Freeway-DQN/
├── config.py              default hyperparameters, shared paths, and the 30-run preset table
├── train.py                training script, fully CLI-driven, supports --preset
├── evaluate.py              runs N episodes headless, reports mean and std reward
├── play.py                 loads a trained model, plays greedily, renders or records
├── compare_policies.py      trains CnnPolicy and MlpPolicy under identical settings
├── requirements.txt
├── EXPERIMENTS.md            full 30-experiment plan: reasoning, predictions, fillable results
├── experiments/
│   └── experiment_log.csv   one row per training run, appended automatically
├── notebooks/
│   └── Freeway_DQN_Colab.ipynb
├── models/                  saved .zip checkpoints (created at runtime)
├── logs/                    TensorBoard logs, one folder per run (created at runtime)
├── plots/                   reward curves and comparison charts (created at runtime)
└── videos/                  recorded gameplay clips (created at runtime)
```

The `models`, `logs`, `plots`, and `videos` folders are not committed empty placeholders. They get created the first time you run any script, and they are meant to be populated by whichever machine actually trains the agent, normally Colab.

## How this is meant to be used

The code lives here and gets version-controlled from a local machine. Training happens on Colab, which has a GPU and cuts training time down substantially. The notebook in `notebooks/` clones this repo, installs the pinned dependencies, and runs the scripts in sequence. Nothing in the codebase assumes a specific machine: all paths are relative to the project root, and every script takes its configuration through CLI flags rather than hardcoded values.

## Setup

### Local (for editing code, and for `play.py --render`)

```bash
git clone <your-repo-url>
cd Freeway-DQN
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Colab (for training)

Open `notebooks/Freeway_DQN_Colab.ipynb`, set the runtime to GPU, and run the setup cell. It clones this repo and installs the same pinned versions listed in `requirements.txt`, so results are reproducible regardless of where they were produced.

## A note on reward clipping

Training uses reward clipping to -1, 0, and 1, following the original DQN paper (Mnih et al., 2015). This keeps Q-value targets on a comparable scale regardless of a game's raw scoring range, which is what makes DQN's hyperparameters transferable across different Atari titles in the first place. Evaluation and playback turn clipping off, so `evaluate.py` and `play.py` report the actual in-game score rather than the clipped training signal. For Freeway specifically this distinction barely matters, since a successful crossing is already worth exactly 1 point, but the code keeps the two paths separate on principle rather than assuming that will always be true.

## Running the scripts

Train a single configuration:

```bash
python train.py --run-id exp01_lr --member kelvin --learning-rate 5e-4 --notes "faster early learning, more variance late"
```

Every run appends one row to `experiments/experiment_log.csv` automatically. Pass `--promote` to also save the run as `models/dqn_model.zip`, the file `play.py` loads by default once you have decided which run is your best one.

Evaluate a trained model over several episodes:

```bash
python evaluate.py --model models/dqn_model.zip --episodes 20
```

Watch the agent play, locally with a live window:

```bash
python play.py --model models/dqn_model.zip --render --episodes 3
```

Record gameplay to video (works on Colab, since it does not need a display):

```bash
python play.py --model models/dqn_model.zip --record --episodes 3
```

Compare CnnPolicy against MlpPolicy under identical conditions:

```bash
python compare_policies.py --timesteps 200000
```

## Policy architecture: CNN vs MLP

Freeway's observations are 84x84 grayscale frames, stacked four deep so the agent can infer motion. CnnPolicy processes this with convolutional layers, which exploit the fact that nearby pixels are related and that the same visual pattern (a car, a lane boundary) can appear anywhere in the frame. MlpPolicy instead flattens the frame into a single vector and has to learn spatial relationships from raw pixel positions, with no shared structure across the image. `compare_policies.py` trains both under the same seed and timestep budget and produces `plots/mlp_vs_cnn.png` so the comparison is based on a measured reward curve rather than an assumption.

## Hyperparameter experiments

Thirty experiments total, ten per group member, one hyperparameter family owned by each member so every sweep tells a coherent story rather than being ten disconnected numbers:

- **Member 1**: learning rate
- **Member 2**: gamma (discount factor) and batch size
- **Member 3**: exploration schedule (epsilon-greedy)

Every experiment is a named preset in `config.py` (`PRESETS`), so running one is a single command instead of retyping flags:

```bash
python train.py --preset m1_lr_02_verylow --notes "slow but stable"
```

Any explicit flag overrides the preset's value for that field, so a preset can still be tweaked without editing `config.py`:

```bash
python train.py --preset m1_lr_02_verylow --total-timesteps 300000 --notes "shorter budget test"
```

The full list of all thirty runs, the reasoning behind each one, and predicted-versus-actual behavior columns to fill in live in **[EXPERIMENTS.md](EXPERIMENTS.md)**. That document and the `PRESETS` dictionary in `config.py` must stay in sync; if a value changes in one, change it in the other. Run `python check_presets.py` to verify they still agree — it exits non-zero and prints the specific mismatch if they have drifted.

**Compute budget:** the 500k-timestep default × 33 runs does not fit in a Colab session. The sweep runs at a reduced `--total-timesteps 150000`, reserving the full 500k for the baseline and the final combined run. See the "Compute budget" section of [EXPERIMENTS.md](EXPERIMENTS.md) for the full breakdown.

`member1`, `member2`, `Birasa` are placeholders in both the presets and the documentation. Replace them with actual names once the group assigns who owns which sweep, in `config.py` and in `EXPERIMENTS.md`.

Every run, regardless of whether it came from a preset or manual flags, appends one row to `experiments/experiment_log.csv` automatically.

## Gameplay video

`videos/freeway_before_training.mp4`: a random policy, included for contrast.
`videos/freeway_after_training.mp4`: the trained agent, produced by `play.py --record`.

## Contributions

| Member | Scripts / components owned | Hyperparameter axis | Experiments run |
|---|---|---|---|
| member1 | | Learning rate | 10 |
| member2 | | Gamma and batch size | 10 |
| Birasa | | Exploration schedule | 10 |

Replace `member1`/`member2`/`Birasa` with real names once assigned, in this table, in `config.py`, and in `EXPERIMENTS.md`.
