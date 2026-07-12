# Training Experiment Plan

This document is the source of truth for the hyperparameter tuning requirement: 30 experiments, 10 per group member, each varying one hyperparameter family from a shared baseline. Every run listed here corresponds to a named preset in `config.py`, so running an experiment is one command:

```bash
python train.py --preset <run_id> --notes "your observation after the run"
```

The metric and `notes` columns below are filled in once results exist in `experiments/experiment_log.csv`. This file is the narrative version of that log: readable in a presentation, with the reasoning behind each run stated up front rather than reconstructed afterward. Read the "How to read the results" section below before deciding which numbers to record — on Freeway, final greedy reward is the wrong headline.

Member names are placeholders (`member1`, `member2`, `Birasa`) until the group assigns real names. Update the `member` field in each relevant preset in `config.py`, and the header of each section below, once that is decided.

Before running anything, `python check_presets.py` confirms every preset in this document still matches `config.py` numerically. Run it after editing either file.

---

## Compute budget: read this before launching the sweep

The default `total_timesteps` in `config.py` is **500,000**, and there are **33 training runs** in the full plan (30 sweep presets + baseline + the 2 policy-comparison runs). At 500k each that is **~16.5M timesteps**, which on a single Colab GPU is roughly **30–90 hours** depending on the runtime you draw. That does not fit in Colab's session limits, and a disconnect loses everything since training does not checkpoint-resume.

Freeway also has a **sparse reward**: the agent sits at 0 for a long stretch before the first crossings appear, and several presets here (`m1_lr_09_extreme2` at 1e-2, `m3_eps_07_zerofloor`) are *designed* to never learn. Spending 500k steps on a run that was always going to fail is wasted budget.

So run the sweep at a **reduced budget** and reserve the full 500k for the two runs whose absolute score you actually quote. Relative hyperparameter ranking is almost always visible well before 500k.

| Phase | Runs | Suggested `--total-timesteps` | Approx GPU time* |
|---|---|---|---|
| Smoke test | 1 | 2,000 (already set in notebook) | seconds |
| Sweep (all 30 presets) | 30 | **150,000** | ~6–15 h total |
| Baseline (quoted number) | 1 | 500,000 | ~1–3 h |
| Final combined run | 1 | 500,000 | ~1–3 h |
| CnnPolicy vs MlpPolicy | 2 | 150,000–200,000 | ~2–6 h |

\*Rough, single mid-tier Colab GPU. Measure your own first run and rescale.

**Total drops to roughly 10–25 h**, and it splits cleanly across sessions and members. Override the budget per run without touching `config.py`:

```bash
python train.py --preset m1_lr_05_modhigh --total-timesteps 150000 --notes "..."
```

Practical guidance:

- **Split by member, not by axis-within-a-session.** Each member runs their own 10 presets in their own Colab session; the shared `experiment_log.csv` merges them because every run appends one row regardless of who ran it.
- **Reduce the buffer with the budget.** `buffer_size` defaults to 100k. At 150k steps a 100k buffer is fine, but if you go lower (e.g. 50k steps for a quick pass) also pass `--buffer-size 50000` so replay memory is not mostly empty.
- **The comparison runs need matched budgets.** `compare_policies.py --timesteps N` already trains both policies at the same `N`; keep it at 150k–200k so MLP has a fair shot before you conclude CNN wins.
- **Save results before the session dies.** Run the "Package results" cell (zip + download) periodically, not just at the very end — a 6-hour sweep that disconnects at hour 5 with nothing saved is a bad afternoon.

Once the sweep identifies each member's best value, the **baseline** and **final combined** runs are the only ones worth the full 500k, because those are the models you demo and quote.

---

## How to read the results: why "final mean reward" is the wrong headline

Freeway is a trap for the obvious metric. A chicken that simply holds "up" already scores ~20, and the greedy (`deterministic=True`), fixed-seed evaluation replays almost the same trajectory no matter what the network learned. In our own runs, `learning_rate=1e-6` and `learning_rate=1e-4` produced **byte-identical** greedy eval rewards (23.30, 22.20, 22.10, 22.40) for the first 100k steps despite a 100× difference in step size. If you rank the 30 experiments by final greedy reward, almost all of them tie at ~22 and the sweep has no story.

The rising `ep_rew_mean` you see during training is mostly **epsilon decaying** (fewer random downward moves), not the value network improving. So the plateau reward is real, but it is nearly the same for every config that trains at all.

What actually separates configurations is **how the reward is earned**, not its final value. Each run now logs these to `experiment_log.csv` (computed in `train.py`; no extra training):

| Column | Meaning | Better = |
|---|---|---|
| `steps_to_threshold` | first eval timestep reaching mean reward ≥ 18 (see `reward_threshold`) | **lower** — faster to competence |
| `auc_reward` | mean of all eval means over training (area under the reward curve) | **higher** — better sample efficiency |
| `late_reward_std` | std of the last ≤5 eval means | **lower** — more stable once settled |
| `stochastic_mean_reward` | non-greedy eval, where the learned action *distribution* matters | higher, and it *varies* between configs |
| `mean_reward` vs `final_mean_reward` | best checkpoint vs end-of-training weights | a large gap flags instability |

**How to frame each sweep in the presentation:**

- **Learning rate (Member 1):** rank by `steps_to_threshold` and `auc_reward`, not final reward. The story is "how fast, and how stably" — the too-high rates (1e-2, 3e-3) should show high `late_reward_std` or a big `mean_reward` vs `final_mean_reward` gap (they reach a peak then diverge), which is exactly the instability the extreme runs exist to demonstrate.
- **Gamma / batch size (Member 2):** gamma's effect is behavioral — read it off the gameplay video (patient gap-waiting vs rushing) plus `auc_reward`. Batch size is a stability-vs-compute story: pair `late_reward_std` with `wall_clock_seconds`.
- **Exploration (Member 3):** `steps_to_threshold` shows the explore/exploit tradeoff directly — fast-decay commits early (low steps, possibly higher `late_reward_std` if it settled wrong), slow-decay wastes budget exploring (high steps). The deliberately-broken `zerofloor` and `alwaysexplore` runs are the contrast cases; keep them.

The deliberately-extreme runs are your clearest evidence precisely because they *break* these metrics while the middle of each range looks flat. Report a documented failure as understanding, not as a gap.

---

## Shared baseline

Every axis below is measured as a deviation from this one reference run.

```bash
python train.py --run-id baseline --member <team> --notes "shared reference point for all three sweeps"
```

| Parameter | Value |
|---|---|
| learning_rate | 1e-4 |
| gamma | 0.99 |
| batch_size | 32 |
| buffer_size | 100000 |
| exploration_initial_eps | 1.0 |
| exploration_final_eps | 0.05 |
| exploration_fraction | 0.10 |

---

## Member 1: Learning rate

**Question this sweep answers:** how sensitive is training to the step size of each gradient update, and where does it break down at the extremes?

**How to run all ten:**

```bash
for preset in m1_lr_01_tiny m1_lr_02_verylow m1_lr_03_low m1_lr_04_baseline \
              m1_lr_05_modhigh m1_lr_06_high m1_lr_07_veryhigh m1_lr_08_extreme \
              m1_lr_09_extreme2 m1_lr_10_lowfixed; do
    python train.py --preset $preset --total-timesteps 150000 --notes "TODO: fill in after run"
done
```

| Run ID | Learning Rate | Predicted Behavior | Steps→18 / AUC / Late-std | Actual Observed Behavior |
|---|---|---|---|---|
| m1_lr_01_tiny | 1e-6 | Learning barely happens, reward stays near random | | |
| m1_lr_02_verylow | 1e-5 | Slow but stable, likely still improving at end of budget | | |
| m1_lr_03_low | 5e-5 | Slower than baseline, possibly more stable | | |
| m1_lr_04_baseline | 1e-4 | Reference point, same as shared baseline's learning rate | | |
| m1_lr_05_modhigh | 3e-4 | Faster early learning | | |
| m1_lr_06_high | 5e-4 | Faster still, watch for reward oscillation | | |
| m1_lr_07_veryhigh | 1e-3 | Likely unstable, Q-values may diverge | | |
| m1_lr_08_extreme | 3e-3 | Expected to fail or collapse, useful negative example | | |
| m1_lr_09_extreme2 | 1e-2 | Almost certainly breaks training entirely | | |
| m1_lr_10_lowfixed | 3e-5 | Stand-in for the tail end of a decayed schedule; see note below | | |

**Note on `m1_lr_10_lowfixed`:** Stable Baselines3's `DQN` accepts a fixed float for `learning_rate` through this CLI, not a decay schedule, even though SB3 itself supports schedule callables internally. This run uses a low fixed rate as an approximation of what a decayed schedule's late-training phase would look like, not a true schedule. State this plainly in the presentation rather than describing it as a schedule experiment; it is honest and still gives you a legitimate data point on low-learning-rate behavior.

**Expected narrative:** too low wastes the training budget without meaningfully learning; too high destabilizes the Bellman targets and can diverge; the useful range sits somewhere between `m1_lr_03_low` and `m1_lr_06_high`. State which specific value performed best once the runs complete.

---

## Member 2: Gamma and batch size

**Question this sweep answers:** how does the discount factor change the agent's willingness to wait for a safe gap versus rushing, and how does batch size trade off gradient noise against compute cost per update?

**How to run all ten:**

```bash
for preset in m2_gamma_01_short m2_gamma_02_shortmed m2_gamma_03_baseline \
              m2_gamma_04_long m2_gamma_05_verylong \
              m2_batch_01_small m2_batch_02_baseline m2_batch_03_mod \
              m2_batch_04_large m2_batch_05_verylarge; do
    python train.py --preset $preset --total-timesteps 150000 --notes "TODO: fill in after run"
done
```

### Gamma (discount factor)

| Run ID | Gamma | Predicted Behavior | Steps→18 / AUC / Late-std | Actual Observed Behavior |
|---|---|---|---|---|
| m2_gamma_01_short | 0.90 | Short-sighted, may cross without waiting for a safe gap | | |
| m2_gamma_02_shortmed | 0.95 | Still fairly reactive | | |
| m2_gamma_03_baseline | 0.99 | Reference point, same as shared baseline's gamma | | |
| m2_gamma_04_long | 0.995 | More patient, better gap prediction, possibly slower early learning | | |
| m2_gamma_05_verylong | 0.999 | Very long horizon, watch for training instability from large target values | | |

### Batch size

| Run ID | Batch Size | Predicted Behavior | Steps→18 / AUC / Late-std | Actual Observed Behavior |
|---|---|---|---|---|
| m2_batch_01_small | 8 | Noisy gradients, more updates per second, less stable | | |
| m2_batch_02_baseline | 32 | Reference point, same as shared baseline's batch size | | |
| m2_batch_03_mod | 64 | Smoother gradients, slightly slower per step | | |
| m2_batch_04_large | 128 | Even smoother, more compute per update | | |
| m2_batch_05_verylarge | 256 | Diminishing returns expected, mainly a wall-clock cost test | | |

**Expected narrative:** Freeway specifically rewards patience, since waiting for a gap beats a rushed crossing, so gamma is not just a numeric tuning knob here, it maps onto a visible behavioral difference in the gameplay footage. Batch size is more of a stability-versus-compute tradeoff; note both final reward and wall-clock training time per run from `experiment_log.csv` when discussing it.

---

## Member 3: Exploration schedule (epsilon-greedy)

**Question this sweep answers:** what is the right balance between exploring the environment early and exploiting a learned policy, and what happens at both extremes of that tradeoff?

**How to run all ten:**

```bash
for preset in m3_eps_01_fastdecay m3_eps_02_baseline m3_eps_03_slowdecay \
              m3_eps_04_veryslow m3_eps_05_highfloor m3_eps_06_lowfloor \
              m3_eps_07_zerofloor m3_eps_08_lowstart m3_eps_09_alwaysexplore \
              m3_eps_10_aggressive; do
    python train.py --preset $preset --total-timesteps 150000 --notes "TODO: fill in after run"
done
```

| Run ID | Eps Start | Eps End | Decay Fraction | Predicted Behavior | Steps→18 / AUC / Late-std | Actual Observed Behavior |
|---|---|---|---|---|---|---|
| m3_eps_01_fastdecay | 1.0 | 0.05 | 0.02 | Exploits early, risks settling on a suboptimal policy before seeing enough traffic patterns | | |
| m3_eps_02_baseline | 1.0 | 0.05 | 0.10 | Reference point, same as shared baseline's exploration schedule | | |
| m3_eps_03_slowdecay | 1.0 | 0.05 | 0.30 | More exploration time, slower convergence, possibly better final policy | | |
| m3_eps_04_veryslow | 1.0 | 0.05 | 0.50 | Spends half of training mostly exploring randomly | | |
| m3_eps_05_highfloor | 1.0 | 0.20 | 0.10 | Never fully commits to the greedy policy, adds noise even late in training | | |
| m3_eps_06_lowfloor | 1.0 | 0.01 | 0.10 | Nearly fully greedy late, minimal residual exploration | | |
| m3_eps_07_zerofloor | 1.0 | 0.0 | 0.10 | No residual exploration at all; can get permanently stuck if a good strategy has not been found yet | | |
| m3_eps_08_lowstart | 0.5 | 0.05 | 0.10 | Starts half-greedy from the beginning, less initial random data collected | | |
| m3_eps_09_alwaysexplore | 1.0 | 0.30 | 0.50 | Heavy, sustained exploration, likely the weakest final performance, useful as a too-much-exploration example | | |
| m3_eps_10_aggressive | 1.0 | 0.02 | 0.05 | Very fast commitment to exploitation, a sharp contrast case against the slow-decay runs | | |

**Expected narrative:** this axis is the clearest illustration of the exploration-exploitation tradeoff the rubric explicitly asks about. `m3_eps_07_zerofloor` and `m3_eps_09_alwaysexplore` are deliberately positioned as failure-mode examples on either end, keep them in the presentation even if they perform badly, a documented bad result is still evidence of understanding.

---

## Final combined run

Once each member has identified their best-performing value, combine them into one configuration and promote it as the model used for the final gameplay video and live demo.

```bash
python train.py --run-id final_combined --member team --promote \
    --learning-rate <member 1's best value> \
    --gamma <member 2's best gamma> \
    --batch-size <member 2's best batch size> \
    --exploration-final-eps <member 3's best eps end> \
    --exploration-fraction <member 3's best decay fraction> \
    --notes "combined best hyperparameters from all three members' independent sweeps"
```

This run is not a preset, since its values depend on results that do not exist until the three sweeps above are complete. Fill in the placeholders once you know them.

| Parameter | Best Value | From Member |
|---|---|---|
| learning_rate | | 1 |
| gamma | | 2 |
| batch_size | | 2 |
| exploration_final_eps | | 3 |
| exploration_fraction | | 3 |
| Final mean reward | | |

---

## Keeping this document and `config.py` in sync

Every preset name and value pair above must match the corresponding entry in `PRESETS` in `config.py` exactly. If a value changes in one place, change it in both. This file is the readable, presentation-facing version of that dictionary; `config.py` is the executable version. They are not allowed to drift apart.
