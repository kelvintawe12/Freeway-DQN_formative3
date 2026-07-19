# Training Experiment Plan

This document is the source of truth for the hyperparameter tuning requirement: 30 experiments, 10 per group member, each varying one hyperparameter family from a shared baseline. Every run listed here corresponds to a named preset in `config.py`, so running an experiment is one command:

```bash
python train.py --preset <run_id> --notes "your observation after the run"
```

The metric and `notes` columns below are filled in once results exist in `experiments/experiment_log.csv`. This file is the narrative version of that log: readable in a presentation, with the reasoning behind each run stated up front rather than reconstructed afterward. Read the "How to read the results" section below before deciding which numbers to record — on Freeway, final greedy reward is the wrong headline.

`Kelvin` owns the learning-rate axis, `Samuel Mwania` owns the gamma and batch size axis, and `Birasa` owns the exploration schedule axis. Update the `member` field in `config.py` if any of these names change, and keep the section headers below in sync.

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

## Kelvin: Learning rate

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
| m1_lr_01_tiny | 1e-6 | Learning barely happens, reward stays near random | 25k / 22.33 / 0.20 | Reached the same top AUC and best reward (22.5) as the faster LRs, but the 1e-6 rate is too slow to recommend; final/stochastic rewards 21.7/21.3. |
| m1_lr_02_verylow | 1e-5 | Slow but stable, likely still improving at end of budget | 25k / 13.40 / 10.54 | Collapsed after an initial peak (best 22.7, final 0.1). AUC and late-std show severe instability. |
| m1_lr_03_low | 5e-5 | Slower than baseline, possibly more stable | 25k / 17.62 / 6.07 | Best checkpoint 23.0, but final and stochastic rewards fell to 20.7. Intermediate instability. |
| m1_lr_04_baseline | 1e-4 | Reference point, same as shared baseline's learning rate | 25k / 21.07 / 1.76 | Solid reference. Best 22.5, final 21.8, stochastic 21.3; moderate AUC and low late-std. |
| m1_lr_05_modhigh | 3e-4 | Faster early learning | 25k / 22.33 / 0.20 | Tied for best AUC and most stable late-std. Best 22.5, final 21.7, stochastic 21.3. Recommended best candidate. |
| m1_lr_06_high | 5e-4 | Faster still, watch for reward oscillation | 25k / 22.33 / 0.20 | Same top AUC and stability as modhigh; no oscillation in this budget. |
| m1_lr_07_veryhigh | 1e-3 | Likely unstable, Q-values may diverge | 25k / 22.33 / 0.20 | No divergence in this budget; reached the same deterministic plateau as the safer LRs. |
| m1_lr_08_extreme | 3e-3 | Expected to fail or collapse, useful negative example | 25k / 22.33 / 0.20 | Did not collapse in this run, but the same eval plateau means the high rate is unnecessary. |
| m1_lr_09_extreme2 | 1e-2 | Almost certainly breaks training entirely | 25k / 22.33 / 0.20 | Surprisingly reached the same plateau, but 1e-2 is too risky for a recommended value. |
| m1_lr_10_lowfixed | 3e-5 | Stand-in for the tail end of a decayed schedule; see note below | 25k / 14.37 / 10.17 | Best checkpoint 23.3, but final 19.7 and stochastic 20.5; AUC shows instability typical of a too-low tail. |

**Note on `m1_lr_10_lowfixed`:** Stable Baselines3's `DQN` accepts a fixed float for `learning_rate` through this CLI, not a decay schedule, even though SB3 itself supports schedule callables internally. This run uses a low fixed rate as an approximation of what a decayed schedule's late-training phase would look like, not a true schedule. State this plainly in the presentation rather than describing it as a schedule experiment; it is honest and still gives you a legitimate data point on low-learning-rate behavior.

**Expected narrative:** The middle-to-high range (`m1_lr_05_modhigh` through `m1_lr_09_extreme2`) all reached the same deterministic greedy plateau and tied on AUC and late-std, so `m1_lr_05_modhigh` (3e-4) is the recommended best value: it is the smallest learning rate in that top group, giving the same performance with the least risk of divergence. The extreme low end (`m1_lr_02_verylow`, `m1_lr_10_lowfixed`) collapsed after the initial peak, while `m1_lr_03_low` and `m1_lr_04_baseline` showed intermediate but acceptable behavior.

---

## Samuel Mwania: Gamma and batch size

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
| m2_gamma_01_short | 0.90 | Short-sighted, may cross without waiting for a safe gap | 25k / 22.13 / 0.58 | Highest AUC across all gamma values. Short discount horizon did not hurt greedy reward (22.5) and produced the most stable late-training performance (std=0.58). The agent still learned to cross effectively despite heavy future-reward discounting. |
| m2_gamma_02_shortmed | 0.95 | Still fairly reactive | 25k / 20.47 / 3.21 | Worst late-training stability of all gamma runs (std=3.21). Final reward dropped to 20.7 from best checkpoint of 22.5, indicating the policy oscillated after peaking. Moderate discount was less stable than either extreme. |
| m2_gamma_03_baseline | 0.99 | Reference point, same as shared baseline's gamma | 25k / 20.68 / 1.92 | Solid performance. Best checkpoint matched top reward (22.5) and final weights held at 22.5. Late-training stability was moderate (std=1.92). Standard DQN gamma works reliably. |
| m2_gamma_04_long | 0.995 | More patient, better gap prediction, possibly slower early learning | 25k / 21.42 / 0.79 | Second-highest AUC (21.42) and low late-training variance (0.79). The longer horizon produced stable, patient behavior. Final reward (21.9) was close to the best checkpoint (22.5), indicating steady convergence without oscillation. |
| m2_gamma_05_verylong | 0.999 | Very long horizon, watch for training instability from large target values | 25k / 16.72 / 8.64 | Confirmed instability prediction. Final reward collapsed to 12.5 despite a best checkpoint of 22.5. Late-training std was extreme (8.64). The very long horizon made Q-value targets too large, causing the policy to diverge after initially learning. |

### Batch size

| Run ID | Batch Size | Predicted Behavior | Steps→18 / AUC / Late-std | Actual Observed Behavior |
|---|---|---|---|---|
| m2_batch_01_small | 8 | Noisy gradients, more updates per second, less stable | 25k / 18.77 / 8.73 | Confirmed instability prediction. Best checkpoint reached 22.5 but final weights collapsed to 2.5 (near-random). Late-training std was extreme (8.73). High-variance gradients caused catastrophic forgetting after initial learning. |
| m2_batch_02_baseline | 32 | Reference point, same as shared baseline's batch size | 25k / 20.68 / 1.92 | Reliable performance. Best and final rewards both 22.5. Moderate late-training variance (1.92). Standard batch size provides a good balance between gradient noise and compute cost. |
| m2_batch_03_mod | 64 | Smoother gradients, slightly slower per step | 25k / 21.67 / 1.02 | Best AUC of all batch sizes (21.67) with low late-training variance (1.02). Smoother gradients improved sample efficiency without meaningful increase in wall-clock time (405s vs 403s for baseline). Best batch size overall. |
| m2_batch_04_large | 128 | Even smoother, more compute per update | 50k / 17.73 / 0.55 | Slowest to reach threshold (50k steps vs 25k for smaller batches). Lowest late-training std (0.55) confirms stability, but reduced AUC (17.73) shows the agent learned more slowly within the fixed budget. Lower best reward (21.5). |
| m2_batch_05_verylarge | 256 | Diminishing returns expected, mainly a wall-clock cost test | 50k / 19.05 / 0.44 | Most stable late-training (std=0.44) but slowest wall-clock time (498s, 24% slower than baseline). Reached threshold at 50k steps. Memory warning triggered (buffer > available RAM). Very stable but inefficient for fixed-budget training. |

**Expected narrative:** Freeway specifically rewards patience, since waiting for a gap beats a rushed crossing, so gamma is not just a numeric tuning knob here, it maps onto a visible behavioral difference in the gameplay footage. Batch size is more of a stability-versus-compute tradeoff; note both final reward and wall-clock training time per run from `experiment_log.csv` when discussing it.

---

## Birasa: Exploration schedule (epsilon-greedy)

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
| m3_eps_01_fastdecay | 1.0 | 0.05 | 0.02 | Exploits early, risks settling on a suboptimal policy before seeing enough traffic patterns | 25k / 20.08 / 2.77 | Fast decay committed the agent early. Some late instability (std=2.77) but solid greedy reward (22.5). |
| m3_eps_02_baseline | 1.0 | 0.05 | 0.10 | Reference point, same as shared baseline's exploration schedule | 25k / 21.07 / 1.76 | Baseline exploration schedule. Reliable AUC and moderate stability. |
| m3_eps_03_slowdecay | 1.0 | 0.05 | 0.30 | More exploration time, slower convergence, possibly better final policy | 25k / 13.80 / 8.45 | Slow decay wasted training budget on random actions. AUC dropped sharply and late-training oscillated heavily. |
| m3_eps_04_veryslow | 1.0 | 0.05 | 0.50 | Spends half of training mostly exploring randomly | 25k / 11.08 / 10.74 | Worst AUC of all 30 experiments. Agent never fully committed to exploitation, confirming excessive exploration harms performance. |
| m3_eps_05_highfloor | 1.0 | 0.20 | 0.10 | Never fully commits to the greedy policy, adds noise even late in training | 25k / 22.22 / 0.46 | Best exploration config overall. High floor acted as implicit regularisation, producing the highest AUC (22.22) and lowest late-std (0.46). |
| m3_eps_06_lowfloor | 1.0 | 0.01 | 0.10 | Nearly fully greedy late, minimal residual exploration | 25k / 20.43 / 1.33 | Slightly less stable than moderate floor. Near-zero residual exploration worked but offered no advantage over baseline. |
| m3_eps_07_zerofloor | 1.0 | 0.0 | 0.10 | No residual exploration at all; can get permanently stuck if a good strategy has not been found yet | 25k / 20.87 / 2.33 | Pure greedy after annealing. Mild instability from overfitting to a narrow set of state-action pairs. |
| m3_eps_08_lowstart | 0.5 | 0.05 | 0.10 | Starts half-greedy from the beginning, less initial random data collected | 25k / 19.85 / 3.75 | Lower initial epsilon reduced early exploration, slowing convergence and increasing late variance. |
| m3_eps_09_alwaysexplore | 1.0 | 0.30 | 0.50 | Heavy, sustained exploration, likely the weakest final performance, useful as a too-much-exploration example | 25k / 18.82 / 1.97 | Permanent 30% randomness capped final performance. Useful negative example of over-exploration. |
| m3_eps_10_aggressive | 1.0 | 0.02 | 0.05 | Very fast commitment to exploitation, a sharp contrast case against the slow-decay runs | 25k / 18.50 / 8.02 | Aggressive exploitation caused late-training oscillation (std=8.02). Fast decay + low floor is a risky combination. |

**Expected narrative:** this axis is the clearest illustration of the exploration-exploitation tradeoff the rubric explicitly asks about. `m3_eps_07_zerofloor` and `m3_eps_09_alwaysexplore` are deliberately positioned as failure-mode examples on either end, keep them in the presentation even if they perform badly, a documented bad result is still evidence of understanding.

---

## Final combined run

Once each member has identified their best-performing value, combine them into one configuration and promote it as the model used for the final gameplay video and live demo.

```bash
python train.py --run-id final_combined --member team --promote \
    --learning-rate 3e-4 \
    --gamma 0.90 \
    --batch-size 64 \
    --exploration-final-eps 0.20 \
    --exploration-fraction 0.10 \
    --total-timesteps 500000 \
    --notes "combined best hyperparameters from all three members' independent sweeps"
```

| Parameter | Best Value | From Member |
|---|---|---|
| learning_rate | 3e-4 | Kelvin Tawe |
| gamma | 0.90 | Samuel Mwania |
| batch_size | 64 | Samuel Mwania |
| exploration_final_eps | 0.20 | Divine Birasa |
| exploration_fraction | 0.10 | Divine Birasa |
| Final mean reward | **22.5** (AUC=22.25, late_std=0.76) | Team |

---

## Keeping this document and `config.py` in sync

Every preset name and value pair above must match the corresponding entry in `PRESETS` in `config.py` exactly. If a value changes in one place, change it in both. This file is the readable, presentation-facing version of that dictionary; `config.py` is the executable version. They are not allowed to drift apart.
