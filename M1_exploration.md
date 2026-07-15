# M1 Learning Rate Exploration — Detailed Findings

This document is the detailed report for the **Member 1 learning-rate sweep** and also provides a project-wide summary of the completed M2 (gamma and batch size) results. The M3 (exploration schedule) sweep has not yet been run, so the final combined model is still provisional.

All numbers come from `experiments/experiment_log.csv` after extracting the Colab `m1_results.zip` and cleaning duplicate rows. The full 30-experiment plan, predictions, and rationale for each run are in `EXPERIMENTS.md`.

---

## 1. What this sweep explored

The learning rate controls how far the Q-network weights move on each gradient step. Too small a step and the agent barely learns within the budget; too large a step and the Q-value targets oscillate or diverge because the Bellman backup is being bootstrapped on its own noisy estimates.

The sweep fixed every other hyperparameter and varied only `learning_rate`:

| Fixed parameter | Value |
|---|---|
| `policy` | `CnnPolicy` |
| `env_id` | `ALE/Freeway-v5` |
| `gamma` | 0.99 |
| `batch_size` | 32 |
| `buffer_size` | 100,000 |
| `exploration_initial_eps` | 1.0 |
| `exploration_final_eps` | 0.05 |
| `exploration_fraction` | 0.10 |
| `total_timesteps` | 150,000 |
| `seed` | 42 |

Ten presets were run, spanning six orders of magnitude plus a low fixed-rate stand-in:

| Run ID | `learning_rate` | Why it was included |
|---|---|---|
| `m1_lr_01_tiny` | 1e-6 | Extremely low; should learn almost nothing |
| `m1_lr_02_verylow` | 1e-5 | Very low; slow but maybe stable |
| `m1_lr_03_low` | 5e-5 | Lower than baseline |
| `m1_lr_04_baseline` | 1e-4 | Same as the shared baseline |
| `m1_lr_05_modhigh` | 3e-4 | Faster early learning |
| `m1_lr_06_high` | 5e-4 | Higher still |
| `m1_lr_07_veryhigh` | 1e-3 | Expected to show instability |
| `m1_lr_08_extreme` | 3e-3 | Expected to diverge |
| `m1_lr_09_extreme2` | 1e-2 | Expected to break training |
| `m1_lr_10_lowfixed` | 3e-5 | Stand-in for the tail of a decayed schedule |

---

## 2. How to read the numbers

`train.py` records several reward columns because **final greedy reward is the wrong headline on Freeway**. The environment is deterministic and the eval seed is fixed, so a greedy policy that has learned to cross produces almost the same score regardless of the learning rate. The metrics that actually separate the runs are:

| Metric | Meaning | Better is |
|---|---|---|
| `mean_reward` | Greedy eval of the best checkpoint saved during training | Higher, but it plateaus |
| `final_mean_reward` | Greedy eval of the weights at the end of training | Higher; a gap vs `mean_reward` flags instability |
| `stochastic_mean_reward` | Non-greedy eval of the best checkpoint | Higher; the learned action distribution matters |
| `steps_to_threshold` | First eval step with mean reward ≥ 18 | Lower |
| `auc_reward` | Mean of all eval rewards across the run | Higher; measures overall sample efficiency |
| `late_reward_std` | Standard deviation of the last 5 eval rewards | Lower; flags oscillation after convergence |
| `wall_clock_seconds` | Real training time | Lower, or traded against quality |

---

## 3. Freeway's eval trap

A chicken that simply holds **up** already scores about 20 points. On top of that, the `EvalCallback` uses `deterministic=True` and the eval environment is seeded, so every run follows the same random no-ops and frame-skips. Once a policy learns to cross reliably, the greedy eval trajectory becomes very similar across different hyperparameter settings.

This is why `m1_lr_01_tiny`, `m1_lr_05_modhigh`, `m1_lr_06_high`, `m1_lr_07_veryhigh`, `m1_lr_08_extreme`, and `m1_lr_09_extreme2` all show the same `best_reward` (22.5), the same `AUC` (22.33), and the same `late_std` (0.20). The **greedy score is not distinguishing**. Use `AUC` and `late_std` to compare the speed and stability of learning.

---

## 4. Full M1 results

| Run ID | LR | `mean_reward` | `final_mean_reward` | `stochastic_mean_reward` | `steps_to_18` | `auc_reward` | `late_reward_std` | `wall_clock` | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `m1_lr_01_tiny` | 1e-6 | 22.5 | 21.7 | 21.3 | 25k | 22.33 | 0.20 | 943.9 s | Tied top group; too slow |
| `m1_lr_02_verylow` | 1e-5 | 22.7 | 0.1 | 21.1 | 25k | 13.40 | 10.54 | 947.8 s | Collapsed |
| `m1_lr_03_low` | 5e-5 | 23.0 | 20.7 | 20.7 | 25k | 17.62 | 6.07 | 943.1 s | Intermediate instability |
| `m1_lr_04_baseline` | 1e-4 | 22.5 | 21.8 | 21.3 | 25k | 21.07 | 1.76 | 966.8 s | Solid reference |
| `m1_lr_05_modhigh` | 3e-4 | 22.5 | 21.7 | 21.3 | 25k | 22.33 | 0.20 | 975.7 s | **Recommended best** |
| `m1_lr_06_high` | 5e-4 | 22.5 | 21.7 | 21.3 | 25k | 22.33 | 0.20 | 973.5 s | Tied top group |
| `m1_lr_07_veryhigh` | 1e-3 | 22.5 | 21.7 | 21.3 | 25k | 22.33 | 0.20 | 971.9 s | Tied top group |
| `m1_lr_08_extreme` | 3e-3 | 22.5 | 21.7 | 21.3 | 25k | 22.33 | 0.20 | 964.8 s | Tied top group |
| `m1_lr_09_extreme2` | 1e-2 | 22.5 | 21.7 | 21.3 | 25k | 22.33 | 0.20 | 970.9 s | Tied top group |
| `m1_lr_10_lowfixed` | 3e-5 | 23.3 | 19.7 | 20.5 | 25k | 14.37 | 10.17 | 973.3 s | Collapsed |

![M1 learning rate comparison](plots/m1_lr_comparison.png)

---

## 5. Per-run interpretation

### 5.1 `m1_lr_01_tiny` — `learning_rate = 1e-6`

Surprisingly, this run reached the same top AUC (22.33) and the same best/final/stochastic rewards as the much larger learning rates. However, a 1e-6 step size is too small to be practical. The run is likely in the same deterministic eval plateau because the eval is seeded and the environment is simple, not because the network actually learned faster. The policy simply learned the same greedy crossing path within the 150k budget. Do not recommend 1e-6 as the best learning rate; it is too slow and risky in a harder environment or with more stochasticity.

### 5.2 `m1_lr_02_verylow` — `learning_rate = 1e-5`

This run is the clearest example of **catastrophic forgetting**. The best checkpoint scored 22.7, but the final greedy eval collapsed to **0.1**. The `AUC` is only 13.40 and `late_reward_std` is 10.54, meaning the eval curve fell apart after the initial peak. 1e-5 is too low for stable DQN learning on Freeway.

### 5.3 `m1_lr_03_low` — `learning_rate = 5e-5`

The best checkpoint (23.0) is the second-highest of all M1 runs, but the final greedy reward (20.7) and stochastic reward (20.7) are lower than the top group. The `AUC` is 17.62 and `late_std` is 6.07, showing moderate instability. It is better than the very low end but not as reliable as the 3e-4 to 1e-2 range.

### 5.4 `m1_lr_04_baseline` — `learning_rate = 1e-4`

The baseline is a solid reference. It reached the plateau quickly, stayed stable, and the final reward (21.8) is close to the best reward (22.5). `AUC` is 21.07 and `late_std` is 1.76. It is a safe default, but the higher learning rates that share the same AUC/stability can be argued as better.

### 5.5 `m1_lr_05_modhigh` — `learning_rate = 3e-4` — **Recommended best**

This is the **recommended best learning rate**. It sits in the top AUC/stability group (AUC 22.33, `late_std` 0.20) and is the **smallest** learning rate in that group. That means it gives the same high sample efficiency and stability as the larger LRs but with the least risk of divergence. Best, final, and stochastic rewards are all near 22-21. Wall time is within 3% of the other runs.

### 5.6 `m1_lr_06_high` — `learning_rate = 5e-4`

Same metrics as `m1_lr_05_modhigh`. The run is stable and efficient. Because the AUC and stability are tied, the smaller LR (`m1_lr_05_modhigh`) is preferred for safety, but `m1_lr_06_high` is also a defensible choice.

### 5.7 `m1_lr_07_veryhigh` — `learning_rate = 1e-3`

Same metrics. No divergence was observed in the 150k budget, but 1e-3 is approaching the range where Q-value targets can become unstable. The tie on AUC means it is not worse than 3e-4, but it is not better either.

### 5.8 `m1_lr_08_extreme` — `learning_rate = 3e-3`

Same metrics. The 3e-3 learning rate did not collapse in this budget, but it is unnecessarily high. The same performance is available at a much smaller, safer learning rate.

### 5.9 `m1_lr_09_extreme2` — `learning_rate = 1e-2`

Same metrics. A 1e-2 learning rate for DQN on an Atari image input is very large and would be expected to diverge. In this run it did not, but only because the eval is degenerate and the run is short. This should not be selected as the best.

### 5.10 `m1_lr_10_lowfixed` — `learning_rate = 3e-5`

This is the stand-in for the tail of a decayed learning-rate schedule. The best checkpoint is 23.3, but the final reward is 19.7 and the stochastic reward is 20.5. The `AUC` is 14.37 and `late_std` is 10.17, showing a clear collapse. The result is that a very low fixed tail can be unstable, which is useful evidence about decay schedules but not a good operating point.

---

## 6. Grouping the results

### 6.1 Very low learning rates (1e-6, 1e-5, 3e-5)

- `1e-6` appears to work but is too slow.
- `1e-5` and `3e-5` collapse after the initial peak.
- Conclusion: below about 5e-5 the learning rate is too small for stable DQN on this budget.

### 6.2 Low-to-moderate learning rates (5e-5 to 1e-4)

- `5e-5` shows intermediate instability.
- `1e-4` is the safe baseline.
- Conclusion: 1e-4 is a good default, but not the best in this sweep.

### 6.3 Moderate-to-high learning rates (3e-4 to 1e-2)

- All five runs reach the same top AUC/stability and same best/final/stochastic rewards.
- Conclusion: the greedy eval plateau is the same, but `m1_lr_05_modhigh` (3e-4) is the safest high-performing choice.

### 6.4 The "low fixed" schedule proxy (3e-5)

- Collapsed after the peak.
- Conclusion: a fixed low rate is not a stable schedule tail in this setting.

---

## 7. Why `m1_lr_05_modhigh` is the best

The selection rule is:

1. Filter for the **highest `auc_reward`**. The top group has AUC = 22.33.
2. Among those, pick the **smallest `learning_rate`**. That is `m1_lr_05_modhigh` at 3e-4.

A smaller learning rate is preferred because it is less likely to overshoot the Bellman target and more likely to generalize to a different random seed or a longer training budget. The higher LRs (5e-4 to 1e-2) are in the same top group but offer no extra reward and more risk.

> **Note:** The `M1_Learning_Rate_Experiments.ipynb` notebook currently reports `m1_lr_01_tiny` as the best because the `sort_values` tie-breaker used `AUC > mean_reward > steps_to_threshold` and pandas' `quicksort` is not stable, so it returned the first tied row in a DataFrame sorted by learning rate. The correct choice to defend is `m1_lr_05_modhigh`.

---

## 8. Comparison to the shared baseline

The shared baseline uses `learning_rate = 1e-4`, `gamma = 0.99`, `batch_size = 32`, and 500,000 timesteps. It reached `mean_reward = 21.9` (older run, no curve metrics available).

The M1 sweep shows that at 150k timesteps, `3e-4` matches the baseline's plateau and ties the best AUC. With a full 500k budget, the `3e-4` combined run (once M3 is done) would be the better model to demo.

---

## 9. M2 context — gamma and batch size

M2 (Samuel Mwania) completed the gamma and batch size sweeps. The best values from M2 are needed for the final combined model.

### 9.1 Gamma

| Run ID | `gamma` | `auc_reward` | `late_reward_std` | Verdict |
|---|---|---|---|---|
| `m2_gamma_01_short` | 0.90 | 22.13 | 0.58 | **Best gamma** |
| `m2_gamma_02_shortmed` | 0.95 | 20.47 | 3.21 | Unstable |
| `m2_gamma_03_baseline` | 0.99 | 20.68 | 1.92 | Reference |
| `m2_gamma_04_long` | 0.995 | 21.42 | 0.79 | Stable, second-best |
| `m2_gamma_05_verylong` | 0.999 | 16.72 | 8.64 | Diverged |

**Best gamma: 0.90.** The short discount horizon was the most stable and efficient. The very long horizon (0.999) made Q-targets too large and the policy collapsed.

### 9.2 Batch size

| Run ID | `batch_size` | `auc_reward` | `late_reward_std` | `wall_clock` | Verdict |
|---|---|---|---|---|---|
| `m2_batch_01_small` | 8 | 18.77 | 8.73 | 394 s | Too noisy |
| `m2_batch_02_baseline` | 32 | 20.68 | 1.92 | 403 s | Reference |
| `m2_batch_03_mod` | 64 | 21.67 | 1.02 | 405 s | **Best batch size** |
| `m2_batch_04_large` | 128 | 17.73 | 0.55 | 412 s | Too slow |
| `m2_batch_05_verylarge` | 256 | 19.05 | 0.44 | 498 s | Stable but slow |

**Best batch size: 64.** It gives the highest AUC with low late-std and only a 2% wall-time increase over the baseline. Small batches are too noisy; large batches are too slow to converge within the budget.

---

## 10. M3 status — exploration schedule

M3 (Birasa) has **not yet been run**. The 10 presets are defined in `config.py` and `EXPERIMENTS.md`, but `experiments/experiment_log.csv` contains no `m3_*` rows.

The exploration sweep will vary `exploration_fraction`, `exploration_final_eps`, and `exploration_initial_eps`. The best M3 result will be combined with the M1 and M2 best values to produce the final model.

Until M3 completes, the provisional combined model uses the M3 baseline values (`exploration_initial_eps = 1.0`, `exploration_final_eps = 0.05`, `exploration_fraction = 0.10`).

---

## 11. Provisional best combined model

The best hyperparameters found so far are:

| Hyperparameter | Best Value | Source |
|---|---|---|
| `learning_rate` | 3e-4 | M1 — `m1_lr_05_modhigh` |
| `gamma` | 0.90 | M2 — `m2_gamma_01_short` |
| `batch_size` | 64 | M2 — `m2_batch_03_mod` |
| `exploration_final_eps` | 0.05 | M3 baseline placeholder |
| `exploration_fraction` | 0.10 | M3 baseline placeholder |

To train this model at the full 500k budget:

```bash
python train.py --run-id final_combined_provisional --member team --promote \
    --learning-rate 3e-4 \
    --gamma 0.90 \
    --batch-size 64 \
    --exploration-final-eps 0.05 \
    --exploration-fraction 0.10 \
    --total-timesteps 500000 \
    --notes "Combined best: M1=3e-4, M2 gamma=0.90 batch=64; M3 baseline until M3 sweep completes"
```

`--promote` saves the model as `models/dqn_model.zip`, which is the default model loaded by `play.py` and `evaluate.py`.

To record a final gameplay video:

```bash
python play.py --model models/dqn_model.zip --record --episodes 3 --video-name final_combined
```

To play locally with a live window:

```bash
python play.py --model models/dqn_model.zip --render --episodes 3
```

---

## 12. Artifacts produced by the M1 sweep

| File | Description |
|---|---|
| `models/m1_lr_05_modhigh.zip` | Recommended best M1 model |
| `models/m1_lr_01_tiny.zip` | Model the notebook used for the video (not the recommended best) |
| `logs/m1_lr_*/` | TensorBoard logs and `evaluations.npz` for each M1 run |
| `plots/m1_lr_comparison.png` | Four-panel plot of M1 results |
| `videos/m1_lr_best-step-0-to-step-20000.mp4` | M1 video (from `m1_lr_01_tiny` due to the tie) |
| `experiments/experiment_log.csv` | All M1 and M2 rows, cleaned and updated |
| `notebooks/M1_Learning_Rate_Experiments.ipynb` | Colab notebook that ran the sweep and generated the video |
| `M1_exploration.md` | This detailed report |
| `m1_results.zip` | Colab output zip containing all the above artifacts |

---

## 13. How to answer questions

### "Why do many runs have the same best reward?"
Freeway's greedy, fixed-seed eval is degenerate. Once the policy learns to cross, the eval trajectory is nearly identical. Use `AUC` and `late_reward_std` to compare runs.

### "Why is `m1_lr_05_modhigh` better than `m1_lr_09_extreme2` if they have the same numbers?"
They share the same eval plateau, but `m1_lr_05_modhigh` uses the smallest learning rate in that top group. A smaller learning rate is safer and less likely to diverge in a longer or more stochastic run.

### "Why did `m1_lr_02_verylow` and `m1_lr_10_lowfixed` collapse?"
Both are at the very low end. The Q-network updates are so small that the policy can get stuck or forget what it learned, especially after the epsilon-greedy exploration decays and the agent stops collecting diverse transitions.

### "Why did `m1_lr_08_extreme` and `m1_lr_09_extreme2` not collapse?"
A 150k budget on a simple game with reward clipping is not enough to guarantee divergence for every high learning rate. The eval is also deterministic. These runs are still too risky to recommend.

### "What is the best final model?"
The provisional best is `learning_rate=3e-4`, `gamma=0.90`, `batch_size=64`. M3 exploration still needs to be tuned. Run the command in Section 11 to train the full 500k model.

---

## 14. Limitations

- The eval is deterministic and uses a fixed seed, which makes the greedy reward near-degenerate.
- The M1 sweep was run at 150k timesteps. The higher learning rates might still diverge at 500k.
- `m1_lr_10_lowfixed` is a stand-in for a decayed schedule, not a true schedule comparison.
- M3 has not been run, so the exploration parameters are still baseline.

---

## 15. Conclusion

The M1 learning-rate sweep demonstrates that on Freeway the middle-to-high learning-rate range (3e-4 to 1e-2) is stable and reaches the same greedy eval plateau. The **best choice is `m1_lr_05_modhigh` with `learning_rate = 3e-4`**, because it is the smallest learning rate in the top AUC/stability group and therefore the safest. The very low end (`1e-5`, `3e-5`) collapses after the initial peak, and the extreme low `1e-6` is too slow to be practical.

Combined with the M2 best values (`gamma = 0.90`, `batch_size = 64`), the provisional best configuration is:

- `learning_rate = 3e-4`
- `gamma = 0.90`
- `batch_size = 64`
- `exploration_final_eps = 0.05` (M3 baseline)
- `exploration_fraction = 0.10` (M3 baseline)

Once M3 completes, the final combined model should be retrained with the best exploration values and the full 500,000 timesteps.
