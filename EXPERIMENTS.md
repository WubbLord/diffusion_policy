# Experiments

## Experiments

### 1. Joint Delta vs Joint Velocity

Compare joint-space action targets for Robomimic lowdim policies.

- **Joint delta:** predict `robot*_joint_pos[t + 1] - robot*_joint_pos[t]` plus the original gripper command.
- **Joint velocity:** predict `obs/robot*_joint_vel[t]` plus the original gripper command.
- Keep dataset, model, seed, observation keys, training schedule, eval seeds, and rollout settings fixed.
- Compare rollout success, videos, validation loss, train action MSE, and per-slice joint / gripper MSE.

### 2. Joint Delta Action Execution Horizon Sweep

Measure how many open-loop joint-delta actions should be executed before replanning.

- Keep the prediction horizon fixed unless intentionally testing interactions.
- Sweep `n_action_steps`, for example `1`, `2`, `4`, `8`, and `12`.
- Keep observation horizon, dataset, seed, model size, and eval seeds fixed.
- Compare rollout success, time-to-success, action smoothness, joint-delta clipping rate, and failure modes in videos.

### 3. Joint Delta Observation Ablation

Test how much privileged kinematic information the joint-delta policy needs.

- **A: Full lowdim state**
  - `object`
  - `robot0_eef_pos`
  - `robot0_eef_quat`
  - `robot0_gripper_qpos`
  - `robot0_joint_pos`
- **B: Object + proprioception**
  - `object`
  - `robot0_gripper_qpos`
  - `robot0_joint_pos`
- **C: Object + EEF state**
  - `object`
  - `robot0_eef_pos`
  - `robot0_eef_quat`
  - `robot0_gripper_qpos`
- Keep action target, dataset, seed, training schedule, eval seeds, and rollout settings fixed.
- Compare whether explicit joint state, explicit EEF pose, or both are needed for good joint-delta control.

### 4. Observation Noise Robustness (CV-Estimate Simulation)

The robomimic lowdim datasets feed ground-truth object pose into the policy. A real
deployment instead consumes a noisy CV pose estimate. Use the new
`task.dataset.obs_noise_std` parameter (per-key Gaussian noise injected at
`__getitem__`, raw / pre-normalization scale) to characterize policy robustness.

- **Train-time deployment shift:** train clean (`obs_noise_std: null`), then evaluate
  on validation batches augmented with `{object: σ}` for σ ∈ {0, 0.001, 0.005, 0.01,
  0.05} m. Measures how much in-distribution position noise the policy tolerates.
- **Train-time augmentation:** retrain with `{object: 0.005}` injected during
  training, then re-run the same evaluation sweep. Tests whether observing noisy
  obs at training time recovers robustness — and whether it costs clean-eval performance.
- Optional: add quaternion-channel noise (`object` slice contains pos+quat+rel
  components for PickPlaceCan; verify the layout per task before claiming "5mm" since
  some dims are rotational).
- Keep action target, model, seed, and schedule fixed.
- Compare val_loss curves, rollout success on the EEF runner, and qualitative
  failure modes in videos.

### 5. Validation-loss Best-Checkpoint Selection

The current workspace selects top-k checkpoints by `test_mean_score` (rollout success).
Joint-action workspaces have `env_runner: null` (no rollout) — so they save only
"latest" checkpoints, even when val_loss has already started rising. Fix and verify:

- Add `monitor_key=val_loss, mode=min` as a config-time override path.
- Re-train one EEF run and one joint-delta run with the override.
- Compare final-checkpoint vs best-val-checkpoint on rollout success (EEF) and on
  held-out val MSE (joint).
- Expected: best-val checkpoint matches or beats final on both, with much larger
  margin on the small-data (`can/ph`) configs.

### 6. Regularization Sweep (Weight Decay)

The default workspace ships `weight_decay=1e-6` (effectively zero). With 200
PickPlaceCan PH demos the policy memorizes the train set within ~50 epochs (see
Results below). Sweep AdamW weight decay:

- Values: `1e-6` (current), `1e-4`, `1e-3`, `1e-2`.
- Hold everything else (LR, batch, EMA, dataset, seed) fixed.
- Train for a fixed step budget that includes the val_loss minimum (≥1000 epochs).
- Compare best-val val_loss and rollout success.

### 7. Multi-Human Dataset Comparison (ph vs mh)

`can/ph` is 200 single-operator demos; `can/mh` is ~300 multi-operator demos with
more behavioral diversity. Test whether the overfitting in Result A is primarily
small-sample-size or low-diversity:

- Configs: identical except `task.dataset_type=ph` vs `mh`.
- Same epoch / step budget.
- Compare best-val val_loss and rollout success on each variant's eval set, and
  cross-evaluate (`ph`-trained on `mh` val and vice versa) for distribution-shift
  signal.

### 8. EMA Power Sweep

The default `policy.ema_power=0.75` produces an effective averaging horizon of
only a few optimizer steps — barely smoothing. Common DDPM-imitation defaults
range 0.99 → 0.9999. Sweep `{0.75, 0.99, 0.999, 0.9999}`. Compare:

- Wall-clock per epoch (negligibly different).
- Best-val val_loss (after enabling Experiment 5's monitor_key fix).
- Rollout success.

### 9. Training Duration Sweep (Empirical Convergence Window)

Already partially run — see Result B. Keep as a baseline reference for Experiments
4–8 so that any future regularizer / EMA / data change is benchmarked against the
known overfitting curve at fixed configs.

---

## Results

> Hardware: CSAIL `csail-shared-h200` partition, 1× H200 per job, 4 CPUs, 32 GB RAM.
> Codebase: this branch + `obs_noise_std` param. WandB project: `diffusion_policy_debug`.

### A. EEF Lowdim Baseline — `can/ph`, 3000 epochs

- Slurm job `767953`. Wall time: 4 h 28 min on `aia-h200-10`.
- Workspace: `train_diffusion_unet_lowdim_can_eef_workspace`, task `can_lowdim`,
  `lr=1e-4` cosine, `weight_decay=1e-6`, `batch_size=256`, EMA power 0.75,
  horizon 16 / n_obs_steps 2 / n_action_steps 8.
- Final metrics:
  - `train_loss` ≈ 0.0 (model fully memorizes the 200-demo training set)
  - `train_action_mse_error` ≈ 0.0
  - `val_loss` minimum ≈ 0.05 around epoch ~50–100, then **monotonically rises to
    ≈ 0.40 by epoch 3000** — classic overfitting on a small dataset with no
    regularization.
- Implication: best generalization checkpoint is far from the final checkpoint.
  This motivates Experiments 5 (best-by-val-loss) and 6 (weight decay).

### B. EEF Lowdim Training Duration Sweep — 500 / 3000 / 6000 epochs

| Job ID | Epochs | Wall time | val_loss min (≈ epoch) | val_loss final | Notes |
|--------|--------|-----------|-------------------------|----------------|-------|
| 770190 | 500    | 55 min    | ~0.05 (~ epoch 50)     | ~0.20          | Already overfitting at finish; best checkpoint mid-run. |
| 767953 | 3000   | 4h 28m    | ~0.05 (~ epoch 50–100) | ~0.40          | Baseline (Result A). |
| 770191 | 6000   | 7h 58m    | ~0.05 (~ epoch 50–100) | ~0.40          | Same shape as 3000-ep run — extra epochs do not improve generalization, only deepen memorization. |

- All three runs have **the same val_loss minimum** at roughly the same epoch.
  More training time produces no benefit and a steadily worsening EMA model.
- The 6000-ep run shows a sharp anomaly in the `lr` panel near step ~30k — the
  cosine schedule resets back to warmup. Most likely cause: the job was preempted
  on `shared-if-available` QoS and the LR scheduler did not resume from saved
  state. Training continues but the LR curve has a discontinuity. Worth tracking
  as a separate cleanup if running long sweeps on preemptible QoS.

### C. Joint-Delta Lowdim Baseline — `can/ph`, 3000 epochs

- Slurm job `772524`. Wall time: 2 h 43 min on `aia-h200-X` (h200 shared).
- Workspace: `train_diffusion_unet_lowdim_joint_delta_workspace`, task
  `can_lowdim_joint_delta`, action_dim 8 (7 joint deltas + 1 gripper),
  `env_runner: null`.
- Joint actions are **derived** from `obs/robot0_joint_pos` deltas in the same
  hdf5 used for the EEF baseline (Result A) — the dataset class
  `RobomimicReplayJointDeltaLowdimDataset` computes `q[t+1] − q[t]` and
  concatenates the original gripper command.
- Final metrics shape mirrors Result A: train_loss → ~0, val_loss minimum
  early then drift upward. Without an env_runner, **val_loss is the only
  generalization signal**; this strengthens the case for Experiment 5.

### D. (placeholder) Observation Noise Robustness

> Not yet run. Will populate after the noise-aware run-set finishes.
> Expected columns: σ_object (m) | val_loss (clean-trained) | val_loss
> (noise-trained) | rollout success.

### E. (placeholder) Best-by-val-loss Checkpointing

> Not yet run. After Experiment 5 ships, compare final vs best-val checkpoints
> on the 3000-epoch baseline (rollout success on EEF; held-out val MSE on joint).

---

## Cross-cutting observations

1. **Dataset size, not epoch count, is the binding constraint.** All three EEF
   runs (Results A and B) hit the val_loss floor by epoch ~50 and degrade past
   that. The fix space is regularization (Exp. 6), augmentation (Exp. 4), or
   more data (Exp. 7) — not longer training.
2. **Joint runs need val_loss-based checkpoint selection.** With `env_runner:
   null`, the only signal of generalization is val_loss; the current top-k by
   `test_mean_score` selector silently does nothing on joint configs.
3. **Preemption breaks LR schedule continuity.** Long runs on
   `shared-if-available` QoS should either checkpoint scheduler state or run on
   a non-preemptible QoS to avoid the warmup-reset artifact in Result B.
