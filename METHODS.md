# Methods And Results Log

This file records concise methods and results for experiments that complete successfully.

## Entry Format

```text
## YYYY-MM-DD: Experiment Name

Status: completed

Methods:
- Dataset:
- Model / controller:
- Training or eval command:
- Key settings:
- Output directory:

Results:
- Primary metric:
- Secondary metrics:
- Notes:
```

## Experiment Entries

## 2026-05-09: Held-Out Adapter Oracle Replay Sweep

Status: completed

Methods:
- Goal: evaluate whether each trained reverse-controller adapter can replay held-out Robomimic demonstrations by converting desired joint transitions into executable `JOINT_POSITION` commands.
- Protocol: for each held-out timestep, compute `desired_delta = q_demo[t + 1] - q_current`, evaluate `u = f(current_lowdim_state, desired_delta)`, send `u` plus logged gripper command through the `JOINT_POSITION` controller, and continue the rollout from the resulting live simulator state.
- Adapters: held-out-demo MLP checkpoints under `data/reverse_controller/*_joint_position_s0.25_n32_heldout_demo/f_mlp_train*/best.pt`.
- Eval split: PH tasks used demos `150:200`; MH tasks used demos `250:300`; 50 demos per run.
- Slurm jobs: `816415` through `816422`, all completed with exit code `0`.
- Output directories: `data/reverse_controller/*_joint_position_s0.25_n32_heldout_demo/oracle_replay_current_state_f_demo*`.

Results:

```text
dataset       success   delta_MAE   mean_q_L2   final_q_L2   mean_EEF_L2   mean_object_L2
can_ph        43/50     0.033875    0.135458    0.254945     0.020710      0.310150
lift_ph       46/50     0.008774    0.029164    0.041462     0.009721      0.051757
lift_mh       43/50     0.011748    0.031646    0.037723     0.009613      0.080071
square_ph     29/50     0.036040    0.144787    0.043297     0.017874      0.353788
square_mh     30/50     0.013600    0.049966    0.036634     0.011382      0.322962
tool_hang_ph   2/50     0.032994    0.127530    0.066986     0.020648      1.140038
transport_ph  17/50     0.017372    0.099036    0.052341     0.023958      0.708106
transport_mh   0/50     0.169473    1.052930    1.880168     0.158153      1.085748
```

- Command saturation was common: per-step saturation rates were `0.65-0.82` for most single-arm tasks, `0.77` for `transport_ph`, and `0.36` for `transport_mh`.
- Interpretation: the adapter works well enough for simpler Can/Lift oracle replay, partially works for Square/Transport PH, and fails on Tool Hang and Transport MH. Long-horizon object/contact drift remains the main weakness.

## 2026-05-08: Joint-Delta DP + Held-Out-Demo Adapter Rollout Eval

Status: completed

Methods:
- Goal: evaluate the trained joint-delta Diffusion Policy when its desired joint-delta outputs are translated through the learned reverse-controller adapter before execution.
- DP checkpoint: `data/outputs/2026.05.03/12.20.41_train_diffusion_unet_lowdim_joint_delta_can_lowdim_joint_delta/checkpoints/latest.ckpt`.
- Adapter checkpoint: `data/reverse_controller/can_mh_joint_position_s0.25_n32_heldout_demo/f_mlp_train000_249_val250_299/best.pt`.
- Dataset/env: Robomimic Can MH lowdim, `data/robomimic/datasets/can/mh/low_dim.hdf5`.
- Execution stack:

```text
DP(obs) -> desired 7D joint delta + gripper
adapter f(full_state, desired_delta) -> physical JOINT_POSITION command
JOINT_POSITION controller -> robosuite env.step
```

- Policy observation keys: `object`, `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, `robot0_joint_pos`.
- Adapter observation keys: policy keys plus `robot0_joint_vel`.
- Eval settings: `n_train=6`, `n_test=50`, `max_steps=500`, `n_action_steps=8`, `n_envs=8`, `num_inference_steps=100`.
- Output directory: `data/outputs/2026.05.03/12.20.41_train_diffusion_unet_lowdim_joint_delta_can_lowdim_joint_delta/eval_latest_with_heldout_adapter`.
- Slurm job: `813171`, completed on `aia-h200-1` in 6:35.

Results:
- Test success: `50/50`, `test/mean_score = 1.0`.
- Train-initial-state success: `6/6`, `train/mean_score = 1.0`.
- Videos written: 6 total, 2 train and 4 test.
- Baseline comparison: the previous latest-checkpoint no-adapter joint eval, `eval_latest_epoch`, had `0/50` test successes and `0/6` train-initial-state successes.
- Interpretation: the latest joint-delta DP can solve Can when its desired joint transitions are passed through the learned adapter; the earlier failure is consistent with action-interface mismatch, not simply a useless joint-delta policy.

## 2026-05-08: Joint-Delta DP Offline Validation MAE

Status: completed

Methods:
- Goal: measure how accurately the trained joint-delta Diffusion Policy predicts desired joint transitions when conditioned on ground-truth teleoperated state.
- Checkpoint: `data/outputs/2026.05.03/12.20.41_train_diffusion_unet_lowdim_joint_delta_can_lowdim_joint_delta/checkpoints/latest.ckpt`.
- Checkpoint state: epoch 4950, global step 1,143,680.
- Model used: EMA policy.
- Dataset: Robomimic Can MH lowdim, `data/robomimic/datasets/can/mh/low_dim.hdf5`.
- Validation split: DP run's own `val_ratio=0.02`, `seed=42` split.
- Validation demos:

```text
demo_26, demo_129, demo_130, demo_194, demo_229, demo_257
```

- Evaluation protocol:
  - No simulator rollout.
  - For each logged timestep, condition policy on ground-truth lowdim observation history `[obs[t - 1], obs[t]]`, padded at `t=0`.
  - Predict the first action from the policy action window.
  - Compare predicted 7D arm joint delta against `robot0_joint_pos[t + 1] - robot0_joint_pos[t]`.
  - Use full 100-step DDPM inference with random seed 42.
- Output directory: `data/outputs/2026.05.03/12.20.41_train_diffusion_unet_lowdim_joint_delta_can_lowdim_joint_delta/offline_joint_delta_mae_dp_val_latest`.
- Slurm job: `810545`, completed on `aia-h200-9` in 1:29.

Results:
- Validation timesteps evaluated: 1,308.
- Overall arm joint-delta MAE: `0.006985` rad.
- Overall arm joint-delta RMSE: `0.010878` rad.
- Mean absolute target joint delta: `0.009394` rad.
- Mean absolute predicted joint delta: `0.008578` rad.
- Per-joint MAE:

```text
j0: 0.002845
j1: 0.008364
j2: 0.003159
j3: 0.009762
j4: 0.005557
j5: 0.008996
j6: 0.010213
```

- Per-demo joint MAE:

```text
demo_26:  0.007663
demo_129: 0.007142
demo_130: 0.007360
demo_194: 0.005087
demo_229: 0.008269
demo_257: 0.004885
```

- Interpretation: the policy predicts desired joint transitions with error on the same order as the target motion magnitude, even when conditioned on ground-truth teleoperated state. This is offline action-label accuracy only; it does not measure whether the predicted deltas are executable by `JOINT_POSITION`.

## 2026-05-06: Reverse Controller Synthetic Probe Data

Status: completed

Methods:
- Dataset: Robomimic Can MH lowdim, `data/robomimic/datasets/can/mh/low_dim.hdf5`.
- Demonstrations used: all 300 teleoperated demo rollouts, `demo_0` through `demo_299`.
- Controller: Robosuite `JOINT_POSITION` with per-joint physical command range `[-0.25, 0.25]` rad.
- Observation/state input:

```text
object
robot0_eef_pos
robot0_eef_quat
robot0_gripper_qpos
robot0_joint_pos
robot0_joint_vel
```

- For each demo timestep:
  - Reset sim to the logged demo simulator state.
  - Sample 32 candidate 7D joint-position commands `u`.
  - Convert physical command to normalized controller action with `u / 0.25`.
  - Use the logged gripper action from the Robomimic action vector.
  - Step the sim once.
  - Record:

```text
state         = lowdim state at demo timestep
command       = sampled physical JOINT_POSITION command, shape (7,)
desired_delta = actual joint delta produced by that command, shape (7,)
demo_delta    = logged q[t + 1] - q[t], shape (7,)
```

- Command sampling used anchor commands based on `demo_delta` plus random commands in the controller range.
- Collection used 28 parallel envs.
- Output directory: `data/reverse_controller/can_mh_joint_position_s0.25_n32/probes`.

Results:
- Generated 300 shard files, one per demo.
- Total demo timesteps probed: 62,756.
- Samples per timestep: 32.
- Total one-step joint-command trials: 2,008,192.
- State dimension: 37.
- Command / joint-delta dimension: 7.
- Data generation completed successfully: `data/reverse_controller/can_mh_joint_position_s0.25_n32/probes/DONE.json`.
