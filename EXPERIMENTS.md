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

## Results

