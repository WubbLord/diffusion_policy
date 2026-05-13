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

## 2026-05-12: FK Joint-Position Full-Pipeline Eval (Partial)

Status: completed for Can/Lift/Square PH/MH; Tool Hang PH and Transport PH/MH still pending at logging time

Methods:
- Goal: evaluate the deterministic FK joint-position adapter in the full deployment stack.
- Execution stack:

```text
joint-delta DP policy -> desired joint delta + gripper
FK joint-position adapter -> JOINT_POSITION command
Robosuite JOINT_POSITION controller -> rollout reward / success
```

- Eval protocol: `n_test=50`, latest corresponding joint-delta DP checkpoint, same rollout settings as the learned-adapter full-pipeline evals.
- Completed tasks: Can PH/MH, Lift PH/MH, and Square PH/MH.

Results:

| Task | Test Success | Train-Init Success | Status |
| --- | ---: | ---: | --- |
| `can_ph` | `0/50` | `0/6` | complete |
| `can_mh` | `0/50` | `0/6` | complete |
| `lift_ph` | `9/50` | `2/6` | complete |
| `lift_mh` | `19/50` | `3/6` | complete |
| `square_ph` | `0/50` | `0/6` | complete |
| `square_mh` | `0/50` | `0/6` | complete |
| `tool_hang_ph` | pending | pending | queued |
| `transport_ph` | pending | pending | queued |
| `transport_mh` | pending | pending | queued |

- Interpretation: FK is not a learned inverse of the joint-position controller's closed-loop response. In full rollouts it compounds policy prediction error, controller tracking error, contact timing error, and state drift, so local/oracle tracking quality does not translate to robust task success.

## 2026-05-12: FK Joint-Position Adapter-Only Tracking Eval

Status: completed

Methods:
- Goal: compare the learned inverse-controller adapter against the analytic forward-kinematics (FK) joint-position adapter on held-out demonstration tracking.
- Protocol: for each held-out demo timestep, compute the desired residual joint transition from the live replay state to the next demo joint state, execute one `JOINT_POSITION` controller step, and measure `|actual_delta - desired_delta|`.
- Split: PH tasks used demos `150:200`; MH tasks used demos `250:300`; 50 demos per task.
- Controller settings: per-joint command scale `0.25`, default Robosuite `JOINT_POSITION` gains and damping.
- Slurm jobs: FK adapter-only jobs `830276`, `830278`, `830280`, `830282`, `830284`, `830286`, `830288`, `830290`, and `830292`; all completed with exit code `0`.
- Output directories: `data/outputs/**/oracle_replay_fk_jp_learnedcfg_scale0p25_defaultkp_demo*`.

Results:

| Dataset | Learned Delta MAE | Learned Rel. MAE | FK Delta MAE | FK Rel. MAE | FK Saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| `can_ph` | `0.033875` | `0.762` | `0.072410` | `0.869` | `0.065` |
| `can_mh` | `0.019667` | `0.767` | `0.046144` | `0.862` | `0.033` |
| `lift_ph` | `0.008774` | `0.570` | `0.039751` | `0.850` | `0.000` |
| `lift_mh` | `0.011748` | `0.744` | `0.029289` | `0.859` | `0.005` |
| `square_ph` | `0.036040` | `0.808` | `0.072662` | `0.887` | `0.075` |
| `square_mh` | `0.013600` | `0.752` | `0.035932` | `0.866` | `0.015` |
| `tool_hang_ph` | `0.032994` | `0.844` | `0.049775` | `0.877` | `0.046` |
| `transport_ph` | `0.017372` | `0.748` | `0.041173` | `0.857` | `0.018` |
| `transport_mh` | `0.169473` | `0.980` | `0.034956` | `0.862` | `0.015` |

- Relative MAE is `mean(abs(actual_delta - desired_delta)) / mean(abs(desired_delta))`.
- Interpretation: the learned adapter tracks better on most single-arm tasks, especially Lift and Can. The FK adapter is more consistent and substantially better on Transport MH, where the learned adapter's local tracking error is very large. Both adapters still have high relative error, so neither is a perfect one-step inverse for the joint-position controller.

## 2026-05-12: Sourish OSC Adapter Study From `origin/sour/obs-noise-param`

Status: completed for summarized runs; some branch jobs were still running or queued in the source writeup

Methods:
- Source read: `origin/sour/obs-noise-param:writeup.md` after `git fetch origin`.
- Goal: compare ways to execute joint-delta Diffusion Policy checkpoints through the standard Robomimic `OSC_POSE` controller rather than the `JOINT_POSITION` controller used by the main adapter experiments.
- Policies: joint-delta lowdim Diffusion Policy checkpoints trained for PH variants of Lift, Can, Square, Tool Hang, and Transport.
- Deterministic adapter: `robomimic_joint_fk_to_eef_runner.py`.
  - Integrate predicted `Δq` onto the current joint state.
  - Run Franka Panda forward kinematics in a standalone MuJoCo model.
  - Convert the target end-effector pose into normalized `OSC_POSE` action space.
  - Execute through the native Robomimic OSC controller.
- Learned OSC adapters:
  - Probe-based NN-to-OSC: Brian-style `InverseControllerMLP`, input `state ⊕ desired_Δq`, target OSC command, synthetic probes collected by stepping random/anchored OSC commands and measuring actual `Δq`.
  - Demo-supervised NN-to-OSC: one pair per demo timestep, `(state_t, q[t+1]-q[t]) -> a_OSC[t]`, where `a_OSC[t]` is the recorded teleoperation command. No simulator probing.
- Eval protocol in the writeup: `n_test=50`, `test_start_seed=100000`, `n_envs=28`; `max_steps=400` for Lift/Can/Square and `700` for Tool Hang/Transport.

Results:

DP training status in Sourish's branch:

| Task | Job ID | Status | Wall Time |
| --- | ---: | --- | --- |
| `lift_ph` | `818751` | completed, 5000 epochs | `6h23m` |
| `can_ph` | `818753` | completed, 5000 epochs | `9h59m` |
| `square_ph` | `818755` | completed, 5000 epochs | `10h34m` |
| `tool_hang_ph` | `818757` | running at writeup time, about 4100/5000 epochs | `17h+` |
| `transport_ph` | `821174` | resumed/running at writeup time, about 3900/5000 epochs | `19h+` |

FK-to-OSC full-pipeline evals:

| Task | OSC `kp` | Test Success | Notes |
| --- | ---: | ---: | --- |
| `lift_ph` | `1000` | `0.94` | strong deterministic adapter result |
| `can_ph` | `1000` | `0.88` | strong deterministic adapter result |
| `square_ph` | `1000` | `0.34` | lower due to controller tracking / task precision |
| `square_ph` | `3000` | `0.50` | higher OSC gain recovered `+0.16` success |
| `tool_hang_ph` | TBD | TBD | DP still training in source writeup |
| `transport_ph` | TBD | TBD | DP/action-layout calibration still WIP in source writeup |

Probe-based NN-to-OSC adapter results:

| Task | Quick Probe Full Pipeline | Brian-Quality Probe Full Pipeline | Quick Probe Oracle Replay |
| --- | ---: | ---: | ---: |
| `lift_ph` | `0.00` | `0.02` | `0.48` |
| `can_ph` | `0.02` | queued | `0.24` |
| `square_ph` | `0.00` | queued | `0.00` |
| `tool_hang_ph` | `0.00` | queued | queued |
| `transport_ph` | queued | queued | queued |

Demo-supervised NN-to-OSC full-pipeline evals:

| Task | Probe NN-to-OSC | Demo-Supervised NN-to-OSC | FK-to-OSC Reference |
| --- | ---: | ---: | ---: |
| `lift_ph` | `0.00` quick / `0.02` Brian-quality | `0.90` | `0.94` |
| `can_ph` | `0.02` quick | `0.64` | `0.88` |
| `square_ph` | `0.00` quick | `0.48` | `0.50` at `kp=3000` |

Action execution horizon / replanning sweep under FK-to-OSC:

| Task | `n_action_steps=8` | `n_action_steps=1` | Notes |
| --- | ---: | ---: | --- |
| `lift_ph` | `0.94` | `0.94` | saturated; replanning more often did not change success |
| `can_ph` | `0.88` | `0.88` | saturated; replanning more often did not change success |
| `square_ph` | `0.34` at `kp=1000` | running in job `827030` | sweep still incomplete in source writeup |

Interpretation:
- For the native Robomimic OSC action interface, deterministic FK-to-OSC is the cleanest adapter: it converts a predicted joint target into the controller's own end-effector target rather than trying to learn a global inverse from arbitrary desired joint deltas to OSC commands.
- Probe-based NN-to-OSC fails even with larger Brian-quality sampling. The writeup argues this is structural: `OSC_POSE -> Δq` is many-to-one and state/Jacobian/nullspace dependent, so the inverse is branch-ambiguous off the demonstration manifold.
- Demo-supervised NN-to-OSC works much better because it learns the teleoperator's chosen OSC command branch on the demo manifold. It nearly matches FK-to-OSC on Lift and Square, but remains below FK-to-OSC on Can.
- OSC gain matters: increasing `kp` from `1000` to `3000` improved Square FK-to-OSC from `0.34` to `0.50`.
- A closed-loop FK variant on Square at `kp=1000` reportedly stayed at `0.34`, suggesting stale FK targets inside the action chunk were not the dominant bottleneck for that setting.

## 2026-05-12: EEF/OSC Probe-Adapter Training And Evals

Status: completed for Can/Lift/Square PH/MH; Tool Hang PH adapter training completed after eval submission; Transport PH/MH probe collection still running

Methods:
- Goal: train a probe-based inverse adapter for the native Robomimic `OSC_POSE` controller, analogous to the `JOINT_POSITION` reverse-controller adapters.
- Synthetic data generation:
  - Reset Robomimic lowdim simulation to demonstration states.
  - Sample `32` normalized `OSC_POSE` commands per timestep, mixing anchored commands around the recorded demo OSC action with random uniform/noisy probes.
  - Step the OSC controller once and record the actual joint transition.
  - Train inverse pairs `(full lowdim state, desired_Δq_actual) -> normalized OSC command`.
- Adapter: `InverseControllerMLP`, input `full lowdim state ⊕ desired_Δq`, output normalized `OSC_POSE` command in `[-1, 1]`.
- Full lowdim state for single-arm tasks: `object`, `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, `robot0_joint_pos`, `robot0_joint_vel`.
- Held-out-demo split: PH demos `0:150` train and `150:200` eval; MH demos `0:250` train and `250:300` eval.
- Training: `100` epochs, batch size `8192`.
- Adapter-only eval: reset to held-out demo states, compute `Δq_demo = q[t+1] - q[t]`, predict OSC command with the adapter, step the OSC controller once, and compare actual vs desired joint delta.
- Full pipeline eval:

```text
latest joint-delta DP checkpoint -> desired joint delta + gripper
EEF/OSC probe adapter f(state, desired joint delta) -> OSC_POSE command
Robomimic OSC_POSE controller -> rollout success
```

- Full pipeline protocol: `n_test=50`, `n_train=6`, latest corresponding joint-delta DP checkpoint.
- Scripts: `reverse_controller/collect_inverse_dataset_osc.py`, `scripts/eval_osc_adapter_one_step.py`, `scripts/eval_joint_delta_with_osc_adapter.py`.
- Output roots:
  - Adapters: `data/reverse_controller_osc_probe/*_osc_pose_n32_heldout_demo`.
  - Full pipeline evals: `data/outputs/**/eval_latest_full50_with_eef_probe_adapter`.

Results:

| Task | Adapter Status | One-Step Delta MAE | One-Step RMSE | Full Pipeline Test | Full Pipeline Train |
| --- | --- | ---: | ---: | ---: | ---: |
| `can_ph` | complete | `0.001141` | `0.002222` | `0/50` | `0/6` |
| `can_mh` | complete | `0.000894` | `0.001607` | `0/50` | `0/6` |
| `lift_ph` | complete | `0.001114` | `0.001946` | `0/50` | `0/6` |
| `lift_mh` | complete | `0.001022` | `0.002107` | `0/50` | `0/6` |
| `square_ph` | complete | `0.001218` | `0.001944` | `0/50` | `0/6` |
| `square_mh` | complete | `0.000987` | `0.001724` | `0/50` | `0/6` |
| `tool_hang_ph` | training complete; eval not submitted in this batch | pending | pending | pending | pending |
| `transport_ph` | probe collection running | pending | pending | pending | pending |
| `transport_mh` | probe collection running | pending | pending | pending | pending |

- Interpretation: the probe-trained OSC adapters locally track held-out demo joint deltas with roughly `1e-3` rad MAE on single-arm tasks, but the full joint-delta DP plus probe-OSC-adapter rollout still gets `0/50` on all completed tasks.
- This matches the Sourish branch observation: probe-based NN-to-OSC can fit one-step local data, but `OSC_POSE -> Δq` is branch-ambiguous and policy-predicted deltas are not guaranteed to lie on the same command branch as the probe labels.

## 2026-05-11: Full DP+Adapter Rollout Evals Across Robomimic

Status: completed for listed runs

Methods:
- Goal: evaluate the full deployment stack where a joint-delta Diffusion Policy predicts desired joint transitions, the held-out-demo adapter maps them to executable `JOINT_POSITION` commands, and Robosuite computes task success.
- Execution stack:

```text
DP lowdim policy -> desired joint delta + gripper
adapter f(current full lowdim state, desired joint delta) -> JOINT_POSITION command
Robosuite JOINT_POSITION controller -> rollout reward / success
```

- Policy horizon settings: `horizon=16`, `n_obs_steps=2`, `n_action_steps=8`, `num_inference_steps=100`.
- Eval protocol: `n_test=50` seeded test rollouts, `n_train=6` train-initial-state sanity rollouts, `n_test_vis=4`, `n_train_vis=2`.
- Test seeds: `100000` through `100049`.
- Output roots:
  - Can MH: `data/outputs/2026.05.03/12.20.41_train_diffusion_unet_lowdim_joint_delta_can_lowdim_joint_delta/eval_latest_full50_with_heldout_adapter`.
  - Sweep tasks: `data/outputs/robomimic_joint_delta_sweep/*/eval_latest_full50_with_heldout_adapter`.
- Final missing-task resubmits: `828492` Square MH, `828493` Tool Hang PH, `828494` Transport MH, all with `n_envs=4` and explicit progress logging.

Results:

| Task | Test Success | Train-Init Success | Max Steps | Status |
| --- | ---: | ---: | ---: | --- |
| `can_ph` | `45/50` | `6/6` | `400` | complete |
| `can_mh` | `50/50` | `6/6` | `500` | complete |
| `lift_ph` | `50/50` | `6/6` | `400` | complete |
| `lift_mh` | `49/50` | `6/6` | `500` | complete |
| `square_ph` | `27/50` | `4/6` | `400` | complete |
| `square_mh` | `23/50` | `5/6` | `500` | complete |
| `tool_hang_ph` | `0/50` | `0/6` | `400` | complete |
| `transport_ph` | `0/50` | `0/6` | `400` | complete |
| `transport_mh` | `5/50` | `1/6` | `500` | complete |

- Interpretation: the full adapter stack works strongly on Can and Lift, partially on Square, fails on Tool Hang and Transport PH, and gets weak but nonzero Transport MH success despite the working two-arm action interface.

## 2026-05-11: Transport Two-Arm Full-Pipeline Eval Smoke

Status: completed

Methods:
- Goal: verify that the full DP+adapter eval path supports Robomimic Transport, which has two robot arms.
- Code path changed: `RobomimicJointAdapterLowdimWrapper` now supports multi-robot action parsing and formats controller actions as:

```text
DP action:         [robot0_dq(7), robot1_dq(7), robot0_gripper, robot1_gripper]
adapter input:    desired_delta_14d
adapter output:   command_14d
robosuite action: [robot0_cmd(7), robot0_gripper, robot1_cmd(7), robot1_gripper]
```

- Smoke command: Transport PH latest joint-delta DP checkpoint plus held-out-demo Transport PH adapter.
- Smoke settings: `n_test=1`, `n_train=0`, `n_envs=1`, `max_steps=8`.
- Slurm job: `818997`, completed in `27s`.
- Output directory: `data/outputs/robomimic_joint_delta_sweep/transport_ph_seed42_offline_816239/eval_transport_adapter_smoke_patch`.

Results:
- Smoke eval completed and wrote `eval_log.json`.
- Confirmed env runner metadata: `n_robots=2`, `joint_dims=[7, 7]`, `gripper_dims=[1, 1]`.
- Smoke success was `0/1`; this run was only an action-interface and runner-shape check, not a meaningful task-performance eval.

## 2026-05-11: Full Pipeline (DP+Adapter) Closed-Loop K Sweep

Status: completed

Methods:
- Goal: evaluate the learned joint-delta DP plus learned adapter when each predicted joint transition is executed as an inner-loop joint-space target.
- Execution stack:

```text
DP(obs) -> desired joint delta + gripper
q_target = q_current + desired joint delta
for k inner steps:
    adapter f(current full lowdim state, q_target - q_current) -> JOINT_POSITION command
    Robosuite JOINT_POSITION controller -> env.step(command)
next policy observation comes from the live simulator state
```

- Sweep: `k = 1..8`, one Slurm array task per `k`.
- Eval protocol: `n_test=50`, `n_train=6`, test seeds `100000` through `100049`, same latest DP checkpoints and held-out-demo adapters as the one-step full-pipeline eval.
- Output roots: `data/outputs/**/eval_closed_loop_adapter_k1_8`.
- Original Slurm arrays: `819020` through `819027`.
- Resubmitted short/preempted indices: `819340` through `819343`, `821132` through `821136`, then `822295` through `822298`.
- Final reliability resubmits with `n_envs=4` and explicit progress logging: `828530` Square PH k3, `828531` Square MH k3, `828532` Tool Hang PH k1-k3, `828533` Transport PH k3/k7, and `828692` Transport MH k1-k8.
- Transport MH k1/k2 were preempted before completion as `828692_1` and `828692_2`; restarted as `829071_[1-2%2]` with the same output root and `n_envs=4`, then completed.

Results:

| Task | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | Best Complete k | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `can_ph` | `45/50` | `31/50` | `2/50` | `8/50` | `4/50` | `2/50` | `1/50` | `1/50` | `k=1` | complete |
| `can_mh` | `50/50` | `40/50` | `12/50` | `9/50` | `11/50` | `6/50` | `5/50` | `10/50` | `k=1` | complete |
| `lift_ph` | `50/50` | `47/50` | `31/50` | `32/50` | `6/50` | `5/50` | `9/50` | `14/50` | `k=1` | complete |
| `lift_mh` | `49/50` | `50/50` | `46/50` | `49/50` | `44/50` | `41/50` | `42/50` | `38/50` | `k=2` | complete |
| `square_ph` | `24/50` | `16/50` | `1/50` | `0/50` | `2/50` | `2/50` | `0/50` | `3/50` | `k=1` | complete |
| `square_mh` | `31/50` | `26/50` | `10/50` | `9/50` | `4/50` | `3/50` | `0/50` | `0/50` | `k=1` | complete |
| `tool_hang_ph` | `0/50` | `0/50` | `0/50` | `0/50` | `0/50` | `0/50` | `0/50` | `0/50` | all k tie | complete |
| `transport_ph` | `0/50` | `2/50` | `0/50` | `0/50` | `0/50` | `0/50` | `0/50` | `0/50` | `k=2` | complete |
| `transport_mh` | `6/50` | `9/50` | `0/50` | `1/50` | `0/50` | `0/50` | `0/50` | `0/50` | `k=2` | complete |

- Interpretation: for Can/Lift, more inner-loop steps usually reduce task success even when the adapter can track residuals. The best full-policy rollout result is generally `k=1`, with Lift-MH as the main exception where `k=2` reaches `50/50`. Transport-MH also improves from `6/50` at `k=1` to `9/50` at `k=2`, but remains weak overall.

## 2026-05-11: EEF Baseline And No-Adapter Joint-Delta Rollout Evals

Status: completed

Methods:
- Goal: compare the original EEF action-interface eval against joint-delta policies executed directly through `JOINT_POSITION` without the learned adapter.
- Original EEF checkpoint: `data/outputs/2026.05.03/11.28.30_train_diffusion_unet_lowdim_can_lowdim/checkpoints/latest.ckpt`.
- Joint-delta checkpoint family: `data/outputs/2026.05.03/12.20.41_train_diffusion_unet_lowdim_joint_delta_can_lowdim_joint_delta/checkpoints/*`.
- Eval protocol: Can MH lowdim, `n_test=50`, `n_train=6`, videos for 4 test and 2 train rollouts.

Results:

| Eval | Checkpoint | Test Success | Train-Init Success | Notes |
| --- | --- | ---: | ---: | --- |
| Original EEF DP | latest | `49/50` | `6/6` | Native OSC action interface |
| Direct joint-delta, no adapter | epoch `0050` | `0/50` | `0/6` | Raw joint deltas sent to `JOINT_POSITION` |
| Direct joint-delta, no adapter | epoch `0100` | `0/50` | `0/6` | Raw joint deltas sent to `JOINT_POSITION` |
| Direct joint-delta, no adapter | epoch `0150` | `0/50` | `0/6` | Raw joint deltas sent to `JOINT_POSITION` |
| Direct joint-delta, no adapter | epoch `0200` | `0/50` | `0/6` | Raw joint deltas sent to `JOINT_POSITION` |
| Direct joint-delta, no adapter | epoch `0250` | `0/50` | `0/6` | Raw joint deltas sent to `JOINT_POSITION` |
| Direct joint-delta, no adapter | latest | `0/50` | `0/6` | Raw joint deltas sent to `JOINT_POSITION` |

- Interpretation: the EEF policy succeeds under its native OSC action interface, while the learned joint deltas are not directly executable as `JOINT_POSITION` commands. This is the core action-interface mismatch the adapter addresses.

## 2026-05-09: Held-Out-Demo Adapter Training Across Robomimic

Status: completed

Methods:
- Goal: train reverse-controller adapters `f(state, desired_delta) -> JOINT_POSITION command` for all available Robomimic lowdim tasks and dataset types.
- Model: MLP with hidden dims `512,512,512`, SiLU activations, layer norm, trained for `100` epochs.
- Synthetic data: `32` sampled joint-position command probes per demo timestep, physical command scale `[-0.25, 0.25]` rad per joint.
- Held-out demo split:
  - PH tasks: train demos `0:150`, validation demos `150:200`.
  - MH tasks: train demos `0:250`, validation demos `250:300`.
- Output roots: `data/reverse_controller/*_joint_position_s0.25_n32_heldout_demo/f_mlp_train*/`.

Results:

| Task | Train Demos | Val Demos | Best Epoch | Best Val Loss | Best Val Command MAE |
| --- | --- | --- | ---: | ---: | ---: |
| `can_mh` | `0:250` | `250:300` | `22` | `0.009629` | `0.004868` |
| `can_ph` | `0:150` | `150:200` | `91` | `0.010516` | `0.005638` |
| `lift_ph` | `0:150` | `150:200` | `98` | `0.011333` | `0.006093` |
| `lift_mh` | `0:250` | `250:300` | `8` | `0.030220` | `0.007666` |
| `square_ph` | `0:150` | `150:200` | `47` | `0.035087` | `0.010626` |
| `square_mh` | `0:250` | `250:300` | `7` | `0.029318` | `0.009184` |
| `tool_hang_ph` | `0:150` | `150:200` | `87` | `0.013958` | `0.005779` |
| `transport_ph` | `0:150` | `150:200` | `28` | `0.020212` | `0.007449` |
| `transport_mh` | `0:250` | `250:300` | `14` | `0.023476` | `0.007266` |

- Each adapter directory contains `best.pt`, `latest.pt`, `history.json`, and `loss.png`.

## 2026-05-09 to 2026-05-11: Joint-Delta DP Training Sweep

Status: completed

Methods:
- Goal: train joint-delta Diffusion Policy checkpoints for Robomimic lowdim tasks using the same lowdim UNet pipeline and the joint-delta dataset configs.
- Config: `train_diffusion_unet_lowdim_joint_delta_workspace`.
- Main settings: `horizon=16`, `n_obs_steps=2`, `n_action_steps=8`, `num_epochs=5000`, `training.seed=42`, offline W&B logging.
- Output roots: `data/outputs/robomimic_joint_delta_sweep/*_seed42_offline_*`.

Results:

| Task | Epoch | Val Loss | Status | Output |
| --- | ---: | ---: | --- | --- |
| `can_mh` | `4999` | `0.199604` | complete | `data/outputs/2026.05.03/12.20.41_train_diffusion_unet_lowdim_joint_delta_can_lowdim_joint_delta` |
| `can_ph` | `4999` | `0.186790` | complete | `data/outputs/robomimic_joint_delta_sweep/can_ph_seed42_offline_816233` |
| `lift_ph` | `4999` | `0.288713` | complete | `data/outputs/robomimic_joint_delta_sweep/lift_ph_seed42_offline_816234` |
| `lift_mh` | `4999` | `0.104599` | complete | `data/outputs/robomimic_joint_delta_sweep/lift_mh_seed42_offline_816235` |
| `square_ph` | `4999` | `0.127605` | complete | `data/outputs/robomimic_joint_delta_sweep/square_ph_seed42_offline_816236` |
| `square_mh` | `4999` | `0.141493` | complete | `data/outputs/robomimic_joint_delta_sweep/square_mh_seed42_offline_816237` |
| `tool_hang_ph` | `4999` | `0.146761` | complete | `data/outputs/robomimic_joint_delta_sweep/tool_hang_ph_seed42_offline_816238` |
| `transport_ph` | `4999` | `0.139775` | complete | `data/outputs/robomimic_joint_delta_sweep/transport_ph_seed42_offline_816239` |
| `transport_mh` | `4999` | `0.110328` | complete | `data/outputs/robomimic_joint_delta_sweep/transport_mh_seed42_offline_816240` |

## 2026-05-11: Joint-Delta DP Training Resumes

Status: completed

Methods:
- Goal: resume interrupted/offline Robomimic joint-delta DP trainings to the planned final epoch.
- Training stack: `train_diffusion_unet_lowdim_joint_delta_workspace` with task-specific `*_lowdim_joint_delta` configs, offline W&B logging, latest-checkpoint resume into the same Hydra output directory.
- Completed resumed jobs:
  - `square_mh`: Slurm `818332`, output `data/outputs/robomimic_joint_delta_sweep/square_mh_seed42_offline_816237`.
  - `tool_hang_ph`: Slurm `818784`, output `data/outputs/robomimic_joint_delta_sweep/tool_hang_ph_seed42_offline_816238`.
  - `transport_ph`: Slurm `818334`, output `data/outputs/robomimic_joint_delta_sweep/transport_ph_seed42_offline_816239`.
  - `transport_mh`: Slurm `822299`, output `data/outputs/robomimic_joint_delta_sweep/transport_mh_seed42_offline_816240`.

Results:

| Task | Slurm Job | Epoch | Final Train Loss | Val Loss | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `square_mh` | `818332` | `4999` | `8.4559e-05` | `0.141493` | complete |
| `tool_hang_ph` | `818784` | `4999` | `7.3580e-05` | `0.146761` | complete |
| `transport_ph` | `818334` | `4999` | `1.2632e-04` | `0.139775` | complete |
| `transport_mh` | `822299` | `4999` | `1.5764e-04` | `0.110328` | complete |

- `latest.ckpt` exists in each completed run's `checkpoints/` directory.
- `transport_mh` Slurm `818335` was preempted at epoch `4879`; Slurm `822299` resumed from `latest.ckpt` into the same output directory and completed to epoch `4999`.

## 2026-05-11: Closed-Loop Adapter Oracle Replay Sweep

Status: completed

Methods:
- Goal: evaluate whether the held-out-demo reverse-controller adapter improves oracle replay when used as an inner-loop joint-space servo.
- Protocol: for each held-out demo timestep, set `q_target = q_current + Δq_demo[t]`; for `k` inner controller steps, compute `residual = q_target - q_current`, evaluate `u = f(current_lowdim_state, residual)`, and step the `JOINT_POSITION` controller.
- Sweep: `k = 1..8`, one Slurm array task per `k`.
- Held-out splits: PH demos `150:200`; MH demos `250:300`; 50 demos per completed task.
- Output roots: `data/reverse_controller/*_joint_position_s0.25_n32_heldout_demo/oracle_replay_closed_loop_sweep_demo*`.
- Completed Slurm arrays in this pass: `818431` through `818437`, with retries `818786` and `818787` for the preempted Square entries.

Results:

| Task | Best Success | Best Success k | Best Delta MAE | Best Delta k | Status |
| --- | ---: | --- | ---: | --- | --- |
| `can_mh` | `36/50` | `k=1` | `0.010816` | `k=2` | complete |
| `can_ph` | `43/50` | `k=1` | `0.010770` | `k=8` | complete |
| `lift_ph` | `45/50` | `k=1` | `0.008754` | `k=1` | complete |
| `lift_mh` | `45/50` | `k=1,2,3` | `0.009657` | `k=7` | complete |
| `square_ph` | `40/50` | `k=2,3` | `0.010343` | `k=2` | complete |
| `square_mh` | `30/50` | `k=1` | `0.009687` | `k=8` | complete |
| `tool_hang_ph` | `2/50` | `k=1,2` | `0.010847` | `k=2` | complete |
| `transport_ph` | `23/50` | `k=1` | `0.009599` | `k=7` | complete |
| `transport_mh` | `6/50` | `k=2` | `0.058283` | `k=7` | complete |

- Interpretation: extra inner-loop steps usually reduce one-step delta tracking error, but they do not reliably improve task success. The best task success is still commonly at `k=1`, suggesting timing/contact drift can outweigh improved joint tracking.
- `transport_mh`: completed `k=1..8`; best success `6/50` at `k=2`; best delta MAE `0.058283` at `k=7`; `k=8` had success `0/50` and delta MAE `0.063957`.

## 2026-05-09: Held-Out Adapter Oracle Replay Sweep

Status: completed

Methods:
- Goal: evaluate whether each trained reverse-controller adapter can replay held-out Robomimic demonstrations by converting desired joint transitions into executable `JOINT_POSITION` commands.
- Protocol: for each held-out timestep, compute `desired_delta = q_demo[t + 1] - q_current`, evaluate `u = f(current_lowdim_state, desired_delta)`, send `u` plus logged gripper command through the `JOINT_POSITION` controller, and continue the rollout from the resulting live simulator state.
- Adapters: held-out-demo MLP checkpoints under `data/reverse_controller/*_joint_position_s0.25_n32_heldout_demo/f_mlp_train*/best.pt`.
- Eval split: PH tasks used demos `150:200`; MH tasks used demos `250:300`; 50 demos per run.
- Slurm jobs: `816415` through `816422` plus the Can-MH held-out replay run, all completed with exit code `0`.
- Output directories: `data/reverse_controller/*_joint_position_s0.25_n32_heldout_demo/oracle_replay_current_state_f_demo*`.

Results:

| Dataset | Success | Delta MAE | Mean q L2 | Final q L2 | Mean EEF L2 | Mean Object L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `can_mh` | `38/50` | `0.019667` | `0.074342` | `0.058373` | `0.011515` | `0.326052` |
| `can_ph` | `43/50` | `0.033875` | `0.135458` | `0.254945` | `0.020710` | `0.310150` |
| `lift_ph` | `46/50` | `0.008774` | `0.029164` | `0.041462` | `0.009721` | `0.051757` |
| `lift_mh` | `43/50` | `0.011748` | `0.031646` | `0.037723` | `0.009613` | `0.080071` |
| `square_ph` | `29/50` | `0.036040` | `0.144787` | `0.043297` | `0.017874` | `0.353788` |
| `square_mh` | `30/50` | `0.013600` | `0.049966` | `0.036634` | `0.011382` | `0.322962` |
| `tool_hang_ph` | `2/50` | `0.032994` | `0.127530` | `0.066986` | `0.020648` | `1.140038` |
| `transport_ph` | `17/50` | `0.017372` | `0.099036` | `0.052341` | `0.023958` | `0.708106` |
| `transport_mh` | `0/50` | `0.169473` | `1.052930` | `1.880168` | `0.158153` | `1.085748` |

- Command saturation was common: per-step saturation rates were `0.65-0.82` for most single-arm tasks, `0.77` for `transport_ph`, and `0.36` for `transport_mh`.
- Interpretation: the adapter works well enough for simpler Can/Lift oracle replay, partially works for Square/Transport PH, and fails on Tool Hang and Transport MH. Long-horizon object/contact drift remains the main weakness.

## 2026-05-08: Full Pipeline (Joint-Delta DP + Held-Out-Demo Adapter) Rollout Eval

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
