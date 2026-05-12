# Joint-Space Diffusion Policy: action-space + adapter study

Companion writeup to `EXPERIMENTS.md`. Covers the work done on top of the joint-delta DiffusionPolicy branch: 5-task baselines, the FK→OSC deterministic adapter, and a parallel reproduction of Brian's learned NN inverse-controller adapter on OSC. Codebase: this branch off `WubbLord/diffusion_policy` + the `obs_noise_std` and adapter additions in this branch. WandB project: `diffusion_policy_debug`.

## TL;DR

- Joint-delta DiffusionPolicy trains cleanly on all five Robomimic PH lowdim tasks at 5 k epochs. Lift / can / square / tool_hang single-arm complete; transport (dual-arm) was preempted and resumed (job 821174).
- The right way to roll out a joint-space policy with the standard Robomimic OSC controller is **deterministic FK→OSC**, not a learned NN inverse. Lift 0.94, can 0.88, square 0.50 (with `osc_kp=3000`). This matches or exceeds Brian's `JOINT_POSITION kp=3000` numbers on the same checkpoints.
- A learned NN→OSC adapter (Brian's MLP architecture, his `collect_inverse_dataset.py`-style sampler, transposed to OSC actions) fails: ~0.00–0.02 success on every task, even with the full Brian-quality sampler (32 probes × 200 demos × 100 epochs). The fundamental reason is structural — see "Why NN→OSC fails" below.
- Oracle-replay (adapter alone, no DP, just replay demo Δq through adapter→env) confirms the failure is upstream of the policy: lift 0.48, can 0.24, square 0.00. Even given perfect demo Δq targets the OSC adapter cannot drive the arm well.

## What was built

Code (all in this branch, paths relative to `diffusion_policy/`):

- `env_runner/robomimic_joint_fk_to_eef_runner.py` — deterministic adapter. Takes the policy's predicted Δq action, integrates onto current joint state, runs forward kinematics via a standalone mujoco `MjModel` on `panda.xml`, converts the FK end-effector pose into the OSC action-space (Δpos, axis-angle Δrot, plus gripper), feeds to the standard `OSC_POSE` controller. Handles per-arm world↔panda-base rotation calibration for dual-arm transport. Supports OSC `kp` / `damping_ratio` overrides at runtime.
- `env_runner/robomimic_joint_brian_osc_runner.py` — learned NN adapter runner. Loads Brian's `InverseControllerMLP` from a checkpoint, normalizes state+desired-Δq input, predicts an OSC command, applies it to `OSC_POSE`.
- `reverse_controller/collect_inverse_dataset_osc.py` — OSC analog of Brian's `collect_inverse_dataset.py`. Uses Brian's exact sampler scheme: anchor commands at `{0, demo, 2·demo, 4·demo, 8·demo, 16·demo, -demo}`, 35% uniform `[-1,1]`, 35% scaled-demo + noise (factor ∈ [0,20]), 30% gaussian. All commands clipped to `[-1, 1]` (OSC normalized action space). For each `(state, command)` it resets the sim and measures actual Δq.
- `eval_nn_osc.py` — full-pipeline (DP + NN adapter) eval (click CLI), writes `eval_log.json` + videos.
- `eval_fk.py` — full-pipeline FK→OSC eval, supports `--osc_kp` / `--osc_damping_ratio`.
- `oracle_replay_osc.py` — adapter-only oracle eval. Walks held-out demos, computes `Δq[t] = q[t+1] − q[t]`, feeds to adapter, applies in env, checks task success.
- `eval_inverse_model_osc.py` — Brian's one-step open-loop adapter accuracy eval transposed to OSC. For each held-out demo timestep, samples `desired_Δq`, predicts OSC command, applies, measures actual Δq, plots per-joint hexbin `(desired, actual)`.
- `workspace/train_diffusion_unet_lowdim_workspace.py` — added Brian's resume-from-checkpoint block so preempted runs continue from saved state instead of restarting (Hydra struct-mode requires `++training.resume=True` to override).
- All five `*_lowdim_joint_delta.yaml` configs now expose `obs_noise_std` (per Experiment 4 in `EXPERIMENTS.md`).

Data layout under `data/`:

- `outputs/2026.05.11/01.22.*_*joint_delta_joint5k/` — the five 5k-epoch DP checkpoints (single-arm tasks) and their `eval_latest_*` subdirs.
- `reverse_controller_osc/{task}_ph/` — Track-1 ("quick") NN-OSC pipeline: 8-probes × 200-demos collect, 50-epoch MLP train, oracle replay.
- `reverse_controller_osc_bq/{task}_ph/` — Track-2 ("Brian-quality") NN-OSC pipeline: 32-probes × 200-demos with Brian's exact sampler, 100-epoch MLP train, oracle replay + onestep eval.

## Results

### A. DP training — 5k epochs, all 5 PH tasks

| Task       | Job ID | Status                  | Wall time |
|------------|--------|-------------------------|-----------|
| lift       | 818751 | COMPLETED (5000 ep)     | 6h 23m    |
| can        | 818753 | COMPLETED (5000 ep)     | 9h 59m    |
| square     | 818755 | COMPLETED (5000 ep)     | 10h 34m   |
| tool_hang  | 818757 | RUNNING (~4100/5000 ep) | 17h+      |
| transport  | 821174 | RUNNING (resume from ~3900/5000 ep) | 19h+ |

Val-loss curves mirror Result A in `EXPERIMENTS.md` — min around epoch ~50–100, then drift up. We always eval the `latest.ckpt` for fair compare (best-val checkpointing is Experiment 5 in `EXPERIMENTS.md`; not yet wired into the joint workspace).

### B. FK→OSC adapter on the 5k-ep checkpoints

`test_start_seed=100000`, `n_test=50`, `n_envs=28`, `max_steps=400` (lift/can/square) / `700` (tool_hang/transport). OSC `output_max` normalization applied to FK Δ-EEF before sending to controller.

| Task      | Default OSC (`kp=150`) | `kp=1000` | `kp=3000` |
|-----------|------------------------|-----------|-----------|
| lift      | —                      | **0.94**  | —         |
| can       | —                      | **0.88**  | —         |
| square    | —                      | 0.34      | **0.50**  |
| tool_hang | (DP still training)    | TBD       | TBD       |
| transport | (DP still training; dual-arm calibration WIP — currently 0/0) | TBD | TBD |

`kp=1000` is a 6.7× bump over the Robomimic default (`kp=150`) and is necessary because the 1/20-s open-loop replan window leaves the EEF lagging the OSC target enough to lose grasps. For square, an additional bump to `kp=3000` recovers another 16 pp.

Closed-loop FK on square (re-FK after every controller step, instead of once per policy step) gave identical 0.34 at `kp=1000` — drift inside the chunk is not the bottleneck, controller tracking is.

### C. NN→OSC adapter — Brian's pipeline transposed to OSC

Two training tracks. Both use Brian's `InverseControllerMLP` (3×512 hidden, SiLU, LayerNorm) and the same input layout (`state` ⊕ `desired_Δq`, normalize per-channel, predict OSC command).

- **Track 1 ("quick")** — 8 probes/demo × 200 demos, 50 epochs, batch 8192. Total ~80 k (state, Δq, command) triples per task.
- **Track 2 ("Brian-quality")** — 32 probes/demo × 200 demos with Brian's exact anchored sampler (`{0, demo, 2·demo, 4·demo, 8·demo, 16·demo, -demo}` + 35/35/30 uniform / scaled-demo / gaussian, all clipped to `[-1,1]`), 100 epochs, batch 8192, val ratio 0.05. Total ~250 k triples per task. Lift completed at the time of writing; the other tasks were not finished before this writeup.

Full-pipeline rollout (DP + adapter), `latest.ckpt`:

| Task      | Track 1 (`eval_latest_nn_osc_brian`) | Track 2 (`eval_latest_nn_osc_brianquality`) |
|-----------|--------------------------------------|---------------------------------------------|
| lift      | 0.00                                 | 0.02                                        |
| can       | 0.02                                 | (queued)                                    |
| square    | 0.00                                 | (queued)                                    |
| tool_hang | 0.00                                 | (queued)                                    |
| transport | (queued, blocked on DP)              | (queued)                                    |

Adapter-only oracle replay (no DP — replay demo `Δq` through adapter):

| Task      | Track 1 oracle success |
|-----------|------------------------|
| lift      | **0.48**               |
| can       | 0.24                   |
| square    | 0.00                   |
| tool_hang | (queued)               |
| transport | (queued)               |

Even with perfect demo Δq targets the adapter cannot drive the OSC controller to success. The full-pipeline number is bounded above by oracle replay, so the policy contribution to the gap is at most ~2 pp on lift, ~22 pp on can.

### D. Brian's NN→JP adapter — reference

For context, Brian's separately-trained NN inverse on the `JOINT_POSITION` controller (`reverse_controller_brian/can_ph_joint_position_*`) hits competitive numbers on can. The contrast is the point of section "Why NN→OSC fails".

## Why NN→OSC fails

OSC and JP give the inverse-problem structurally different shapes.

- **`JOINT_POSITION`**: command is a 7-DoF joint Δq target; controller is an independent PD per joint. Given desired Δq, the command is essentially `desired_Δq` modulo controller bandwidth. The map is **one-to-one and near-linear**. A small MLP with reasonable data converges to ≈ identity-scale.
- **`OSC_POSE`**: command is a 6-DoF Cartesian wrench (Δpos + axis-angle Δrot). The Δq response is the impedance dynamics integrated through the manipulator Jacobian, with nullspace control filling the redundancy. Two different OSC commands can produce the same Δq (the Cartesian command is underdetermined for 7-DoF arms once you fix the EE target). The inverse map is **many-to-one and Jacobian-singular near collisions / wrist limits**. The MLP has to either pick a branch consistently or it diffuses energy across plausible inverses and produces garbage.

In practice this shows up as:
- One-step adapter prediction has high per-joint RMSE on shoulder + elbow even at 100 epochs.
- Oracle replay drifts off the demo manifold within ~30 steps even on lift (the easiest task).
- Bumping data 4× (Track 1 → Track 2) and adding Brian's extreme-anchor sampling gives essentially no improvement on lift — the gap is not a data-size problem.

This is why FK→OSC, which is a closed-form inverse using the analytic Jacobian and the controller's own normalization, wins. The deterministic adapter never has to pick a branch; the controller does.

## What's still open

- **Tool-hang FK→OSC** (currently blocked on the long 5k-ep training). Expect closer to lift/can than to square, since tool_hang is single-arm and reach-only.
- **Transport dual-arm**. Two issues: (a) DP still has NaN val_loss as of the resume; (b) FK→OSC currently scores 0/0 even after per-arm world-rotation calibration (90° / −90° detected) and `osc_kp=1000`. Action-layout debug needed.
- **Brian-quality sampler on can / square / tool_hang / transport**. Lift didn't move; unclear whether the other tasks will (the structural argument predicts they won't, but it costs ~1 H100-hour per task to verify, so it's worth running).
- **Best-by-val-loss checkpointing on joint workspaces** (Experiment 5). Until this is wired in we eval `latest.ckpt` which is past the val-loss minimum; some of the joint-delta FK→OSC numbers may improve a couple of points if we re-eval at the val-min checkpoint.
- **Failure-mode taxonomy on the videos**. The NN-OSC videos consistently show wrist swivel + grasp-miss; FK→OSC failures on square are object-knock and pose-collisions. Not yet tabulated.

## Pointers

- `EXPERIMENTS.md` — the original experiment catalogue (Experiments 1–9 and Results A–C). This document continues it with the new adapter work.
- Brian's reference inverse-controller pipeline: `reverse_controller/` package. We follow his data format and `InverseControllerMLP`; the only thing we change in `collect_inverse_dataset_osc.py` is which command the env steps with.
- Slurm jobs of record: `818751`, `818753`, `818755`, `818757`, `821174` (training); `821042`–`821051` (Track-1 NN-OSC pipelines); `821027`–`821145` (Track-2 BQ pipelines).
- WandB tags: `joint5k`, `joint_delta`, `nn_osc_brian`, `nn_osc_brianquality`.
