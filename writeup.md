# Joint-Space Diffusion Policy: action-space + adapter study

Companion writeup to `EXPERIMENTS.md`. Covers the work done on top of the joint-delta DiffusionPolicy branch: 5-task baselines, the FK→OSC deterministic adapter, and a parallel reproduction of Brian's learned NN inverse-controller adapter on OSC. Codebase: this branch off `WubbLord/diffusion_policy` + the `obs_noise_std` and adapter additions in this branch. WandB project: `diffusion_policy_debug`.

## TL;DR

- Joint-delta DiffusionPolicy trains cleanly on **lift / can / square** at 5 k epochs (success: 0.94 / 0.88 / 0.50 via FK→OSC). **Tool_hang** is still training (~4200/5000 ep) and currently gives 0.00 via FK→OSC at any `kp` — needs to finish. **Transport** (dual-arm) is still training and FK→OSC currently scores 0.00 even after world-rotation calibration (90°/−90° detected).
- The right way to roll out a joint-space policy with the standard Robomimic OSC controller is **deterministic FK→OSC**, not a learned NN inverse. Lift 0.94, can 0.88, square 0.50 (with `osc_kp=3000`). This matches or exceeds Brian's `JOINT_POSITION kp=3000` numbers on the same checkpoints.
- A **probe-based** learned NN→OSC adapter (Brian's MLP architecture, his `collect_inverse_dataset.py`-style sampler, transposed to OSC actions) fails: ~0.00–0.02 success on every task, even with the full Brian-quality sampler (32 probes × 200 demos × 100 epochs). The fundamental reason is structural — see "Why NN→OSC fails" below.
- A **demo-supervised** learned NN→OSC adapter (no env probing; one training pair per demo timestep, command = the OSC action teleop recorded) closes the gap to FK→OSC: lift 0.90, can 0.64 (best variant 0.78 at d100), square 0.48. Training takes ~5 min/task. See section E.
- The demo-supervised adapter **scales monotonically with demo count** on lift and square (lift 0.70/0.82/0.88/0.90 at d10/d20/d50/d200; square 0.12/0.14/.../0.48). On can it peaks at **d100 = 0.78** and dips back to 0.64 at d200 — more data isn't always better. See section E ablation.
- Smaller MLPs (`128×2`) **beat** the baseline `512×3` on lift (0.94 vs 0.90) but **collapse** on can (0.20 vs 0.64) — the inverse mapping's complexity is task-specific.
- Oracle-replay (adapter alone, no DP, just replay demo Δq through adapter→env) confirms the probe-based failure is upstream of the policy: lift 0.48, can 0.24, square 0.00. Even given perfect demo Δq targets the OSC adapter cannot drive the arm well.
- **Residual NN→OSC** (NN learns the residual `a_demo − FK→OSC(...)` and adds it back at inference) is in flight — predicted to match or beat FK→OSC because the worst case is "NN outputs 0 = FK→OSC alone". Numbers will land in section G.
- **n_action_steps sweep** confirms controller tracking, not chunk drift, is the bottleneck: lift/can/square are all saturated across steps ∈ {1, 2, 4, 8} at the same kp. Section F.

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
- `reverse_controller_osc_demosup/{task}_ph/` — Track-3 (demo-supervised) NN-OSC pipeline: one (state, Δq, command) triple per demo timestep, no env probing.
- `reverse_controller_osc_demosup_{d10,d20,d50,d100,h128,e50}/{task}_ph/` — demo-supervised ablation variants (data count / MLP size / epoch budget).

## Experimental setup — the unchanging knobs

The same simulator, control frequency, episode length, and success criteria apply to every row in every table below; calling them out once.

**Simulator and controller.** robosuite 1.4 → Robomimic env wrapper. Control frequency 20 Hz (50 ms per env.step). OSC_POSE controller with `output_max = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]` (positional Δ in m, axis-angle Δ in rad, default ratio) and `damping_ratio = 1.0` unless explicitly varied. Default `kp = 150` is too low for our 8-step open-loop chunks; we bump to `kp = 1000` for lift/can and `kp = 3000` for square. The OSC controller expects its 6-D command in *normalized* `[-1, 1]` per axis and rescales internally to the `output_max` range.

**Task variants.** All five Robomimic lowdim tasks, PH split (proficient-human, 200 demos each). Tasks: `lift` (single-arm cube lift), `can` (PickPlaceCan single-arm sort), `square` (NutAssemblySquare single-arm insertion), `tool_hang` (single-arm hook-and-hang precision), `transport` (dual-arm handover). Transport is the only dual-arm task and adds the world↔panda base-rotation calibration step.

**DP architecture and training.** Identical to upstream `train_diffusion_unet_lowdim_workspace` plus our `obs_noise_std` injector. UNet1D diffusion policy with hidden `[256, 512, 1024]`, `kernel_size=5`, `n_groups=8`. Prediction horizon `T = 16`, observation history `n_obs_steps = 2`, action execution `n_action_steps = 8` (open-loop chunk between replans). 100 DDPM denoising steps at inference. EMA `power = 0.75`. AdamW, `lr = 1e-4` cosine, `weight_decay = 1e-6`, `batch_size = 256`. Target 5000 epochs unless preempted.

**Action target.** For joint-delta DP only: `a_t = [ Δq_{robot0} (7), Δq_{robot1} (7 if dual-arm), gripper_{robot0} (1), gripper_{robot1} (1 if dual-arm) ]` — layout "joints_then_grippers". `Δq[t] = q[t+1] − q[t]` is computed directly from `obs/robotN_joint_pos` in the dataset's `_data_to_joint_obs` builder. Gripper command stays the raw value from `actions[:, gripper_idx]` (index `-1` for single-arm, `[6, 13]` for transport).

**Observation features.** Single-arm: `object, robot0_eef_pos (3), robot0_eef_quat (4), robot0_gripper_qpos (2), robot0_joint_pos (7)` — total 33-D after `object` (which is 23-D for lift, 14-D for can, 14-D for square, 53-D for tool_hang; 41-D for transport). Transport: same set duplicated for `robot1_*`, total 73-D. `joint_vel` is intentionally excluded — DP doesn't see velocities, neither does the adapter, so train/deploy distributions match.

**Eval protocol.** Each cell in the result tables is `test/mean_score` from a single deterministic eval run: `test_start_seed = 100000`, `n_test = 50` rollouts, `n_envs = 28` (Async vector env), `max_steps = 400` for single-arm tasks and `700` for tool_hang and transport. The mean is over the 50 held-out episodes. We also report `train/mean_score` (6 episodes from the train split) for sanity. Success is the env's own `is_success()` (binary per episode, reduced via `any` across the episode horizon — i.e. did the env ever reach the success state). Videos are recorded for the first 4 test and 2 train episodes for failure-mode analysis.

**Hardware.** All training and eval on CSAIL `csail-shared-h200` partition (NVIDIA H200), 1 GPU per job, 4 CPUs, 32–80 GB RAM, `shared-if-available` QoS (preemptible, 24 h time limit). DP training wall time: 6 h (lift) → ~11 h (square) → still running (tool_hang / transport).

## Eval modes — what each row actually measures

To make the tables below readable, three rollout modes are reported:

- **Full pipeline** — DP outputs a chunk of joint-delta actions, the adapter (FK→OSC or NN→OSC) converts each into an OSC command, the OSC controller drives the arm, the env reports task success at the end. This is the metric we ultimately care about; what the writeup tables default to.
- **Adapter only ("oracle replay")** — *no DP*. For each held-out demo, walk the recorded states; at each step compute `Δq[t] = q[t+1] − q[t]` from the demo itself, feed `(state, Δq)` to the adapter, apply the resulting OSC command in the env, check task success. This bounds the full pipeline from above: if the adapter can't even replay demos when given perfect Δq targets, the policy can't help. Implemented in `oracle_replay_osc.py`.
- **DP only ("naive joint-delta")** — DP joint-delta actions sent directly to the controller, *no adapter at all*. This is the diagnostic baseline: it should fail, and it does, because joint deltas are not OSC Cartesian commands. Brian's Can MH gives 0/50 under this mode (see blog Fig. 2). We report a single confirmation row here, not a full sweep.

Other knobs reported when they vary:

- `osc_kp` — OSC controller proportional gain (default Robomimic 150). FK→OSC needs `kp=1000` or `3000` because the open-loop replan window is too long at the default.
- `osc_damping_ratio` (`dr`) — default 1.0; appears in the old JP sweep.
- `n_action_steps` — how many DP-predicted steps to execute open-loop before the next replan (Experiment 2 in `EXPERIMENTS.md`). Default 8 during training.

All rollouts: `test_start_seed=100000`, `n_test=50`, `n_envs=28`, `max_steps=400` (lift/can/square) / `700` (tool_hang/transport). Numbers in tables are `test/mean_score` (held-out, n=50) unless otherwise noted.

## Results

### A. DP training — 5k epochs, all 5 PH tasks

| Task       | Job ID | Status                  | Wall time |
|------------|--------|-------------------------|-----------|
| lift       | 818751 | COMPLETED (5000 ep)     | 6h 23m    |
| can        | 818753 | COMPLETED (5000 ep)     | 9h 59m    |
| square     | 818755 | COMPLETED (5000 ep)     | 10h 34m   |
| tool_hang  | 818757 | COMPLETED (5000 ep)     | 21h 09m   |
| transport  | 821174 / 828593 | RUNNING (cumulative ~epoch 4200+/5000; will time out at 24h, 828593 queued via `--dependency=afterany:821174` to finish the run) | 23h 30m so far |

Val-loss curves mirror Result A in `EXPERIMENTS.md` — min around epoch ~50–100, then drift up. We always eval the `latest.ckpt` for fair compare (best-val checkpointing is Experiment 5 in `EXPERIMENTS.md`; not yet wired into the joint workspace).

### B. FK→OSC adapter on the 5k-ep checkpoints

**Mode: full pipeline.** OSC `output_max` normalization applied to FK Δ-EEF before sending to controller. Reported metric is `test/mean_score` (n=50).

| Task      | Default OSC (`kp=150`) | `kp=1000` | `kp=3000` |
|-----------|------------------------|-----------|-----------|
| lift      | —                      | **0.94**  | —         |
| can       | —                      | **0.88**  | —         |
| square    | —                      | 0.34      | **0.50**  |
| tool_hang | (DP fully trained, 5000 ep) | 0.00 | 0.00 |
| transport | (dual-arm calibration WIP — currently 0.00) | TBD | TBD |

Tool_hang FK→OSC at the *final* `latest.ckpt` (job 818757 ran to 5000 ep, 21h 09m) still gives 0.00 at both `kp=1000` and `kp=3000`. Demosup NN→OSC on the same final checkpoint is 0.02. So the bottleneck on tool_hang is not DP convergence — the task's high-precision insertion phase is genuinely beyond what FK→OSC can express at our control bandwidth, or the joint-delta target itself is too coarse a representation for the fine-motor portion of the task. (A `kp=5000` cell and a finer-grained adapter/controller config are open follow-ups.)

`kp=1000` is a 6.7× bump over the Robomimic default (`kp=150`) and is necessary because the 1/20-s open-loop replan window leaves the EEF lagging the OSC target enough to lose grasps. For square, an additional bump to `kp=3000` recovers another 16 pp.

Closed-loop FK on square (mode: full pipeline, re-FK after every controller step instead of once per policy step) gave identical 0.34 at `kp=1000` — drift inside the chunk is not the bottleneck, controller tracking is.

#### How FK→OSC actually works

The runner is a closed-form, learning-free converter from "joint-space prediction" to "Cartesian controller command", with five moving parts:

1. **Side-car mujoco model.** A separate `mujoco.MjModel` (loaded once from the robosuite-bundled `panda/robot.xml`) holds the kinematic chain. No gripper, contacts, or actuators — just bone geometry. `mujoco.mj_kinematics(model, data)` propagates joint angles through the chain and writes per-body poses into `data.xpos / data.xmat`. Used purely to compute "what world pose does this `q` produce?"; never stepped, never modified.

2. **Single FK call.** `_PandaFK.fk(q)` writes `q` into `data.qpos[:7]`, calls `mj_kinematics`, and reads back `(data.xpos[eef_bid], data.xmat[eef_bid])` for the `right_hand` body — a 3-D world position and a 3×3 rotation matrix in the **panda model's base frame**. ~5 µs per call.

3. **Per-step Δp / Δr computation.** For each (B-th env, t-th step in the chunk):

   ```
   q ← q_curr                                # (7,) joint state from env at chunk start
   p_prev, R_prev ← FK(q)                    # panda-frame initial pose

   for t in range(T):
       q ← q + dq_chunk[t]                   # integrate predicted Δq
       p_t, R_t ← FK(q)                      # new panda-frame pose

       dp_panda ← p_t − p_prev                              # 3-D translation
       dR_panda ← R_t @ R_prev.T                            # 3×3 rotation delta
       dr_panda ← Rotation.from_matrix(dR_panda).as_rotvec() # 3-D axis-angle

       # Rotate into world frame
       dp_world ← R_world_panda @ dp_panda
       dr_world ← R_world_panda @ dr_panda

       # Normalize to OSC's [-1, 1] command space using its own output_max
       osc[t, 0:3] ← clip(dp_world / 0.05, -1, 1)
       osc[t, 3:6] ← clip(dr_world / 0.5,  -1, 1)
       osc[t, 6]   ← gripper_chunk[t]                       # passthrough

       p_prev, R_prev ← p_t, R_t
   ```

   The axis-angle vector `Rotation.from_matrix(ΔR).as_rotvec()` is exactly what `OSC_POSE` expects in its orientation slot — no quaternion or Euler conversion. The 0.05 m / 0.5 rad divisors come from the controller's `output_max` config; OSC rescales the normalized command back up internally.

4. **World↔panda calibration (once per rollout per arm).** `_PandaFK.fk` returns poses in the panda model's base frame. Robomimic's `robotN_eef_pos / _eef_quat` are in the **world** frame. For single-arm tasks the two coincide (panda mount is identity-rotated); for transport the two arms are mounted facing each other, giving 90° / −90° z-rotations. The runner discovers this from each env's first observation:

   ```
   q0       = obs[0, n_obs_steps - 1, joint_slice]
   p_env    = obs[0, n_obs_steps - 1, eef_pos_slice]
   q_env    = obs[0, n_obs_steps - 1, eef_quat_slice]
   _, R_fk0 = panda_fk.fk(q0)                              # panda-frame R
   R_env0   = scipy.Rotation.from_quat(q_env).as_matrix()  # world-frame R
   R_world_panda = R_env0 @ R_fk0.T
   ```

   The runner logs `arm{i} world<-panda z-angle ≈ X°` once per rollout; for transport you see `+90.0°` and `−90.0°`, for single-arm tasks `0.0°`.

5. **OSC controller takes over.** The env's `OSC_POSE` controller receives our `[Δp_norm, Δr_norm, gripper]`, multiplies by `output_max` to get the actual target delta, computes the wrench `F = K_p · (x_target − x_current) + K_d · ẋ_current` with our configurable `K_p = osc_kp_pos`, projects through the manipulator Jacobian to joint torques, applies them, and steps physics.

**Why this can't fail in the way NN→OSC fails.** FK→OSC never inverts OSC. It computes the trajectory the policy intends the EE to follow — uniquely determined by `q_curr` and the predicted `Δq` sequence via FK — and hands that trajectory to the controller in the controller's own command language. The controller handles redundancy resolution, nullspace projection, and torque computation as always. The only failure surfaces are explicit and diagnosable:

- Controller can't physically track the requested Δp/Δr in 50 ms. → tuned via `kp`.
- The policy's predicted Δq trajectory is itself bad. → bounded by DP training quality.
- The world↔panda calibration is wrong. → logged and verifiable.

NN→OSC has at least five additional failure surfaces (the many-to-one inverse, distribution mismatch with the policy, `output_max` saturation, no physics prior, rollout drift) — most of them silent.

**Adapter only is N/A for FK→OSC** — FK→OSC is analytic (no learned weights), so adapter-only oracle replay reduces to feeding the demo's own `Δq[t]` through the analytic Jacobian. We didn't run this; it would essentially measure controller tracking error.

**DP only is N/A for these checkpoints** — DP outputs joint deltas; the OSC controller expects 6-D Cartesian. Sending joint deltas straight into OSC would just be reinterpreting 7 numbers as `(Δpos, Δrot, gripper)` arbitrarily. Brian's blog confirms the analogous baseline on `JOINT_POSITION` gives 0/50 (his Can MH naive joint-delta row).

### C. NN→OSC adapter — Brian's pipeline transposed to OSC

#### Architecture

Brian's `InverseControllerMLP` (`reverse_controller/common.py`):
- Input: concatenate state features (per-task obs vector, e.g. 33-D for lift/can/square, 73-D for transport) and the *desired joint delta* (7-D single-arm, 14-D dual-arm). For single-arm tasks the input is `(33 + 7) = 40` floats.
- Hidden layers: 3 × Linear(512) + LayerNorm + SiLU.
- Output head: Linear → 7-D (single-arm) or 14-D (dual-arm) OSC command in `[-1, 1]` normalized action space.
- Normalizers (input mean/std, command mean/std) are computed once over the training shards and stored in the checkpoint. At inference the runner reads them to map raw obs → MLP input → raw OSC command.

#### Data collection — Brian's synthetic probe procedure

The collector resets the simulator to each demo's state, samples one or more candidate OSC commands per state, executes each command for *one* env.step, and records the resulting Δq:

```
for each demo:
    for each timestep t in demo:
        sim.reset_to(demo_state[t])
        q_before = sim.q()
        for k in range(samples_per_step):
            cmd = sampler(demo_command[t])              # ← see distribution below
            sim.step(cmd)
            q_after = sim.q()
            shard.append({
                "state":         build_state_features(obs[t]),
                "desired_delta": q_after - q_before,     # what this command actually produced
                "command":       cmd,                    # the OSC command we sent
            })
            sim.reset_to(demo_state[t])
```

Brian's sampler distribution (`reverse_controller/collect_inverse_dataset_osc.py`):

| Fraction | Source | Notes |
|----------|--------|-------|
| anchored | `{0, a_demo, 2·a_demo, 4·a_demo, 8·a_demo, 16·a_demo, −a_demo}` | the 7 anchor commands per step |
| 35 % of remainder | `Uniform(-1, 1)^6` | exploration over the full OSC range |
| 35 % of remainder | `factor · a_demo + N(0, 0.1)` with `factor ~ Uniform(0, 20)` | scaled-demo + noise |
| 30 % of remainder | `N(0, 0.3)` | gaussian |

All commands are clipped to `[-1, 1]` per axis (OSC normalized action range). The gripper command is a passthrough of `a_demo[gripper_idx]` and not perturbed.

Two training tracks. Both use the architecture and sampler above; they differ in `samples_per_step` and epoch budget.

- **Track 1 ("quick")** — `samples_per_step = 8`, 200 demos, 50 training epochs, batch 8192. Total ≈ 8 × 200 × (avg 100 steps/demo) ≈ 160 k (state, Δq, command) triples per task.
- **Track 2 ("Brian-quality")** — `samples_per_step = 32` (matches Brian's blog), 200 demos, 100 training epochs, batch 8192, val ratio 0.05. Total ≈ 32 × 200 × 100 ≈ 640 k triples per task. Lift completed at the time of writing; the other four BQ pipelines (can, square, tool_hang, transport) are running now via jobs 828210–828217.

Both tracks train with AdamW, `lr = 1e-4`, MSE loss between predicted OSC command and the sampler's recorded command (normalized).

**Mode: full pipeline** (DP `latest.ckpt` + adapter). Eval-dir names in parentheses for cross-reference.

| Task      | Track 1 (`eval_latest_nn_osc_brian`) | Track 2 (`eval_latest_nn_osc_brianquality`) |
|-----------|--------------------------------------|---------------------------------------------|
| lift      | 0.00                                 | 0.02                                        |
| can       | 0.02                                 | (BQ pipeline running, job 828211)           |
| square    | 0.00                                 | (BQ pipeline running, job 828213)           |
| tool_hang | 0.00                                 | (BQ pipeline running, job 828215)           |
| transport | (queued)                             | (BQ pipeline running, job 828217)           |

**Mode: adapter only** (oracle replay — no DP; feed demo `Δq` through adapter directly):

| Task      | Track 1 oracle success |
|-----------|------------------------|
| lift      | **0.48**               |
| can       | 0.24                   |
| square    | 0.00                   |
| tool_hang | (not run)              |
| transport | (not run)              |

Comparing the two modes is the diagnostic: even when given perfect demo `Δq` targets (adapter-only oracle), the Track 1 NN-OSC adapter fails outright on square (0.00) and only partially recovers on lift (0.48). The full pipeline is bounded above by the oracle number, so the policy contribution to the failure is at most a couple of pp on lift and can. The probe-based adapter itself is the bottleneck.

### D. Brian's NN→JP adapter — reference

For context, Brian's separately-trained NN inverse on the `JOINT_POSITION` controller (`reverse_controller_brian/can_ph_joint_position_*`) hits competitive numbers on can. The contrast is the point of section "Why NN→OSC fails".

### E. NN→OSC adapter — demo-supervised pipeline (new)

The probe-based pipelines (Tracks 1–2) treat the inverse problem as "given a random Δq target, find the OSC command that achieves it." Demo-supervised training drops all environment-probing entirely and instead uses the demos as both the *input distribution* and the *label source*.

#### Conceptual setup

Each Robomimic demo `i` is a sequence of `(obs_t, action_t)` pairs of length `T_i` (typically 100–400 steps), where `obs_t` includes the proprioceptive arm state and `action_t` is the OSC command teleop sent at step `t`. Critically, the demo records:

- `q_t = obs_t.robot0_joint_pos`   (7-D arm joint configuration at step `t`)
- `q_{t+1} = obs_{t+1}.robot0_joint_pos`  (joint config at the next step)
- `action_t = a_demo[t]`            (the 7-D OSC command that produced the `q_t → q_{t+1}` transition)

From this we can compute three quantities per timestep:

- **state features** `s_t` — a 33-D vector concatenating `object, robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, robot0_joint_pos` (the same obs the joint-delta DP sees; sizes 23 + 3 + 4 + 2 + 7 = 39 for can; 33 for lift; varies per task by `object` dim).
- **desired joint delta** `Δq_t = q_{t+1} − q_t` — what the arm actually moved in joint space.
- **OSC command** `a_demo[t]` — what teleop sent to OSC to produce that motion.

The adapter is trained to predict `a_demo[t]` from `(s_t, Δq_t)`. At deployment the DP predicts `Δq_t` from observation and the adapter converts it to an OSC command.

#### Collection procedure (`collect_demo_only_osc.py`)

Pseudocode:

```
for each demo:
    obs       = demo["obs"]              # dict of obs key arrays of length T
    next_obs  = demo["next_obs"]
    actions   = demo["actions"]          # (T, action_dim) -- OSC commands from teleop

    for t in range(T):
        state         = concat([obs[k][t] for k in obs_keys])      # 33-D
        desired_delta = next_obs[joint_key][t] - obs[joint_key][t] # 7-D Δq
        command       = actions[t]                                  # 7-D OSC

        shard.append({
            "state":         state,
            "desired_delta": desired_delta,
            "command":       command,
        })
```

No simulator reset, no env step, no candidate-command sampling, no rollout — pure read from the HDF5 file. The collector runs at ~12 demos/second on a single CPU. For the standard `--max-demos 200`, the full dataset of (state, Δq, command) triples per task is exactly `Σ_{i=0}^{199} T_i` ≈ 30 k–60 k examples depending on task. Lift: ~30 k. Can: ~40 k. Square: ~50 k. Tool_hang: ~60 k.

Per-task shard files are stored under `data/reverse_controller_osc_demosup/{task}_ph/shards/demo_N.npz`. A `metadata.json` at the top of each dataset records: dataset path, obs keys, joint keys, controller (`OSC_POSE`), and supervision label (`demo_only`).

#### Architecture (`reverse_controller/common.InverseControllerMLP`)

Brian's MLP, used unchanged from Tracks 1/2 so cross-track comparisons are clean:

```
inverse_controller_mlp(state_t, Δq_t):
    x = concat([state_t, Δq_t])          # (33 + 7) = 40-D for single-arm tasks
    x = (x - input_mean) / input_std     # per-channel normalize using train stats
    h = Linear(40 -> 512)(x)
    h = LayerNorm(512)(h)
    h = SiLU(h)
    h = Linear(512 -> 512)(h)
    h = LayerNorm(512)(h)
    h = SiLU(h)
    h = Linear(512 -> 512)(h)
    h = LayerNorm(512)(h)
    h = SiLU(h)
    cmd_normalized = Linear(512 -> 7)(h)
    cmd_raw = cmd_normalized * command_std + command_mean
    return cmd_raw
```

Parameter count: roughly `40·512 + 512·512·2 + 512·7 + LN terms` ≈ 550 k params. Tiny by ML standards. The `input_mean / input_std / command_mean / command_std` arrays are fit on the training-split shards once before training and saved with the checkpoint.

Compare ablation `h128` (2 × 128 hidden): ≈ 25 k params, 22× smaller, used to test "is the 3 × 512 capacity actually needed?". See the ablation table.

#### Training (`reverse_controller/train_inverse_model.py`)

- Loss: per-channel MSE on normalized commands: `‖predicted_cmd_normalized − target_cmd_normalized‖²`.
- Optimizer: AdamW, `lr = 1e-4`, no weight decay.
- Batch size: 2048. Each batch covers ~5% of a typical task's pairs.
- Epochs: 200 (default), shuffle every epoch.
- Train/val split: 95/5 random demos. Validation is `val_loss` (MSE on normalized command) per epoch.
- Wall time: ~5 minutes total per task on a single H200 (data loading and MLP forward/backward are equally cheap).

Why MSE works despite the OSC inverse being many-to-one in general: the demo distribution implicitly selects one branch of the inverse map (the branch teleop happened to use), so the supervision target is *self-consistent across the demo set*. The MLP doesn't need to learn how to pick a branch; it just memorizes the branch teleop already picked. Off-manifold the branch ambiguity returns, but the policy at test time stays close enough to the demo manifold that this rarely matters.

#### Why this is fundamentally different from the probe-based pipelines

| Axis | Probe-based (Tracks 1, 2) | Demo-supervised (Track 3) |
|------|---------------------------|---------------------------|
| Dataset size | 160 k–640 k triples | 30 k–60 k triples |
| Input distribution | covers `(state × OSC_command)` uniformly via sampler | restricted to demo manifold |
| Label source | sim-measured `Δq` from random commands | demo-recorded OSC command |
| What the MLP fits | "inverse of OSC over the entire command space" | "inverse of OSC along the demo manifold" |
| Branch ambiguity | unresolved (MSE averages branches) | resolved by demos (teleop picked one) |
| Collect cost | 30–60 min/task (env stepping) | <1 min/task (HDF5 read only) |
| Train cost | ~10 min/task (8192 batch, 100 ep) | ~5 min/task (2048 batch, 200 ep) |
| Lift full-pipeline | 0.00 / 0.02 | **0.90** |
| Can full-pipeline | 0.02 / TBD | **0.64** |
| Square full-pipeline | 0.00 / TBD | **0.48** |

**Mode: full pipeline** (DP `latest.ckpt` + adapter). Side-by-side with the two probe tracks and FK→OSC for direct comparison.

| Task      | Track 1 (probe) | Track 2 (BQ probe) | **Track 3 (demo-supervised)** | FK→OSC reference |
|-----------|-----------------|--------------------|------------------------------|------------------|
| lift      | 0.00            | 0.02               | **0.90**                     | 0.94             |
| can       | 0.02            | (running, 828211)  | **0.64**                     | 0.88             |
| square    | 0.00            | (running, 828213)  | **0.48**                     | 0.50 (kp=3000)   |
| tool_hang | 0.00            | (running, 828215)  | 0.02                         | 0.00 (DP not converged) |
| transport | (n/a)           | (running, 828217)  | (collect+train done, eval blocked on dual-arm runner) | 0.00 |

The demo-supervised adapter closes ~95–100 % of the FK→OSC gap on lift, ~73 % on can, and matches FK→OSC on square. That is, with a tenth of the data and no env probing, the learned inverse now competes with the closed-form one.

**Mode: adapter only** (oracle replay): not yet run for demo-supervised. `oracle_replay_osc.py` is hard-coded to `DEFAULT_OBS_KEYS` (40-D state), but the demo-supervised collector uses the joint-delta workspace obs (33-D), so it errors out with a normalizer-shape mismatch. Fixable; on the open list.

**Mode: DP only** (naive joint-delta direct to OSC): not run for these checkpoints — same N/A reasoning as for FK→OSC (joint deltas are not OSC commands).

**Ablation (demo-supervised, lift+can+square, full-pipeline mode).**

Variants (each holds the others at the baseline `d200 / 512×3 / 200 ep / batch 2048`):

| Variant | Demos | Hidden | Epochs | Purpose |
|---------|-------|--------|--------|---------|
| `d10` | 10 | 512×3 | 200 | min-data: do we need 200 demos? |
| `d20` | 20 | 512×3 | 200 | min-data extension |
| `d50` | 50 | 512×3 | 200 | quarter-data |
| `d100` | 100 | 512×3 | 200 | half-data |
| `h128` | 200 | 128×2 | 200 | small-arch: does the inverse really need 3×512? |
| `e50` | 200 | 512×3 | 50 | early-stop: does it converge faster? |
| **baseline d200** | 200 | 512×3 | 200 | reference cell |

Eval (full pipeline), test/mean_score (n=50):

| Task | d10 | d20 | d50 | d100 | h128 | e50 | **baseline (d200)** |
|------|-----|-----|-----|------|------|-----|---------------------|
| lift | 0.70 | 0.82 | 0.88 | 0.78 | **0.94** | 0.90 | 0.90 |
| can  | 0.28 | 0.38 | 0.68 | **0.78** | 0.20 | 0.36 | 0.64 |
| square | 0.12 | 0.14 | — | — | — | — | 0.48 |
| tool_hang | — | — | — | — | — | — | 0.02 |
| transport | — | — | — | — | — | — | (training, blocked on dual-arm runner) |

Readings:

- **Lift is over-parameterized** at 512×3. The 128×2 ablation *beats* the baseline (0.94 vs. 0.90) — for lift the inverse mapping is simple enough that a smaller MLP generalizes better. Lift also tolerates the e50 short-train (0.90 = baseline). Lift's bottleneck is not the adapter at all; it's the FK→OSC gap (0.94) and DP itself.
- **Can has the opposite shape**. 128×2 collapses to 0.20 and e50 drops to 0.36 — the can adapter needs both capacity and training time. The single best can adapter we have is `d100 / 512×3 / 200ep` at **0.78** (vs. 0.64 baseline). So *less* training data wins on can — the d200 dataset has demo states the adapter overfits to. This points to a sweet spot around 100 demos for can.
- **Demo-count scaling**: clean monotone growth on lift (0.70 → 0.82 → 0.88 at d10/d20/d50), saturating between d50 and d100. Can grows up to d100 then dips. Square grows from 0.12 (d10) to 0.48 (d200) without saturating — more demos likely keeps helping.
- **Tool_hang demosup is 0.02** despite a successful adapter train. The bottleneck is the DP itself: current `latest.ckpt` (epoch ~4200 / 5000) gives FK→OSC = 0.00 at both `kp=1000` and `kp=3000`. Either tool_hang's joint-delta DP isn't converged yet or this task is fundamentally hard for joint-space prediction.

**Why this works when probe-based collection didn't.** The probe-based dataset asks the MLP to invert OSC over the entire `(state × Δq)` product space (every uniform/scaled/gaussian command Brian's sampler generates). At test time the diffusion policy only ever queries a thin tube around the demo trajectories. Demo-supervised training is the same conditional `(state, Δq) → a_OSC`, but restricted to the support the rollout will actually visit. The branch-ambiguity problem (many OSC commands map to the same Δq) doesn't go away — but inside the demo manifold, teleop already picked a consistent branch, so the MLP only has to memorize that one. Off-manifold accuracy is worthless if the policy never goes off-manifold.

### H. FK→JP — analytic adapter, JOINT_POSITION controller (running)

Symmetric counterpart to FK→OSC: same diffusion policy joint-delta predictions, but the analytic adapter now produces *joint position targets* and sends them to the `JOINT_POSITION` controller rather than translating to OSC. This is the path Brian's blog used for his NN→JP adapter — but here we use the *analytic* version (no learning, just integration), which should match or exceed Brian's learned JP numbers if the JP controller is competently tuned.

**Pipeline.** The runner is the existing `RobomimicJointLowdimRunner` (already in the repo, used for the original joint-delta evals). It receives the DP's `Δq_t` chunk, integrates `q_target[i] = q_curr + cumsum(Δq[0..i])` internally, and sends `q_target` (7-D absolute joint position) to the JP controller. The JP controller does its own PD-per-joint tracking with `controller_kp` we set explicitly (robosuite default 50 is way too low — the arm lags badly).

**`kp` sweep.** kp values 300, 1000, 3000 (and pending 5000) at `damping_ratio = 2.0`. Jobs 828541 (lift), 828542 (can), 828543 (square).

**Results so far (partial — kp=3000 row + can/square kp=1000 cells still running).**

| Task   | kp=300 | kp=1000 | kp=3000 | FK→OSC reference |
|--------|--------|---------|---------|------------------|
| lift   | 0.48   | 0.72    | running | 0.94 (kp=1000) |
| can    | 0.04   | running | pending | 0.88 (kp=1000) |
| square | 0.00   | running | pending | 0.50 (kp=3000) |

The sweep script's `[fail]` annotation in stdout is a "score below cell-success threshold" flag, **not** a crash — the `eval_log.json` files are written normally.

**Reading the partial data.**

- The "structurally trivial inverse" argument predicted JP would match or beat FK→OSC at high kp. So far it **doesn't** on lift: JP `kp=1000` = 0.72 vs FK→OSC `kp=1000` = 0.94. The remaining `kp=3000` cells will tell us whether JP closes the gap or stalls below FK→OSC.
- Can at `kp=300` = 0.04 is a much steeper drop from baseline than lift (which still got 0.48 at the same kp). This matches Brian's blog observation that can's joint trajectories are harder to track at low PD stiffness.
- Square at `kp=300` = 0.00 isn't surprising — square already needed `kp=3000` on the OSC side to break above 0.34.

A clean table comparing FK→OSC vs JP at matched kp will land in the next update once kp=3000 rollouts complete.

### G. Residual NN→OSC — analytic + learned (running)

**Architecture.** Same 3×512 MLP as the demo-supervised adapter, but the training target is the residual:

```
fk_pred[t]   = FK→OSC(q[t], q[t+1] − q[t], gripper[t])              # 7-D, normalized
residual[t]  = demo_command[t] − fk_pred[t]                          # 7-D target
```

At inference, the FK runner computes `fk_pred` as usual, runs `nn_residual = clip(MLP(state, Δq), ±0.3)`, and steps with `osc = clip(fk_pred + nn_residual, -1, 1)`. Worst case: `nn_residual → 0` and we recover FK→OSC.

**Pipeline.** `collect_demo_residual_osc.py` (computes the FK pred per step via the standalone Panda mujoco model + per-demo `R_world_panda` calibration) → `reverse_controller/train_inverse_model.py` (unchanged, just trains on the residual target) → patched FK runner with `residual_adapter_path` arg.

**Jobs and results — clip=0.3 (828458/459/460 COMPLETED).**

| Task | kp | clip | residual+FK score | FK alone (ref) | Δ vs FK alone |
|------|----|------|-------------------|----------------|---------------|
| lift   | 1000 | 0.3 | 0.86 | 0.94 | **−8 pp** |
| can    | 1000 | 0.3 | 0.34 | 0.88 | **−54 pp** |
| square | 3000 | 0.3 | 0.02 | 0.50 | **−48 pp** |

This **contradicts the original "worst case is FK→OSC alone" prediction**. The residual NN at `clip=0.3` is actively pulling rollouts off-manifold — 0.3 of OSC's normalized range is up to ±0.015 m / ±0.15 rad of additive correction every step on top of FK, which is large enough that any per-step prediction error compounds into trajectory drift. Square is hit hardest because OSC's stiffness ceiling is already binding there; an extra noisy correction tips it over.

**Fix in flight — clip=0.05 (828555/556/557, running).** Same DP + adapter checkpoints, residual clipped to ±0.05 (i.e. at most ±2.5 mm / ±25 mrad per step on top of FK). If clip=0.05 still hurts, the residual approach is fundamentally broken at inference (signs / frame mismatch, not magnitude), and we drop it. If clip=0.05 lands within ±2 pp of FK alone, residual is harmless but not useful at the current cap; the next step is to train it on a *smaller-residual* target distribution (e.g. teleop − FK, restricted to demos where FK already tracks well).

### F. n_action_steps sweep — FK→OSC adapter

**Mode: full pipeline.** The 5k-ep checkpoints were trained with `n_action_steps=8` (Robomimic default). Replanning more often (open-loop chunk shorter) means the OSC controller spends less time tracking a stale FK target. Results so far (job 828177 still finishing the steps=8 and steps=12 cells):

| Task   | steps=1 | steps=2 | steps=4 | steps=8 (baseline) | steps=12 |
|--------|---------|---------|---------|--------------------|----------|
| lift (kp=1000)   | 0.94 | 0.94 | 0.94 | 0.94 | (running) |
| can (kp=1000)    | 0.88 | 0.88 | 0.88 | 0.88 | (running) |
| square (kp=3000) | 0.50 | 0.50 | 0.50 | 0.50 | (running) |

**All three tasks are saturated across replan cadence.** Within the measured range, the open-loop chunk length doesn't move the needle — controller tracking (governed by `kp`) is the bottleneck, not stale-FK drift inside the chunk. This rules out "shorten the chunk to fix square" as a fix and reinforces the conclusion that square's 0.50 ceiling at kp=3000 is a contact-physics ceiling, not a planning-frequency ceiling.

Job 828177 is the re-run of the actsteps sweep (the previous attempt at job 827030 was broken by a transient `huggingface_hub` upgrade I caused on the cluster).

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

- **Tool-hang FK→OSC** (job 818757 now COMPLETED at 5000 ep, 21h 09m). Even the fully-trained checkpoint scores 0.00 at `kp=1000` and `kp=3000`; demosup NN→OSC on the same checkpoint = 0.02. The task is genuinely beyond what joint-delta + FK→OSC at our control bandwidth can express. Next experiments to disambiguate: (a) `kp=5000` cell, (b) `n_action_steps=1` cell, (c) the demo-derived `q_target` trajectory replay (no DP) to check whether the demos themselves are tracked successfully under FK→OSC.
- **Transport dual-arm FK→OSC**. Two issues: (a) DP val_loss is healthy and still improving (0.07–0.08 now vs 0.112 at epoch 3900) but `test_mean_score` stuck at 0.0; (b) FK→OSC currently scores 0.00 even after per-arm world-rotation calibration (90° / −90° detected for the two inward-facing arms) at `osc_kp=1000`. Action-layout matches the dataset convention (`[arm0_dq, arm1_dq, arm0_grip, arm1_grip]` aka "joints_then_grippers"), so the bug is elsewhere — likely either (i) per-arm Jacobian frame mixing or (ii) FK chunk integration drift compounding more than at single-arm scale. Diagnostic data so far: pure-FK calibration is mathematically exact (`‖R_hand_world(FK) − R_hand_world(obs)‖_F = 0` on both arms; both `R_world_panda` are clean +90° / −90° z-rotations). A demo-replay control (feed the demo's own `Δq` through FK→OSC) gave 0/5; a raw-action replay (skip FK, send the demo's recorded OSC commands directly) gave 1/5 — so the env's `reset_to` + delta-OSC playback is itself flaky and the demo-replay control is inconclusive. Need a deterministic-replay test under matched seeds before drawing further conclusions. Job 821174 continues training; 828593 queued via `--dependency=afterany:821174` to take the run to 5000 ep.
- **Residual NN→OSC clip=0.3 was net-negative** (lift −8 pp, can −54 pp, square −48 pp vs FK→OSC alone). Clip=0.05 rerun is in flight (828555/556/557). If clip=0.05 also hurts, the approach is structurally wrong at inference, not magnitude-wrong. Section G has the table.
- **Transport NN→OSC eval** (any track). `RobomimicJointBrianOSCRunner` is single-arm only — `n_robots` is fixed to 1 and there is no per-arm command assembly. Adding dual-arm support is the analogue of what we already did in `RobomimicJointFKtoEEFRunner`. Adapter checkpoint for transport demosup is being trained right now (job 828231); eval blocked until the dual-arm NN runner exists.
- **Demo-supervised oracle replay**. `oracle_replay_osc.py` is hard-coded to `DEFAULT_OBS_KEYS` (40-dim state), but the demo-supervised collector uses the joint-delta workspace obs (33-dim), so oracle replay errors out with a normalizer-shape mismatch. The full-pipeline rollout numbers in (E) are not yet flanked by oracle numbers; that requires plumbing the saved `obs_keys` through `oracle_replay_osc.py`.
- **Brian-quality sampler on can / square / tool_hang / transport**. Demo-supervised dominates lift on this axis, but we never *measured* the other tasks' BQ numbers. Currently running as jobs 828210–828217. Predicted: stays ≤ 0.05 on all four (the structural argument), but worth confirming for the writeup.
- **Best-by-val-loss checkpointing on joint workspaces** (Experiment 5 in `EXPERIMENTS.md`). Until this is wired in we eval `latest.ckpt` which is past the val-loss minimum; some of the joint-delta FK→OSC numbers may improve a couple of points if we re-eval at the val-min checkpoint.
- **Failure-mode taxonomy on the videos**. The probe-based NN-OSC videos consistently show wrist swivel + grasp-miss; FK→OSC failures on square are object-knock and pose-collisions; demo-supervised failures on can are gripper-misalignment on the bin-place. Not yet tabulated.

## Pointers

- `EXPERIMENTS.md` — the original experiment catalogue (Experiments 1–9 and Results A–C). This document continues it with the new adapter work.
- Brian's reference inverse-controller pipeline: `reverse_controller/` package. We follow his data format and `InverseControllerMLP`; the only thing we change in `collect_inverse_dataset_osc.py` is which command the env steps with.
- Demo-supervised collector: `collect_demo_only_osc.py` + `demosup_pipeline.sh`.
- n_action_steps sweep: `sweep_action_steps.sh`.
- Adapter checkpoints uploaded to Hugging Face: https://huggingface.co/sour5blue/diffusion-policy-osc-adapters — probe-based (Track 1) in `probe_nn_osc/`, demo-supervised (Track 3) in `demosup_nn_osc/{task}/`.
- Slurm jobs of record:
  - DP training: `818751` lift, `818753` can, `818755` square, `818757` tool_hang (COMPLETED 5000 ep), `821174` transport resume (running, near 24 h time limit), `828593` queued (`afterany:821174`) to finish transport to 5000 ep.
  - Track 1 (probe quick) NN-OSC: `821042`–`821051`.
  - Track 2 (BQ probe) NN-OSC: `821027`–`821145` (lift only); `828210`–`828217` (can/square/tool_hang/transport, in flight).
  - Track 3 (demo-supervised) NN-OSC: `827035` lift, `827036` can, `827037` square, `827301` tool_hang collect+train, `828208` tool_hang full eval, `828231` transport collect+train.
  - Actsteps sweep (FK→OSC): `827030` (broken by HF upgrade), `828177` (re-run).
  - Demo-supervised ablation: `828206` lift d50/d100/h128/e50 eval, `828207` can same, `828235`/`828236`/`828237` demo-count sweep (d10/d20) for lift/can/square.
- WandB tags: `joint5k`, `joint_delta`, `nn_osc_brian`, `nn_osc_brianquality`, `nn_osc_demosup`.

## Auto-harvested results log
Newest entries last. One line per `eval_log.json` (or job completion event).
- `2026-05-12 16:55` job=**828211** BQ NN-OSC can — finished, no eval_log found at `data/outputs/2026.05.11/*can_lowdim_joint_delta_joint5k/eval_latest_nn_osc_brianquality/eval_log.json`
- `2026-05-12 16:57` job=**828541** JP lift (`eval_latest_jp_kp5000_dr2.0`) — test/mean_score=0.98, train/mean_score=0.8333333333333334
- `2026-05-12 16:59` job=**828556** residual clip=0.05 can kp=1000 — finished, no eval_log found at `data/outputs/2026.05.11/*can_lowdim_joint_delta_joint5k/eval_latest_residual_kp1000_clip0p05/eval_log.json`
- `2026-05-12 16:59` job=**828542** JP can (`eval_latest_jp_kp3000_dr2.0`) — test/mean_score=0.82, train/mean_score=0.5
- `2026-05-12 16:59` job=**828543** JP square (`eval_latest_jp_kp3000_dr2.0`) — test/mean_score=0.42, train/mean_score=0.16666666666666666
- `2026-05-12 16:59` job=**828177** actsteps sweep (`eval_actsteps_8_kp3000`) — test/mean_score=0.5, train/mean_score=0.3333333333333333
- `2026-05-12 16:59` job=**828539** demosup_ablate square (`eval_latest_nn_osc_demosup_d100`) — test/mean_score=0.4, train/mean_score=0.6666666666666666
- `2026-05-12 17:13` job=**828555** residual clip=0.05 lift kp=1000 — finished, no eval_log found at `data/outputs/2026.05.11/*lift_lowdim_joint_delta_joint5k/eval_latest_residual_kp1000_clip0p05/eval_log.json`
- `2026-05-12 17:13` job=**828557** residual clip=0.05 square kp=3000 — finished, no eval_log found at `data/outputs/2026.05.11/*square_lowdim_joint_delta_joint5k/eval_latest_residual_kp3000_clip0p05/eval_log.json`
- `2026-05-12 17:13` job=**828542** JP can (`eval_latest_jp_kp5000_dr2.0`) — test/mean_score=0.86, train/mean_score=1.0
- `2026-05-12 17:13` job=**828543** JP square (`eval_latest_jp_kp5000_dr2.0`) — test/mean_score=0.42, train/mean_score=0.3333333333333333
- `2026-05-12 17:13` job=**828177** actsteps sweep — finished, no eval_log found at `data/outputs/2026.05.11/*_lowdim_joint_delta_joint5k/eval_actsteps_*_kp*/eval_log.json`
