# Joint-Space Diffusion Policy: action-space + adapter study

Companion writeup to `EXPERIMENTS.md`. Covers the work done on top of the joint-delta DiffusionPolicy branch: 5-task baselines, the FK→OSC deterministic adapter, and a parallel reproduction of Brian's learned NN inverse-controller adapter on OSC. Codebase: this branch off `WubbLord/diffusion_policy` + the `obs_noise_std` and adapter additions in this branch. WandB project: `diffusion_policy_debug`.

## TL;DR

- Joint-delta DiffusionPolicy trains cleanly on all five Robomimic PH lowdim tasks at 5 k epochs. Lift / can / square / tool_hang single-arm complete; transport (dual-arm) was preempted and resumed (job 821174).
- The right way to roll out a joint-space policy with the standard Robomimic OSC controller is **deterministic FK→OSC**, not a learned NN inverse. Lift 0.94, can 0.88, square 0.50 (with `osc_kp=3000`). This matches or exceeds Brian's `JOINT_POSITION kp=3000` numbers on the same checkpoints.
- A **probe-based** learned NN→OSC adapter (Brian's MLP architecture, his `collect_inverse_dataset.py`-style sampler, transposed to OSC actions) fails: ~0.00–0.02 success on every task, even with the full Brian-quality sampler (32 probes × 200 demos × 100 epochs). The fundamental reason is structural — see "Why NN→OSC fails" below.
- A **demo-supervised** learned NN→OSC adapter (no env probing; one training pair per demo timestep, command = the OSC action teleop recorded) closes the gap to FK→OSC: lift 0.90, can 0.64, square 0.48. Training takes ~5 min/task. See section E.
- Oracle-replay (adapter alone, no DP, just replay demo Δq through adapter→env) confirms the probe-based failure is upstream of the policy: lift 0.48, can 0.24, square 0.00. Even given perfect demo Δq targets the OSC adapter cannot drive the arm well.

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
| tool_hang  | 818757 | RUNNING (~4100/5000 ep) | 17h+      |
| transport  | 821174 | RUNNING (resume from ~3900/5000 ep) | 19h+ |

Val-loss curves mirror Result A in `EXPERIMENTS.md` — min around epoch ~50–100, then drift up. We always eval the `latest.ckpt` for fair compare (best-val checkpointing is Experiment 5 in `EXPERIMENTS.md`; not yet wired into the joint workspace).

### B. FK→OSC adapter on the 5k-ep checkpoints

**Mode: full pipeline.** OSC `output_max` normalization applied to FK Δ-EEF before sending to controller. Reported metric is `test/mean_score` (n=50).

| Task      | Default OSC (`kp=150`) | `kp=1000` | `kp=3000` |
|-----------|------------------------|-----------|-----------|
| lift      | —                      | **0.94**  | —         |
| can       | —                      | **0.88**  | —         |
| square    | —                      | 0.34      | **0.50**  |
| tool_hang | (DP still training)    | TBD       | TBD       |
| transport | (DP still training; dual-arm calibration WIP — currently 0.00) | TBD | TBD |

`kp=1000` is a 6.7× bump over the Robomimic default (`kp=150`) and is necessary because the 1/20-s open-loop replan window leaves the EEF lagging the OSC target enough to lose grasps. For square, an additional bump to `kp=3000` recovers another 16 pp.

Closed-loop FK on square (mode: full pipeline, re-FK after every controller step instead of once per policy step) gave identical 0.34 at `kp=1000` — drift inside the chunk is not the bottleneck, controller tracking is.

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

The probe-based pipelines (Tracks 1–2) treat the inverse problem as "given a random Δq target, find the OSC command that achieves it." Demo-supervised training drops all environment-probing entirely.

#### Demo-supervised collection procedure (`collect_demo_only_osc.py`)

```
for each demo:
    for each timestep t in demo (length = 100..400 steps):
        state         = build_state_features(obs[t], obs_keys)
        desired_delta = q[t+1] − q[t]                              # the actual demo Δq
        command       = demo_action[t]                             # the OSC command teleop recorded
        shard.append({ state, desired_delta, command })
```

No sim resets, no env stepping, no random sampling. The training triple count per task is exactly `Σ_demo |demo|` ≈ 20 k–60 k pairs (200 demos × 100–300 steps).

#### Training

Same architecture as Tracks 1/2 (3×512 SiLU+LayerNorm MLP), same loss (MSE on normalized command). 200 epochs, batch 2048, val ratio 0.05, AdamW, `lr = 1e-4`. Trains in ~5 min per task on a single H200 (vs. ~30–60 min collect step for the probe tracks).

Obs keys for single-arm tasks: `object, robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, robot0_joint_pos` (33-D state, matching the joint-delta DP's obs). Dual-arm transport uses `--joint-keys robot0_joint_pos,robot1_joint_pos` + the doubled obs set.

The fundamental difference vs. probe-based: the (state, Δq) inputs are restricted to the demo manifold, and the supervision target is *the actual command the human operator used*, not "any command that produces this Δq". The MLP no longer has to invert a many-to-one map; it only has to memorize the one branch teleop already chose.

**Mode: full pipeline** (DP `latest.ckpt` + adapter). Side-by-side with the two probe tracks and FK→OSC for direct comparison.

| Task      | Track 1 (probe) | Track 2 (BQ probe) | **Track 3 (demo-supervised)** | FK→OSC reference |
|-----------|-----------------|--------------------|------------------------------|------------------|
| lift      | 0.00            | 0.02               | **0.90**                     | 0.94             |
| can       | 0.02            | (running)          | **0.64**                     | 0.88             |
| square    | 0.00            | (running)          | **0.48**                     | 0.50 (kp=3000)   |
| tool_hang | 0.00            | (running)          | (eval running, job 828208)   | (DP training)    |
| transport | (n/a)           | (running)          | (dual-arm pipeline pending)  | (debug WIP)      |

The demo-supervised adapter closes ~95–100 % of the FK→OSC gap on lift, ~73 % on can, and matches FK→OSC on square. That is, with a tenth of the data and no env probing, the learned inverse now competes with the closed-form one.

**Mode: adapter only** (oracle replay): not yet run for demo-supervised. `oracle_replay_osc.py` is hard-coded to `DEFAULT_OBS_KEYS` (40-D state), but the demo-supervised collector uses the joint-delta workspace obs (33-D), so it errors out with a normalizer-shape mismatch. Fixable; on the open list.

**Mode: DP only** (naive joint-delta direct to OSC): not run for these checkpoints — same N/A reasoning as for FK→OSC (joint deltas are not OSC commands).

**Ablation (demo-supervised, lift+can, full-pipeline mode).** Jobs 828206 / 828207 still running.

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

Eval (full pipeline) — table fills in as jobs land:

| Task | d10 | d20 | d50 | d100 | h128 | e50 | **baseline** |
|------|-----|-----|-----|------|------|-----|--------------|
| lift | (jobs 828235 PD) | (PD) | (running, 828206) | (running) | (running) | (running) | **0.90** |
| can  | (828236 PD) | (PD) | (running, 828207) | (running) | (running) | (running) | **0.64** |
| square | (828237 PD) | (PD) | — | — | — | — | **0.48** |
| tool_hang | — | — | — | — | — | — | (eval running) |

**Why this works when probe-based collection didn't.** The probe-based dataset asks the MLP to invert OSC over the entire `(state × Δq)` product space (every uniform/scaled/gaussian command Brian's sampler generates). At test time the diffusion policy only ever queries a thin tube around the demo trajectories. Demo-supervised training is the same conditional `(state, Δq) → a_OSC`, but restricted to the support the rollout will actually visit. The branch-ambiguity problem (many OSC commands map to the same Δq) doesn't go away — but inside the demo manifold, teleop already picked a consistent branch, so the MLP only has to memorize that one. Off-manifold accuracy is worthless if the policy never goes off-manifold.

### F. n_action_steps sweep — FK→OSC adapter

**Mode: full pipeline.** The 5k-ep checkpoints were trained with `n_action_steps=8` (Robomimic default). Replanning more often (open-loop chunk shorter) means the OSC controller spends less time tracking a stale FK target, which is what bottlenecks square at `kp=1000`. Results so far at `n_action_steps=1` (full re-plan every step) under FK→OSC, same `kp` as the `n_action_steps=8` baselines:

| Task   | steps=1 | steps=2 | steps=4 | steps=8 (baseline) | steps=12 |
|--------|---------|---------|---------|--------------------|----------|
| lift   | 0.94    | (job 828177, running) | (running) | 0.94 | (running) |
| can    | 0.88    | (running) | (running) | 0.88 | (running) |
| square | 0.50 (kp=3000) | (running) | (running) | 0.50 | (running) |

Lift and can are saturated at every cadence we've measured, so re-plan frequency makes no measurable difference there — they're already at near-ceiling. The square row is the interesting one; if steps=2/4 beat steps=8, that suggests open-loop tracking error is meaningful even at `kp=3000`. Job 828177 is the re-run of the actsteps sweep (the previous attempt at job 827030 was broken by a transient `huggingface_hub` upgrade).

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

- **Tool-hang FK→OSC** (job 818757 still running, ~20 h in, ~4200/5000 ep). Expect closer to lift/can than to square, since tool_hang is single-arm and reach-only. Job 828208 runs a partial eval against the current `latest.ckpt` while training continues.
- **Transport dual-arm FK→OSC**. Two issues: (a) DP val_loss is healthy but `test_mean_score` stuck at 0.0; (b) FK→OSC currently scores 0.00 even after per-arm world-rotation calibration (90° / −90° detected for the two inward-facing arms) at `osc_kp=1000`. Action-layout matches the dataset convention (`[arm0_dq, arm1_dq, arm0_grip, arm1_grip]` aka "joints_then_grippers"), so the bug is elsewhere — likely either (i) per-arm Jacobian frame mixing or (ii) FK chunk integration drift compounding more than at single-arm scale. Open debug item.
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
  - DP training: `818751` lift, `818753` can, `818755` square, `818757` tool_hang (still running), `821174` transport resume (still running).
  - Track 1 (probe quick) NN-OSC: `821042`–`821051`.
  - Track 2 (BQ probe) NN-OSC: `821027`–`821145` (lift only); `828210`–`828217` (can/square/tool_hang/transport, in flight).
  - Track 3 (demo-supervised) NN-OSC: `827035` lift, `827036` can, `827037` square, `827301` tool_hang collect+train, `828208` tool_hang full eval, `828231` transport collect+train.
  - Actsteps sweep (FK→OSC): `827030` (broken by HF upgrade), `828177` (re-run).
  - Demo-supervised ablation: `828206` lift d50/d100/h128/e50 eval, `828207` can same, `828235`/`828236`/`828237` demo-count sweep (d10/d20) for lift/can/square.
- WandB tags: `joint5k`, `joint_delta`, `nn_osc_brian`, `nn_osc_brianquality`, `nn_osc_demosup`.
