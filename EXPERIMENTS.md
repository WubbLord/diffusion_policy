# Experiments

## Experiment 1: Whole-Pipeline Success Rate Eval

**Goal:** Measure end-to-end task success for the joint-delta Diffusion Policy pipeline.

**Setup:**

- Task: Robomimic Can lowdim.
- Policy: joint-delta Diffusion Policy trained on demonstration-derived desired joint transitions.
- Adapter: learned inverse controller `f(state, dq_desired) -> u`.
- Controller: Robosuite `JOINT_POSITION`.
- Eval: simulation rollouts from the standard Diffusion Policy env runner.

**Pipeline:**

```text
obs[t] -> DP -> dq_desired[t], gripper[t]
state[t], dq_desired[t] -> adapter f -> joint-position command u[t]
u[t], gripper[t] -> JOINT_POSITION controller -> env.step
```

**Metrics:**

- Test success rate: `test/mean_score`.
- Train-initial-state success rate: `train/mean_score`.
- Per-rollout max reward.
- Rollout videos.
- Command saturation rate.

**Expected result:**

- The full joint-delta + adapter pipeline succeeds on the same seeded rollout protocol used by the original Diffusion Policy env runner.
- The same joint-delta DP without the adapter has much lower success.

**Main claim:** A joint-delta policy can be evaluated by standard task success once its desired transitions are translated into executable joint-controller commands.

## Experiment 2: Inverter Input Ablation

**Goal:** Determine whether the action adapter needs state information or is just learning a constant rescaling.

**Adapters:**

```text
f1(dq_desired)
f2(q, dq_desired)
f3(q, qdot, dq_desired)
f4(full_lowdim_obs, dq_desired)
f5(full_lowdim_obs, dq_desired, previous_command)
```

**Training data generation:**

- Reset sim to demo states.
- For each state, choose desired `dq` targets.
- Find or sample joint-position controller commands `u`.
- Record:

```text
input  = state, dq_desired
target = u
```

- Train `f` to predict `u`.

**Evaluation:**

- One-step tracking: reset to `state[t]`, command `u = f(state[t], dq_demo[t])`, measure `dq_actual`.
- Multi-step oracle replay: roll through full demo using `f(state[t], dq_demo[t])`.

**Metrics:**

- One-step `|dq_actual - dq_desired|`.
- Multi-step joint trajectory error.
- EEF trajectory error.
- Success rate.
- Saturation rate.
- Stability failures.

**Expected result:**

- If `f1` works well, mismatch is mostly gain/normalization.
- If `f3`, `f4`, or `f5` are much better, mismatch is state-dependent and justifies a learned adapter.

**Main claim:** Executable joint commands depend on controller state and robot configuration, not only on desired joint displacement.

## Experiment 3: Open-Loop vs Closed-Loop Adapter Execution

**Goal:** Test whether the adapter should be a one-step translator or an inner-loop joint-space servo.

**Open-loop runner:**

```text
for each policy/control step:
    dq_desired = dq_demo[t] or pi(obs[t])
    u = f(state, dq_desired)
    env.step(u)
```

**Closed-loop runner:**

```text
for each policy step:
    q_target = q_current + dq_desired
    for k inner steps:
        residual = q_target - q_current
        u = f(state, residual)
        env.step(u)
    return next observation to policy
```

**Sweep:**

```text
k = 1, 2, 4, 8
```

**Evaluation:**

- First use oracle labels: `dq_desired = q_demo[t + 1] - q_demo[t]`.
- Optionally repeat with learned Diffusion Policy outputs.

**Metrics:**

- Joint target tracking error.
- EEF pose tracking error.
- Task success.
- Trajectory drift.
- Control smoothness.
- Episode length/control cost.
- Latency from extra inner steps.

**Expected result:**

- Closed-loop execution improves target tracking and reduces accumulated drift.
- Too many inner steps may improve tracking but hurt speed or alter task timing.

**Main claim:** Joint-transition actions are better interpreted as local state targets, not one-shot low-level commands.

## Experiment 4: Adapter Supervision Strategy

**Goal:** Compare practical ways to obtain pseudo ground-truth joint-controller commands for the inverter.

**Strategies:**

- **A. Demo-state local search:** For each demo state and desired `dq_demo`, optimize `u` so one-step `dq_actual` matches `dq_demo`.
- **B. Random probing:** Reset to demo states, sample many `u` values, observe resulting `dq_actual`, and train inverse model `(state, dq_actual) -> u`.
- **C. Forward model + optimization:** Train `g(state, u) -> dq_actual`. At test time choose:

```text
u* = argmin_u ||g(state, u) - dq_desired||
```

- **D. Analytic baseline:** Use known Robosuite controller scaling, clipping, and gains to hand-design `u`.
- **E. Hybrid residual adapter:** Use:

```text
u = u_analytic + f_residual(state, dq_desired)
```

**Evaluation:**

- Use the same held-out demo states and oracle replay protocol.
- Compare data efficiency and replay quality.

**Metrics:**

- Number of simulator probes required.
- One-step `dq` tracking error.
- Multi-step joint trajectory error.
- EEF trajectory error.
- Task success.
- Runtime cost at eval.
- Generalization to held-out trajectories.
- Generalization to Diffusion Policy-predicted `dq`.

**Expected result:**

- Analytic and scaling baselines may help but likely fail in nonlinear/contact-heavy states.
- Random probing is simple but data-hungry.
- Forward-model optimization may track well but be slower.
- Hybrid residual adapter may give the best tradeoff.

**Main claim:** A small amount of simulator probing can convert demonstration-derived joint transitions into executable joint-controller commands.
