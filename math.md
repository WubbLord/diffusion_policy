# Math companion: kinematic redundancy, FK adapters, and the OSC inverse problem

Standalone math document for the joint-delta DP work in `writeup.md`. Covers:

1. Setup and FK basics
2. Manipulator Jacobian and the 1-D Panda null space
3. The FK→OSC adapter — forward map and why it works
4. World↔panda-base frame calibration
5. OSC dynamics: impedance, Jacobian-transpose, posture nullspace
6. Failure modes (controller lag, singularities, posture mismatch)
7. FK→JP contrast (Brian's analytic JP adapter)
8. NN→OSC structural failure — the rank-deficient inverse
9. Why NN→JP succeeds where NN→OSC doesn't
10. Big-picture comparison table

---

## 1. Setup

**Robot.** 7-DoF Panda. Joint state `q ∈ ℝ⁷`. End-effector pose `x = (p, R) ∈ ℝ³ × SO(3)` — 3 position dimensions + 3 orientation dimensions = **6-DoF in EE space**.

**Mismatch.** `dim(q) = 7 > dim(x) = 6`. The arm is **kinematically redundant**: for almost any target EE pose there is a 1-parameter family of joint configurations achieving it. The free parameter is physically the "elbow swivel" — rotate the elbow around the shoulder-wrist line without disturbing the gripper.

This 1-D redundancy is the root cause of every adapter pathology in our writeup. Every adapter has to either (a) commit to a specific branch of this 1-D family, (b) let the controller pick it at runtime, or (c) live with the resulting ambiguity in training. The four adapters in our writeup correspond exactly to those three strategies.

## 2. Forward kinematics

`FK: ℝ⁷ → ℝ³ × SO(3)` defined by chained rigid-body transforms along the kinematic chain:

```
T_world_eef(q) = T_world_base · ∏_{i=1..7} T_{i-1, i}(q_i)
```

Each `T_{i-1,i}(q_i)` is a homogeneous transform parameterized by one revolute-joint angle (all Panda joints are revolute). Multiplying the chain gives the 4×4 transform of the EE frame in the world.

In our runner: a **standalone mujoco model** of the Panda (no gripper, no contacts) does this. One call to `mj_kinematics(model, data)` propagates `data.qpos[:7] = q` through the chain and writes `(data.xpos[eef_body], data.xmat[eef_body])` into a 3-D position + 3×3 rotation matrix. Pure FK, no physics; ~5 μs per call.

We use the **standalone-XML base frame** — call it the "panda" frame. A separate calibration (§4) converts that to the robot's actual mount frame in world.

## 3. The Jacobian and the 1-D null space

The differential FK is the manipulator Jacobian:

```
ẋ = J(q) · q̇,    J(q) ∈ ℝ⁶ˣ⁷
```

Stacking linear and angular parts:

```
       ⎡ v ⎤   ⎡ J_v(q) ⎤
ẋ  =  ⎢   ⎥ = ⎢        ⎥ q̇
       ⎣ ω ⎦   ⎣ J_w(q) ⎦
```

where `v ∈ ℝ³` is EE linear velocity, `ω ∈ ℝ³` is EE angular velocity (body twist).

**Rank theorem.** Away from kinematic singularities, `rank(J(q)) = 6`. By rank-nullity:

```
dim ker J(q) = 7 − 6 = 1
```

So there exists `n(q) ∈ ℝ⁷` (unique up to scale) with `J(q) · n(q) = 0`. Joint velocity along `n(q)` produces **zero EE velocity** — that's the elbow swivel.

**Pseudo-inverse and projectors.** Define the Moore–Penrose right pseudo-inverse:

```
J⁺(q) = J(q)ᵀ · (J(q) · J(q)ᵀ)⁻¹    ∈ ℝ⁷ˣ⁶          (well-defined since J has full row rank)
```

It satisfies `J · J⁺ = I_6` (right-inverse on EE space) but `J⁺ · J ≠ I_7` (not a left-inverse on joint space). The deviation defines two complementary orthogonal projectors:

```
P_range(q)    =  J⁺(q) · J(q)        rank 6, projects q̇ onto the "row space of J" — the directions that move the EE
P_null(q)     =  I_7 − J⁺(q) · J(q)  rank 1, projects q̇ onto ker J — the elbow-swivel direction
```

`P_null(q)` is exactly the 1-D null-space projector. Any joint velocity `q̇ ∈ ℝ⁷` decomposes uniquely:

```
q̇ = P_range(q) · q̇ + P_null(q) · q̇
  = (component that moves the EE) + (component that only moves the elbow)
```

For a given desired EE velocity `ẋ_des ∈ ℝ⁶`, the minimum-norm joint solution is `q̇ = J⁺ · ẋ_des`. The full solution family is:

```
q̇ = J⁺ · ẋ_des + P_null · γ        for arbitrary γ ∈ ℝ⁷               (3.1)
```

Equation (3.1) is the entire game. **Forward** (`q̇ → ẋ`) collapses 7-D to 6-D by dropping the `P_null · γ` term. **Inverse** (`ẋ → q̇`) is underdetermined by exactly that 1-D family.

## 4. Forward direction: joint trajectory → EE trajectory

Given a chunked joint trajectory `q_target[0..T]` from the policy (with `q_target[t] = q_curr + Σ_{s≤t} Δq[s]`), the EE trajectory is just FK applied per timestep:

```
p_target_panda[t] = FK_pos(q_target[t])
R_target_panda[t] = FK_rot(q_target[t])
```

Per-step deltas in the panda base frame:

```
Δp_panda[t] = p_target_panda[t] − p_target_panda[t−1]                          (3-vec)
ΔR_panda[t] = R_target_panda[t] · R_target_panda[t−1]ᵀ                        (3×3 rotation)
Δr_panda[t] = log(ΔR_panda[t])         (axis-angle, 3-vec via Rodrigues)
```

These are **uniquely determined** — FK is a function, no ambiguity. The 1-D nullspace component of `Δq` projects to zero in EE space by definition of `P_null`, so the EE trajectory `(Δp, Δr)` carries only the 6 dimensions of information in `P_range · Δq`; the seventh joint dimension is **silently dropped** by FK.

## 5. World ↔ panda-base calibration

The standalone Panda XML's "world" frame is the panda **base** frame (mount origin, identity rotation). The actual robosuite env mounts the Panda somewhere in the world with some pose. For single-arm tasks the mount is identity-rotated → frames coincide. For TwoArmTransport, the two arms are mounted facing each other → ±90° z-rotation each.

Let `T_world_pandabase ∈ SE(3)` be the fixed mount transform. At the first env reset we observe one (q₀, p_env₀, R_env₀) from obs:

```
R_env₀ = R_world_pandabase · R_FK_panda(q₀)
```

Solving for the rotation part:

```
R_world_pandabase = R_env₀ · R_FK_panda(q₀)⁻¹                                  (5.1)
```

implemented as `R_world_panda = R_env0 @ R_fk0.T`. Logged per arm at rollout start (`arm0 world<-panda z-angle ≈ X°`).

To rotate the per-step deltas from panda frame to world frame:

```
Δp_world[t] = R_world_pandabase · Δp_panda[t]                                  (5.2)
Δr_world[t] = R_world_pandabase · Δr_panda[t]                                  (5.3)
```

Equation (5.3) uses the fact that axis-angle vectors transform like ordinary vectors under a frame change: if a rotation in frame A has axis-angle representation `α`, the same rotation in frame B has representation `R_BA · α`.

**Why (5.2) is exact even with a nonzero mount translation**: `Δp_panda` is a *displacement* (difference of two positions), so the translation component of `T_world_pandabase` cancels and only the rotation matters.

## 6. OSC normalization and command

The `OSC_POSE` controller takes a 7-D action `[Δp_n, Δr_n, grip]` in normalized `[−1, 1]⁷`, then internally scales by `output_max = [0.05 m, 0.05 m, 0.05 m, 0.5 rad, 0.5 rad, 0.5 rad]` to get a physical delta target. We invert that scaling on our side:

```
Δp_n[t] = clip(Δp_world[t] / 0.05, −1, 1)
Δr_n[t] = clip(Δr_world[t] / 0.5,  −1, 1)
osc_cmd[t] = [Δp_n[t], Δr_n[t], gripper[t]]                                    (7-vec per arm)
```

For two-arm transport we concatenate per-arm OSC commands `[osc_arm0, osc_arm1]` (14-dim total).

## 7. OSC dynamics: where the nullspace gets filled

OSC implements an **operational-space impedance** controller. Given the commanded EE pose target `x_target = x_current + Δx_cmd`, it computes a wrench `F ∈ ℝ⁶`:

```
F = K_p · (x_target − x_current) + K_d · ẋ_current                            (operational-space PD)
```

It then projects to joint torques via Jacobian-transpose for the task part, and adds a posture term in the null space:

```
τ_task    = J(q)ᵀ · F                                                          (Jacobian-transpose)
τ_posture = P_null(q)ᵀ · τ_posture_des                                          (nullspace projection of secondary task)
τ         = τ_task + τ_posture                                                  (combined torque)
```

Here `τ_posture_des` is the secondary-task torque (default robosuite: drive `q` toward a fixed reference posture `q_ref` with PD: `τ_posture_des = K_p^post · (q_ref − q) − K_d^post · q̇`). The projector `P_null(q)ᵀ = (I_7 − J⁺ J)ᵀ` ensures `τ_posture` lives in the joint directions that don't disturb the EE — i.e. the 1-D nullspace direction `n(q)` from §3.

**This is the crucial coupling.** OSC accepts our 6-D EE delta command, executes it via Jacobian-transpose in the 6-D EE-relevant subspace, and fills in the elbow-swivel motion from its own posture controller in the 1-D nullspace. We never have to specify what the elbow should do; OSC picks something reasonable.

**Why FK→OSC works without inverting OSC**:
- The 6 dimensions OSC reads (the `Δx` command) are exactly what FK produced from the policy's `Δq` trajectory.
- The 1 dimension OSC fills (the nullspace torque) is one we never had access to in EE space anyway — the FK map silently dropped it.
- The policy *implicitly* chose some nullspace component (by predicting `Δq` rather than `Δx`), but we discard that implicit choice and let OSC's posture controller make its own choice at runtime.

Empirically this is consistent because the demo trajectories were OSC-generated to begin with — the demo's "intended" nullspace component is whatever OSC's posture controller naturally produces from the same starting state. Replaying that gives a consistent trajectory.

## 8. Failure modes (where the math breaks)

Three places the FK→OSC pipeline can fail:

1. **Controller tracking lag.** OSC's PD with finite `K_p` can't perfectly track a target that's moving at 50 ms / step. The realized motion lags the commanded one by a few ms. We bump `K_p` from the Robomimic default 150 to 1000 (lift / can) or 3000 (square) to close this gap. Higher `K_p` → tighter tracking, but past a point the controller becomes stiff and contact-unsafe.

2. **Kinematic singularity.** When `q` is near a singularity (e.g. arm fully extended), `J(q)` loses rank, `J⁺` blows up, and the Cartesian command requires unbounded joint torques in some direction. OSC saturates and stalls. Our FK trajectory can pass through singular regions if the policy's `Δq` does; OSC clips.

3. **Posture mismatch.** OSC's default posture controller drives toward a fixed `q_ref`. If the demo's chosen nullspace component disagrees with this default (e.g. the demo wanted "elbow up" but OSC pulls "elbow down"), the elbow drifts off the demo trajectory even though the EE pose tracks. On tight workspaces (square: peg threading; transport: arms in narrow gap) this can crash the elbow into geometry the joint-space demo avoided.

Failure 1 is tunable via `K_p`. Failures 2–3 are structural for FK→OSC and explain why **square ceilings at 0.50 (kp=3000)** and **transport ceilings at 0.00** despite a well-trained joint-delta policy.

## 9. Contrast — FK→JP (Brian's analytic JP adapter)

JP controller accepts a 7-D **absolute joint target** `q_cmd ∈ ℝ⁷`. The forward dynamics are independent per-joint PD:

```
τ_i = k_p_i (q_cmd_i − q_i) − k_d_i · q̇_i,    for i ∈ 1..7   (no coupling at command interface)
```

No Jacobian, no operational-space projection, **no nullspace projector**. The adapter is trivial:

```
q_cmd[t] = q_curr + cumsum(Δq[0..t])           (analytic integration in joint space)
```

This is what Brian called "FK→JP" but is really just **integration → JP**. No FK is needed because we're not converting to a different action space — `Δq` is already the right currency.

**Why his analytic FK→JP fails at full pipeline (his Figure 5: Can/Square 0/50, Lift nonzero only)**:
- JP's per-joint PD is **open-loop with respect to the EE pose**. Small joint-tracking errors don't get corrected via Cartesian feedback; they integrate over time.
- The DP's `Δq` predictions have ~10–30 mrad noise per joint. JP's PD tries to track each prediction's `q_cmd`, but the realized joint trajectory drifts in 7-D, and the EE wanders relative to the demo trajectory.
- OSC, by contrast, gets *direct Cartesian feedback* — even with noisy joint targets, the OSC-realized EE position stays close to the FK-projected target because the impedance term `K_p · (x_target − x_current)` measures error in the task-relevant space.

So:
- **FK→OSC (ours)**: closed-loop in EE pose, the task-relevant subspace. Joint nullspace drifts allowed; EE stays on target. Works on lift / can / square.
- **FK→JP (Brian's)**: closed-loop in joints, open-loop in EE pose. Joint errors compound, EE drifts. Fails on all but lift.

**Why Brian's learned NN→JP works (his Figure 4, our Section D)**: the learned MLP doesn't just integrate — it predicts `q_cmd` from `(state, Δq_des)`. The "state" input lets the MLP correct for accumulated PD-tracking error by reading current `q` and pulling `q_cmd` toward where it actually needs to be. It implicitly learns a closed-loop correction that the analytic integrator lacks.

## 10. Why NN→OSC fails: the rank-deficient inverse

The NN→OSC adapter is asked to learn

```
f: (state, Δq_des) → OSC_cmd                                                  (8.1)
```

For each `(state, Δq_des)` we'd want `f` to output the OSC command whose forward map produces `Δq_des`. But the forward OSC map is

```
Δq = J(q)⁺ · OSC_cmd_scaled + P_null(q) · q̇_posture_response · Δt              (8.2)
```

where `OSC_cmd_scaled = output_max ⊙ OSC_cmd` and `q̇_posture_response` is the realized joint motion from `τ_posture`. The second term is the **unobservable** part — it's a function of the OSC controller's internal posture state, which the adapter never sees.

For a given `Δq` there are infinitely many `OSC_cmd` solutions parameterized by the unobserved posture response. The probe-based sampler trains on synthetic `(state, OSC_cmd, Δq)` triples where the second term varies with state and posture in a way the network can't reconstruct.

**MSE training averages branches:**

```
L(θ) = E_{state, Δq_des} [ ‖f_θ(state, Δq_des) − OSC_cmd_target‖² ]
```

If multiple `OSC_cmd_target` values pair with the same `(state, Δq_des)` (which happens because of (8.2)), MSE drives `f_θ` to the *mean* of those targets. The mean of a multi-modal distribution in `[−1, 1]^6` is just noise centered at the projected center — relative MAE ≈ 1 (which is what we observed for our probe NN→OSC, and what Brian's Table 1 shows for our adapter where his NN→JP stays at 0.57–0.98).

**The demo-supervised NN→OSC works** because demos resolve the branch: every `(state, Δq_demo)` is paired with the specific `OSC_cmd_demo` that teleop happened to produce. The MLP only has to memorize teleop's branch choice, which is *consistent across the demo manifold*. Off-manifold it'd be broken; the policy mostly stays on-manifold so it doesn't matter in practice.

## 11. Why NN→JP succeeds: well-posed inverse

For JP the forward map is component-wise diagonal:

```
q_realized_i = q_cmd_i + (PD-tracking residual)_i                              (9.1)
```

No coupling, no nullspace projector. The inverse is essentially `q_cmd_i = q_realized_i − residual_i`, which is almost identity scale + small correction. A learned MLP converges to that mapping in a few epochs because:

- The function class is well-conditioned (near-identity).
- The PD residual is a deterministic function of `(state, Δq)`, so it can be learned.
- There's no second underdetermined term to memorize.

Brian's MLP on Can MH reaches `relative MAE = 0.767` (above) and full-pipeline `50/50 success` at the same training budget our probe NN→OSC settles at `relative MAE ≈ 1` and `0/50` success. Same data volume, same architecture, same sampler — the only difference is *which inverse* the network is asked to compute.

## 12. Picture in one paragraph

OSC's 6-D command interface throws away the 1-D Panda kinematic redundancy by construction. A learned inverse on OSC has to put it back, which is an ill-posed regression on synthetic probe data and a memorization task on demo-only data. JP's 7-D command interface preserves all joint dimensions explicitly, so the learned inverse is well-posed regardless of data source. FK→OSC sidesteps the entire issue by never inverting OSC — it forwards through FK and lets OSC choose its own nullspace component at runtime, accepting that the policy's implicit nullspace intent is silently discarded. FK→JP also avoids the inverse problem, but JP's per-joint PD provides no Cartesian closed-loop feedback, so joint-target noise integrates into EE-pose drift and the pipeline fails on everything except trivial tasks. The four adapters in our writeup are exactly the four points in the 2×2 design space `{learned vs analytic} × {Cartesian-controller vs joint-controller}`.

## 13. Comparison table

| Adapter | Map | Closed-loop on | Null-space handling | Inverse problem | Empirically |
|---|---|---|---|---|---|
| **FK→OSC** (ours) | `Δq → Δx_world` via FK, send to OSC | EE pose (Cartesian) | Dropped by FK; OSC's posture controller fills it at runtime | None (forward map only) | lift 0.94, can 0.88, square 0.50; transport 0.00 (bug); tool_hang 0.00 |
| **FK→JP** (Brian's analytic) | `Δq → q_cmd = q_curr + Δq`, send to JP | Joints only (per-joint PD) | None needed; all 7 joints commanded | None (forward map only) | Lift nonzero; Can/Square 0/50 — joint errors → EE drift |
| **NN→OSC (probe)** | learned `(state, Δq) → OSC_cmd` | EE pose | Tries to predict OSC's posture choice → fails | Rank-deficient by 1; underdetermined | rel MAE ≈ 1, full-pipeline ≈ 0/50 — branch ambiguity |
| **NN→OSC (demo-supervised)** | learned `(state, Δq) → OSC_cmd` on demos | EE pose | Memorizes teleop's branch (one-branch consistent) | Resolved per-state by demo support | lift 0.90, can 0.64, square 0.48 — works on-manifold |
| **NN→JP** (Brian's learned) | learned `(state, Δq) → q_cmd` | Joints | None needed | Well-posed, near-identity | Can MH 50/50 — trivial inverse + closed-loop correction |

The FK→OSC adapter wins on **lift/can/square** because the closed-loop Cartesian feedback is exactly what those tasks need. The FK→JP adapter fails because there's no Cartesian feedback. The NN→OSC adapter fails on synthetic data because the inverse it's asked to learn is rank-deficient by exactly 1. The NN→JP adapter succeeds because its inverse is well-posed.

The whole story is the 1-D null space of the manipulator Jacobian, and which adapter pretends it isn't there.
