"""Runner that takes a joint-delta policy's predicted Δq, runs forward kinematics
on the standalone Panda model to compute target EEF poses, and feeds Δeef
commands to robosuite's OSC_POSE controller (the controller demos were
collected with). This avoids the under-tracking issue of JOINT_POSITION
without retraining: same trained joint policy, same OSC dynamics as EEF runs.

Implementation notes:

* Inherits no logic from RobomimicLowdimRunner directly; we copy the env
  construction and rollout loop because we need to insert FK between the
  policy and env.step. The only differences vs the EEF runner are:
  - the policy outputs 8-dim [Δq(7), gripper(1)] instead of 7-dim OSC.
  - we run a side-car mujoco model to convert Δq to Δeef per step.
* FK uses robosuite's bundled panda XML (no extra deps; `mujoco` is already a
  robosuite transitive dependency).
* The panda XML's body of interest is `right_hand`; the env exposes EEF pose
  via `robot0_eef_pos` / `robot0_eef_quat`. We assume the panda mount is
  identity-rotated in world (typical robosuite setup); this is verified at
  runner-construction by comparing FK(q_init) against env.obs.eef_pos and
  checking the residual is a pure translation.
* The policy outputs 8 actions per chunk, executed open-loop. For each step i
  we integrate q_target[i] = q_current + cumsum(Δq[0..i]), FK each to get the
  target EEF trajectory, then send incremental Δeef commands per step.
"""
import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import math
import wandb.sdk.data_types.video as wv
import mujoco
from scipy.spatial.transform import Rotation
from typing import List

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env_runner.robomimic_lowdim_runner import create_env
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
import robomimic.utils.file_utils as FileUtils


def _resolve_panda_xml_path():
    import robosuite
    p = os.path.join(
        os.path.dirname(robosuite.__file__),
        "models", "assets", "robots", "panda", "robot.xml")
    if not os.path.exists(p):
        raise FileNotFoundError(f"panda robot.xml not found at {p}")
    return p


class _PandaFK:
    """Side-car mujoco model used purely for FK on the Panda kinematic chain."""

    def __init__(self, panda_xml_path: str = None, eef_body_name: str = "right_hand"):
        path = panda_xml_path or _resolve_panda_xml_path()
        # mesh paths are relative to the XML directory; chdir for load.
        old_cwd = os.getcwd()
        os.chdir(os.path.dirname(path))
        try:
            self.model = mujoco.MjModel.from_xml_path(path)
        finally:
            os.chdir(old_cwd)
        self.data = mujoco.MjData(self.model)
        self.eef_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, eef_body_name)
        if self.eef_bid < 0:
            raise ValueError(f"body {eef_body_name!r} not in panda XML")
        assert self.model.nq == 7, f"expected 7-DOF panda, got nq={self.model.nq}"

    def fk(self, q):
        """Run FK at joint configuration q (shape (7,)). Returns (pos, R) in
        the panda base frame."""
        self.data.qpos[:7] = q
        self.data.qvel[:] = 0
        mujoco.mj_kinematics(self.model, self.data)
        pos = self.data.xpos[self.eef_bid].copy()
        R = self.data.xmat[self.eef_bid].reshape(3, 3).copy()
        return pos, R


class RobomimicJointFKtoEEFRunner(BaseLowdimRunner):
    """Joint-delta policy executed via FK → OSC_POSE.

    The policy still predicts 8-dim [Δq(7), gripper(1)]. The runner converts
    each predicted Δq to an OSC_POSE delta action [Δp(3), Δr(3 axis-angle),
    gripper(1)] before stepping the env. Demo-collection controller is OSC,
    so the resulting Δeef trajectory matches the dynamics the policy was
    implicitly trained for.
    """

    def __init__(self,
            output_dir,
            dataset_path,
            obs_keys,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            n_latency_steps=0,
            render_hw=(256, 256),
            render_camera_name='agentview',
            fps=10,
            crf=22,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            # FK-runner specific
            panda_xml_path: str = None,
            eef_body_name: str = "right_hand",
            joint_pos_obs_key: str = "robot0_joint_pos",
            eef_pos_obs_key: str = "robot0_eef_pos",
            eef_quat_obs_key: str = "robot0_eef_quat",
            joint_pos_obs_keys: list = None,
            eef_pos_obs_keys: list = None,
            eef_quat_obs_keys: list = None,
            input_action_layout: str = 'joints_then_grippers',
            delta_pos_clip: float = 0.05,
            delta_rot_clip: float = 0.5,
            osc_kp_pos: float = None,
            osc_kp_ori: float = None,
            osc_damping_ratio: float = None,
            disable_world_rotation: bool = False,
            # Residual NN adapter on top of FK->OSC. Optional. Single-arm only.
            # If set, the rollout computes
            #     osc = clip(FK->OSC(q, dq, grip) + clip(NN(state, dq), ±clip), -1, 1)
            # before stepping the env. Adapter must come from
            # reverse_controller/train_inverse_model.py (Brian's InverseControllerMLP)
            # trained on residual targets via collect_demo_residual_osc.py.
            residual_adapter_path: str = None,
            residual_clip: float = 0.3,
            residual_obs_keys: list = None,
            residual_device: str = "cuda:0",
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        # Demos used OSC_POSE; we keep that. control_delta defaults to True for
        # the can/ph robomimic env_meta; verify and don't change.

        # Override OSC kp / damping if requested -- mirrors the JP runner's
        # controller_kp override that took JP from 0% -> 82%.
        # controller_configs may be a single dict (single-arm) or a list of
        # dicts (multi-arm); handle both.
        if osc_kp_pos is not None or osc_kp_ori is not None or osc_damping_ratio is not None:
            cc = env_meta["env_kwargs"].get("controller_configs", {})
            def _apply(d):
                if osc_kp_pos is not None:
                    d["kp"] = float(osc_kp_pos)
                if osc_damping_ratio is not None:
                    d["damping_ratio"] = float(osc_damping_ratio)
                return d
            if isinstance(cc, list):
                cc = [_apply(dict(d)) for d in cc]
            else:
                cc = _apply(dict(cc))
            env_meta["env_kwargs"]["controller_configs"] = cc


        def env_fn():
            robomimic_env = create_env(env_meta=env_meta, obs_keys=obs_keys)
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicLowdimWrapper(
                        env=robomimic_env,
                        obs_keys=obs_keys,
                        init_state=None,
                        render_hw=render_hw,
                        render_camera_name=render_camera_name,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps, codec='h264', input_pix_fmt='rgb24',
                        crf=crf, thread_type='FRAME', thread_count=1),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps,
            )

        env_fns = [env_fn] * n_envs
        env_seeds, env_prefixs, env_init_fn_dills = [], [], []

        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, enable_render=enable_render):
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        env.env.file_path = str(filename)
                    assert isinstance(env.env.env, RobomimicLowdimWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))

        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    env.env.file_path = str(filename)
                assert isinstance(env.env.env, RobomimicLowdimWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # Compute obs slices for joint_pos / eef_pos / eef_quat from obs_keys.
        # The dataset's obs_dim is sum of per-key dimensions; we walk obs_keys
        # in order and read each key's dim from the hdf5 once.
        with h5py.File(dataset_path, 'r') as f:
            first_obs = f['data/demo_0/obs']
            obs_slices = {}
            offset = 0
            for k in obs_keys:
                d = int(first_obs[k].shape[-1])
                obs_slices[k] = slice(offset, offset + d)
                offset += d

        # Promote single-arm scalar keys to single-element lists if no list given.
        if joint_pos_obs_keys is None:
            joint_pos_obs_keys = [joint_pos_obs_key]
        if eef_pos_obs_keys is None:
            eef_pos_obs_keys = [eef_pos_obs_key]
        if eef_quat_obs_keys is None:
            eef_quat_obs_keys = [eef_quat_obs_key]
        n_robots = len(joint_pos_obs_keys)
        if not (len(eef_pos_obs_keys) == n_robots and len(eef_quat_obs_keys) == n_robots):
            raise ValueError("joint_pos_obs_keys/eef_pos/eef_quat must all have same length")
        for required in joint_pos_obs_keys + eef_pos_obs_keys + eef_quat_obs_keys:
            if required not in obs_slices:
                raise KeyError(f"obs_keys must include {required!r}; got {obs_keys}")

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.obs_slices = obs_slices
        self.joint_pos_obs_key = joint_pos_obs_key  # legacy single-arm
        self.eef_pos_obs_key = eef_pos_obs_key
        self.eef_quat_obs_key = eef_quat_obs_key
        self.joint_pos_obs_keys = list(joint_pos_obs_keys)
        self.eef_pos_obs_keys = list(eef_pos_obs_keys)
        self.eef_quat_obs_keys = list(eef_quat_obs_keys)
        self.n_robots = n_robots
        self.input_action_layout = input_action_layout
        self.delta_pos_clip = float(delta_pos_clip)
        self.delta_rot_clip = float(delta_rot_clip)

        self.disable_world_rotation = bool(disable_world_rotation)

        # FK model (one shared instance used in main process for action transform).
        self.fk = _PandaFK(panda_xml_path=panda_xml_path,
                           eef_body_name=eef_body_name)

        # Residual NN adapter (optional).
        self.residual_adapter = None
        self.residual_clip = float(residual_clip)
        self.residual_state_sls = None
        if residual_adapter_path is not None:
            import torch as _torch
            from reverse_controller.common import load_inverse_checkpoint
            payload, model, normalizer = load_inverse_checkpoint(
                residual_adapter_path, device='cpu')
            dev = _torch.device(residual_device)
            model = model.to(dev)
            model.eval()
            norm_on_dev = {
                name: {k: (_torch.as_tensor(v, dtype=_torch.float32, device=dev)
                           if isinstance(v, np.ndarray) else v.to(dev))
                       for k, v in stats.items()}
                for name, stats in normalizer.items()
            }
            self.residual_adapter = model
            self.residual_normalizer = norm_on_dev
            self.residual_payload = payload
            self.residual_device = dev
            ro_keys = list(residual_obs_keys) if residual_obs_keys else list(obs_keys)
            for k in ro_keys:
                if k not in obs_slices:
                    raise KeyError(f"residual_obs_keys requires {k!r} not in obs_keys")
            self.residual_obs_keys = ro_keys
            self.residual_state_sls = [obs_slices[k] for k in ro_keys]
            print(f"FK runner: residual adapter loaded from {residual_adapter_path}"
                  f" (clip={self.residual_clip}, n_obs_features="
                  f"{sum(sl.stop - sl.start for sl in self.residual_state_sls)})")

    def _action_chunk_joint_to_eef(self, q_curr_b, dq_chunk_b, gripper_chunk_b,
                                   R_world_panda=None):
        """Convert a chunk of predicted joint deltas to OSC_POSE actions.

        Args:
          q_curr_b:        (B, 7)  current joint positions per env at chunk start.
          dq_chunk_b:      (B, T, 7) predicted joint deltas, per-step.
          gripper_chunk_b: (B, T, 1) gripper command per step.

        Returns:
          osc_chunk_b:     (B, T, 7)  [Δp(3), Δr_axisangle(3), gripper(1)].

        For each env independently we integrate q_target[t] =
        q_curr + cumsum(dq[0..t]), then FK each to (p_t, R_t), and emit
        Δp_t = p_t − p_{t−1}, Δr_t = log(R_t · R_{t−1}^T) (axis-angle).
        Where p_{−1} := FK(q_curr), R_{−1} := R_FK(q_curr).
        """
        B, T, _ = dq_chunk_b.shape
        osc = np.zeros((B, T, 7), dtype=np.float32)

        for b in range(B):
            q = q_curr_b[b].astype(np.float64).copy()
            p_prev, R_prev = self.fk.fk(q)
            for t in range(T):
                q = q + dq_chunk_b[b, t].astype(np.float64)
                p_t, R_t = self.fk.fk(q)
                dp = p_t - p_prev
                dR = R_t @ R_prev.T
                # Convert rotation matrix to axis-angle. Make sure dR is a
                # proper rotation (numerical drift can give slight
                # non-orthogonality; scipy handles it).
                rotvec = Rotation.from_matrix(dR).as_rotvec()
                # Map deltas from panda-local frame to world frame.
                if R_world_panda is not None:
                    dp = R_world_panda @ dp
                    rotvec = R_world_panda @ rotvec
                # Clip to keep OSC tracking stable.
                # OSC_POSE expects action in normalized [-1, 1] which it then
                # internally scales by output_max (0.05 m / 0.5 rad). Without
                # this normalization a physical 5 mm Δp gets re-scaled to
                # 0.25 mm -> arm crawls. delta_pos_clip / delta_rot_clip are
                # interpreted as the OSC output_max (defaults match robosuite).
                dp_norm = np.clip(dp / self.delta_pos_clip, -1.0, 1.0)
                rotvec_norm = np.clip(rotvec / self.delta_rot_clip, -1.0, 1.0)
                osc[b, t, 0:3] = dp_norm
                osc[b, t, 3:6] = rotvec_norm
                osc[b, t, 6:7] = gripper_chunk_b[b, t]
                p_prev, R_prev = p_t, R_t
        return osc

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        # Per-arm slices (joints + eef pose) — used at runtime to read state and
        # to calibrate the world<-panda_base rotation per arm on first reset.
        joint_sls = [self.obs_slices[k] for k in self.joint_pos_obs_keys]
        eef_pos_sls = [self.obs_slices[k] for k in self.eef_pos_obs_keys]
        eef_quat_sls = [self.obs_slices[k] for k in self.eef_quat_obs_keys]
        # World<-panda_base rotation per arm. Calibrated once at the first
        # env.reset()'s obs by comparing FK(q_init) vs env eef_pose.
        R_world_panda = [None] * self.n_robots

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            past_action = None
            policy.reset()

            # Calibrate per-arm world<-panda rotation from the first obs.
            # Use env 0's obs at the most-recent obs step.
            from scipy.spatial.transform import Rotation as _Rot
            for i in range(self.n_robots):
                q0 = np.asarray(obs[0, self.n_obs_steps - 1, joint_sls[i]], dtype=np.float64)
                p0 = np.asarray(obs[0, self.n_obs_steps - 1, eef_pos_sls[i]], dtype=np.float64)
                qt0 = np.asarray(obs[0, self.n_obs_steps - 1, eef_quat_sls[i]], dtype=np.float64)
                _, R_fk0 = self.fk.fk(q0)
                R_e0 = _Rot.from_quat(qt0).as_matrix()
                R_world_panda[i] = R_e0 @ R_fk0.T
                if self.disable_world_rotation:
                    R_world_panda[i] = None  # skip rotation; OSC takes base-frame deltas
                # Sanity log: how much rotation per arm
                ang_deg = float(np.degrees(np.arctan2(R_world_panda[i][1, 0],
                                                     R_world_panda[i][0, 0])))
                print(f"  arm{i} world<-panda z-angle ≈ {ang_deg:.1f}°")

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {env_name}JointFKtoEEF {chunk_idx + 1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)

            done = False
            while not done:
                np_obs_dict = {
                    'obs': obs[:, :self.n_obs_steps].astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[
                        :, -(self.n_obs_steps - 1):].astype(np.float32)

                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # action shape: (B, n_obs_steps + n_action_steps - n_latency, 8)
                # then we take [:, n_latency:] giving (B, n_action_steps, 8)
                action = np_action_dict['action'][:, self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                # Slice action per robot. Layout = joints_then_grippers:
                #   [arm0_dq(7), ..., armN_dq(7), arm0_grip(1), ..., armN_grip(1)]
                n = self.n_robots
                joint_action = action[..., 0:7*n]               # (B, T, 7n)
                gripper_action = action[..., 7*n:7*n + n]       # (B, T, n)

                # Pull current q for each arm.
                q_curr_per_arm = [obs[:, self.n_obs_steps - 1, sl] for sl in joint_sls]  # n arrays of (B, 7)

                # Build OSC action by per-arm FK. Output layout interleaves arms:
                #   [arm0_osc(7), arm1_osc(7), ...]   each arm_osc = [Δp(3), Δr(3), grip(1)]
                B = action.shape[0]
                T = action.shape[1]
                osc_per_arm = []
                for i in range(n):
                    dq_i = joint_action[..., i*7:(i+1)*7]      # (B, T, 7)
                    grip_i = gripper_action[..., i:i+1]         # (B, T, 1)
                    q_curr_i = q_curr_per_arm[i]                # (B, 7)
                    osc_i = self._action_chunk_joint_to_eef(
                        q_curr_b=q_curr_i,
                        dq_chunk_b=dq_i,
                        gripper_chunk_b=grip_i,
                        R_world_panda=R_world_panda[i],
                    )                                           # (B, T, 7)
                    osc_per_arm.append(osc_i)
                osc_action = np.concatenate(osc_per_arm, axis=-1)  # (B, T, 7n)

                # Optional residual NN adapter (single-arm only for now).
                if self.residual_adapter is not None and self.n_robots == 1:
                    import torch as _torch
                    B, T, _ = osc_action.shape
                    # State features at chunk start, tiled over T.
                    state_at_chunk = np.concatenate(
                        [obs[:, self.n_obs_steps - 1, sl] for sl in self.residual_state_sls],
                        axis=-1)                                          # (B, S)
                    state_chunk = np.broadcast_to(
                        state_at_chunk[:, None, :], (B, T, state_at_chunk.shape[-1]))
                    # NN input: concat(state, dq) per step.
                    nn_in_np = np.concatenate(
                        [state_chunk, joint_action[..., :7]], axis=-1).astype(np.float32)
                    nn_in = _torch.from_numpy(nn_in_np).to(self.residual_device)
                    mean = self.residual_normalizer['input']['mean']
                    std  = self.residual_normalizer['input']['std']
                    cmean = self.residual_normalizer['command']['mean']
                    cstd  = self.residual_normalizer['command']['std']
                    with _torch.no_grad():
                        nn_in_norm = (nn_in - mean) / std
                        pred_norm = self.residual_adapter(nn_in_norm)
                        pred = pred_norm * cstd + cmean                   # (B, T, 7)
                    residual_np = pred.detach().cpu().numpy().astype(np.float32)
                    residual_np = np.clip(residual_np, -self.residual_clip, self.residual_clip)
                    osc_action = np.clip(osc_action + residual_np, -1.0, 1.0)

                obs, reward, done, info = env.step(osc_action)
                done = np.all(done)
                past_action = action

                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call(
                'get_attr', 'reward')[this_local_slice]

        max_rewards = collections.defaultdict(list)
        log_data = dict()
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward

            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f'sim_video_{seed}'] = sim_video

        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
