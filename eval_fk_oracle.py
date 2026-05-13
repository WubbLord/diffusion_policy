"""FK→OSC adapter oracle replay (no diffusion policy).

For each held-out demo, walk the recorded joint trajectory, compute
Δq[t] = q[t+1] − q[t] from the demo's own joint_pos observations, push
through the FK→OSC adapter (same math as RobomimicJointFKtoEEFRunner), and
step the env. Reports test/mean_score over n_test demos — directly
comparable to Brian's blog Table 1 (NN→JP oracle replay).

Usage:
    python eval_fk_oracle.py --task can --osc_kp 1000 --output_dir ...
    python eval_fk_oracle.py --task transport --osc_kp 1000 --output_dir ...
"""
import os
import sys
import json
import pathlib
import click
import h5py
import numpy as np
import dill
import torch
import mujoco
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import robosuite
import robomimic.utils.file_utils as FU
import robomimic.utils.env_utils as EU
import robomimic.utils.obs_utils as OU


def _resolve_panda_xml():
    return os.path.join(os.path.dirname(robosuite.__file__),
                        "models", "assets", "robots", "panda", "robot.xml")


class _PandaFK:
    def __init__(self):
        xml = _resolve_panda_xml()
        cwd = os.getcwd(); os.chdir(os.path.dirname(xml))
        self.model = mujoco.MjModel.from_xml_path(xml)
        os.chdir(cwd)
        self.data = mujoco.MjData(self.model)
        self.bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand")

    def fk(self, q):
        self.data.qpos[:7] = q; self.data.qvel[:] = 0
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.xpos[self.bid].copy(), self.data.xmat[self.bid].reshape(3, 3).copy()


def _fk_to_osc(q_curr, dq, R_world_panda, fk_model,
               dp_clip=0.05, dr_clip=0.5):
    """Convert (q_curr, dq) -> (Δp_world_norm, Δr_world_norm)."""
    p_prev, R_prev = fk_model.fk(q_curr)
    p_t, R_t = fk_model.fk(q_curr + dq)
    dp_panda = p_t - p_prev
    dR_panda = R_t @ R_prev.T
    rotvec_panda = Rotation.from_matrix(dR_panda).as_rotvec()
    dp_world = R_world_panda @ dp_panda
    dr_world = R_world_panda @ rotvec_panda
    return (np.clip(dp_world / dp_clip, -1.0, 1.0),
            np.clip(dr_world / dr_clip, -1.0, 1.0))


@click.command()
@click.option('--task', required=True,
              type=click.Choice(['lift', 'can', 'square', 'tool_hang', 'transport']))
@click.option('--split', default='ph', type=click.Choice(['ph', 'mh']))
@click.option('--osc_kp', type=float, default=1000.0)
@click.option('--osc_damping_ratio', type=float, default=1.0)
@click.option('--n_test', type=int, default=50)
@click.option('--test_start_seed', type=int, default=100000)
@click.option('--output_dir', required=True)
def main(task, split, osc_kp, osc_damping_ratio, n_test, test_start_seed, output_dir):
    dataset = f'data/robomimic/datasets/{task}/{split}/low_dim.hdf5'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    em = FU.get_env_metadata_from_dataset(dataset)
    # OSC kp override (per-arm-broadcast if dict-form)
    cc = em['env_kwargs'].get('controller_configs', {})
    def _apply(d):
        d = dict(d); d['kp'] = float(osc_kp); d['damping_ratio'] = float(osc_damping_ratio); return d
    em['env_kwargs']['controller_configs'] = (
        [_apply(d) for d in cc] if isinstance(cc, list) else _apply(cc))

    n_robots = len(em['env_kwargs'].get('robots', ['Panda']))
    is_dual = n_robots == 2

    obs_keys = ['object',
                'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_joint_pos']
    if is_dual:
        obs_keys += ['robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos', 'robot1_joint_pos']
    OU.initialize_obs_utils_with_obs_specs(obs_modality_specs={"obs": {"low_dim": obs_keys}})

    env = EU.create_env_from_metadata(env_meta=em, render=False, render_offscreen=False)

    fk = _PandaFK()

    # Compute per-arm world←panda rotation per env from each demo's first obs.
    rewards = []
    n_demos_used = 0
    with h5py.File(dataset, 'r') as f:
        demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')],
                           key=lambda k: int(k.split('_')[1]))
        # Held-out demos: matches the writeup convention test_start_seed=100000 maps to
        # "test demos" but in the existing FK runner test demos are seeded randomly. For
        # adapter oracle replay we instead walk a held-out tail of the demos: take the
        # last n_test demos.
        held_out = demo_keys[-n_test:]
        print(f"Replaying {len(held_out)} held-out demos for {task}/{split} at osc_kp={osc_kp}")

        for di, dk in enumerate(held_out):
            d = f['data/' + dk]
            states = d['states'][:]
            q_seqs = [d[f'obs/robot{i}_joint_pos'][:] for i in range(n_robots)]
            # gripper: arm0 idx=-1 for single, [6, 13] for transport
            if is_dual:
                grip = d['actions'][:, [6, 13]]
            else:
                grip = d['actions'][:, [-1]]

            env.reset()
            env.reset_to({'states': states[0]})

            # calibrate per-arm R_world_panda from obs
            R_world_panda = []
            obs_now = env.env._get_observations()
            for i in range(n_robots):
                q0 = obs_now[f'robot{i}_joint_pos']
                qt = obs_now[f'robot{i}_eef_quat']
                _, R_fk0 = fk.fk(q0)
                R_e0 = Rotation.from_quat(qt).as_matrix()
                R_world_panda.append(R_e0 @ R_fk0.T)

            T = min(states.shape[0], min(q.shape[0] for q in q_seqs)) - 1
            max_reward = 0.0
            for t in range(T):
                # build action: per-arm [dp(3), dr(3), grip(1)] concatenated
                obs_t = env.env._get_observations()
                a_parts = []
                for i in range(n_robots):
                    q_curr = obs_t[f'robot{i}_joint_pos'].astype(np.float64)
                    dq = (q_seqs[i][t + 1] - q_seqs[i][t]).astype(np.float64)
                    dp_n, dr_n = _fk_to_osc(q_curr, dq, R_world_panda[i], fk)
                    a_parts.extend([*dp_n, *dr_n, float(grip[t, i])])
                a = np.array(a_parts, dtype=np.float64)
                _, r, done, _ = env.step(a)
                max_reward = max(max_reward, float(r))
                if done:
                    break
            rewards.append(max_reward)
            n_demos_used += 1
            if di < 5 or di % 10 == 0:
                print(f"  demo {dk} -> max_reward={max_reward:.3f} ({t+1} steps)")

    mean_score = float(np.mean(rewards))
    print(f"\nFINAL: test/mean_score = {mean_score:.4f} over {n_demos_used} demos")
    log = {
        "test/mean_score": mean_score,
        "task": task,
        "split": split,
        "osc_kp": osc_kp,
        "n_demos": n_demos_used,
        "rewards": rewards,
    }
    with open(os.path.join(output_dir, 'eval_log.json'), 'w') as f:
        json.dump(log, f, indent=2)
    print(f"wrote {output_dir}/eval_log.json")


if __name__ == '__main__':
    main()
