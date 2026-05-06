# Reverse Controller

This folder learns a state-conditioned inverse for robosuite `JOINT_POSITION`:

```text
u = f(s, desired_joint_delta)
```

where `u` is the physical joint-position command in radians. The collector probes
Robomimic simulator states with sampled `JOINT_POSITION` commands and records the
actual joint delta produced by the controller. The trainer fits an MLP that maps
full lowdim state plus desired actual joint delta back to the executable command.

For Can MH, the default full state is:

```text
object
robot0_eef_pos
robot0_eef_quat
robot0_gripper_qpos
robot0_joint_pos
robot0_joint_vel
```

The scripts are:

```text
collect_inverse_dataset.py  # build one shard per Robomimic demo
train_inverse_model.py      # train f on collected probes
eval_inverse_model.py       # one-step sim eval on demo deltas
```

The generated commands are pseudo-labels for the chosen `JOINT_POSITION`
interface, not original human teleop commands.
