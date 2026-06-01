#!/usr/bin/env python3
"""
Teleoperation constraints for the Trossen AI stationary setup.

Lets you record data while keeping the end-effector at a fixed Cartesian height
(z-axis lock) and/or freezing wrist joints, all enforced *during* teleoperation so
the recorded `action`/`observation.state` faithfully reflect the constrained motion.

Two independent constraints, both opt-in:

1. Wrist lock (joint space, no IK):
   The listed joint indices (default joint_3, joint_4, joint_5) are overwritten with
   reference values captured when teleoperation starts. The gripper (joint_6) is left free.

2. z-axis lock (Cartesian, requires pinocchio IK):
   Each step, forward kinematics gives the leader's intended end-effector (x, y); z is
   replaced by a fixed `locked_z`. A damped least-squares position IK over the remaining
   free position joints (those among joint_0..joint_5 that are not wrist-locked, i.e.
   joint_0, joint_1, joint_2 by default) drives the follower to (x, y, locked_z).

Install dependency (recording machine only): pip install pin

This module is only imported when constraints are enabled, so pinocchio is not required
for normal (unconstrained) recording.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

# Number of actuated arm joints (joint_0..joint_5); joint_6 is the gripper.
NUM_ARM_JOINTS = 6


def _load_urdf_model(urdf_path: str | Path, package_root: str | Path | None):
    """Build a pinocchio model from the URDF, resolving package:// URIs if needed."""
    import pinocchio as pin

    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    urdf_str = urdf_path.read_text()
    if package_root is not None:
        package_root = Path(package_root).resolve()
        urdf_str = urdf_str.replace("package://trossen_arm_description", str(package_root))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as tmp:
            tmp.write(urdf_str)
            tmp_path = tmp.name
        try:
            return pin.buildModelFromUrdf(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    return pin.buildModelFromUrdf(str(urdf_path))


class StationaryTeleopConstraint:
    """Per-arm teleoperation constraint (wrist lock + z-axis lock).

    Usage:
        c = StationaryTeleopConstraint(urdf_path=..., ee_link="follower_right_ee_gripper_link",
                                       joint_prefix="follower_right",
                                       lock_wrist_joints=[3, 4, 5], lock_z=True)
        c.initialize_refs(follower_present_pos)   # call once, with current follower joints
        goal = c.apply(leader_goal_pos)            # call every teleop step
    """

    def __init__(
        self,
        urdf_path: str | Path,
        ee_link: str,
        joint_prefix: str,
        package_root: str | Path | None = None,
        lock_wrist_joints: list[int] | None = (3, 4, 5),
        lock_z: bool = True,
        locked_z: float | None = None,
        ik_iters: int = 30,
        ik_tol: float = 5e-4,
        ik_damping: float = 1e-3,
    ):
        import pinocchio as pin

        self._pin = pin
        self.lock_wrist_joints = sorted(set(lock_wrist_joints)) if lock_wrist_joints else []
        self.lock_z = lock_z
        self.locked_z = locked_z
        self.ik_iters = ik_iters
        self.ik_tol = ik_tol
        self.ik_damping = ik_damping

        for j in self.lock_wrist_joints:
            if not 0 <= j < NUM_ARM_JOINTS:
                raise ValueError(f"lock_wrist_joints index {j} out of range [0, {NUM_ARM_JOINTS}).")

        self.model = _load_urdf_model(urdf_path, package_root)
        self.data = self.model.createData()

        if not self.model.existFrame(ee_link):
            raise ValueError(f"EE link '{ee_link}' not found in URDF.")
        self.ee_frame_id = self.model.getFrameId(ee_link)

        # Map arm joint index (0..5) -> (q index, v index) in the pinocchio model.
        self.joint_q_idx: dict[int, int] = {}
        self.joint_v_idx: dict[int, int] = {}
        for j in range(NUM_ARM_JOINTS):
            name = f"{joint_prefix}_joint_{j}"
            if not self.model.existJointName(name):
                raise ValueError(f"Joint '{name}' not found in URDF.")
            jid = self.model.getJointId(name)
            self.joint_q_idx[j] = self.model.joints[jid].idx_q
            self.joint_v_idx[j] = self.model.joints[jid].idx_v

        # Free position joints for IK = arm joints that are NOT wrist-locked.
        self.free_pos_joints = [j for j in range(NUM_ARM_JOINTS) if j not in self.lock_wrist_joints]

        self.wrist_refs: dict[int, float] = {}
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize_refs(self, follower_present_pos: np.ndarray) -> None:
        """Capture reference values from the follower's current joints.

        - Wrist reference values are taken from the current follower joints (unless
          a locked value was explicitly provided via the config in the future).
        - If `locked_z` was not set, capture the current end-effector height as z0.
        """
        follower_present_pos = np.asarray(follower_present_pos, dtype=np.float64)
        for j in self.lock_wrist_joints:
            self.wrist_refs[j] = float(follower_present_pos[j])

        if self.lock_z and self.locked_z is None:
            q = self._build_q(follower_present_pos)
            self.locked_z = float(self._fk_position(q)[2])

        self._initialized = True

    def _build_q(self, arm_joints: np.ndarray) -> np.ndarray:
        """Embed the 6 arm joints into a full pinocchio configuration vector (others = 0)."""
        q = np.zeros(self.model.nq)
        for j in range(NUM_ARM_JOINTS):
            q[self.joint_q_idx[j]] = float(arm_joints[j])
        return q

    def _fk_position(self, q: np.ndarray) -> np.ndarray:
        pin = self._pin
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return np.array(self.data.oMf[self.ee_frame_id].translation, dtype=np.float64)

    def _solve_position_ik(self, q: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """Damped least-squares IK on free position joints to reach target_pos (x, y, z)."""
        pin = self._pin
        free_v = [self.joint_v_idx[j] for j in self.free_pos_joints]
        I = np.eye(3)
        for _ in range(self.ik_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            current = np.array(self.data.oMf[self.ee_frame_id].translation, dtype=np.float64)
            err = target_pos - current
            if np.linalg.norm(err) < self.ik_tol:
                break
            J = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J_pos = J[:3, free_v]  # (3, n_free)
            # dq_free = J^T (J J^T + lambda I)^-1 err
            dq_free = J_pos.T @ np.linalg.solve(J_pos @ J_pos.T + self.ik_damping * I, err)
            dq = np.zeros(self.model.nv)
            for k, vidx in enumerate(free_v):
                dq[vidx] = dq_free[k]
            q = pin.integrate(self.model, q, dq)
        return q

    def apply(self, leader_goal_pos: np.ndarray) -> np.ndarray:
        """Return a constrained goal position (7,) given the leader's goal joints (7,)."""
        if not self._initialized:
            raise RuntimeError("Call initialize_refs() before apply().")

        goal = np.asarray(leader_goal_pos, dtype=np.float64).copy()

        # 1) Wrist lock (joint space).
        for j, ref in self.wrist_refs.items():
            goal[j] = ref

        # 2) z-axis lock (Cartesian, via IK over free position joints).
        if self.lock_z:
            q = self._build_q(goal[:NUM_ARM_JOINTS])
            target = self._fk_position(q)  # leader's (x, y, z) with wrist already overridden
            target[2] = self.locked_z
            q = self._solve_position_ik(q, target)
            for j in self.free_pos_joints:
                goal[j] = q[self.joint_q_idx[j]]

        return goal.astype(np.float32)
