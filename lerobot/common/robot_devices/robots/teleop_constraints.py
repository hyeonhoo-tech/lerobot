#!/usr/bin/env python3
"""
Teleoperation constraints for the Trossen AI stationary setup.

Lets you record data while constraining the follower's end-effector, enforced *during*
teleoperation so the recorded `action`/`observation.state` faithfully reflect the
constrained motion.

Three independent, opt-in constraints:

1. z-axis lock (`lock_z`):
   Keep the end-effector at a fixed Cartesian height. The leader's intended (x, y) is
   followed; z is replaced by `locked_z`.

2. Orientation lock (`lock_orientation`):
   Keep the end-effector orientation fixed (the gripper keeps pointing the same way) while
   it translates in the x-y plane. This uses all 6 arm joints in the IK (wrist included),
   so it CANNOT be combined with joint-space wrist locking — when enabled, `lock_wrist_joints`
   is ignored. The orientation reference is captured from the leader's pose on the first
   teleop step, so hold the leader's gripper in the desired orientation when you start.

3. Wrist lock (`lock_wrist_joints`, joint space):
   Freeze the listed joint indices (e.g. joint_3, joint_4, joint_5) at the values held when
   teleoperation starts. The gripper (joint_6) is always left free. Ignored if
   `lock_orientation` is on.

IK solver:
- Orientation lock on  -> full 6-DOF pose IK (position + orientation) over all 6 arm joints.
- Orientation lock off -> position-only IK over the free (non-wrist-locked) arm joints.

Install dependency (recording machine only): pip install pin
This module is only imported when constraints are enabled.
"""

from __future__ import annotations

import tempfile
import time
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
    """Per-arm teleoperation constraint (z-axis lock / orientation lock / wrist lock).

    Usage:
        c = StationaryTeleopConstraint(urdf_path=..., ee_link="follower_right_ee_gripper_link",
                                       joint_prefix="follower_right",
                                       lock_z=True, locked_z=0.10, lock_orientation=True)
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
        lock_orientation: bool = False,
        locked_pitch_offset_deg: float = 0.0,
        soft_start_s: float = 2.0,
        max_xy_step: float = 0.01,
        feasible_z_tol: float = 0.02,
        max_joint_step: float = 0.08,
        ik_iters: int = 80,
        ik_tol: float = 1e-3,
        ik_damping: float = 1e-2,
    ):
        import pinocchio as pin

        self._pin = pin
        self.lock_z = lock_z
        self.locked_z = locked_z
        self.lock_orientation = lock_orientation
        self.locked_pitch_offset_deg = locked_pitch_offset_deg
        self.soft_start_s = soft_start_s
        self.max_xy_step = max_xy_step
        self.feasible_z_tol = feasible_z_tol
        self.max_joint_step = max_joint_step
        self.ik_iters = ik_iters
        self.ik_tol = ik_tol
        self.ik_damping = ik_damping

        # Orientation lock needs the wrist free for IK; it overrides joint-space wrist locking.
        if lock_orientation:
            self.lock_wrist_joints: list[int] = []
        else:
            self.lock_wrist_joints = sorted(set(lock_wrist_joints)) if lock_wrist_joints else []
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

        # Joints the IK is allowed to move. Orientation lock uses all 6; otherwise only the
        # arm joints that are not wrist-locked (e.g. joint_0, joint_1, joint_2).
        if self.lock_orientation:
            self.free_arm_joints = list(range(NUM_ARM_JOINTS))
        else:
            self.free_arm_joints = [j for j in range(NUM_ARM_JOINTS) if j not in self.lock_wrist_joints]

        self.wrist_refs: dict[int, float] = {}
        self.locked_R: np.ndarray | None = None  # orientation reference (set on first apply)
        self._q_start_arm: np.ndarray | None = None  # follower arm joints when teleop starts
        self._t0: float | None = None  # soft-start ramp clock
        self._ramp_anchor: np.ndarray | None = None  # arm pose the current ramp starts from
        self._last_apply_t: float | None = None  # wall time of the previous apply() call
        self._last_returned_arm: np.ndarray | None = None  # last arm goal we returned
        self._prev_xy: np.ndarray | None = None  # last commanded EE (x, y) for rate limiting
        self._last_good_arm: np.ndarray | None = None  # last feasible arm goal (z held)
        self._warned_infeasible = False
        # If apply() has not been called for longer than this (loop paused, e.g. between
        # warmup and recording), re-arm the soft-start from the current pose to avoid a jump.
        self._gap_s = 0.3
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize_refs(self, follower_present_pos: np.ndarray) -> None:
        """Capture reference values from the follower's current joints."""
        follower_present_pos = np.asarray(follower_present_pos, dtype=np.float64)
        self._q_start_arm = follower_present_pos[:NUM_ARM_JOINTS].copy()
        for j in self.lock_wrist_joints:
            self.wrist_refs[j] = float(follower_present_pos[j])

        if self.lock_z and self.locked_z is None:
            q = self._build_q(follower_present_pos)
            self.locked_z = float(self._fk_pose(q).translation[2])

        if self.lock_z:
            z_now = float(self._fk_pose(self._build_q(follower_present_pos)).translation[2])
            print(
                f"[TeleopConstraint] start z = {z_now:.4f} m -> locked z = {self.locked_z:.4f} m"
                f" (soft-start over {self.soft_start_s:.1f}s)"
            )
        self._initialized = True

    def _build_q(self, arm_joints: np.ndarray) -> np.ndarray:
        """Embed the 6 arm joints into a full pinocchio configuration vector (others = 0)."""
        q = np.zeros(self.model.nq)
        for j in range(NUM_ARM_JOINTS):
            q[self.joint_q_idx[j]] = float(arm_joints[j])
        return q

    def _fk_pose(self, q: np.ndarray):
        """Return the EE placement (SE3) for configuration q."""
        pin = self._pin
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return pin.SE3(self.data.oMf[self.ee_frame_id])

    def _solve_ik(self, q: np.ndarray, oMdes, free_v: list[int]) -> np.ndarray:
        """Damped least-squares IK to reach the desired EE placement oMdes.

        If `lock_orientation` is set, matches the full 6-DOF pose; otherwise position only.
        Only the velocity components in `free_v` are allowed to change.
        """
        pin = self._pin
        if self.lock_orientation:
            I6 = np.eye(6)
            for _ in range(self.ik_iters):
                pin.forwardKinematics(self.model, self.data, q)
                pin.updateFramePlacements(self.model, self.data)
                oMcur = self.data.oMf[self.ee_frame_id]
                iMd = oMcur.actInv(oMdes)  # desired pose expressed in current EE frame
                err = pin.log(iMd).vector  # 6-vector (LOCAL frame)
                if np.linalg.norm(err) < self.ik_tol:
                    break
                J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id)  # LOCAL
                J = -pin.Jlog6(iMd.inverse()).dot(J)  # 6 x nv
                Jf = J[:, free_v]
                v_free = -Jf.T.dot(np.linalg.solve(Jf.dot(Jf.T) + self.ik_damping * I6, err))
                v = np.zeros(self.model.nv)
                for k, vidx in enumerate(free_v):
                    v[vidx] = v_free[k]
                q = pin.integrate(self.model, q, v)
        else:
            I3 = np.eye(3)
            target_pos = oMdes.translation
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
                J_pos = J[:3, free_v]
                v_free = J_pos.T.dot(np.linalg.solve(J_pos.dot(J_pos.T) + self.ik_damping * I3, err))
                v = np.zeros(self.model.nv)
                for k, vidx in enumerate(free_v):
                    v[vidx] = v_free[k]
                q = pin.integrate(self.model, q, v)
        return q

    def apply(self, leader_goal_pos: np.ndarray, follower_present_pos: np.ndarray | None = None) -> np.ndarray:
        """Return a constrained goal position (7,) given the leader's goal joints (7,).

        `follower_present_pos` (the follower's actual joints right now) is used to re-anchor
        the soft-start whenever the teleop loop has been paused — e.g. after `teleop_safety_stop`
        resets the arm to its home pose between warmup and recording. Without it, the ramp would
        ease from a stale pose and the follower would jerk.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize_refs() before apply().")

        pin = self._pin
        goal = np.asarray(leader_goal_pos, dtype=np.float64).copy()

        # 0) Detect a loop pause (gap between apply calls) and re-arm the soft-start from the
        # follower's ACTUAL current pose, since it may have been moved (e.g. reset to home by
        # teleop_safety_stop) while the loop was not running.
        now = time.perf_counter()
        paused = self._last_apply_t is not None and (now - self._last_apply_t) > self._gap_s
        if self._t0 is None or paused:
            self._t0 = now
            if follower_present_pos is not None:
                present_arm = np.asarray(follower_present_pos, dtype=np.float64)[:NUM_ARM_JOINTS]
                self._ramp_anchor = present_arm.copy()
                self._prev_xy = self._fk_pose(self._build_q(present_arm)).translation[:2].copy()
                self._last_good_arm = None  # don't compare jumps against a stale pre-pause pose
            else:
                anchor = self._last_returned_arm if self._last_returned_arm is not None else self._q_start_arm
                self._ramp_anchor = None if anchor is None else anchor.copy()
        self._last_apply_t = now

        # 1) Joint-space wrist lock (skipped when orientation lock is on).
        for j, ref in self.wrist_refs.items():
            goal[j] = ref

        # 2) Cartesian constraints via IK.
        if self.lock_z or self.lock_orientation:
            q = self._build_q(goal[:NUM_ARM_JOINTS])
            oMcur = self._fk_pose(q)  # leader's intended pose (wrist already overridden)

            target_pos = np.array(oMcur.translation, dtype=np.float64)
            if self.lock_z:
                target_pos[2] = self.locked_z

            # Rate-limit the target in the x-y plane so a fast teleop motion does not jump
            # the target so far that the per-step IK cannot converge (which would let the EE
            # leave the locked-z plane and dive). Keeps each step's target close to the last.
            if self.max_xy_step and self.max_xy_step > 0:
                if self._prev_xy is None:
                    self._prev_xy = target_pos[:2].copy()
                delta = target_pos[:2] - self._prev_xy
                dist = float(np.linalg.norm(delta))
                if dist > self.max_xy_step:
                    target_pos[:2] = self._prev_xy + delta * (self.max_xy_step / dist)
                self._prev_xy = target_pos[:2].copy()

            target_R = np.array(oMcur.rotation, dtype=np.float64)
            if self.lock_orientation:
                if self.locked_R is None:
                    captured_R = np.array(oMcur.rotation, dtype=np.float64)
                    # Tilt the captured orientation by a fixed pitch (about the EE-local Y
                    # axis). Positive value pitches one way; flip the sign if it tilts the
                    # wrong way for your setup.
                    theta = np.deg2rad(self.locked_pitch_offset_deg)
                    R_off = pin.rpy.rpyToMatrix(0.0, theta, 0.0)
                    self.locked_R = captured_R @ R_off
                    rpy = pin.rpy.matrixToRpy(self.locked_R)
                    print(
                        "[TeleopConstraint] orientation locked at "
                        f"roll/pitch/yaw = {np.round(rpy, 4).tolist()} rad "
                        f"(pitch offset {self.locked_pitch_offset_deg:+.1f} deg)"
                    )
                target_R = self.locked_R

            oMdes = pin.SE3(target_R, target_pos)
            free_v = [self.joint_v_idx[j] for j in self.free_arm_joints]
            q = self._solve_ik(q, oMdes, free_v)

            # Candidate arm goal from the IK solution (wrist already overridden in `goal`).
            candidate_arm = goal[:NUM_ARM_JOINTS].copy()
            for j in self.free_arm_joints:
                candidate_arm[j] = q[self.joint_q_idx[j]]

            # Safety guards. Reject the IK solution (hold the last good pose) when:
            #  (a) it cannot hold the locked height (target unreachable / under-converged), or
            #  (b) it jumps more than `max_joint_step` rad on any joint in one step (IK branch
            #      flip / divergence) — this prevents sudden fast/violent motion.
            achieved_z = float(self._fk_pose(q).translation[2])
            reject = self.lock_z and abs(achieved_z - self.locked_z) > self.feasible_z_tol
            reason = "out of reach at locked z" if reject else ""
            if self._last_good_arm is not None and self.max_joint_step > 0:
                jump = float(np.max(np.abs(candidate_arm - self._last_good_arm)))
                if jump > self.max_joint_step:
                    reject = True
                    reason = f"joint jump {jump:.3f} rad > {self.max_joint_step:.3f}"

            if reject and self._last_good_arm is not None:
                goal[:NUM_ARM_JOINTS] = self._last_good_arm  # freeze arm; gripper still follows
                if not self._warned_infeasible:
                    print(f"[TeleopConstraint] rejecting IK solution ({reason}); holding position.")
                    self._warned_infeasible = True
            else:
                goal[:NUM_ARM_JOINTS] = candidate_arm
                self._last_good_arm = candidate_arm.copy()
                self._warned_infeasible = False

        # 3) Soft start: ease the arm joints from the (re-armed) anchor pose to the constrained
        # goal over `soft_start_s`. Re-arming is handled at the top of apply(). The gripper
        # (joint_6) is left untouched.
        if self.soft_start_s > 0 and self._ramp_anchor is not None:
            alpha = (now - self._t0) / self.soft_start_s
            if alpha < 1.0:
                alpha = max(alpha, 0.0)
                goal[:NUM_ARM_JOINTS] = (
                    (1.0 - alpha) * self._ramp_anchor + alpha * goal[:NUM_ARM_JOINTS]
                )

        self._last_returned_arm = goal[:NUM_ARM_JOINTS].copy()
        return goal.astype(np.float32)
