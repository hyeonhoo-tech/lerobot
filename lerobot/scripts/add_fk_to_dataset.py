#!/usr/bin/env python3
"""
Compute forward kinematics (FK) for a recorded LeRobot dataset and save a new dataset
with EEF poses added as extra features.

Uses pinocchio for FK with the stationary_ai URDF.
Install dependencies: pip install pin scipy

Example usage:
    python lerobot/scripts/add_fk_to_dataset.py \
        --repo-id your_user/your_dataset \
        --output-repo-id your_user/your_dataset_with_fk \
        --urdf /path/to/stationary_ai.urdf \
        --root /path/to/local/dataset
"""

import argparse
import copy
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

try:
    import pinocchio as pin
except ImportError:
    raise ImportError("This script requires pinocchio. Install with: pip install pin")

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


LEFT_EE_LINK = "follower_left_ee_gripper_link"
RIGHT_EE_LINK = "follower_right_ee_gripper_link"
HEAD_CAMERA_LINK = "cam_high_color_optical_frame"


def _load_urdf(urdf_path: str | Path, package_root: str | Path | None = None) -> pin.Model:
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    with open(urdf_path) as f:
        urdf_str = f.read()

    if package_root is not None:
        package_root = Path(package_root).resolve()
        urdf_str = urdf_str.replace("package://trossen_arm_description", str(package_root))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as tmp:
            tmp.write(urdf_str)
            tmp_path = tmp.name
        try:
            model = pin.buildModelFromUrdf(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        model = pin.buildModelFromUrdf(str(urdf_path))

    return model


def _build_state_to_q_mapping(model: pin.Model, state_names: list[str]) -> list[tuple[int, int]]:
    name_to_state_idx = {n: i for i, n in enumerate(state_names)}
    mapping = []
    for side, prefix in [("left", "left"), ("right", "right")]:
        for j in range(6):
            state_key = f"{side}_joint_{j}"
            urdf_name = f"follower_{side}_joint_{j}"
            if state_key in name_to_state_idx and model.existJointName(urdf_name):
                joint_id = model.getJointId(urdf_name)
                mapping.append((name_to_state_idx[state_key], model.joints[joint_id].idx_q))
    return mapping


def _compute_fk(
    model: pin.Model,
    data: pin.Data,
    state: np.ndarray,
    state_to_q: list[tuple[int, int]],
    frame_ids: dict[str, int],
) -> dict:
    q = np.zeros(model.nq)
    for state_idx, q_idx in state_to_q:
        q[q_idx] = float(state[state_idx])

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    out = {}
    for link_name, prefix in [
        (LEFT_EE_LINK, "left_ee"),
        (RIGHT_EE_LINK, "right_ee"),
        (HEAD_CAMERA_LINK, "head_camera"),
    ]:
        placement = data.oMf[frame_ids[link_name]]
        pos = np.array(placement.translation, dtype=np.float32)
        R = placement.rotation
        r = Rotation.from_matrix(R)
        quat = r.as_quat().astype(np.float32)   # [qx, qy, qz, qw]
        euler = r.as_euler("xyz").astype(np.float32)  # [roll, pitch, yaw]
        out[f"{prefix}_position"] = pos
        out[f"{prefix}_quat_xyzw"] = quat
        out[f"{prefix}_euler_xyz"] = euler

    return out


def add_fk_to_dataset(
    dataset: LeRobotDataset,
    urdf_path: str | Path,
    output_repo_id: str,
    output_root: str | Path,
    package_root: str | Path | None = None,
    push_to_hub: bool = False,
):
    model = _load_urdf(urdf_path, package_root=package_root)
    data = model.createData()

    state_key = "observation.state"
    state_names = dataset.features[state_key]["names"]

    state_to_q = _build_state_to_q_mapping(model, state_names)
    if len(state_to_q) != 12:
        raise ValueError(
            f"Expected 12 arm joints (6 left + 6 right), got {len(state_to_q)}. "
            "Check that state names contain left_joint_0..5 and right_joint_0..5."
        )

    frame_ids = {}
    for link_name in [LEFT_EE_LINK, RIGHT_EE_LINK, HEAD_CAMERA_LINK]:
        frame_ids[link_name] = model.getFrameId(link_name)

    # Build new feature schema: original features + FK features (observation + action)
    fk_features = {
        # Observation EEF (FK of current joint positions)
        "obs_left_ee_position":      {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "obs_left_ee_quat_xyzw":     {"dtype": "float32", "shape": (4,), "names": ["qx", "qy", "qz", "qw"]},
        "obs_left_ee_euler_xyz":     {"dtype": "float32", "shape": (3,), "names": ["roll", "pitch", "yaw"]},
        "obs_right_ee_position":     {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "obs_right_ee_quat_xyzw":    {"dtype": "float32", "shape": (4,), "names": ["qx", "qy", "qz", "qw"]},
        "obs_right_ee_euler_xyz":    {"dtype": "float32", "shape": (3,), "names": ["roll", "pitch", "yaw"]},
        "obs_head_camera_position":  {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "obs_head_camera_quat_xyzw": {"dtype": "float32", "shape": (4,), "names": ["qx", "qy", "qz", "qw"]},
        # Action EEF (FK of goal joint positions)
        "action_left_ee_position":   {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "action_left_ee_quat_xyzw":  {"dtype": "float32", "shape": (4,), "names": ["qx", "qy", "qz", "qw"]},
        "action_left_ee_euler_xyz":  {"dtype": "float32", "shape": (3,), "names": ["roll", "pitch", "yaw"]},
        "action_right_ee_position":  {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "action_right_ee_quat_xyzw": {"dtype": "float32", "shape": (4,), "names": ["qx", "qy", "qz", "qw"]},
        "action_right_ee_euler_xyz": {"dtype": "float32", "shape": (3,), "names": ["roll", "pitch", "yaw"]},
    }
    features_with_fk = copy.deepcopy(dataset.meta.info["features"])
    for k, v in fk_features.items():
        features_with_fk[k] = copy.deepcopy(v)

    new_dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=dataset.meta.fps,
        root=output_root,
        robot_type=dataset.meta.robot_type,
        features=features_with_fk,
        use_videos=True,
    )

    ep_from = dataset.episode_data_index["from"]
    ep_to = dataset.episode_data_index["to"]

    for ep_idx in tqdm(range(dataset.num_episodes), desc="Episodes"):
        from_idx = int(ep_from[ep_idx].item())
        to_idx = int(ep_to[ep_idx].item())

        for global_idx in tqdm(range(from_idx, to_idx), desc="Frames", leave=False):
            item = dataset[global_idx]

            state = item[state_key].numpy().astype(np.float64)
            obs_fk = _compute_fk(model, data, state, state_to_q, frame_ids)

            action = item["action"].numpy().astype(np.float64)
            action_fk = _compute_fk(model, data, action, state_to_q, frame_ids)

            frame = {
                "action": item["action"],
                "observation.state": item["observation.state"],
                "task": item["task"],
                "timestamp": np.float32(item["timestamp"].item()),
                **{f"obs_{k}": v for k, v in obs_fk.items()},
                **{f"action_{k}": v for k, v in action_fk.items() if "head_camera" not in k},
            }
            for cam_key in dataset.meta.camera_keys:
                frame[cam_key] = item[cam_key].permute(1, 2, 0)

            new_dataset.add_frame(frame)

        new_dataset.save_episode()

    if push_to_hub:
        print("Pushing to Hub...")
        new_dataset.push_to_hub()
        print(f"Pushed to https://huggingface.co/datasets/{output_repo_id}")
    else:
        print(f"Saved locally to {output_root}")

    return new_dataset


def main():
    parser = argparse.ArgumentParser(description="Add FK-computed EEF poses to a LeRobot dataset.")
    parser.add_argument("--repo-id", type=str, required=True, help="Source dataset repo id.")
    parser.add_argument("--output-repo-id", type=str, required=True, help="Output dataset repo id.")
    parser.add_argument("--root", type=str, default=None, help="Local path to source dataset.")
    parser.add_argument("--output-root", type=str, required=True, help="Local path to save output dataset.")
    parser.add_argument("--urdf", type=str, required=True, help="Path to stationary_ai.urdf.")
    parser.add_argument("--package-root", type=str, default=None, help="Path to trossen_arm_description for resolving package:// URIs.")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Episode indices to process (default: all).")
    parser.add_argument("--push-to-hub", action="store_true", help="Push output dataset to HuggingFace Hub.")
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=args.episodes)
    print(f"Loaded dataset: {dataset.num_frames} frames, {dataset.num_episodes} episodes.")

    add_fk_to_dataset(
        dataset=dataset,
        urdf_path=args.urdf,
        output_repo_id=args.output_repo_id,
        output_root=args.output_root,
        package_root=args.package_root,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
