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
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

try:
    import pinocchio as pin
except ImportError:
    raise ImportError("This script requires pinocchio. Install with: pip install pin")

from lerobot.common.datasets.compute_stats import sample_indices
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import decode_video_frames


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


def _build_state_to_q_mapping(model: pin.Model, state_names: list[str]) -> tuple[list[tuple[int, int]], dict[str, int]]:
    name_to_state_idx = {n: i for i, n in enumerate(state_names)}
    mapping = []
    for side in ["left", "right"]:
        for j in range(6):
            state_key = f"{side}_joint_{j}"
            urdf_name = f"follower_{side}_joint_{j}"
            if state_key in name_to_state_idx and model.existJointName(urdf_name):
                joint_id = model.getJointId(urdf_name)
                mapping.append((name_to_state_idx[state_key], model.joints[joint_id].idx_q))

    gripper_indices = {}
    for side in ["left", "right"]:
        key = f"{side}_joint_6"
        if key in name_to_state_idx:
            gripper_indices[side] = name_to_state_idx[key]

    return mapping, gripper_indices


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

    state_to_q, gripper_indices = _build_state_to_q_mapping(model, state_names)

    frame_ids = {}
    for link_name in [LEFT_EE_LINK, RIGHT_EE_LINK, HEAD_CAMERA_LINK]:
        frame_ids[link_name] = model.getFrameId(link_name)

    # Build new feature schema: original features + FK features (observation + action)
    fk_features = {
        # Observation EEF (FK of current joint positions)
        "obs_left_ee_position":      {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "obs_left_ee_quat_xyzw":     {"dtype": "float32", "shape": (4,), "names": ["qx", "qy", "qz", "qw"]},
        "obs_left_ee_euler_xyz":     {"dtype": "float32", "shape": (3,), "names": ["roll", "pitch", "yaw"]},
        #"obs_left_gripper":          {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
        "obs_right_ee_position":     {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "obs_right_ee_quat_xyzw":    {"dtype": "float32", "shape": (4,), "names": ["qx", "qy", "qz", "qw"]},
        "obs_right_ee_euler_xyz":    {"dtype": "float32", "shape": (3,), "names": ["roll", "pitch", "yaw"]},
        "obs_right_gripper":         {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
        "obs_head_camera_position":  {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "obs_head_camera_quat_xyzw": {"dtype": "float32", "shape": (4,), "names": ["qx", "qy", "qz", "qw"]},
        "obs_head_camera_euler_xyz": {"dtype": "float32", "shape": (3,), "names": ["roll", "pitch", "yaw"]},
        # Action EEF (FK of goal joint positions)
        "action_left_ee_position":   {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "action_left_ee_quat_xyzw":  {"dtype": "float32", "shape": (4,), "names": ["qx", "qy", "qz", "qw"]},
        "action_left_ee_euler_xyz":  {"dtype": "float32", "shape": (3,), "names": ["roll", "pitch", "yaw"]},
        #"action_left_gripper":       {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
        "action_right_ee_position":  {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "action_right_ee_quat_xyzw": {"dtype": "float32", "shape": (4,), "names": ["qx", "qy", "qz", "qw"]},
        "action_right_ee_euler_xyz": {"dtype": "float32", "shape": (3,), "names": ["roll", "pitch", "yaw"]},
        "action_right_gripper":      {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
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
    # Non-video image features (e.g. depth maps) are re-saved as PNGs one frame at a
    # time with no shortcut like the video-copy path below, so parallelize the writes.
    new_dataset.start_image_writer(num_processes=0, num_threads=8)

    ep_from = dataset.episode_data_index["from"]
    ep_to = dataset.episode_data_index["to"]

    for ep_idx in tqdm(range(dataset.num_episodes), desc="Episodes"):
        from_idx = int(ep_from[ep_idx].item())
        to_idx = int(ep_to[ep_idx].item())
        num_frames = to_idx - from_idx

        # Read all tabular data for this episode at once (no video decoding)
        ep_rows = dataset.hf_dataset.select(range(from_idx, to_idx))
        states     = np.array(ep_rows["observation.state"], dtype=np.float64)  # (N, 14)
        actions    = np.array(ep_rows["action"],            dtype=np.float64)  # (N, 14)
        timestamps = np.array(ep_rows["timestamp"],         dtype=np.float32)  # (N,)
        task_indices = ep_rows["task_index"]

        # Build episode_buffer manually — avoids add_frame touching video at all
        episode_buffer = new_dataset.create_episode_buffer()

        for i in tqdm(range(num_frames), desc="Frames", leave=False):
            obs_fk    = _compute_fk(model, data, states[i],  state_to_q, frame_ids)
            action_fk = _compute_fk(model, data, actions[i], state_to_q, frame_ids)

            task_str = dataset.meta.tasks[int(task_indices[i])]

            episode_buffer["frame_index"].append(i)
            episode_buffer["timestamp"].append(float(timestamps[i]))
            episode_buffer["task"].append(task_str)
            episode_buffer["action"].append(actions[i].astype(np.float32))
            episode_buffer["observation.state"].append(states[i].astype(np.float32))

            for k, v in obs_fk.items():
                key = f"obs_{k}"
                if key in episode_buffer:
                    episode_buffer[key].append(v)
            for k, v in action_fk.items():
                key = f"action_{k}"
                if key in episode_buffer:
                    episode_buffer[key].append(v)

            # Gripper values (joint_6) stored separately from FK
            for side, state_idx in gripper_indices.items():
                if f"obs_{side}_gripper" in episode_buffer:
                    episode_buffer[f"obs_{side}_gripper"].append(
                        np.array([states[i][state_idx]], dtype=np.float32)
                    )
                if f"action_{side}_gripper" in episode_buffer:
                    episode_buffer[f"action_{side}_gripper"].append(
                        np.array([actions[i][state_idx]], dtype=np.float32)
                    )

            episode_buffer["size"] += 1

        # Handle videos: copy file directly + decode a small sample for stats computation.
        # encode_episode_videos skips encoding when the destination file already exists.
        for vid_key in dataset.meta.video_keys:
            src = dataset.root / dataset.meta.get_video_file_path(ep_idx, vid_key)
            dst = new_dataset.root / new_dataset.meta.get_video_file_path(ep_idx, vid_key)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

            # Decode a sample of frames so compute_episode_stats has real image paths.
            # Use integer frame indices → exact i/fps timestamps to avoid tolerance errors.
            idxs = sample_indices(num_frames)
            sample_ts = [idx / dataset.meta.fps for idx in idxs]
            frames = decode_video_frames(src, sample_ts, dataset.tolerance_s)  # (n, C, H, W)

            img_paths = []
            for frame_i, frame in enumerate(frames):
                img_path = new_dataset._get_image_file_path(ep_idx, vid_key, frame_i)
                img_path.parent.mkdir(parents=True, exist_ok=True)
                new_dataset._save_image(frame, img_path)
                img_paths.append(str(img_path))

            episode_buffer[vid_key] = img_paths

        # Handle non-video image features (e.g. depth maps): these aren't stored as a
        # separate video file, so every frame must be re-saved from the decoded source.
        # Use `with_format(None)` to bypass the dataset's torch/ToTensor transform, which
        # would otherwise mangle 16-bit single-channel depth PNGs into (1, H, W) int16
        # tensors instead of leaving them as PIL images.
        raw_ep_rows = dataset.hf_dataset.select(range(from_idx, to_idx)).with_format(None)
        for img_key in dataset.meta.image_keys:
            src_images = raw_ep_rows[img_key]
            img_paths = []
            for frame_i, img in enumerate(src_images):
                img_path = new_dataset._get_image_file_path(ep_idx, img_key, frame_i)
                img_path.parent.mkdir(parents=True, exist_ok=True)
                new_dataset._save_image(img, img_path)
                img_paths.append(str(img_path))
            episode_buffer[img_key] = img_paths

        new_dataset.save_episode(episode_data=episode_buffer)

    new_dataset.stop_image_writer()

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
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help="Timestamp sync tolerance in seconds when loading the source dataset. Raise this "
        "(e.g. 0.01) if loading fails with a timestamp sync error due to recording fps jitter. "
        "Timestamps are not used by the FK computation, so loosening this is safe here.",
    )
    args = parser.parse_args()

    dataset = LeRobotDataset(
        args.repo_id, root=args.root, episodes=args.episodes, tolerance_s=args.tolerance_s
    )
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