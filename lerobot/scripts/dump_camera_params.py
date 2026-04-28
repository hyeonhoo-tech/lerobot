"""Dump RealSense intrinsics + depth scale for the trossen_ai_stationary rig
into a single JSON file (Phantom-ready).

Usage:
    python lerobot/scripts/dump_camera_params.py \
        --output ~/trossen_camera_params.json

The output file collects everything Phantom (or any RGBD pose pipeline) needs
in one place, so you can copy it off this public PC. The cam->robot extrinsics
are NOT auto-detected; the script writes a placeholder you fill in after a
hand-eye calibration.
"""

import argparse
import datetime
import json
from pathlib import Path

import pyrealsense2 as rs

from lerobot.common.robot_devices.robots.configs import TrossenAIStationaryRobotConfig


def _intrinsics_to_dict(intr: rs.intrinsics) -> dict:
    return {
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "width": int(intr.width),
        "height": int(intr.height),
        "distortion_model": str(intr.model).split(".")[-1],
        "distortion_coeffs": [float(c) for c in intr.coeffs],
    }


def _phantom_intrinsics_block(intr: rs.intrinsics) -> dict:
    """Phantom expects a {"left": {...}, "right": {...}} dict (stereo legacy).
    With a single mono RGBD camera we duplicate the same intrinsics into both."""
    block = {
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "disto": [0.0] * 12,
    }
    return {"left": dict(block), "right": dict(block)}


def _query_camera(serial: str, width: int, height: int, fps: int) -> dict:
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(str(serial))
    cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(cfg)
    try:
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_intr = color_stream.get_intrinsics()
        depth_intr = depth_stream.get_intrinsics()
        depth_to_color = depth_stream.get_extrinsics_to(color_stream)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        return {
            "serial_number": str(serial),
            "resolution": {"width": int(width), "height": int(height)},
            "fps": int(fps),
            "color_intrinsics": _intrinsics_to_dict(color_intr),
            "depth_intrinsics": _intrinsics_to_dict(depth_intr),
            "depth_scale_meters_per_unit": float(depth_scale),
            "depth_unit_in_saved_npy": "uint16 millimeters (RealSense raw)",
            "convert_to_meters": "depth_in_meters = npy.astype(float32) * depth_scale_meters_per_unit",
            "depth_to_color_extrinsics": {
                "rotation_3x3_row_major": [
                    list(depth_to_color.rotation[0:3]),
                    list(depth_to_color.rotation[3:6]),
                    list(depth_to_color.rotation[6:9]),
                ],
                "translation_xyz_meters": list(depth_to_color.translation),
            },
            "phantom_camera_intrinsics_json": _phantom_intrinsics_block(color_intr),
        }
    finally:
        pipeline.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path.home() / "trossen_camera_params.json"),
        help="Where to write the consolidated params file (move off this PC after).",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"],
        help="Which configured cameras to query.",
    )
    args = parser.parse_args()

    robot_cfg = TrossenAIStationaryRobotConfig(camera_interface="intel_realsense")

    cameras_out: dict = {}
    for name in args.cameras:
        cam_cfg = robot_cfg.cameras[name]
        print(f"[{name}] querying RealSense (serial={cam_cfg.serial_number})...")
        try:
            cameras_out[name] = _query_camera(
                serial=cam_cfg.serial_number,
                width=cam_cfg.width,
                height=cam_cfg.height,
                fps=cam_cfg.fps,
            )
        except Exception as e:
            cameras_out[name] = {
                "serial_number": str(cam_cfg.serial_number),
                "error": f"{type(e).__name__}: {e}",
            }

    bundle = {
        "_about": (
            "Camera parameters for trossen_ai_stationary. Move this file off the "
            "public PC. Fill in `cam_to_robot_base_extrinsics_TODO` after running "
            "a hand-eye calibration."
        ),
        "robot_type": "trossen_ai_stationary",
        "captured_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "cameras": cameras_out,
        "cam_to_robot_base_extrinsics_TODO": {
            "_status": "NOT CALIBRATED — placeholder identity transform.",
            "_what_phantom_expects": (
                "A 4x4 transform mapping points expressed in the camera frame "
                "into the robot base frame. Phantom loads this from "
                "configs.camera_extrinsics (default: phantom/camera/camera_extrinsics.json)."
            ),
            "_recipe": [
                "1. Fix an AprilTag (or charuco board) rigidly to the left follower EE.",
                "2. Move the arm through ~20 distinct poses; for each: log joint state and capture cam_high frame with the tag clearly visible.",
                "3. Run forward kinematics (Trossen URDF) to get T_base_ee for every pose.",
                "4. Detect the tag in each cam_high frame to get T_cam_tag.",
                "5. Solve hand-eye (e.g. cv2.calibrateHandEye, Park or Daniilidis method) for T_cam_base.",
                "6. Invert if needed and paste the 4x4 result into T_cam_to_base_4x4 below.",
            ],
            "T_cam_to_base_4x4": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "_phantom_camera_extrinsics_json_template": [
                {
                    "camera_base_ori": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "camera_base_pos": [0.0, 0.0, 0.0],
                    "camera_base_quat": [1.0, 0.0, 0.0, 0.0],
                    "p_marker_ee": [0.0, 0.0, 0.0],
                }
            ],
        },
        "depth_npy_layout": {
            "shape": "(num_frames, height, width)",
            "dtype": "uint16",
            "unit": "millimeters",
            "where": "videos/chunk-XXX/observation.images.cam_high_depth/episode_NNNNNN.npy",
            "phantom_path_expectation": "<demo>/<episode_idx>/depth.npy (in meters as float32) — divide by 1000 before symlinking",
        },
    }

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"\nWrote {out_path}")
    print("Move this file off the public PC (USB / private cloud / your laptop).")


if __name__ == "__main__":
    main()
