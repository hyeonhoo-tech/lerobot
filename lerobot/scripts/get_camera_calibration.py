"""Dump RGB/Depth intrinsics and RGB<->Depth extrinsics for the connected RealSense cameras.

These values are baked into each camera unit at the factory and don't depend on how the
camera is mounted/aimed, so they're valid regardless of the physical rig pose.

Usage:
    python lerobot/scripts/get_camera_calibration.py --output calibration.json
"""

import argparse
import json

import pyrealsense2 as rs

CAMERAS = {
    "cam_high": 146322077603,
    "cam_low": 207222071967,
}
WIDTH = 640
HEIGHT = 480
FPS = 30


def intrinsics_to_dict(i: rs.intrinsics) -> dict:
    return {
        "width": i.width,
        "height": i.height,
        "fx": i.fx,
        "fy": i.fy,
        "ppx": i.ppx,
        "ppy": i.ppy,
        "distortion_model": str(i.model),
        "coeffs": list(i.coeffs),
    }


def extrinsics_to_dict(e: rs.extrinsics) -> dict:
    return {
        "rotation": list(e.rotation),  # row-major 3x3
        "translation": list(e.translation),  # meters
    }


def get_camera_calibration(serial: str) -> dict:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(str(serial))
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    profile = pipeline.start(config)
    try:
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()

        color_intrinsics = color_profile.get_intrinsics()
        depth_intrinsics = depth_profile.get_intrinsics()

        depth_to_color = depth_profile.get_extrinsics_to(color_profile)
        color_to_depth = color_profile.get_extrinsics_to(depth_profile)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        return {
            "serial_number": serial,
            "color_intrinsics": intrinsics_to_dict(color_intrinsics),
            "depth_intrinsics": intrinsics_to_dict(depth_intrinsics),
            "depth_to_color_extrinsics": extrinsics_to_dict(depth_to_color),
            "color_to_depth_extrinsics": extrinsics_to_dict(color_to_depth),
            "depth_scale_meters_per_unit": depth_scale,
        }
    finally:
        pipeline.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None, help="Path to write JSON output to.")
    args = parser.parse_args()

    result = {}
    for name, serial in CAMERAS.items():
        print(f"Reading calibration from {name} (serial {serial})...")
        result[name] = get_camera_calibration(serial)

    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
