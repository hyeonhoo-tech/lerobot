"""Grab one photo each from cam_high and cam_low, for use with label_marker_corners.py and
solve_camera_extrinsics.py. Run this after the calibration board has been placed and is not
going to move again until the whole calibration process (touch points + these photos) is done.

Captures at a higher resolution than the normal 640x480 recording stream (default 1920x1080)
because the ArUco marker is small enough in frame that 640x480 doesn't have enough pixels per
bit cell to decode reliably. Also saves the color intrinsics AT THAT RESOLUTION next to each
photo (as <name>_board_photo_intrinsics.json), since RealSense intrinsics differ per resolution
and camera_calibration.json only has the 640x480 values.

Usage:
    python lerobot/scripts/capture_board_photos.py --output-dir .
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

CAMERAS = {
    "cam_high": 146322077603,
    "cam_low": 207222071967,
}
WIDTH = 1280
HEIGHT = 720
FPS = 15
WARMUP_FRAMES = 30  # let auto-exposure settle before grabbing the keeper frame


def grab_photo_and_intrinsics(serial: str) -> tuple[np.ndarray, dict]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(str(serial))
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(config)
    try:
        for _ in range(WARMUP_FRAMES):
            frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        image = np.asanyarray(color_frame.get_data())

        i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intrinsics = {
            "width": i.width,
            "height": i.height,
            "fx": i.fx,
            "fy": i.fy,
            "ppx": i.ppx,
            "ppy": i.ppy,
            "coeffs": list(i.coeffs),
        }
        return image, intrinsics
    finally:
        pipeline.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    for name, serial in CAMERAS.items():
        print(f"Capturing {name} (serial {serial}) at {WIDTH}x{HEIGHT}...")
        img, intrinsics = grab_photo_and_intrinsics(serial)
        out_path = out_dir / f"{name}_board_photo.png"
        cv2.imwrite(str(out_path), img)
        intrinsics_path = out_dir / f"{name}_board_photo_intrinsics.json"
        with open(intrinsics_path, "w") as f:
            json.dump(intrinsics, f, indent=2)
        print(f"  saved {out_path}")
        print(f"  saved {intrinsics_path}")


if __name__ == "__main__":
    main()
