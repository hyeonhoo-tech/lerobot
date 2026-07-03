"""Live tool to help re-align a bumped RealSense camera back to the angle it had during
a previous dataset recording.

Shows the current live view overlaid with edges from a reference frame pulled out of the
recorded dataset's video, so you can visually nudge the camera until the edges line up again.
Also prints an approximate rotation estimate (assumes the camera moved by pure rotation, i.e.
no translation) from ORB feature matching + homography decomposition, using each camera's real
intrinsics from calibration.json (see get_camera_calibration.py) if present.

Usage:
    python lerobot/scripts/match_camera_angle.py --camera cam_high --ref cam_high_ref_frame.png --serial 146322077603
    python lerobot/scripts/match_camera_angle.py --camera cam_low --ref cam_low_ref_frame.png --serial 207222071967

Press 'q' to quit, 'b' to toggle blend/edge-overlay display mode.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

WIDTH = 640
HEIGHT = 480
FPS = 30


def load_intrinsics(calibration_json: str | None, camera_name: str) -> np.ndarray | None:
    if calibration_json is None or not Path(calibration_json).exists():
        return None
    with open(calibration_json) as f:
        calib = json.load(f)
    if camera_name not in calib:
        return None
    ci = calib[camera_name]["color_intrinsics"]
    return np.array(
        [[ci["fx"], 0, ci["ppx"]], [0, ci["fy"], ci["ppy"]], [0, 0, 1]], dtype=np.float64
    )


def estimate_rotation_deg(ref_gray: np.ndarray, live_gray: np.ndarray, K: np.ndarray | None):
    """Approximate rotation-only camera motion between ref and live frame via ORB + homography.

    Returns (roll, pitch, yaw) in degrees, or None if not enough matches / no K available.
    """
    if K is None:
        return None

    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(live_gray, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 15:
        return None
    matches = sorted(matches, key=lambda m: m.distance)[:200]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if H is None or mask is None or mask.sum() < 12:
        return None

    num, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
    # Pick the candidate with smallest translation magnitude (closest to pure rotation).
    best = min(range(num), key=lambda i: np.linalg.norm(translations[i]))
    R = rotations[best]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    pitch = np.degrees(np.arctan2(-R[2, 0], sy))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", required=True, help="Camera name, e.g. cam_high")
    parser.add_argument("--serial", required=True, help="RealSense serial number")
    parser.add_argument("--ref", required=True, help="Path to reference frame image")
    parser.add_argument(
        "--calibration-json", default=None, help="Path to get_camera_calibration.py output JSON"
    )
    args = parser.parse_args()

    ref = cv2.imread(args.ref)
    if ref is None:
        raise FileNotFoundError(args.ref)
    ref = cv2.resize(ref, (WIDTH, HEIGHT))
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref_edges = cv2.Canny(ref_gray, 60, 150)

    K = load_intrinsics(args.calibration_json, args.camera)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(str(args.serial))
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(config)

    blend_mode = False
    window = f"match_camera_angle: {args.camera} (q=quit, b=toggle view)"
    print("Move/adjust the camera until the red (reference) and green (live) edges line up.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            live = np.asanyarray(color_frame.get_data())
            live_gray = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)

            if blend_mode:
                display = cv2.addWeighted(ref, 0.5, live, 0.5, 0)
            else:
                live_edges = cv2.Canny(live_gray, 60, 150)
                display = live.copy()
                display[ref_edges > 0] = (0, 0, 255)  # reference edges in red
                display[live_edges > 0] = (0, 255, 0)  # live edges in green

            rot = estimate_rotation_deg(ref_gray, live_gray, K)
            if rot is not None:
                roll, pitch, yaw = rot
                text = f"approx delta roll={roll:+.1f} pitch={pitch:+.1f} yaw={yaw:+.1f} deg"
            else:
                text = "approx delta: not enough matches"
            cv2.putText(display, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("b"):
                blend_mode = not blend_mode
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
