"""Compute the actual (measured) camera-to-robot-base pose for cam_high/cam_low using:
  - 4 gripper touch points on one ArUco marker corner each (see capture_touch_points.py),
    forward-kinematics'd through the URDF to ground the marker in the robot base frame.
  - solvePnP of that same marker as seen by each camera (see capture_board_photos.py),
    using the intrinsics captured alongside each photo (<photo>_intrinsics.json).

Compares the result against the URDF's hardcoded nominal mount pose for the same camera link,
so you can see how far the CAD assumption is from the physically measured pose.

Also composes the measured color-to-base pose with each camera's factory-calibrated
depth-to-color extrinsics (from get_camera_calibration.py's output) to get depth-to-base,
since the marker was only photographed in the color stream — no separate depth measurement
is needed, the depth<->color relationship inside each camera unit is already known.

Usage:
    python lerobot/scripts/solve_camera_extrinsics.py \
        --touch-points touch_points.json \
        --marker-id 146 --marker-length-m 0.043 \
        --photo cam_high=cam_high_board_photo.png --photo cam_low=cam_low_board_photo.png \
        --urdf stationary_ai.urdf \
        --calibration-json camera_calibration.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

RIGHT_EE_LINK = "follower_right_ee_gripper_link"
LEFT_EE_LINK = "follower_left_ee_gripper_link"
CAMERA_LINKS = {
    "cam_high": "cam_high_color_optical_frame",
    "cam_low": "cam_low_color_optical_frame",
}
DICTIONARY = cv2.aruco.DICT_ARUCO_MIP_36H12


def load_urdf(urdf_path: str) -> pin.Model:
    return pin.buildModelFromUrdf(str(urdf_path))


def fk_ee_position(model: pin.Model, data: pin.Data, arm: str, joint_values: list[float]) -> np.ndarray:
    q = np.zeros(model.nq)
    for j in range(6):
        joint_name = f"follower_{arm}_joint_{j}"
        if model.existJointName(joint_name):
            q[model.joints[model.getJointId(joint_name)].idx_q] = float(joint_values[j])
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    link = RIGHT_EE_LINK if arm == "right" else LEFT_EE_LINK
    return np.array(data.oMf[model.getFrameId(link)].translation)


def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve R, t minimizing sum ||R @ src[i] + t - dst[i]||^2. src/dst: (N, 3)."""
    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = dst.mean(axis=0) - R @ src.mean(axis=0)
    return R, t


def marker_object_points(marker_length_m: float) -> np.ndarray:
    s = marker_length_m / 2
    return np.array(
        [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64
    )  # 0=TL, 1=TR, 2=BR, 3=BL — matches label_marker_corners.py


def detect_marker_corners(image_path: str, marker_id: int) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or marker_id not in ids.flatten():
        raise RuntimeError(f"Marker id {marker_id} not found in {image_path}")
    idx = list(ids.flatten()).index(marker_id)
    return corners[idx][0].astype(np.float64)  # (4, 2)


def camera_intrinsics_matrix(photo_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads intrinsics from the "<photo>_intrinsics.json" saved by capture_board_photos.py,
    which matches the actual resolution the photo was taken at (unlike camera_calibration.json,
    which is fixed at the 640x480 recording resolution)."""
    intrinsics_path = str(Path(photo_path).with_suffix("")) + "_intrinsics.json"
    with open(intrinsics_path) as f:
        ci = json.load(f)
    K = np.array([[ci["fx"], 0, ci["ppx"]], [0, ci["fy"], ci["ppy"]], [0, 0, 1]], dtype=np.float64)
    dist = np.array(ci["coeffs"], dtype=np.float64)
    return K, dist


def rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    R_rel = R1.T @ R2
    angle = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
    return angle


def load_depth_to_color(calibration_json: str, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    """librealsense's rs2_extrinsics.rotation is a COLUMN-major flat 3x3, so reshape with
    order='F' (not the default row-major 'C') or this silently comes out transposed."""
    with open(calibration_json) as f:
        calib = json.load(f)
    ext = calib[camera_name]["depth_to_color_extrinsics"]
    R = np.array(ext["rotation"], dtype=np.float64).reshape(3, 3, order="F")  # P_color = R @ P_depth + t
    t = np.array(ext["translation"], dtype=np.float64)
    return R, t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--touch-points", required=True)
    parser.add_argument("--marker-id", type=int, required=True)
    parser.add_argument("--marker-length-m", type=float, required=True)
    parser.add_argument(
        "--photo",
        action="append",
        required=True,
        help="camera_name=path/to/photo.png, repeatable (once per camera)",
    )
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--calibration-json", default=None, help="get_camera_calibration.py output, for depth-to-base")
    parser.add_argument(
        "--output", default=None, help="Write one consolidated JSON (intrinsics + measured base-frame poses)"
    )
    args = parser.parse_args()

    calib_full = None
    if args.calibration_json:
        with open(args.calibration_json) as f:
            calib_full = json.load(f)
    result = {}

    with open(args.touch_points) as f:
        touch_data = json.load(f)
    arm = touch_data["arm"]
    corners = touch_data["corners"]  # {"0": [...7 joint values...], "1": [...], ...}

    model = load_urdf(args.urdf)
    data = model.createData()

    p_base = np.array([fk_ee_position(model, data, arm, corners[str(i)]) for i in range(4)])
    obj_points = marker_object_points(args.marker_length_m)

    R_board2base, t_board2base = kabsch(obj_points, p_base)
    print("Touch-point residuals (mm), board-to-base fit:")
    for i in range(4):
        pred = R_board2base @ obj_points[i] + t_board2base
        print(f"  corner {i}: residual = {np.linalg.norm(pred - p_base[i]) * 1000:.2f} mm")

    photos = dict(p.split("=", 1) for p in args.photo)

    for cam_name, photo_path in photos.items():
        img_points = detect_marker_corners(photo_path, args.marker_id)
        K, dist = camera_intrinsics_matrix(photo_path)

        ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)
        if not ok:
            print(f"{cam_name}: solvePnP failed")
            continue
        R_tag2cam, _ = cv2.Rodrigues(rvec)
        t_tag2cam = tvec.flatten()

        R_cam2base = R_board2base @ R_tag2cam.T
        t_cam2base = t_board2base - R_board2base @ R_tag2cam.T @ t_tag2cam
        quat_measured = Rotation.from_matrix(R_cam2base).as_quat()  # xyzw

        # URDF nominal (CAD) value for comparison
        pin.forwardKinematics(model, data, np.zeros(model.nq))
        pin.updateFramePlacements(model, data)
        nominal_placement = data.oMf[model.getFrameId(CAMERA_LINKS[cam_name])]
        t_nominal = np.array(nominal_placement.translation)
        R_nominal = np.array(nominal_placement.rotation)

        translation_delta_mm = np.linalg.norm(t_cam2base - t_nominal) * 1000
        rotation_delta_deg = rotation_angle_deg(R_nominal, R_cam2base)

        print(f"\n=== {cam_name} (color) ===")
        print(f"Measured camera-to-base position (m): {t_cam2base.tolist()}")
        print(f"Measured camera-to-base quat (xyzw):   {quat_measured.tolist()}")
        print(f"URDF nominal position (m):             {t_nominal.tolist()}")
        print(f"Delta vs URDF nominal: {translation_delta_mm:.1f} mm translation, {rotation_delta_deg:.2f} deg rotation")

        cam_result = {
            "color_to_base": {"position": t_cam2base.tolist(), "quat_xyzw": quat_measured.tolist()},
            "urdf_nominal_color_position": t_nominal.tolist(),
            "delta_vs_urdf_nominal": {"translation_mm": translation_delta_mm, "rotation_deg": rotation_delta_deg},
        }

        if args.calibration_json:
            R_d2c, t_d2c = load_depth_to_color(args.calibration_json, cam_name)
            R_depth2base = R_cam2base @ R_d2c
            t_depth2base = R_cam2base @ t_d2c + t_cam2base
            quat_depth2base = Rotation.from_matrix(R_depth2base).as_quat()
            print(f"--- {cam_name} (depth, composed via factory depth->color) ---")
            print(f"Measured depth-to-base position (m): {t_depth2base.tolist()}")
            print(f"Measured depth-to-base quat (xyzw):   {quat_depth2base.tolist()}")
            cam_result["depth_to_base"] = {
                "position": t_depth2base.tolist(),
                "quat_xyzw": quat_depth2base.tolist(),
            }

        if calib_full is not None and cam_name in calib_full:
            cam_result.update(calib_full[cam_name])  # serial, intrinsics, depth_scale, internal extrinsics

        result[cam_name] = cam_result

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved consolidated calibration to {args.output}")


if __name__ == "__main__":
    main()
