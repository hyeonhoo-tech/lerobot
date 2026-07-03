"""Detect one ArUco marker in a photo and label its 4 corners (0=top-left, 1=top-right,
2=bottom-right, 3=bottom-left, in OpenCV's canonical detection order) so you know exactly
which physical corner to touch with the gripper for each index in capture_touch_points.py.

Usage:
    python lerobot/scripts/label_marker_corners.py --image cam_high_board_photo.png --marker-id 146
"""

import argparse
from pathlib import Path

import cv2

DICTIONARY = cv2.aruco.DICT_ARUCO_MIP_36H12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--marker-id", type=int, required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or args.marker_id not in ids.flatten():
        raise RuntimeError(f"Marker id {args.marker_id} not found in {args.image}")

    idx = list(ids.flatten()).index(args.marker_id)
    pts = corners[idx][0]  # shape (4, 2), order: top-left, top-right, bottom-right, bottom-left

    labels = ["0 (top-left)", "1 (top-right)", "2 (bottom-right)", "3 (bottom-left)"]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for i, (pt, label, color) in enumerate(zip(pts, labels, colors)):
        pt = tuple(int(v) for v in pt)
        cv2.circle(img, pt, 6, color, -1)
        cv2.putText(img, label, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_path = args.output or str(Path(args.image).with_suffix("")) + f"_marker{args.marker_id}_labeled.png"
    cv2.imwrite(out_path, img)
    print(f"Saved labeled image to {out_path}")
    print("Corner pixel coords (0=TL, 1=TR, 2=BR, 3=BL):")
    for i, pt in enumerate(pts):
        print(f"  {i}: {pt.tolist()}")


if __name__ == "__main__":
    main()
