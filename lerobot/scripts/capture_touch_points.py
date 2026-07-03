"""Teleop the right arm and, for each of the marker's 4 corners (see label_marker_corners.py
for which physical corner is which index), press SPACE when the gripper tip is touching that
corner to log the follower's joint state. Used by solve_camera_extrinsics.py to ground the
board in the robot base frame via forward kinematics.

Controls: SPACE = capture current corner and advance, ESC = quit early.

Usage:
    python lerobot/scripts/capture_touch_points.py --output touch_points.json
"""

import argparse
import json
import time

from pynput import keyboard

from lerobot.common.robot_devices.robots.configs import TrossenAIStationaryRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

CORNER_LABELS = ["0 (top-left)", "1 (top-right)", "2 (bottom-right)", "3 (bottom-left)"]
FPS = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="touch_points.json")
    parser.add_argument("--arm", default="right", choices=["right", "left"])
    args = parser.parse_args()

    events = {"capture": False, "quit": False}

    def on_press(key):
        if key == keyboard.Key.space:
            events["capture"] = True
        elif key == keyboard.Key.esc:
            events["quit"] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    cfg = TrossenAIStationaryRobotConfig()
    cfg.cameras = {}
    robot = ManipulatorRobot(cfg)
    robot.connect()

    touches = {}
    try:
        corner_idx = 0
        print("Teleoperating. Move the leader arm to drive the follower.")
        while corner_idx < len(CORNER_LABELS) and not events["quit"]:
            print(f"\n>>> Touch corner {CORNER_LABELS[corner_idx]} with the gripper tip, then press SPACE.")
            events["capture"] = False
            while not events["capture"] and not events["quit"]:
                t0 = time.perf_counter()
                robot.teleop_step(record_data=False)
                dt = time.perf_counter() - t0
                time.sleep(max(0.0, 1 / FPS - dt))

            if events["quit"]:
                break

            state = robot.follower_arms[args.arm].read("Present_Position")
            touches[str(corner_idx)] = state.tolist()
            print(f"Captured corner {corner_idx}: {state.tolist()}")
            corner_idx += 1

        if len(touches) == len(CORNER_LABELS):
            with open(args.output, "w") as f:
                json.dump({"arm": args.arm, "corners": touches}, f, indent=2)
            print(f"\nSaved {len(touches)} touch points to {args.output}")
        else:
            print(f"\nQuit early with only {len(touches)}/{len(CORNER_LABELS)} corners captured. Not saved.")
    finally:
        robot.disconnect()
        listener.stop()


if __name__ == "__main__":
    main()
