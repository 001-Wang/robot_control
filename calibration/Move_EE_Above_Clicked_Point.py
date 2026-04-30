#!/usr/bin/env python3
"""
Validate hand-eye calibration by clicking a point on the table in the RGB image
and moving the Franka only in x/y so the current EE origin is above that point.

This script is intentionally conservative:
- the clicked target is computed from RGB ray + known table plane
- the EE keeps its current z height
- the EE keeps its current orientation
- only x/y translation changes

Use this when you want to validate x/y alignment on the table without moving
down toward contact.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from Click_To_Robot_Target import (
    ClickSelector,
    FrankaRobotInterface,
    backproject_pixel_to_camera,
    capture_aligned_rgbd,
    get_robust_depth,
    intersect_camera_ray_with_plane,
    load_gripper_T_camera,
    load_table_plane,
    pose_with_same_orientation_and_new_translation,
    start_realsense_pipeline,
)
from franka_utils import exit_with_error


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def build_vertical_rotation(current_rotation: np.ndarray, vertical_direction: str) -> np.ndarray:
    target_z = np.array([0.0, 0.0, -1.0 if vertical_direction == "down" else 1.0], dtype=np.float64)
    candidate_x = current_rotation[:, 0]
    projected_x = candidate_x - np.dot(candidate_x, target_z) * target_z
    if np.linalg.norm(projected_x) < 1e-6:
        candidate_y = current_rotation[:, 1]
        projected_x = np.cross(candidate_y, target_z)
    target_x = normalize(projected_x)
    target_y = normalize(np.cross(target_z, target_x))
    target_rotation = np.column_stack([target_x, target_y, target_z])
    if np.linalg.det(target_rotation) < 0.0:
        target_y = -target_y
        target_rotation = np.column_stack([target_x, target_y, target_z])
    return target_rotation


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Click a table point in RGB and move the Franka EE only in x/y so the "
            "EE origin stays at its current z height while going above the clicked point."
        )
    )
    parser.add_argument("--robot-ip", type=str, default=None, help="Robot IP address.")
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Path to calibration JSON containing gripper_T_camera.",
    )
    parser.add_argument(
        "--plane",
        type=Path,
        default=Path("table_plane.json"),
        help="Table plane JSON in base frame.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--motion-duration",
        type=float,
        default=4.0,
        help="x/y motion duration in seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the target without moving the robot.",
    )
    parser.add_argument(
        "--skip-motion-confirmation",
        action="store_true",
        help="Skip the interactive confirmation before moving the robot.",
    )
    parser.add_argument(
        "--vertical-direction",
        choices=["down", "up"],
        default="down",
        help="Desired EE local +Z direction in the base frame during the validation move.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if args.robot_ip is None:
        exit_with_error("--robot-ip is required.")

    try:
        gripper_T_camera = load_gripper_T_camera(args.calibration)
        plane_normal, plane_offset = load_table_plane(args.plane)
    except Exception as exc:
        exit_with_error(str(exc))

    print("[INFO] Loaded gripper_T_camera:")
    print(np.array2string(gripper_T_camera, precision=6, suppress_small=True))
    print(f"[INFO] Loaded plane normal={plane_normal.tolist()} offset={plane_offset:.9f}")

    pipeline = None
    try:
        pipeline, align, intrinsics = start_realsense_pipeline(args.width, args.height, args.fps)
        print("[INFO] Color intrinsics:")
        print(json.dumps({k: intrinsics[k] for k in ["fx", "fy", "cx", "cy"]}, indent=2))

        frame = capture_aligned_rgbd(pipeline, align, intrinsics)
        selector = ClickSelector(frame.color_bgr)
        clicked = selector.run()
        if clicked is None:
            print("[INFO] No point selected. Exiting.")
            return 0

        u, v = clicked
        print(f"[INFO] Clicked pixel: u={u}, v={v}")

        robot = FrankaRobotInterface(args.robot_ip)
        robot.connect()
        base_T_gripper = robot.read_base_T_gripper()
        base_T_camera = base_T_gripper @ gripper_T_camera

        print("[INFO] Current base_T_gripper:")
        print(np.array2string(base_T_gripper, precision=6, suppress_small=True))
        print("[INFO] base_T_camera:")
        print(np.array2string(base_T_camera, precision=6, suppress_small=True))

        table_xyz = intersect_camera_ray_with_plane(
            base_T_camera=base_T_camera,
            u=u,
            v=v,
            fx=frame.fx,
            fy=frame.fy,
            cx=frame.cx,
            cy=frame.cy,
            plane_normal=plane_normal,
            plane_offset=plane_offset,
        )
        print(f"[RESULT] Clicked table point in base frame [m]: {table_xyz}")

        depth_m = get_robust_depth(frame.depth_m, u, v, radius=2)
        if np.isfinite(depth_m):
            depth_point_camera = backproject_pixel_to_camera(
                u=u,
                v=v,
                depth_m=depth_m,
                fx=frame.fx,
                fy=frame.fy,
                cx=frame.cx,
                cy=frame.cy,
            )
            depth_point_base = (base_T_camera @ depth_point_camera)[:3]
            print(f"[INFO] Robust depth: {depth_m:.4f} m")
            print(f"[INFO] Depth-based point in base frame [m]: {depth_point_base}")
            print(f"[INFO] Depth-vs-plane delta [m]: {(depth_point_base - table_xyz).tolist()}")
        else:
            print("[WARN] No valid depth near clicked pixel. Plane target still valid for motion.")

        target_xyz = base_T_gripper[:3, 3].copy()
        target_xyz[0] = table_xyz[0]
        target_xyz[1] = table_xyz[1]
        print(f"[INFO] XY-only EE target [m]: {target_xyz}")
        print(
            f"[INFO] Current EE z is held fixed at {base_T_gripper[2, 3]:.6f} m; "
            f"table z at target is {table_xyz[2]:.6f} m."
        )
        target_rotation = build_vertical_rotation(base_T_gripper[:3, :3], args.vertical_direction)
        print(f"[INFO] Target EE vertical direction: {args.vertical_direction}")
        print("[INFO] Target rotation:")
        print(np.array2string(target_rotation, precision=6, suppress_small=True))

        if args.dry_run:
            print("[INFO] Dry run enabled. Not moving robot.")
            return 0

        if not args.skip_motion_confirmation:
            print("WARNING: This will move the robot only in x/y while keeping current z and orientation.")
            print("Verify the printed target first and keep the stop button at hand.")
            try:
                response = input("Type 'move' to continue: ").strip().lower()
            except EOFError:
                print(
                    "[INFO] No interactive stdin is available for motion confirmation. "
                    "Re-run with --skip-motion-confirmation if you want to execute the move."
                )
                return 1
            if response != "move":
                print("[INFO] Motion cancelled by user.")
                return 0

        base_T_target = pose_with_same_orientation_and_new_translation(base_T_gripper, target_xyz)
        base_T_target[:3, :3] = target_rotation
        print("[INFO] Commanded base_T_target:")
        print(np.array2string(base_T_target, precision=6, suppress_small=True))
        print("[INFO] Sending XY-only + vertical-orientation target pose to robot...")
        start_time = time.time()
        robot.move_to_pose(base_T_target, duration=args.motion_duration)
        print(f"[INFO] Motion command completed in {time.time() - start_time:.2f} s.")
        return 0

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        return 130
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
