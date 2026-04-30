#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from franka_utils import connect_robot, exit_with_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the current Franka pose and compare the end-effector origin and "
            "camera position against a fitted plane."
        )
    )
    parser.add_argument(
        "--robot-ip",
        default="192.168.1.11",
        help="Franka robot IP address.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("data/calibration_result_fixed.json"),
        help="Path to calibration_result JSON.",
    )
    parser.add_argument(
        "--plane",
        type=Path,
        default=Path("table_plane.json"),
        help="Path to fitted plane JSON.",
    )
    return parser.parse_args()


def load_matrix(transform: dict) -> np.ndarray:
    matrix = np.asarray(transform["matrix"], dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("Transform matrix must have shape 4x4.")
    return matrix


def signed_distance_to_plane(point: np.ndarray, normal: np.ndarray, offset: float) -> float:
    return float(normal @ point + offset)


def main() -> None:
    args = parse_args()

    calibration = json.loads(args.calibration.read_text())
    plane_data = json.loads(args.plane.read_text())

    gripper_T_camera = load_matrix(calibration["gripper_T_camera"])
    normal = np.asarray(plane_data["plane"]["normal"], dtype=float)
    offset = float(plane_data["plane"]["offset"])

    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0.0:
        raise ValueError("Plane normal must be non-zero.")
    normal = normal / normal_norm
    offset = offset / normal_norm

    try:
        robot = connect_robot(args.robot_ip)
    except RuntimeError as exc:
        exit_with_error(str(exc))

    state = robot.read_once()
    base_T_gripper = np.array(state.O_T_EE, dtype=float).reshape(4, 4, order="F")
    base_T_camera = base_T_gripper @ gripper_T_camera

    ee_position = base_T_gripper[:3, 3]
    camera_position = base_T_camera[:3, 3]

    camera_offset_in_gripper = gripper_T_camera[:3, 3]
    camera_offset_in_base = base_T_gripper[:3, :3] @ camera_offset_in_gripper

    ee_distance = signed_distance_to_plane(ee_position, normal, offset)
    camera_distance = signed_distance_to_plane(camera_position, normal, offset)
    predicted_delta = float(normal @ camera_offset_in_base)

    np.set_printoptions(precision=6, suppress=True)
    print("Current base_T_gripper:")
    print(base_T_gripper)
    print()
    print("Calibrated gripper_T_camera:")
    print(gripper_T_camera)
    print()
    print(f"Plane normal: {normal.tolist()}")
    print(f"Plane offset: {offset:.9f}")
    print()
    print(f"EE origin in base frame: {ee_position.tolist()}")
    print(f"Camera position in base frame: {camera_position.tolist()}")
    print()
    print(f"Camera offset in gripper frame: {camera_offset_in_gripper.tolist()}")
    print(f"Camera offset in base frame: {camera_offset_in_base.tolist()}")
    print()
    print(f"Signed EE-origin distance to plane: {ee_distance:.6f} m")
    print(f"Signed camera distance to plane:    {camera_distance:.6f} m")
    print(f"Camera-EE plane distance delta:     {camera_distance - ee_distance:.6f} m")
    print(f"Predicted delta from rotated offset:{predicted_delta:.6f} m")


if __name__ == "__main__":
    main()
