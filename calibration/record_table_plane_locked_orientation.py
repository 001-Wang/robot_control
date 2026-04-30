#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from franka_utils import connect_robot, exit_with_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively record end-effector points while keeping the EE rotation "
            "close to the current reference orientation, then fit a plane."
        )
    )
    parser.add_argument(
        "--robot-ip",
        default="192.168.1.11",
        help="Franka robot IP address.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=3,
        help="Minimum number of points to capture before fitting the plane.",
    )
    parser.add_argument(
        "--max-angle-deg",
        type=float,
        default=2.0,
        help="Maximum allowed rotation drift from the reference orientation in degrees.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Reject points whose orientation drift exceeds --max-angle-deg.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("table_plane_locked_orientation.json"),
        help="JSON file to store captured points and fitted plane.",
    )
    return parser.parse_args()


def read_base_T_ee(robot: object) -> np.ndarray:
    state = robot.read_once()
    return np.array(state.O_T_EE, dtype=float).reshape(4, 4, order="F")


def rotation_angle_deg(reference_rotation: np.ndarray, current_rotation: np.ndarray) -> float:
    delta = reference_rotation.T @ current_rotation
    value = (np.trace(delta) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]

    norm = np.linalg.norm(normal)
    if norm == 0.0:
        raise ValueError("Plane fit failed because the captured points are degenerate.")
    normal = normal / norm
    if normal[2] < 0.0:
        normal = -normal

    offset = -float(normal @ centroid)
    return normal, offset, centroid


def to_jsonable_matrix(matrix: np.ndarray) -> list[list[float]]:
    return matrix.tolist()


def to_jsonable_vector(vector: np.ndarray) -> list[float]:
    return vector.tolist()


def save_partial_result(
    *,
    output_path: Path,
    robot_ip: str,
    reference_pose: np.ndarray,
    reference_position: np.ndarray,
    max_angle_deg: float,
    strict: bool,
    captured: list[dict[str, Any]],
) -> None:
    result: dict[str, Any] = {
        "robot_ip": robot_ip,
        "num_points": len(captured),
        "reference_base_T_ee": to_jsonable_matrix(reference_pose),
        "reference_position_xyz": to_jsonable_vector(reference_position),
        "max_angle_deg": max_angle_deg,
        "strict": strict,
        "points": captured,
    }

    if len(captured) >= 3:
        points = np.asarray([entry["point_xyz"] for entry in captured], dtype=float)
        normal, offset, centroid = fit_plane(points)
        result["plane"] = {
            "normal": to_jsonable_vector(normal),
            "offset": offset,
            "centroid": to_jsonable_vector(centroid),
            "equation": {
                "a": float(normal[0]),
                "b": float(normal[1]),
                "c": float(normal[2]),
                "d": offset,
            },
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))


def main() -> None:
    args = parse_args()
    if args.num_points < 3:
        raise ValueError("--num-points must be at least 3.")
    if args.max_angle_deg < 0.0:
        raise ValueError("--max-angle-deg must be non-negative.")

    try:
        robot = connect_robot(args.robot_ip)
    except RuntimeError as exc:
        exit_with_error(str(exc))

    reference_pose = read_base_T_ee(robot)
    reference_rotation = reference_pose[:3, :3].copy()
    reference_position = reference_pose[:3, 3].copy()

    np.set_printoptions(precision=6, suppress=True)
    print("Locked-orientation table plane capture")
    print("The current EE rotation is now the reference orientation.")
    print(f"Reference position: {reference_position}")
    print(f"Max allowed drift: {args.max_angle_deg:.3f} deg")
    if args.strict:
        print("Strict mode: points beyond the angle threshold are rejected.")
    else:
        print("Non-strict mode: points beyond the threshold are kept but flagged.")
    print()
    print("Reference base_T_ee:")
    print(reference_pose)
    print()
    print("Move parallel to the table while keeping the same wrist orientation.")
    print("Press Enter to record the current pose, or type q to finish.")
    print()

    captured: list[dict[str, Any]] = []
    point_index = 1
    while True:
        user_input = input(f"[P{point_index}] Press Enter to record, or type q to finish: ").strip()
        if user_input.lower() == "q":
            if len(captured) < 3:
                print("Need at least 3 accepted points before fitting a plane.")
                continue
            break

        try:
            base_T_ee = read_base_T_ee(robot)
        except RuntimeError as exc:
            print()
            print(f"Robot read failed while recording P{point_index}: {exc}")
            if captured:
                save_partial_result(
                    output_path=args.output,
                    robot_ip=args.robot_ip,
                    reference_pose=reference_pose,
                    reference_position=reference_position,
                    max_angle_deg=args.max_angle_deg,
                    strict=args.strict,
                    captured=captured,
                )
                print(f"Saved partial results to {args.output}")
            else:
                print("No points were recorded yet, so nothing was saved.")
            print("Reconnect FCI / robot communication, then restart the script.")
            raise SystemExit(1)

        rotation = base_T_ee[:3, :3]
        point = base_T_ee[:3, 3].copy()
        angle_deg = rotation_angle_deg(reference_rotation, rotation)

        if args.strict and angle_deg > args.max_angle_deg:
            print(
                f"Rejected P{point_index}: rotation drift {angle_deg:.3f} deg "
                f"exceeds threshold {args.max_angle_deg:.3f} deg."
            )
            print()
            continue

        captured.append(
            {
                "id": f"P{point_index}",
                "base_T_ee": to_jsonable_matrix(base_T_ee),
                "point_xyz": to_jsonable_vector(point),
                "rotation_drift_deg": angle_deg,
                "within_angle_threshold": angle_deg <= args.max_angle_deg,
            }
        )

        status = "OK" if angle_deg <= args.max_angle_deg else "OUT_OF_RANGE"
        print(f"Recorded P{point_index}: {point} | rotation drift = {angle_deg:.3f} deg | {status}")
        print(base_T_ee)
        print()

        point_index += 1
        if len(captured) >= args.num_points:
            print(
                f"Minimum point count reached ({args.num_points}). "
                "Add more points or type q to fit the plane."
            )

    points = np.asarray([entry["point_xyz"] for entry in captured], dtype=float)
    normal, offset, centroid = fit_plane(points)

    print("Fitted plane in base frame")
    print(f"Points used: {len(captured)}")
    print(f"Centroid: {centroid}")
    print(f"Normal:   {normal}")
    print(
        "Equation: "
        f"{normal[0]:.9f} * x + {normal[1]:.9f} * y + {normal[2]:.9f} * z + {offset:.9f} = 0"
    )

    save_partial_result(
        output_path=args.output,
        robot_ip=args.robot_ip,
        reference_pose=reference_pose,
        reference_position=reference_position,
        max_angle_deg=args.max_angle_deg,
        strict=args.strict,
        captured=captured,
    )
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
