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
            "Interactively record end-effector touch points and fit a plane in the "
            "robot base frame."
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
        help="Minimum number of touch points to capture before fitting the plane.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file to store captured points and fitted plane.",
    )
    return parser.parse_args()


def read_base_T_ee(robot: object) -> np.ndarray:
    state = robot.read_once()
    return np.array(state.O_T_EE, dtype=float).reshape(4, 4, order="F")


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]

    norm = np.linalg.norm(normal)
    if norm == 0.0:
        raise ValueError("Plane fit failed because the captured points are degenerate.")
    normal = normal / norm

    # Keep the normal direction stable for readability.
    if normal[2] < 0.0:
        normal = -normal

    offset = -float(normal @ centroid)
    return normal, offset, centroid


def to_jsonable_matrix(matrix: np.ndarray) -> list[list[float]]:
    return matrix.tolist()


def to_jsonable_vector(vector: np.ndarray) -> list[float]:
    return vector.tolist()


def main() -> None:
    args = parse_args()
    if args.num_points < 3:
        raise ValueError("--num-points must be at least 3.")

    try:
        robot = connect_robot(args.robot_ip)
    except RuntimeError as exc:
        exit_with_error(str(exc))

    captured: list[dict[str, Any]] = []

    print("Interactive table touch capture")
    print("Move the robot tip to the table contact point, then press Enter to record.")
    print("Type 'q' and press Enter to stop once at least 3 points are recorded.")
    print()

    point_index = 1
    while True:
        user_input = input(
            f"[P{point_index}] Press Enter to record the current pose, or type q to finish: "
        ).strip()
        if user_input.lower() == "q":
            if len(captured) < 3:
                print("Need at least 3 points before fitting a plane.")
                continue
            break

        base_T_ee = read_base_T_ee(robot)
        point = base_T_ee[:3, 3].copy()

        captured.append(
            {
                "id": f"P{point_index}",
                "base_T_ee": to_jsonable_matrix(base_T_ee),
                "point_xyz": to_jsonable_vector(point),
            }
        )

        np.set_printoptions(precision=6, suppress=True)
        print(f"Recorded P{point_index}: {point}")
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

    result = {
        "robot_ip": args.robot_ip,
        "num_points": len(captured),
        "points": captured,
        "plane": {
            "normal": to_jsonable_vector(normal),
            "offset": offset,
            "centroid": to_jsonable_vector(centroid),
            "equation": {
                "a": float(normal[0]),
                "b": float(normal[1]),
                "c": float(normal[2]),
                "d": offset,
            },
        },
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
