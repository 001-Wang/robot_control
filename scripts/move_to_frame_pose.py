#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from pylibfranka import CartesianPose, ControllerMode, RealtimeConfig, Robot


def pose_list_to_matrix(pose: list[float]) -> np.ndarray:
    return np.array(pose, dtype=float).reshape(4, 4, order="F")


def matrix_to_pose_list(matrix: np.ndarray) -> list[float]:
    return ensure_transform(matrix, name="Cartesian pose").reshape(16, order="F").tolist()


def project_rotation_to_so3(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=float))
    projected = u @ vt
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


def ensure_transform(matrix: Any, *, name: str) -> np.ndarray:
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix.")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{name} final row must be [0, 0, 0, 1].")
    transform = transform.copy()
    transform[:3, :3] = project_rotation_to_so3(transform[:3, :3])
    return transform


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=float)[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    quaternion = np.array([x, y, z, w], dtype=float)
    return quaternion / np.linalg.norm(quaternion)


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def slerp_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    return (
        np.sin(theta_0 - theta) / sin_theta_0 * q0
        + np.sin(theta) / sin_theta_0 * q1
    )


def smoothstep5(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha**2)


def interpolate_pose(start: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    scaled = smoothstep5(alpha)
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = (1.0 - scaled) * start[:3, 3] + scaled * target[:3, 3]
    q0 = matrix_to_quaternion_xyzw(start)
    q1 = matrix_to_quaternion_xyzw(target)
    pose[:3, :3] = quaternion_xyzw_to_matrix(slerp_xyzw(q0, q1, scaled))
    return pose


def rotation_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a, dtype=float)[:3, :3].T @ np.asarray(b, dtype=float)[:3, :3]
    cos_angle = (float(np.trace(relative)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def apply_collision_behavior(robot: Robot) -> None:
    lower_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
    upper_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
    lower_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
    upper_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
    robot.set_collision_behavior(
        lower_torque_thresholds,
        upper_torque_thresholds,
        lower_force_thresholds,
        upper_force_thresholds,
    )


def parse_frame_index_from_path(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if match is None:
        raise ValueError(f"Could not parse frame index from image name: {path}")
    return int(match.group(1))


def run_dir_from_image_path(image_path: Path) -> Path:
    if image_path.parent.name in {"rgb", "depth"}:
        return image_path.parent.parent
    return image_path.parent


def load_frame_entry(run_dir: Path, image_path: Path | None, frame_index: int) -> dict[str, Any]:
    poses_path = run_dir / "frame_poses.jsonl"
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing pose log: {poses_path}")

    relative_image = None
    if image_path is not None:
        try:
            relative_image = image_path.resolve().relative_to(run_dir.resolve()).as_posix()
        except ValueError:
            relative_image = None

    fallback = None
    for line in poses_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if relative_image is not None and entry.get("rgb_path") == relative_image:
            return entry
        if int(entry.get("frame_index", -1)) == frame_index:
            fallback = entry

    if fallback is not None:
        return fallback
    raise ValueError(f"Frame index {frame_index} was not found in {poses_path}")


def target_matrix_from_entry(entry: dict[str, Any]) -> np.ndarray:
    if "base_T_gripper" in entry:
        payload = entry["base_T_gripper"]
    elif "base_T_ee" in entry:
        payload = entry["base_T_ee"]
    else:
        raise ValueError("Frame entry does not contain base_T_gripper or base_T_ee.")
    matrix = payload["matrix"] if isinstance(payload, dict) and "matrix" in payload else payload
    return ensure_transform(matrix, name="target frame pose")


def main() -> int:
    parser = argparse.ArgumentParser(description="Move Franka EE/gripper to a saved frame pose.")
    parser.add_argument("--robot-ip", "--ip", default="192.168.1.11", help="Franka robot IP address.")
    parser.add_argument("--image", type=Path, help="Path to rgb_*.png for the target frame.")
    parser.add_argument("--run-dir", type=Path, help="Run directory containing frame_poses.jsonl.")
    parser.add_argument("--frame-index", type=int, help="Frame index to load when --image is not enough.")
    parser.add_argument("--duration-sec", type=float, default=8.0, help="Motion duration in seconds.")
    parser.add_argument("--hold-sec", type=float, default=1.0, help="Hold target pose after motion.")
    parser.add_argument("--dry-run", action="store_true", help="Print target pose and exit before moving.")
    parser.add_argument("--yes", action="store_true", help="Skip interactive safety confirmation.")
    args = parser.parse_args()

    if args.duration_sec <= 0.0:
        raise ValueError("--duration-sec must be positive.")
    if args.hold_sec < 0.0:
        raise ValueError("--hold-sec must be non-negative.")
    if args.image is None and (args.run_dir is None or args.frame_index is None):
        raise ValueError("Provide --image, or provide both --run-dir and --frame-index.")

    image_path = args.image.resolve() if args.image is not None else None
    run_dir = args.run_dir.resolve() if args.run_dir is not None else run_dir_from_image_path(image_path)
    frame_index = args.frame_index if args.frame_index is not None else parse_frame_index_from_path(image_path)
    entry = load_frame_entry(run_dir, image_path, frame_index)
    target = target_matrix_from_entry(entry)

    print(f"Run dir: {run_dir}")
    print(f"Frame index: {entry.get('frame_index', frame_index)}")
    print(f"RGB path: {entry.get('rgb_path')}")
    print("Target base_T_gripper:")
    print(target)
    print("Target xyz:", target[:3, 3].tolist())

    if args.dry_run:
        return 0

    print("WARNING: this will move the robot.")
    if not args.yes:
        try:
            input("Press Enter only if the workspace is clear and the user stop is ready...")
        except EOFError:
            print(
                "No interactive stdin is available. Re-run with --yes only after "
                "confirming the workspace is clear and the user stop is ready."
            )
            return 1

    robot = Robot(args.robot_ip, RealtimeConfig.kIgnore)
    apply_collision_behavior(robot)
    initial_state = robot.read_once()
    current = pose_list_to_matrix(initial_state.O_T_EE)
    translation_error_m = float(np.linalg.norm(current[:3, 3] - target[:3, 3]))
    rotation_error_deg = rotation_distance_deg(current, target)
    print("Current base_T_gripper:")
    print(current)
    print(
        "Target error: "
        f"{translation_error_m * 1000.0:.2f} mm, {rotation_error_deg:.2f} deg"
    )

    active_control = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)
    robot_state, period = active_control.readOnce()
    start_pose = pose_list_to_matrix(robot_state.O_T_EE)

    elapsed = 0.0
    motion_done = False
    while not motion_done:
        elapsed += period.to_sec()
        alpha = min(elapsed / args.duration_sec, 1.0)
        commanded_pose = interpolate_pose(start_pose, target, alpha)
        command = CartesianPose(matrix_to_pose_list(commanded_pose))
        if elapsed >= args.duration_sec:
            motion_done = True
            if args.hold_sec <= 0.0:
                command.motion_finished = True
        active_control.writeOnce(command)
        if not motion_done or args.hold_sec > 0.0:
            robot_state, period = active_control.readOnce()

    hold_elapsed = 0.0
    while hold_elapsed < args.hold_sec:
        hold_elapsed += period.to_sec()
        command = CartesianPose(matrix_to_pose_list(target))
        if hold_elapsed >= args.hold_sec:
            command.motion_finished = True
        active_control.writeOnce(command)
        if hold_elapsed < args.hold_sec:
            robot_state, period = active_control.readOnce()

    print("Finished moving to frame pose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
