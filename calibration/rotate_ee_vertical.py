#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import numpy as np
from pylibfranka import ControllerMode, RealtimeConfig, Robot, CartesianPose

from franka_utils import exit_with_error


def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(
        description=(
            "Rotate the current end-effector orientation so its local Z axis aligns "
            "with base vertical while keeping the current EE position fixed."
        )
    )
    parser.add_argument(
        "--robot-ip",
        default="192.168.1.11",
        help="Franka robot IP address.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Move duration in seconds.",
    )
    parser.add_argument(
        "--vertical-direction",
        choices=["down", "up"],
        default="down",
        help="Desired EE local +Z direction in the base frame.",
    )
    parser.add_argument(
        "--print-rate-hz",
        type=float,
        default=5.0,
        help="Status print rate in Hz while the motion is running.",
    )
    return parser.parse_args()


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector.reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def so3_log(rotation: np.ndarray) -> np.ndarray:
    value = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(value))
    if angle < 1e-9:
        return np.zeros(3, dtype=float)
    omega_hat = (rotation - rotation.T) / (2.0 * np.sin(angle))
    return angle * np.array(
        [omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]],
        dtype=float,
    )


def so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-9:
        return np.eye(3, dtype=float)
    axis = rotation_vector / angle
    axis_hat = skew(axis)
    return (
        np.eye(3, dtype=float)
        + np.sin(angle) * axis_hat
        + (1.0 - np.cos(angle)) * (axis_hat @ axis_hat)
    )


def interpolate_rotation(start_rotation: np.ndarray, target_rotation: np.ndarray, alpha: float) -> np.ndarray:
    delta = start_rotation.T @ target_rotation
    return start_rotation @ so3_exp(alpha * so3_log(delta))


def rotation_angle_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    delta = rotation_a.T @ rotation_b
    value = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))


def build_vertical_rotation(current_rotation: np.ndarray, vertical_direction: str) -> np.ndarray:
    target_z = np.array([0.0, 0.0, -1.0 if vertical_direction == "down" else 1.0], dtype=float)

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


def rotation_from_pose_list(pose: list[float]) -> np.ndarray:
    return np.array(pose, dtype=float).reshape(4, 4, order="F")[:3, :3]


def translation_from_pose_list(pose: list[float]) -> np.ndarray:
    return np.array(pose, dtype=float).reshape(4, 4, order="F")[:3, 3]


def write_rotation_into_pose_list(pose: list[float], rotation: np.ndarray) -> list[float]:
    transform = np.array(pose, dtype=float).reshape(4, 4, order="F").copy()
    transform[:3, :3] = rotation
    return transform.reshape(16, order="F").tolist()


def main() -> None:
    args = parse_args()
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive.")
    if args.print_rate_hz <= 0.0:
        raise ValueError("--print-rate-hz must be positive.")

    robot = None
    try:
        robot = Robot(args.robot_ip, RealtimeConfig.kIgnore)
    except Exception as exc:
        exit_with_error(f"Failed to connect to the Franka robot at {args.robot_ip}: {exc}")

    try:
        print("WARNING: This example will move the robot.")
        print("Keep the user stop button at hand.")
        input("Press Enter to continue...")

        active_control = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)

        robot_state, duration = active_control.readOnce()
        initial_pose = robot_state.O_T_EE.copy()
        start_rotation = rotation_from_pose_list(initial_pose)
        fixed_translation = translation_from_pose_list(initial_pose)
        target_rotation = build_vertical_rotation(start_rotation, args.vertical_direction)
        total_angle_deg = rotation_angle_deg(start_rotation, target_rotation)

        np.set_printoptions(precision=6, suppress=True)
        print("Rotating EE to vertical orientation")
        print(f"Duration: {args.duration:.3f} s")
        print(f"Vertical direction: {args.vertical_direction}")
        print(f"Total orientation change: {total_angle_deg:.3f} deg")
        print("Position will be held fixed during the move.")
        print()
        print("Initial base_T_ee:")
        print(np.array(initial_pose, dtype=float).reshape(4, 4, order='F'))
        print()
        print("Target rotation:")
        print(target_rotation)
        print()

        time_elapsed = 0.0
        motion_finished = False
        last_print_time = 0.0
        print_period_s = 1.0 / args.print_rate_hz

        while not motion_finished:
            robot_state, duration = active_control.readOnce()
            current_pose = robot_state.O_T_EE.copy()
            current_rotation = rotation_from_pose_list(current_pose)

            angle_to_target_deg = rotation_angle_deg(current_rotation, target_rotation)
            if time_elapsed - last_print_time >= print_period_s or time_elapsed == 0.0:
                translation_error = translation_from_pose_list(current_pose) - fixed_translation
                print(
                    f"progress={min(1.0, time_elapsed / args.duration):.3f} "
                    f"angle_to_target_deg={angle_to_target_deg:.3f} "
                    f"translation_error={translation_error.tolist()}"
                )
                last_print_time = time_elapsed

            alpha = min(1.0, time_elapsed / args.duration)
            smooth_alpha = 3.0 * alpha * alpha - 2.0 * alpha * alpha * alpha
            commanded_rotation = interpolate_rotation(start_rotation, target_rotation, smooth_alpha)

            new_pose = write_rotation_into_pose_list(initial_pose, commanded_rotation)
            cartesian_pose = CartesianPose(new_pose)

            time_elapsed += duration.to_sec()

            if time_elapsed >= args.duration:
                final_pose = write_rotation_into_pose_list(initial_pose, target_rotation)
                cartesian_pose = CartesianPose(final_pose)
                cartesian_pose.motion_finished = True
                motion_finished = True

            active_control.writeOnce(cartesian_pose)

        final_state = robot.read_once()
        final_rotation = np.array(final_state.O_T_EE, dtype=float).reshape(4, 4, order="F")[:3, :3]
        final_angle_error_deg = rotation_angle_deg(final_rotation, target_rotation)
        print(f"Final measured angle-to-target: {final_angle_error_deg:.3f} deg")
        print("Finished EE vertical reorientation.")

    except Exception as exc:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        exit_with_error(f"Franka control error: {exc}")


if __name__ == "__main__":
    main()
