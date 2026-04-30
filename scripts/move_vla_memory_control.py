#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from pylibfranka import CartesianPose, ControllerMode, RealtimeConfig, Robot


DEFAULT_MEMORY_DIR = Path("data/vla_traj/memory")


def project_rotation_to_so3(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=float))
    projected = u @ vt
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


def ensure_transform(matrix: Any, *, name: str) -> np.ndarray:
    transform = np.asarray(matrix, dtype=float)
    if transform.shape == (3, 4):
        transform = np.vstack([transform, np.array([0.0, 0.0, 0.0, 1.0])])
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 transform, or a 3x4 transform without the final row.")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{name} final row must be [0, 0, 0, 1].")
    transform = transform.copy()
    transform[:3, :3] = project_rotation_to_so3(transform[:3, :3])
    return transform


def pose_list_to_matrix(pose: list[float]) -> np.ndarray:
    return np.array(pose, dtype=float).reshape(4, 4, order="F")


def matrix_to_pose_list(matrix: np.ndarray) -> list[float]:
    return ensure_transform(matrix, name="Cartesian pose").reshape(16, order="F").tolist()


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


def poses_are_close(
    current: np.ndarray,
    target: np.ndarray,
    translation_tolerance_m: float,
    rotation_tolerance_deg: float,
) -> bool:
    translation_error_m = float(np.linalg.norm(current[:3, 3] - target[:3, 3]))
    rotation_error_deg = rotation_distance_deg(current, target)
    return translation_error_m <= translation_tolerance_m and rotation_error_deg <= rotation_tolerance_deg


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


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    try:
        return [jsonable(item) for item in value]
    except TypeError:
        return str(value)


def robot_state_to_entry(
    robot_state: Any,
    *,
    sample_index: int,
    phase: str,
    target_index: int | None,
    target_label: str | None,
    monotonic_time_sec: float,
    wall_time_sec: float,
) -> dict[str, Any]:
    base_T_ee = pose_list_to_matrix(robot_state.O_T_EE)
    entry: dict[str, Any] = {
        "sample_index": sample_index,
        "timestamp_wall_sec": wall_time_sec,
        "timestamp_monotonic_sec": monotonic_time_sec,
        "phase": phase,
        "target_index": target_index,
        "target_label": target_label,
        "base_T_ee": {"matrix": base_T_ee.tolist()},
        "base_T_gripper": {"matrix": base_T_ee.tolist()},
    }
    for field in (
        "q",
        "dq",
        "q_d",
        "dq_d",
        "ddq_d",
        "theta",
        "dtheta",
        "tau_J",
        "tau_J_d",
        "dtau_J",
        "joint_contact",
        "joint_collision",
        "cartesian_contact",
        "cartesian_collision",
        "O_F_ext_hat_K",
        "K_F_ext_hat_K",
        "O_dP_EE_d",
        "O_T_EE_d",
        "F_T_EE",
        "EE_T_K",
        "m_ee",
        "m_load",
        "m_total",
    ):
        if hasattr(robot_state, field):
            entry[field] = jsonable(getattr(robot_state, field))
    return entry


class JointStatusLogger:
    def __init__(self, path: Path | None, fps: float) -> None:
        if fps <= 0.0:
            raise ValueError("--status-fps must be positive.")
        self.path = path
        self.interval_sec = 1.0 / fps
        self.sample_index = 0
        self.last_log_monotonic_sec = float("-inf")
        self.file = None

    def __enter__(self) -> "JointStatusLogger":
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def maybe_log(
        self,
        robot_state: Any,
        *,
        phase: str,
        target_index: int | None,
        target_label: str | None,
        force: bool = False,
    ) -> bool:
        if self.file is None:
            return False
        now_mono = time.monotonic()
        if not force and now_mono - self.last_log_monotonic_sec < self.interval_sec:
            return False
        entry = robot_state_to_entry(
            robot_state,
            sample_index=self.sample_index,
            phase=phase,
            target_index=target_index,
            target_label=target_label,
            monotonic_time_sec=now_mono,
            wall_time_sec=time.time(),
        )
        self.file.write(json.dumps(entry) + "\n")
        self.file.flush()
        self.sample_index += 1
        self.last_log_monotonic_sec = now_mono
        return True


def resolve_memory_path(memory: Path, memory_id: str | None) -> Path:
    if memory_id is not None:
        candidate = memory / f"{memory_id}.json" if memory.is_dir() else memory.parent / f"{memory_id}.json"
        if not candidate.exists():
            raise FileNotFoundError(f"Could not find memory id {memory_id!r} at {candidate}")
        return candidate.resolve()
    if memory.is_file():
        return memory.resolve()
    if not memory.exists():
        raise FileNotFoundError(f"Memory path does not exist: {memory}")
    candidates = sorted(memory.glob("*.json"), key=lambda path: (path.stat().st_mtime, path.name))
    if not candidates:
        raise FileNotFoundError(f"No JSON memory files found in {memory}")
    return candidates[-1].resolve()


def keyframe_base_T_gripper(keyframe: dict[str, Any], index: int) -> np.ndarray:
    if "base_T_gripper" not in keyframe:
        raise ValueError(f"keyframes[{index}] does not contain base_T_gripper.")
    payload = keyframe["base_T_gripper"]
    matrix = payload["matrix"] if isinstance(payload, dict) and "matrix" in payload else payload
    return ensure_transform(matrix, name=f"keyframes[{index}].base_T_gripper")


def load_vla_targets(memory_path: Path) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    data = json.loads(memory_path.read_text(encoding="utf-8"))
    keyframes = data.get("keyframes")
    if not isinstance(keyframes, list):
        raise ValueError(f"{memory_path} must contain a keyframes list.")
    if len(keyframes) < 2:
        raise ValueError(f"{memory_path} must contain at least a start keyframe and one motion keyframe.")
    transforms = [keyframe_base_T_gripper(keyframe, index) for index, keyframe in enumerate(keyframes)]
    return keyframes, transforms


def print_target_summary(keyframes: list[dict[str, Any]], targets: list[np.ndarray]) -> None:
    for index, target in enumerate(targets):
        label = keyframes[index].get("name", f"keyframe_{index}")
        keyframe_index = keyframes[index].get("keyframe_index", index)
        print(
            f"[{index}] {label} keyframe_index={keyframe_index} "
            f"xyz={target[:3, 3].tolist()}"
        )


def prompt_float(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number in seconds.")
            continue
        if value <= 0.0:
            print("Duration must be positive.")
            continue
        return value


def start_cartesian_control(robot: Robot) -> Any:
    try:
        return robot.start_cartesian_pose_control(ControllerMode.JointImpedance)
    except Exception as exc:
        message = str(exc)
        if "cannot start at singular pose" in message:
            raise RuntimeError(
                "Failed to start Cartesian pose control because the robot is currently at or near "
                "a singular pose. Use Franka Desk to bend the elbow/wrist away from a straight "
                "or fully aligned configuration, then retry."
            ) from exc
        raise


def move_to_target(
    *,
    active_control: Any,
    start_pose: np.ndarray,
    target_pose: np.ndarray,
    duration_sec: float,
    hold_sec: float,
    logger: JointStatusLogger,
    target_index: int,
    target_label: str,
    finish_control: bool = False,
    already_there_translation_tolerance_m: float = 0.005,
    already_there_rotation_tolerance_deg: float = 2.0,
) -> np.ndarray:
    if poses_are_close(
        start_pose,
        target_pose,
        already_there_translation_tolerance_m,
        already_there_rotation_tolerance_deg,
    ):
        print(f"Target [{target_index}] already reached; skipping motion.")
        robot_state, period = active_control.readOnce()
    else:
        robot_state, period = active_control.readOnce()
        start_pose = pose_list_to_matrix(robot_state.O_T_EE)
        active_control.writeOnce(CartesianPose(matrix_to_pose_list(start_pose)))
        robot_state, period = active_control.readOnce()
        elapsed = 0.0
        while elapsed < duration_sec:
            dt = period.to_sec()
            elapsed += dt
            alpha = min(elapsed / duration_sec, 1.0)
            commanded_pose = interpolate_pose(start_pose, target_pose, alpha)
            command = CartesianPose(matrix_to_pose_list(commanded_pose))
            if finish_control and hold_sec <= 0.0 and elapsed >= duration_sec:
                command.motion_finished = True
            active_control.writeOnce(command)
            logger.maybe_log(
                robot_state,
                phase="motion",
                target_index=target_index,
                target_label=target_label,
            )
            if elapsed < duration_sec or hold_sec > 0.0:
                robot_state, period = active_control.readOnce()

    hold_elapsed = 0.0
    while hold_elapsed < hold_sec:
        hold_elapsed += period.to_sec()
        command = CartesianPose(matrix_to_pose_list(target_pose))
        if finish_control and hold_elapsed >= hold_sec:
            command.motion_finished = True
        active_control.writeOnce(command)
        logger.maybe_log(
            robot_state,
            phase="hold",
            target_index=target_index,
            target_label=target_label,
        )
        if hold_elapsed < hold_sec:
            robot_state, period = active_control.readOnce()

    logger.maybe_log(
        robot_state,
        phase="target_reached",
        target_index=target_index,
        target_label=target_label,
        force=True,
    )
    return target_pose.copy()


def move_through_targets(
    *,
    active_control: Any,
    targets: list[np.ndarray],
    keyframes: list[dict[str, Any]],
    total_duration_sec: float,
    logger: JointStatusLogger,
) -> np.ndarray:
    if not targets:
        raise ValueError("Second-round target list must not be empty.")

    robot_state, period = active_control.readOnce()
    initial_pose = pose_list_to_matrix(robot_state.O_T_EE)
    active_control.writeOnce(CartesianPose(matrix_to_pose_list(initial_pose)))
    robot_state, period = active_control.readOnce()

    segment_duration_sec = total_duration_sec / len(targets)
    segment_starts = [initial_pose, *[target.copy() for target in targets[:-1]]]
    elapsed = 0.0
    last_logged_target_index = 1
    last_logged_target_label = str(keyframes[0].get("name", "motion_1"))

    while elapsed < total_duration_sec:
        dt = period.to_sec()
        elapsed = min(elapsed + dt, total_duration_sec)
        segment_index = min(int(elapsed / segment_duration_sec), len(targets) - 1)
        segment_start_time = segment_index * segment_duration_sec
        local_elapsed = elapsed - segment_start_time
        if segment_index == len(targets) - 1:
            local_elapsed = min(local_elapsed, segment_duration_sec)
        alpha = min(local_elapsed / segment_duration_sec, 1.0)

        target_index = segment_index + 1
        target_label = str(keyframes[segment_index].get("name", f"motion_{target_index}"))
        commanded_pose = interpolate_pose(
            segment_starts[segment_index],
            targets[segment_index],
            alpha,
        )
        command = CartesianPose(matrix_to_pose_list(commanded_pose))
        motion_done = elapsed >= total_duration_sec
        if motion_done:
            command.motion_finished = True
        active_control.writeOnce(command)

        logger.maybe_log(
            robot_state,
            phase="motion",
            target_index=target_index,
            target_label=target_label,
        )
        last_logged_target_index = target_index
        last_logged_target_label = target_label

        if not motion_done:
            robot_state, period = active_control.readOnce()

    logger.maybe_log(
        robot_state,
        phase="target_reached",
        target_index=last_logged_target_index,
        target_label=last_logged_target_label,
        force=True,
    )
    return targets[-1].copy()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Move Franka through VLA memory poses: first keyframe as start, then one or two "
            "motion keyframes, while exporting robot joint status."
        )
    )
    parser.add_argument("--robot-ip", "--ip", default="192.168.1.11", help="Franka robot IP address.")
    parser.add_argument(
        "--memory",
        type=Path,
        default=DEFAULT_MEMORY_DIR,
        help="Memory JSON file or directory. If a directory is given, the newest JSON is used.",
    )
    parser.add_argument("--memory-id", help="Memory JSON id/name inside --memory, for example 27.")
    parser.add_argument("--first-duration-sec", type=float, default=8.0, help="Duration to move to the start pose.")
    parser.add_argument(
        "--second-round-duration-sec",
        type=float,
        help="Total duration for the motion targets. If omitted, you will be prompted after start pose.",
    )
    parser.add_argument("--hold-sec", type=float, default=0.5, help="Hold time after the start pose.")
    parser.add_argument(
        "--status-output",
        type=Path,
        help="JSONL output for robot joint status. Defaults to <memory_stem>_joint_status.jsonl.",
    )
    parser.add_argument("--status-fps", type=float, default=30.0, help="Robot joint status logging frequency.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected poses and exit before moving.")
    parser.add_argument("--yes", action="store_true", help="Skip the initial interactive safety confirmation.")
    parser.add_argument(
        "--already-there-translation-tolerance-m",
        type=float,
        default=0.005,
        help="Translation tolerance for treating a target pose as already reached.",
    )
    parser.add_argument(
        "--already-there-rotation-tolerance-deg",
        type=float,
        default=2.0,
        help="Rotation tolerance for treating a target pose as already reached.",
    )
    args = parser.parse_args()

    if args.first_duration_sec <= 0.0:
        raise ValueError("--first-duration-sec must be positive.")
    if args.second_round_duration_sec is not None and args.second_round_duration_sec <= 0.0:
        raise ValueError("--second-round-duration-sec must be positive.")
    if args.hold_sec < 0.0:
        raise ValueError("--hold-sec must be non-negative.")
    if args.status_fps <= 0.0:
        raise ValueError("--status-fps must be positive.")

    memory_path = resolve_memory_path(args.memory, args.memory_id)
    keyframes, transforms = load_vla_targets(memory_path)
    start_target = transforms[0]
    motion_targets = transforms[1:] if len(transforms) <= 3 else transforms[-2:]
    motion_keyframes = keyframes[1:] if len(keyframes) <= 3 else keyframes[-2:]
    status_output = args.status_output
    if status_output is None:
        status_output = memory_path.with_name(f"{memory_path.stem}_joint_status.jsonl")

    print(f"Memory file: {memory_path}")
    print(f"Joint status output: {status_output}")
    print("Selected targets:")
    selected_keyframes = [keyframes[0], *motion_keyframes]
    selected_targets = [start_target, *motion_targets]
    print_target_summary(selected_keyframes, selected_targets)
    print(f"Motion target count after start: {len(motion_targets)}")

    if args.dry_run:
        return 0

    print("WARNING: this will move the robot.")
    if not args.yes:
        try:
            input("Press Enter only if the workspace is clear and the user stop is ready...")
        except EOFError as exc:
            raise RuntimeError(
                "No interactive stdin is available for the safety confirmation. "
                "Run with --yes only after confirming the workspace is clear and the user stop is ready."
            ) from exc

    robot = Robot(args.robot_ip, RealtimeConfig.kIgnore)
    active_control = None
    try:
        apply_collision_behavior(robot)
        initial_state = robot.read_once()
        current_pose = pose_list_to_matrix(initial_state.O_T_EE)
        print("Current base_T_gripper:")
        print(current_pose)
        print(
            "Start target error: "
            f"{np.linalg.norm(current_pose[:3, 3] - start_target[:3, 3]) * 1000.0:.2f} mm, "
            f"{rotation_distance_deg(current_pose, start_target):.2f} deg"
        )

        with JointStatusLogger(None, args.status_fps) as start_logger:
            active_control = start_cartesian_control(robot)
            current_pose = move_to_target(
                active_control=active_control,
                start_pose=current_pose,
                target_pose=start_target,
                duration_sec=args.first_duration_sec,
                hold_sec=args.hold_sec,
                logger=start_logger,
                target_index=0,
                target_label=str(keyframes[0].get("name", "start")),
                finish_control=True,
                already_there_translation_tolerance_m=args.already_there_translation_tolerance_m,
                already_there_rotation_tolerance_deg=args.already_there_rotation_tolerance_deg,
            )
            active_control = None
            print("Reached start pose.")

            if args.second_round_duration_sec is None:
                input("Press Enter when you are ready to run the VLA motion targets...")
                second_round_duration_sec = prompt_float("Total time for second round motion, in seconds: ")
            elif args.yes:
                second_round_duration_sec = args.second_round_duration_sec
            else:
                print("Press Enter when you are ready to run the VLA motion targets.")
                input()
                second_round_duration_sec = args.second_round_duration_sec

            robot_state = robot.read_once()
            current_pose = pose_list_to_matrix(robot_state.O_T_EE)
            active_control = start_cartesian_control(robot)
            with JointStatusLogger(status_output, args.status_fps) as logger:
                for offset, target_pose in enumerate(motion_targets, start=1):
                    keyframe = motion_keyframes[offset - 1]
                    target_label = str(keyframe.get("name", f"motion_{offset}"))
                    print(
                        f"Moving to motion target {offset}/{len(motion_targets)} "
                        f"inside one continuous {second_round_duration_sec:.3f} sec trajectory"
                    )
                current_pose = move_through_targets(
                    active_control=active_control,
                    targets=motion_targets,
                    keyframes=motion_keyframes,
                    total_duration_sec=second_round_duration_sec,
                    logger=logger,
                )

        print(f"Finished VLA memory motion. Wrote {status_output}")
        return 0
    except Exception:
        robot.stop()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
