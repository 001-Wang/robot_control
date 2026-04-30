#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from pylibfranka import CartesianPose, ControllerMode, RealtimeConfig, Robot


TARGET_BASE_T_GRIPPER = np.array(
[
        [
          0.9981175791351335,
          0.05773628814568739,
          0.020683792026080387,
          0.39213055047281886
        ],
        [
          0.05822255263036859,
          -0.9980214712254366,
          -0.023733464522247866,
          0.09146229484958546
        ],
        [
          0.019272586402036614,
          0.024893051323270132,
          -0.9995043288596593,
          0.13892831616743234
        ],
        [
          0,
          0,
          0,
          1
        ]
],
    dtype=float,
)


def project_rotation_to_so3(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=float))
    projected = u @ vt
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


def ensure_homogeneous_transform(
    matrix: np.ndarray,
    *,
    name: str,
    orthonormalize_rotation: bool = False,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape == (3, 4):
        matrix = np.vstack([matrix, np.array([0.0, 0.0, 0.0, 1.0], dtype=float)])
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 transform, or a 3x4 transform without the final row.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{name} final row must be [0, 0, 0, 1].")
    matrix = matrix.copy()
    if orthonormalize_rotation:
        matrix[:3, :3] = project_rotation_to_so3(matrix[:3, :3])
    return matrix


TARGET_BASE_T_GRIPPER = ensure_homogeneous_transform(
    TARGET_BASE_T_GRIPPER,
    name="TARGET_BASE_T_GRIPPER",
    orthonormalize_rotation=True,
)


def pose_list_to_matrix(pose: list[float]) -> np.ndarray:
    return np.array(pose, dtype=float).reshape(4, 4, order="F")


def matrix_to_pose_list(matrix: np.ndarray) -> list[float]:
    return ensure_homogeneous_transform(matrix, name="Cartesian pose").reshape(16, order="F").tolist()


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
    alpha = smoothstep5(alpha)
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = (1.0 - alpha) * start[:3, 3] + alpha * target[:3, 3]
    q0 = matrix_to_quaternion_xyzw(start)
    q1 = matrix_to_quaternion_xyzw(target)
    pose[:3, :3] = quaternion_xyzw_to_matrix(slerp_xyzw(q0, q1, alpha))
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
    return (
        translation_error_m <= translation_tolerance_m
        and rotation_error_deg <= rotation_tolerance_deg
    )


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


def jsonable_matrix(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=float).tolist()


def safe_camera_info(device_or_sensor: Any, info: Any, fallback: str) -> str:
    try:
        return device_or_sensor.get_info(info)
    except Exception:
        return fallback


def candidate_stream_modes(width: int, height: int, fps: int) -> list[tuple[int, int, int]]:
    requested = (width, height, fps)
    common_modes = [
        (424, 240, 15),
        (640, 480, 15),
        (640, 480, 30),
        (1280, 720, 30),
    ]
    return [requested] + [mode for mode in common_modes if mode != requested]


def start_color_pipeline(
    *,
    rs: Any,
    pipeline: Any,
    width: int,
    height: int,
    fps: int,
    allow_stream_fallback: bool,
) -> tuple[Any, tuple[int, int, int]]:
    modes = candidate_stream_modes(width, height, fps) if allow_stream_fallback else [(width, height, fps)]
    errors: list[str] = []
    for candidate_width, candidate_height, candidate_fps in modes:
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            candidate_width,
            candidate_height,
            rs.format.bgr8,
            candidate_fps,
        )
        try:
            profile = pipeline.start(config)
            return profile, (candidate_width, candidate_height, candidate_fps)
        except RuntimeError as exc:
            errors.append(
                f"{candidate_width}x{candidate_height} @ {candidate_fps} FPS -> {exc}"
            )
    raise RuntimeError("Failed to start RealSense color stream.\n" + "\n".join(errors))


def intrinsics_to_json(video_stream_profile: Any) -> dict[str, Any]:
    intrinsics = video_stream_profile.get_intrinsics()
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "model": str(intrinsics.model),
        "coeffs": [float(value) for value in intrinsics.coeffs],
    }


def make_video_writer(cv2: Any, path: Path, fps: float, width: int, height: int) -> Any:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {path}")
    return writer


def find_h264_ffmpeg() -> str | None:
    candidates: list[str] = []
    system_ffmpeg = Path("/usr/bin/ffmpeg")
    if system_ffmpeg.exists():
        candidates.append(str(system_ffmpeg))
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg is not None and path_ffmpeg not in candidates:
        candidates.append(path_ffmpeg)

    for ffmpeg in candidates:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and "libx264" in result.stdout:
            return ffmpeg
    return None


def transcode_to_playable_mp4(raw_path: Path, output_path: Path) -> bool:
    ffmpeg = find_h264_ffmpeg()
    if ffmpeg is None:
        if raw_path != output_path:
            raw_path.replace(output_path)
        print("No ffmpeg with libx264 found; kept OpenCV MP4 without H.264 transcoding.")
        return False

    tmp_output_path = output_path.with_name(f"{output_path.stem}_h264_tmp{output_path.suffix}")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(raw_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_output_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg H.264 transcode failed; keeping raw OpenCV MP4.")
        if result.stderr:
            print(result.stderr[-2000:])
        if raw_path != output_path:
            raw_path.replace(output_path)
        return False

    tmp_output_path.replace(output_path)
    if raw_path.exists() and raw_path != output_path:
        raw_path.unlink()
    return True


def warmup_color_camera(
    *,
    pipeline: Any,
    warmup_frames: int,
    timeout_ms: int,
    startup_grace_sec: float,
    warmup_retries: int,
) -> None:
    if startup_grace_sec > 0.0:
        time.sleep(startup_grace_sec)

    last_error: Exception | None = None
    for attempt in range(1, max(1, warmup_retries) + 1):
        try:
            for _ in range(warmup_frames):
                pipeline.wait_for_frames(timeout_ms=timeout_ms)
            return
        except Exception as exc:
            last_error = exc
            if attempt >= warmup_retries:
                break
            print(f"Camera warmup attempt {attempt}/{warmup_retries} failed: {exc}")
            time.sleep(1.0)
    raise RuntimeError(f"RealSense color warmup failed: {last_error}")


class LatestColorFrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, pipeline: Any) -> None:
        if self._thread is not None:
            return

        def worker() -> None:
            while not self._stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except Exception:
                    continue
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data()).copy()
                if color_image.ndim != 3:
                    continue
                latest = {
                    "color_image": color_image,
                    "realsense_timestamp_ms": float(color_frame.get_timestamp()),
                    "realsense_frame_number": int(color_frame.get_frame_number()),
                }
                with self._lock:
                    self._latest = latest

        self._thread = threading.Thread(target=worker, name="realsense-color-buffer", daemon=True)
        self._thread.start()

    def get_latest(self) -> dict[str, Any] | None:
        with self._lock:
            if self._latest is None:
                return None
            return {
                "color_image": self._latest["color_image"].copy(),
                "realsense_timestamp_ms": self._latest["realsense_timestamp_ms"],
                "realsense_frame_number": self._latest["realsense_frame_number"],
            }

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def record_rgb_frame(
    *,
    frame_buffer: LatestColorFrameBuffer,
    video_writer: Any,
    pose_log_file: Any,
    robot_state: Any,
    monotonic_time_sec: float,
    wall_time_sec: float,
    label: str,
    frame_counter: int,
) -> int:
    latest_frame = frame_buffer.get_latest()
    if latest_frame is None:
        return frame_counter

    base_T_ee = pose_list_to_matrix(robot_state.O_T_EE)
    video_writer.write(latest_frame["color_image"])
    entry = {
        "frame_index": frame_counter,
        "timestamp_wall_sec": wall_time_sec,
        "timestamp_monotonic_sec": monotonic_time_sec,
        "realsense_timestamp_ms": latest_frame["realsense_timestamp_ms"],
        "realsense_frame_number": latest_frame["realsense_frame_number"],
        "waypoint_index": 0,
        "waypoint_label": label,
        "base_T_ee": {"matrix": jsonable_matrix(base_T_ee)},
        "base_T_gripper": {"matrix": jsonable_matrix(base_T_ee)},
    }
    pose_log_file.write(json.dumps(entry) + "\n")
    pose_log_file.flush()
    return frame_counter + 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Move Franka EE/gripper to keyframe 0 pose.")
    parser.add_argument("--robot-ip", default="192.168.1.11", help="Franka robot IP address.")
    parser.add_argument("--duration-sec", type=float, default=8.0, help="Motion duration in seconds.")
    parser.add_argument("--hold-sec", type=float, default=1.0, help="Hold target pose after motion.")
    parser.add_argument("--dry-run", action="store_true", help="Print target pose and exit before moving.")
    parser.add_argument(
        "--record-output-dir",
        type=Path,
        help="If set, record a RealSense RGB video and per-frame pose log here.",
    )
    parser.add_argument("--width", type=int, default=1280, help="RealSense RGB stream width.")
    parser.add_argument("--height", type=int, default=720, help="RealSense RGB stream height.")
    parser.add_argument("--fps", type=int, default=30, help="RealSense RGB stream FPS.")
    parser.add_argument(
        "--record-fps",
        type=float,
        default=30.0,
        help="Maximum pose/video recording rate while moving and holding.",
    )
    parser.add_argument(
        "--allow-stream-fallback",
        action="store_true",
        help="Try nearby known-safe RealSense modes if the requested mode is unavailable.",
    )
    parser.add_argument("--warmup-frames", type=int, default=30, help="RealSense warmup frames.")
    parser.add_argument("--frame-timeout-ms", type=int, default=5000, help="RealSense frame timeout.")
    parser.add_argument("--startup-grace-sec", type=float, default=2.0, help="Delay after camera start.")
    parser.add_argument("--warmup-retries", type=int, default=3, help="RealSense warmup retries.")
    parser.add_argument(
        "--already-there-translation-tolerance-m",
        type=float,
        default=0.005,
        help="Translation tolerance for treating the target pose as already reached.",
    )
    parser.add_argument(
        "--already-there-rotation-tolerance-deg",
        type=float,
        default=2.0,
        help="Rotation tolerance for treating the target pose as already reached.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive safety confirmation. Use only when the workspace is clear.",
    )
    args = parser.parse_args()

    if args.duration_sec <= 0.0:
        raise ValueError("--duration-sec must be positive.")
    if args.hold_sec < 0.0:
        raise ValueError("--hold-sec must be non-negative.")
    if args.record_fps <= 0.0:
        raise ValueError("--record-fps must be positive.")
    if args.already_there_translation_tolerance_m < 0.0:
        raise ValueError("--already-there-translation-tolerance-m must be non-negative.")
    if args.already_there_rotation_tolerance_deg < 0.0:
        raise ValueError("--already-there-rotation-tolerance-deg must be non-negative.")

    print("Target base_T_gripper:")
    print(TARGET_BASE_T_GRIPPER)
    print("Target xyz:", TARGET_BASE_T_GRIPPER[:3, 3].tolist())
    if args.record_output_dir is not None:
        print(f"Recording output: {args.record_output_dir}")

    if args.dry_run:
        return 0

    print("WARNING: this will move the robot.")
    if not args.yes:
        try:
            input("Press Enter only if the workspace is clear and the user stop is ready...")
        except EOFError as exc:
            raise RuntimeError(
                "No interactive stdin is available for the safety confirmation. "
                "Run with --yes after confirming the workspace is clear and the user stop is ready."
            ) from exc

    robot = Robot(args.robot_ip, RealtimeConfig.kIgnore)
    pipeline = None
    frame_buffer = None
    video_writer = None
    pose_log_file = None
    frame_counter = 0
    active_mode = None
    color_intrinsics = None
    run_start_wall_sec = time.time()
    run_start_monotonic_sec = time.monotonic()
    try:
        apply_collision_behavior(robot)
        if args.record_output_dir is not None:
            import cv2
            import pyrealsense2 as rs

            args.record_output_dir.mkdir(parents=True, exist_ok=True)
            video_path = args.record_output_dir / "trajectory.mp4"
            raw_video_path = args.record_output_dir / "trajectory_raw_opencv.mp4"
            pose_log_path = args.record_output_dir / "frame_poses.jsonl"

            pipeline = rs.pipeline()
            profile, active_mode = start_color_pipeline(
                rs=rs,
                pipeline=pipeline,
                width=args.width,
                height=args.height,
                fps=args.fps,
                allow_stream_fallback=args.allow_stream_fallback,
            )
            active_device = profile.get_device()
            active_name = safe_camera_info(active_device, rs.camera_info.name, "<unknown>")
            active_serial = safe_camera_info(
                active_device,
                rs.camera_info.serial_number,
                "<unknown>",
            )
            print(
                f"Streaming from: {active_name} | serial={active_serial} | "
                f"mode={active_mode[0]}x{active_mode[1]} @ {active_mode[2]} FPS"
            )

            color_stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            color_intrinsics = intrinsics_to_json(color_stream_profile)

            warmup_color_camera(
                pipeline=pipeline,
                warmup_frames=args.warmup_frames,
                timeout_ms=args.frame_timeout_ms,
                startup_grace_sec=args.startup_grace_sec,
                warmup_retries=args.warmup_retries,
            )
            frame_buffer = LatestColorFrameBuffer()
            frame_buffer.start(pipeline)
            video_writer = make_video_writer(
                cv2,
                raw_video_path,
                args.record_fps,
                active_mode[0],
                active_mode[1],
            )
            pose_log_file = pose_log_path.open("w", encoding="utf-8")

        last_record_monotonic_sec = float("-inf")
        record_interval_sec = 1.0 / args.record_fps
        initial_state = robot.read_once()
        start_pose = pose_list_to_matrix(initial_state.O_T_EE)
        translation_error_m = float(np.linalg.norm(start_pose[:3, 3] - TARGET_BASE_T_GRIPPER[:3, 3]))
        rotation_error_deg = rotation_distance_deg(start_pose, TARGET_BASE_T_GRIPPER)
        print("Current base_T_gripper:")
        print(start_pose)
        print(
            "Target error: "
            f"{translation_error_m * 1000.0:.2f} mm, {rotation_error_deg:.2f} deg"
        )

        target_already_reached = poses_are_close(
            start_pose,
            TARGET_BASE_T_GRIPPER,
            args.already_there_translation_tolerance_m,
            args.already_there_rotation_tolerance_deg,
        )

        if target_already_reached:
            print("Target pose is already reached; skipping Cartesian motion control.")
            hold_start = time.monotonic()
            while time.monotonic() - hold_start < args.hold_sec:
                if video_writer is None or pose_log_file is None or frame_buffer is None:
                    time.sleep(0.02)
                    continue
                now_mono = time.monotonic()
                if now_mono - last_record_monotonic_sec < record_interval_sec:
                    time.sleep(0.001)
                    continue
                before = frame_counter
                frame_counter = record_rgb_frame(
                    frame_buffer=frame_buffer,
                    video_writer=video_writer,
                    pose_log_file=pose_log_file,
                    robot_state=robot.read_once(),
                    monotonic_time_sec=now_mono,
                    wall_time_sec=time.time(),
                    label="keyframe0_hold",
                    frame_counter=frame_counter,
                )
                if frame_counter > before:
                    last_record_monotonic_sec = now_mono
        else:
            try:
                active_control = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)
            except Exception as exc:
                message = str(exc)
                if "cannot start at singular pose" in message:
                    raise RuntimeError(
                        "Failed to start Cartesian pose control because the robot is currently "
                        "at or near a singular pose.\n"
                        "Use Franka Desk to move the arm to a less singular joint configuration, "
                        "then retry. Keep the tool pose similar, but bend the elbow/wrist away "
                        "from a straight or fully aligned configuration."
                    ) from exc
                raise

            robot_state, period = active_control.readOnce()
            start_pose = pose_list_to_matrix(robot_state.O_T_EE)

            elapsed = 0.0
            motion_done = False
            while not motion_done:
                elapsed += period.to_sec()
                alpha = min(elapsed / args.duration_sec, 1.0)
                commanded_pose = interpolate_pose(start_pose, TARGET_BASE_T_GRIPPER, alpha)
                command = CartesianPose(matrix_to_pose_list(commanded_pose))
                if elapsed >= args.duration_sec:
                    motion_done = True
                    if args.hold_sec <= 0.0:
                        command.motion_finished = True
                active_control.writeOnce(command)

                if video_writer is not None and pose_log_file is not None and frame_buffer is not None:
                    now_mono = time.monotonic()
                    if now_mono - last_record_monotonic_sec >= record_interval_sec:
                        before = frame_counter
                        frame_counter = record_rgb_frame(
                            frame_buffer=frame_buffer,
                            video_writer=video_writer,
                            pose_log_file=pose_log_file,
                            robot_state=robot_state,
                            monotonic_time_sec=now_mono,
                            wall_time_sec=time.time(),
                            label="keyframe0_motion",
                            frame_counter=frame_counter,
                        )
                        if frame_counter > before:
                            last_record_monotonic_sec = now_mono

                if not motion_done or args.hold_sec > 0.0:
                    robot_state, period = active_control.readOnce()

            if args.hold_sec > 0.0:
                hold_elapsed = 0.0
                hold_finished = False
                while not hold_finished:
                    hold_elapsed += period.to_sec()
                    command = CartesianPose(matrix_to_pose_list(TARGET_BASE_T_GRIPPER))
                    if hold_elapsed >= args.hold_sec:
                        command.motion_finished = True
                        hold_finished = True
                    active_control.writeOnce(command)

                    if video_writer is not None and pose_log_file is not None and frame_buffer is not None:
                        now_mono = time.monotonic()
                        if now_mono - last_record_monotonic_sec >= record_interval_sec:
                            before = frame_counter
                            frame_counter = record_rgb_frame(
                                frame_buffer=frame_buffer,
                                video_writer=video_writer,
                                pose_log_file=pose_log_file,
                                robot_state=robot_state,
                                monotonic_time_sec=now_mono,
                                wall_time_sec=time.time(),
                                label="keyframe0_hold",
                                frame_counter=frame_counter,
                            )
                            if frame_counter > before:
                                last_record_monotonic_sec = now_mono

                    if not hold_finished:
                        robot_state, period = active_control.readOnce()

        if args.record_output_dir is not None:
            if pose_log_file is not None:
                pose_log_file.close()
                pose_log_file = None
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            video_transcoded_h264 = transcode_to_playable_mp4(
                args.record_output_dir / "trajectory_raw_opencv.mp4",
                args.record_output_dir / "trajectory.mp4",
            )
            metadata = {
                "robot_ip": args.robot_ip,
                "target_name": "keyframe0",
                "target_base_T_gripper": {"matrix": TARGET_BASE_T_GRIPPER.tolist()},
                "duration_sec": args.duration_sec,
                "hold_sec": args.hold_sec,
                "video_path": str((args.record_output_dir / "trajectory.mp4").resolve()),
                "pose_log_path": str((args.record_output_dir / "frame_poses.jsonl").resolve()),
                "requested_width": args.width,
                "requested_height": args.height,
                "requested_fps": args.fps,
                "record_fps": args.record_fps,
                "video_codec": "h264" if video_transcoded_h264 else "opencv_raw_mp4",
                "stream_width": active_mode[0] if active_mode else None,
                "stream_height": active_mode[1] if active_mode else None,
                "stream_fps": active_mode[2] if active_mode else None,
                "color_intrinsics": color_intrinsics,
                "frames_recorded": frame_counter,
                "run_start_wall_sec": run_start_wall_sec,
                "run_end_wall_sec": time.time(),
                "run_duration_sec": time.monotonic() - run_start_monotonic_sec,
            }
            (args.record_output_dir / "run_metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Recorded {frame_counter} frames to {args.record_output_dir}")
        return 0
    except Exception:
        robot.stop()
        raise
    finally:
        if pose_log_file is not None:
            pose_log_file.close()
        if video_writer is not None:
            video_writer.release()
        if frame_buffer is not None:
            frame_buffer.stop()
        if pipeline is not None:
            pipeline.stop()


if __name__ == "__main__":
    raise SystemExit(main())
