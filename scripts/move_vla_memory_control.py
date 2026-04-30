#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from pylibfranka import CartesianPose, ControllerMode, Gripper, RealtimeConfig, Robot


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


def jsonable_matrix(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=float).tolist()


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


def gripper_state_to_json(gripper_state: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    for field in ("width", "max_width", "is_grasped", "temperature", "time"):
        if hasattr(gripper_state, field):
            entry[field] = jsonable(getattr(gripper_state, field))
    return entry


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

    def get_latest(self, *, copy_image: bool = True) -> dict[str, Any] | None:
        with self._lock:
            if self._latest is None:
                return None
            color_image = self._latest["color_image"]
            return {
                "color_image": color_image.copy() if copy_image else color_image,
                "realsense_timestamp_ms": self._latest["realsense_timestamp_ms"],
                "realsense_frame_number": self._latest["realsense_frame_number"],
            }

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class RgbRecorder:
    def __init__(
        self,
        *,
        output_dir: Path | None,
        width: int,
        height: int,
        fps: int,
        record_fps: float,
        allow_stream_fallback: bool,
        warmup_frames: int,
        frame_timeout_ms: int,
        startup_grace_sec: float,
        warmup_retries: int,
        gripper_state: dict[str, Any] | None = None,
    ) -> None:
        if record_fps <= 0.0:
            raise ValueError("--record-fps must be positive.")
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.record_fps = record_fps
        self.allow_stream_fallback = allow_stream_fallback
        self.warmup_frames = warmup_frames
        self.frame_timeout_ms = frame_timeout_ms
        self.startup_grace_sec = startup_grace_sec
        self.warmup_retries = warmup_retries
        self.gripper_state = dict(gripper_state) if gripper_state is not None else None
        self.pipeline = None
        self.frame_buffer: LatestColorFrameBuffer | None = None
        self.video_writer = None
        self.pose_log_file = None
        self.active_mode: tuple[int, int, int] | None = None
        self.color_intrinsics: dict[str, Any] | None = None
        self.frame_counter = 0
        self.next_frame_index = 0
        self.last_record_monotonic_sec = float("-inf")
        self.record_interval_sec = 1.0 / record_fps
        self._write_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=8)
        self._writer_thread: threading.Thread | None = None
        self.frames_dropped = 0

    def __enter__(self) -> "RgbRecorder":
        if self.output_dir is None:
            return self

        import cv2
        import pyrealsense2 as rs

        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_video_path = self.output_dir / "trajectory_raw_opencv.mp4"

        self.pipeline = rs.pipeline()
        profile, self.active_mode = start_color_pipeline(
            rs=rs,
            pipeline=self.pipeline,
            width=self.width,
            height=self.height,
            fps=self.fps,
            allow_stream_fallback=self.allow_stream_fallback,
        )
        active_device = profile.get_device()
        active_name = safe_camera_info(active_device, rs.camera_info.name, "<unknown>")
        active_serial = safe_camera_info(active_device, rs.camera_info.serial_number, "<unknown>")
        print(
            f"Streaming from: {active_name} | serial={active_serial} | "
            f"mode={self.active_mode[0]}x{self.active_mode[1]} @ {self.active_mode[2]} FPS"
        )

        color_stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intrinsics = intrinsics_to_json(color_stream_profile)
        warmup_color_camera(
            pipeline=self.pipeline,
            warmup_frames=self.warmup_frames,
            timeout_ms=self.frame_timeout_ms,
            startup_grace_sec=self.startup_grace_sec,
            warmup_retries=self.warmup_retries,
        )
        self.frame_buffer = LatestColorFrameBuffer()
        self.frame_buffer.start(self.pipeline)
        self.video_writer = make_video_writer(
            cv2,
            raw_video_path,
            self.record_fps,
            self.active_mode[0],
            self.active_mode[1],
        )
        self.pose_log_file = (self.output_dir / "frame_poses.jsonl").open("w", encoding="utf-8")
        self._start_writer_thread()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop_writer_thread()
        if self.pose_log_file is not None:
            self.pose_log_file.close()
            self.pose_log_file = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.frame_buffer is not None:
            self.frame_buffer.stop()
            self.frame_buffer = None
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

    def _start_writer_thread(self) -> None:
        if self._writer_thread is not None:
            return

        def worker() -> None:
            while True:
                item = self._write_queue.get()
                try:
                    if item is None:
                        return
                    if self.video_writer is None or self.pose_log_file is None:
                        continue
                    color_image = item.pop("color_image")
                    self.video_writer.write(color_image)
                    self.pose_log_file.write(json.dumps(item) + "\n")
                    self.pose_log_file.flush()
                    self.frame_counter += 1
                finally:
                    self._write_queue.task_done()

        self._writer_thread = threading.Thread(target=worker, name="rgb-record-writer", daemon=True)
        self._writer_thread.start()

    def _stop_writer_thread(self) -> None:
        if self._writer_thread is None:
            return
        self._write_queue.join()
        self._write_queue.put(None)
        self._write_queue.join()
        self._writer_thread.join(timeout=5.0)
        self._writer_thread = None

    def maybe_record(
        self,
        robot_state: Any,
        *,
        waypoint_index: int,
        waypoint_label: str,
        force: bool = False,
    ) -> bool:
        if self.video_writer is None or self.pose_log_file is None or self.frame_buffer is None:
            return False
        now_mono = time.monotonic()
        if not force and now_mono - self.last_record_monotonic_sec < self.record_interval_sec:
            return False
        latest_frame = self.frame_buffer.get_latest(copy_image=False)
        if latest_frame is None:
            return False

        base_T_ee = pose_list_to_matrix(robot_state.O_T_EE)
        entry = {
            "frame_index": self.next_frame_index,
            "color_image": latest_frame["color_image"],
            "timestamp_wall_sec": time.time(),
            "timestamp_monotonic_sec": now_mono,
            "realsense_timestamp_ms": latest_frame["realsense_timestamp_ms"],
            "realsense_frame_number": latest_frame["realsense_frame_number"],
            "waypoint_index": waypoint_index,
            "waypoint_label": waypoint_label,
            "base_T_ee": {"matrix": jsonable_matrix(base_T_ee)},
            "base_T_gripper": {"matrix": jsonable_matrix(base_T_ee)},
        }
        for field in ("q", "dq"):
            if hasattr(robot_state, field):
                entry[field] = jsonable(getattr(robot_state, field))
        if self.gripper_state is not None:
            entry["gripper_state"] = self.gripper_state
        try:
            self._write_queue.put_nowait(entry)
        except queue.Full:
            self.frames_dropped += 1
            self.last_record_monotonic_sec = now_mono
            return False
        self.next_frame_index += 1
        self.last_record_monotonic_sec = now_mono
        return True

    def write_metadata(
        self,
        *,
        robot_ip: str,
        memory_path: Path,
        motion_keyframes: list[dict[str, Any]],
        motion_targets: list[np.ndarray],
        second_round_duration_sec: float,
        status_output: Path,
        run_start_wall_sec: float,
        run_start_monotonic_sec: float,
    ) -> None:
        if self.output_dir is None:
            return
        video_transcoded_h264 = transcode_to_playable_mp4(
            self.output_dir / "trajectory_raw_opencv.mp4",
            self.output_dir / "trajectory.mp4",
        )
        metadata = {
            "robot_ip": robot_ip,
            "memory_path": str(memory_path.resolve()),
            "target_names": [
                str(keyframe.get("name", f"motion_{index + 1}"))
                for index, keyframe in enumerate(motion_keyframes)
            ],
            "target_base_T_gripper": [
                {"matrix": target.tolist()} for target in motion_targets
            ],
            "second_round_duration_sec": second_round_duration_sec,
            "status_output": str(status_output.resolve()),
            "video_path": str((self.output_dir / "trajectory.mp4").resolve()),
            "pose_log_path": str((self.output_dir / "frame_poses.jsonl").resolve()),
            "requested_width": self.width,
            "requested_height": self.height,
            "requested_fps": self.fps,
            "record_fps": self.record_fps,
            "video_codec": "h264" if video_transcoded_h264 else "opencv_raw_mp4",
            "stream_width": self.active_mode[0] if self.active_mode else None,
            "stream_height": self.active_mode[1] if self.active_mode else None,
            "stream_fps": self.active_mode[2] if self.active_mode else None,
            "color_intrinsics": self.color_intrinsics,
            "frames_recorded": self.frame_counter,
            "frames_dropped": self.frames_dropped,
            "run_start_wall_sec": run_start_wall_sec,
            "run_end_wall_sec": time.time(),
            "run_duration_sec": time.monotonic() - run_start_monotonic_sec,
        }
        (self.output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )


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
    recorder: RgbRecorder | None = None,
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
        if recorder is not None:
            recorder.maybe_record(
                robot_state,
                waypoint_index=target_index,
                waypoint_label=target_label,
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
    if recorder is not None:
        recorder.maybe_record(
            robot_state,
            waypoint_index=last_logged_target_index,
            waypoint_label=last_logged_target_label,
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
    parser.add_argument(
        "--record-output-dir",
        type=Path,
        help="If set, record a RealSense RGB video and per-frame pose log for the second-round motion here.",
    )
    parser.add_argument("--width", type=int, default=1280, help="RealSense RGB stream width.")
    parser.add_argument("--height", type=int, default=720, help="RealSense RGB stream height.")
    parser.add_argument("--fps", type=int, default=30, help="RealSense RGB stream FPS.")
    parser.add_argument(
        "--record-fps",
        type=float,
        default=30.0,
        help="Maximum pose/video recording rate during the second-round motion.",
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
    if args.record_fps <= 0.0:
        raise ValueError("--record-fps must be positive.")

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
    if args.record_output_dir is not None:
        print(f"Second-round recording output: {args.record_output_dir}")
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
    gripper = Gripper(args.robot_ip) if args.record_output_dir is not None else None
    active_control = None
    try:
        run_start_wall_sec = time.time()
        run_start_monotonic_sec = time.monotonic()
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
            gripper_state = None
            if gripper is not None:
                gripper_state = gripper_state_to_json(gripper.read_once())
                gripper_state["timestamp_wall_sec"] = time.time()
                gripper_state["timestamp_monotonic_sec"] = time.monotonic()
            recorder: RgbRecorder
            with RgbRecorder(
                output_dir=args.record_output_dir,
                width=args.width,
                height=args.height,
                fps=args.fps,
                record_fps=args.record_fps,
                allow_stream_fallback=args.allow_stream_fallback,
                warmup_frames=args.warmup_frames,
                frame_timeout_ms=args.frame_timeout_ms,
                startup_grace_sec=args.startup_grace_sec,
                warmup_retries=args.warmup_retries,
                gripper_state=gripper_state,
            ) as recorder, JointStatusLogger(status_output, args.status_fps) as logger:
                active_control = start_cartesian_control(robot)
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
                    recorder=recorder,
                )
            recorder.write_metadata(
                robot_ip=args.robot_ip,
                memory_path=memory_path,
                motion_keyframes=motion_keyframes,
                motion_targets=motion_targets,
                second_round_duration_sec=second_round_duration_sec,
                status_output=status_output,
                run_start_wall_sec=run_start_wall_sec,
                run_start_monotonic_sec=run_start_monotonic_sec,
            )
            if args.record_output_dir is not None:
                print(f"Recorded {recorder.frame_counter} frames to {args.record_output_dir}")

        print(f"Finished VLA memory motion. Wrote {status_output}")
        return 0
    except Exception:
        robot.stop()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
