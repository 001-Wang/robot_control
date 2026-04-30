#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
from pylibfranka import CartesianPose, ControllerMode, RealtimeConfig, Robot

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move the Franka through Cartesian waypoints while recording a RealSense "
            "RGB video, aligned depth frames, and robot/camera poses for each recorded frame."
        )
    )
    parser.add_argument(
        "--robot-ip",
        default="192.168.1.11",
        help="Franka robot IP address.",
    )
    parser.add_argument(
        "--waypoints",
        type=Path,
        
        required=True,
        help="JSON file describing absolute base_T_ee waypoint poses.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where video and pose logs are written.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help=(
            "Optional hand-eye calibration JSON containing gripper_T_camera.matrix. "
            "When provided, base_T_camera is logged for each frame too."
        ),
    )
    parser.add_argument("--width", type=int, default=640, help="RGB stream width.")
    parser.add_argument("--height", type=int, default=480, help="RGB stream height.")
    parser.add_argument("--fps", type=int, default=30, help="RGB stream FPS.")
    parser.add_argument(
        "--allow-stream-fallback",
        action="store_true",
        help="If the requested RealSense stream mode is unsupported, try nearby known-safe color modes.",
    )
    parser.add_argument(
        "--frame-timeout-ms",
        type=int,
        default=5000,
        help="Timeout for RealSense waits while warming up.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Frames to discard before recording starts.",
    )
    parser.add_argument(
        "--startup-grace-sec",
        type=float,
        default=2.0,
        help="Extra delay after starting the RealSense pipeline before warmup begins.",
    )
    parser.add_argument(
        "--warmup-retries",
        type=int,
        default=3,
        help="How many times to retry camera warmup before failing.",
    )
    parser.add_argument(
        "--hold-after-each",
        type=float,
        default=0.25,
        help="Extra hold time in seconds after reaching each waypoint.",
    )
    parser.add_argument(
        "--record-during-motion",
        action="store_true",
        help=(
            "Also record frames while the robot is moving between waypoints. "
            "By default, recording happens during hold phases only to keep the "
            "Franka control loop lighter."
        ),
    )
    parser.add_argument(
        "--video-name",
        default="trajectory.mp4",
        help="Output video filename inside --output-dir.",
    )
    parser.add_argument(
        "--depth-preview-video-name",
        default="depth_preview.mp4",
        help="Output colorized depth preview video filename inside --output-dir.",
    )
    parser.add_argument(
        "--poses-name",
        default="frame_poses.jsonl",
        help="Output per-frame pose log filename inside --output-dir.",
    )
    parser.add_argument(
        "--depth-dir-name",
        default="depth",
        help="Directory inside --output-dir where per-waypoint raw depth stills are written as 16-bit PNGs.",
    )
    parser.add_argument(
        "--rgb-dir-name",
        default="rgb",
        help="Directory inside --output-dir where per-waypoint RGB stills are written as PNGs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate robot connection, camera streaming, waypoint loading, and print the "
            "current base_T_ee without starting Cartesian control or moving the robot."
        ),
    )
    parser.add_argument(
        "--use-current-pose-as-start",
        action="store_true",
        help=(
            "Rebase the waypoint trajectory onto the robot's live base_T_ee at runtime. "
            "Waypoint 0 becomes the current pose, and later waypoints keep their original "
            "relative transform from waypoint 0."
        ),
    )
    parser.add_argument(
        "--hold-capture-fps",
        type=float,
        default=3.0,
        help=(
            "Maximum capture rate during waypoint hold phases. Lower values reduce control-loop "
            "load while keeping videos and waypoint stills."
        ),
    )
    return parser.parse_args()


def safe_camera_info(device_or_sensor: Any, info: Any, fallback: str) -> str:
    try:
        return device_or_sensor.get_info(info)
    except Exception:
        return fallback


def candidate_stream_modes(width: int, height: int, fps: int) -> list[tuple[int, int, int]]:
    requested = (width, height, fps)
    common_modes = [
        (424, 240, 6),
        (424, 240, 15),
        (640, 480, 6),
        (640, 480, 15),
        (640, 480, 30),
    ]
    ordered = [requested]
    for mode in common_modes:
        if mode != requested:
            ordered.append(mode)
    return ordered


def start_pipeline_with_mode(
    *,
    pipeline: rs.pipeline,
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
            rs.stream.depth,
            candidate_width,
            candidate_height,
            rs.format.z16,
            candidate_fps,
        )
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

    raise RuntimeError(
        "Failed to start the RealSense pipeline for all attempted stream modes.\n"
        + "\n".join(errors)
    )


def wait_for_frames(pipeline: rs.pipeline, timeout_ms: int, frame_index: int) -> rs.composite_frame:
    try:
        return pipeline.wait_for_frames(timeout_ms=timeout_ms)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Timed out waiting for frame {frame_index} after {timeout_ms} ms.\n"
            "Checks:\n"
            "  - Reconnect the RealSense and prefer a direct USB 3 port.\n"
            "  - Close realsense-viewer, ROS nodes, and other camera users.\n"
            "  - Try a lower stream load such as --width 424 --height 240 --fps 15.\n"
            f"Original error: {exc}"
        ) from exc


def warmup_camera(
    *,
    pipeline: rs.pipeline,
    warmup_frames: int,
    timeout_ms: int,
    startup_grace_sec: float,
    warmup_retries: int,
) -> None:
    if startup_grace_sec > 0.0:
        time.sleep(startup_grace_sec)

    last_error: Exception | None = None
    attempts = max(1, warmup_retries)
    for attempt in range(1, attempts + 1):
        try:
            for frame_index in range(1, warmup_frames + 1):
                wait_for_frames(pipeline, timeout_ms, frame_index)
            return
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            print(
                f"Camera warmup attempt {attempt}/{attempts} failed: {exc}\n"
                "Retrying RealSense warmup..."
            )
            time.sleep(1.0)

    assert last_error is not None
    raise RuntimeError(f"RealSense warmup failed after {attempts} attempts: {last_error}")


def pose_list_to_matrix(pose: list[float]) -> np.ndarray:
    return np.array(pose, dtype=float).reshape(4, 4, order="F")


def matrix_to_pose_list(matrix: np.ndarray) -> list[float]:
    return np.asarray(matrix, dtype=float).reshape(16, order="F").tolist()


def validate_transform_matrix(matrix: Any) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.shape != (4, 4):
        raise ValueError("Pose matrix must have shape 4x4.")
    return array


def quaternion_xyzw_to_matrix(quaternion_xyzw: list[float]) -> np.ndarray:
    x, y, z, w = [float(value) for value in quaternion_xyzw]
    norm = np.linalg.norm([x, y, z, w])
    if norm == 0.0:
        raise ValueError("Quaternion norm must be non-zero.")
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


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
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def transform_from_payload(payload: Any) -> np.ndarray:
    if isinstance(payload, dict):
        if "matrix" in payload:
            return validate_transform_matrix(payload["matrix"])
        if "translation" in payload and "quaternion_xyzw" in payload:
            translation = np.asarray(payload["translation"], dtype=float)
            if translation.shape != (3,):
                raise ValueError("Translation must have exactly 3 values.")
            transform = np.eye(4, dtype=float)
            transform[:3, :3] = quaternion_xyzw_to_matrix(payload["quaternion_xyzw"])
            transform[:3, 3] = translation
            return transform
    return validate_transform_matrix(payload)


def load_waypoints(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    items = payload["waypoints"] if isinstance(payload, dict) and "waypoints" in payload else payload
    if not isinstance(items, list) or not items:
        raise ValueError("Waypoint file must contain a non-empty list or {'waypoints': [...]} .")

    waypoints: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if isinstance(item, dict) and any(key in item for key in {"pose", "matrix", "translation"}):
            pose_payload = item.get("pose", item)
            duration_sec = float(item.get("duration_sec", 3.0))
            hold_sec = float(item.get("hold_sec", 0.0))
            label = str(item.get("label", f"wp_{index:03d}"))
        else:
            pose_payload = item
            duration_sec = 3.0
            hold_sec = 0.0
            label = f"wp_{index:03d}"

        if duration_sec <= 0.0:
            raise ValueError(f"Waypoint {index} has non-positive duration_sec.")
        if hold_sec < 0.0:
            raise ValueError(f"Waypoint {index} has negative hold_sec.")

        waypoints.append(
            {
                "label": label,
                "duration_sec": duration_sec,
                "hold_sec": hold_sec,
                "target_matrix": transform_from_payload(pose_payload),
            }
        )
    return waypoints


def load_gripper_T_camera(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    if "gripper_T_camera" not in payload:
        raise ValueError("Calibration JSON must contain gripper_T_camera.")
    return transform_from_payload(payload["gripper_T_camera"])


def apply_default_collision_behavior(robot: Robot) -> None:
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


def slerp_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def quintic_time_scaling(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    # 10 a^3 - 15 a^4 + 6 a^5 gives zero velocity and acceleration at endpoints.
    return alpha * alpha * alpha * (10.0 + alpha * (-15.0 + 6.0 * alpha))


def pose_interp(start_matrix: np.ndarray, target_matrix: np.ndarray, alpha: float) -> np.ndarray:
    alpha = quintic_time_scaling(alpha)
    start_translation = start_matrix[:3, 3]
    target_translation = target_matrix[:3, 3]
    translation = (1.0 - alpha) * start_translation + alpha * target_translation

    q0 = matrix_to_quaternion_xyzw(start_matrix)
    q1 = matrix_to_quaternion_xyzw(target_matrix)
    rotation = quaternion_xyzw_to_matrix(slerp_xyzw(q0, q1, alpha).tolist())

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def rotation_angle_deg(rotation_delta: np.ndarray) -> float:
    trace = float(np.trace(rotation_delta))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def poses_are_close(
    start_matrix: np.ndarray,
    target_matrix: np.ndarray,
    *,
    translation_tol_m: float = 1e-3,
    rotation_tol_deg: float = 0.5,
) -> bool:
    translation_error = float(np.linalg.norm(target_matrix[:3, 3] - start_matrix[:3, 3]))
    rotation_delta = start_matrix[:3, :3].T @ target_matrix[:3, :3]
    angle_error_deg = rotation_angle_deg(rotation_delta)
    return translation_error <= translation_tol_m and angle_error_deg <= rotation_tol_deg


def jsonable_matrix(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=float).tolist()


def intrinsics_to_json(profile: rs.video_stream_profile) -> dict[str, Any]:
    intrinsics = profile.get_intrinsics()
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


def make_video_writer(path: Path, fps: int, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {path}")
    return writer


def depth_frame_to_png(depth_frame: rs.depth_frame, path: Path) -> None:
    depth_image = np.asanyarray(depth_frame.get_data())
    if depth_image.ndim != 2 or depth_image.dtype != np.uint16:
        raise RuntimeError("Expected aligned depth frame as a 2D uint16 image.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), depth_image):
        raise RuntimeError(f"Failed to write depth image to {path}")


def image_bgr_to_png(image_bgr: np.ndarray, path: Path) -> None:
    if image_bgr.ndim != 3 or image_bgr.dtype != np.uint8:
        raise RuntimeError("Expected RGB image as a 3D uint8 array.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Failed to write RGB image to {path}")


def depth_image_to_preview(depth_image: np.ndarray) -> np.ndarray:
    valid = depth_image[depth_image > 0]
    if valid.size == 0:
        return np.zeros((*depth_image.shape, 3), dtype=np.uint8)

    near = float(np.percentile(valid, 1))
    far = float(np.percentile(valid, 99))
    if far <= near:
        far = near + 1.0

    clipped = np.clip(depth_image.astype(np.float32), near, far)
    normalized = ((clipped - near) / (far - near) * 255.0).astype(np.uint8)
    normalized[depth_image == 0] = 0
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


class LatestAlignedFrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, pipeline: rs.pipeline, align_to_color: rs.align) -> None:
        if self._thread is not None:
            return

        def worker() -> None:
            while not self._stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except Exception:
                    continue

                aligned_frames = align_to_color.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data()).copy()
                depth_image = np.asanyarray(depth_frame.get_data()).copy()
                if color_image.ndim != 3 or depth_image.ndim != 2:
                    continue

                latest = {
                    "color_image": color_image,
                    "depth_image": depth_image,
                    "realsense_timestamp_ms": float(color_frame.get_timestamp()),
                    "realsense_frame_number": int(color_frame.get_frame_number()),
                    "depth_realsense_frame_number": int(depth_frame.get_frame_number()),
                }
                with self._lock:
                    self._latest = latest

        self._thread = threading.Thread(target=worker, name="realsense-frame-buffer", daemon=True)
        self._thread.start()

    def get_latest(self) -> dict[str, Any] | None:
        with self._lock:
            if self._latest is None:
                return None
            return {
                "color_image": self._latest["color_image"].copy(),
                "depth_image": self._latest["depth_image"].copy(),
                "realsense_timestamp_ms": self._latest["realsense_timestamp_ms"],
                "realsense_frame_number": self._latest["realsense_frame_number"],
                "depth_realsense_frame_number": self._latest["depth_realsense_frame_number"],
            }

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class AsyncRecordingWriter:
    def __init__(
        self,
        *,
        video_writer: cv2.VideoWriter,
        depth_preview_writer: cv2.VideoWriter,
        pose_log_file: Any,
    ) -> None:
        self._video_writer = video_writer
        self._depth_preview_writer = depth_preview_writer
        self._pose_log_file = pose_log_file
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="recording-writer", daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            packet = self._queue.get()
            if packet is None:
                self._queue.task_done()
                break

            color_image = packet["color_image"]
            depth_image = packet["depth_image"]
            self._video_writer.write(color_image)
            self._depth_preview_writer.write(depth_image_to_preview(depth_image))

            rgb_path: Path | None = packet["rgb_path"]
            depth_path: Path | None = packet["depth_path"]
            if rgb_path is not None:
                image_bgr_to_png(color_image, rgb_path)
            if depth_path is not None:
                depth_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(depth_path), depth_image):
                    raise RuntimeError(f"Failed to write depth image to {depth_path}")

            self._pose_log_file.write(json.dumps(packet["entry"]) + "\n")
            self._pose_log_file.flush()
            self._queue.task_done()

        self._video_writer.release()
        self._depth_preview_writer.release()
        self._pose_log_file.close()

    def enqueue(self, packet: dict[str, Any]) -> None:
        self._queue.put(packet)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=10.0)


def maybe_record_frame(
    *,
    frame_buffer: LatestAlignedFrameBuffer,
    recording_writer: AsyncRecordingWriter,
    robot_state: Any,
    monotonic_time_sec: float,
    wall_time_sec: float,
    waypoint_index: int,
    waypoint_label: str,
    gripper_T_camera: np.ndarray | None,
    camera_T_gripper: np.ndarray | None,
    depth_dir: Path,
    rgb_dir: Path,
    still_basename: str | None,
    frame_counter: int,
) -> int:
    latest_frame = frame_buffer.get_latest()
    if latest_frame is None:
        return frame_counter

    image_bgr = latest_frame["color_image"]
    if image_bgr.ndim != 3:
        return frame_counter

    depth_image = latest_frame["depth_image"]

    depth_path_relative = None
    rgb_path_relative = None
    rgb_path = None
    depth_path = None
    if still_basename is not None:
        rgb_path = rgb_dir / f"{still_basename}.png"
        depth_path = depth_dir / f"{still_basename}.png"
        rgb_path_relative = str(rgb_path.relative_to(rgb_dir.parent))
        depth_path_relative = str(depth_path.relative_to(depth_dir.parent))

    base_T_ee = pose_list_to_matrix(robot_state.O_T_EE)
    entry: dict[str, Any] = {
        "frame_index": frame_counter,
        "timestamp_wall_sec": wall_time_sec,
        "timestamp_monotonic_sec": monotonic_time_sec,
        "realsense_timestamp_ms": latest_frame["realsense_timestamp_ms"],
        "realsense_frame_number": latest_frame["realsense_frame_number"],
        "depth_realsense_frame_number": latest_frame["depth_realsense_frame_number"],
        "waypoint_index": waypoint_index,
        "waypoint_label": waypoint_label,
        "base_T_ee": {"matrix": jsonable_matrix(base_T_ee)},
        "base_T_gripper": {"matrix": jsonable_matrix(base_T_ee)},
    }
    if rgb_path_relative is not None:
        entry["rgb_path"] = rgb_path_relative
    if depth_path_relative is not None:
        entry["depth_path"] = depth_path_relative
    if gripper_T_camera is not None:
        base_T_camera = base_T_ee @ gripper_T_camera
        entry["base_T_camera"] = {"matrix": jsonable_matrix(base_T_camera)}
        if camera_T_gripper is not None:
            entry["camera_T_gripper"] = {"matrix": jsonable_matrix(camera_T_gripper)}
        entry["camera_T_base"] = {"matrix": jsonable_matrix(np.linalg.inv(base_T_camera))}

    recording_writer.enqueue(
        {
            "color_image": image_bgr,
            "depth_image": depth_image,
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "entry": entry,
        }
    )
    return frame_counter + 1


def print_waypoint_summary(waypoints: list[dict[str, Any]]) -> None:
    print("Loaded waypoints:")
    for index, waypoint in enumerate(waypoints):
        translation = waypoint["target_matrix"][:3, 3]
        print(
            f"  [{index}] {waypoint['label']}: "
            f"xyz={translation.tolist()} "
            f"duration={waypoint['duration_sec']:.3f}s "
            f"hold={waypoint['hold_sec']:.3f}s"
        )


def print_transform(label: str, matrix: np.ndarray) -> None:
    np.set_printoptions(precision=6, suppress=True)
    print(f"{label}:")
    print(np.asarray(matrix, dtype=float))


def rebase_waypoints_to_current_pose(
    waypoints: list[dict[str, Any]],
    current_base_T_ee: np.ndarray,
) -> list[dict[str, Any]]:
    if not waypoints:
        return waypoints

    original_start = waypoints[0]["target_matrix"]
    start_inv = np.linalg.inv(original_start)
    rebased: list[dict[str, Any]] = []
    for waypoint in waypoints:
        relative_transform = start_inv @ waypoint["target_matrix"]
        rebased_waypoint = dict(waypoint)
        rebased_waypoint["target_matrix"] = current_base_T_ee @ relative_transform
        rebased.append(rebased_waypoint)
    return rebased


def main() -> None:
    args = parse_args()
    waypoints = load_waypoints(args.waypoints)
    gripper_T_camera = load_gripper_T_camera(args.calibration) if args.calibration else None
    camera_T_gripper = np.linalg.inv(gripper_T_camera) if gripper_T_camera is not None else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / args.video_name
    depth_preview_video_path = args.output_dir / args.depth_preview_video_name
    pose_log_path = args.output_dir / args.poses_name
    metadata_path = args.output_dir / "run_metadata.json"
    depth_dir = args.output_dir / args.depth_dir_name
    rgb_dir = args.output_dir / args.rgb_dir_name

    print_waypoint_summary(waypoints)
    print(f"Video output: {video_path}")
    print(f"Depth preview video: {depth_preview_video_path}")
    print(f"RGB still output: {rgb_dir}")
    print(f"Depth still output: {depth_dir}")
    print(f"Pose log:     {pose_log_path}")
    if gripper_T_camera is not None:
        print("Camera pose logging enabled via gripper_T_camera.")
    if args.record_during_motion:
        print("Frame capture mode: during motion and holds.")
    else:
        print("Frame capture mode: hold phases only.")
    print()
    print("WARNING: This script will move the robot.")
    print("Make sure the path is collision-free and keep the stop button at hand.")
    input("Press Enter to continue...")

    pipeline = rs.pipeline()
    frame_buffer = LatestAlignedFrameBuffer()
    robot = None
    video_writer = None
    depth_preview_writer = None
    pose_log_file = None
    recording_writer = None
    active_control = None
    frame_counter = 0
    last_hold_capture_time_sec = float("-inf")
    run_start_wall_sec = time.time()
    run_start_monotonic_sec = time.monotonic()
    align_to_color = rs.align(rs.stream.color)
    color_intrinsics = None
    depth_intrinsics = None
    depth_scale = None

    try:
        robot = Robot(args.robot_ip, RealtimeConfig.kIgnore)
        apply_default_collision_behavior(robot)

        profile, active_mode = start_pipeline_with_mode(
            pipeline=pipeline,
            width=args.width,
            height=args.height,
            fps=args.fps,
            allow_stream_fallback=args.allow_stream_fallback,
        )
        active_device = profile.get_device()
        active_name = safe_camera_info(active_device, rs.camera_info.name, "<unknown>")
        active_serial = safe_camera_info(active_device, rs.camera_info.serial_number, "<unknown>")
        print(
            f"Streaming from: {active_name} | serial={active_serial} | "
            f"mode={active_mode[0]}x{active_mode[1]} @ {active_mode[2]} FPS"
        )

        color_stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_stream_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_intrinsics = intrinsics_to_json(color_stream_profile)
        depth_intrinsics = intrinsics_to_json(depth_stream_profile)
        first_depth_sensor = active_device.first_depth_sensor()
        depth_scale = float(first_depth_sensor.get_depth_scale())

        warmup_camera(
            pipeline=pipeline,
            warmup_frames=args.warmup_frames,
            timeout_ms=args.frame_timeout_ms,
            startup_grace_sec=args.startup_grace_sec,
            warmup_retries=args.warmup_retries,
        )
        frame_buffer.start(pipeline, align_to_color)

        robot_state = robot.read_once()
        current_base_T_ee = pose_list_to_matrix(robot_state.O_T_EE)
        print_transform("Current base_T_ee before Cartesian control", current_base_T_ee)

        if args.use_current_pose_as_start:
            waypoints = rebase_waypoints_to_current_pose(waypoints, current_base_T_ee)
            print("Rebased waypoint trajectory onto current base_T_ee.")
            print_waypoint_summary(waypoints)

        if args.dry_run:
            print("Dry run succeeded. Exiting before Cartesian control and motion.")
            return

        video_writer = make_video_writer(video_path, active_mode[2], active_mode[0], active_mode[1])
        depth_preview_writer = make_video_writer(
            depth_preview_video_path,
            active_mode[2],
            active_mode[0],
            active_mode[1],
        )
        pose_log_file = pose_log_path.open("w", encoding="utf-8")
        recording_writer = AsyncRecordingWriter(
            video_writer=video_writer,
            depth_preview_writer=depth_preview_writer,
            pose_log_file=pose_log_file,
        )
        depth_dir.mkdir(parents=True, exist_ok=True)
        rgb_dir.mkdir(parents=True, exist_ok=True)

        try:
            active_control = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)
        except Exception as exc:
            message = str(exc)
            if "cannot start at singular pose" in message:
                raise RuntimeError(
                    "Failed to start Cartesian pose control because the robot is currently at "
                    "or near a singular pose.\n"
                    "Use Franka Desk to move the arm to a less singular joint configuration, "
                    "then retry.\n"
                    "Tip: keep the tool pose similar, but bend the elbow/wrist away from a "
                    "straight or fully aligned configuration."
                ) from exc
            raise
        robot_state, duration = active_control.readOnce()
        current_segment_start = pose_list_to_matrix(robot_state.O_T_EE)
        saved_waypoint_stills: set[int] = set()

        for waypoint_index, waypoint in enumerate(waypoints):
            target_matrix = waypoint["target_matrix"]
            segment_duration = waypoint["duration_sec"]
            segment_hold = waypoint["hold_sec"] + args.hold_after_each
            waypoint_label = waypoint["label"]
            segment_elapsed = 0.0

            print(f"Moving to waypoint [{waypoint_index}] {waypoint_label}")
            if poses_are_close(current_segment_start, target_matrix):
                print(f"Waypoint [{waypoint_index}] already reached; skipping motion segment.")
            else:
                while segment_elapsed < segment_duration:
                    robot_state, duration = active_control.readOnce()
                    dt = duration.to_sec()
                    segment_elapsed += dt
                    alpha = min(segment_elapsed / segment_duration, 1.0)
                    commanded_matrix = pose_interp(current_segment_start, target_matrix, alpha)
                    cartesian_pose = CartesianPose(matrix_to_pose_list(commanded_matrix))
                    active_control.writeOnce(cartesian_pose)

                    if args.record_during_motion:
                        now_wall = time.time()
                        now_mono = time.monotonic()
                        frame_counter = maybe_record_frame(
                            frame_buffer=frame_buffer,
                            recording_writer=recording_writer,
                            robot_state=robot_state,
                            monotonic_time_sec=now_mono,
                            wall_time_sec=now_wall,
                            waypoint_index=waypoint_index,
                            waypoint_label=waypoint_label,
                            gripper_T_camera=gripper_T_camera,
                            camera_T_gripper=camera_T_gripper,
                            depth_dir=depth_dir,
                            rgb_dir=rgb_dir,
                            still_basename=None,
                            frame_counter=frame_counter,
                        )

            hold_elapsed = 0.0
            while hold_elapsed < segment_hold:
                robot_state, duration = active_control.readOnce()
                dt = duration.to_sec()
                hold_elapsed += dt
                cartesian_pose = CartesianPose(matrix_to_pose_list(target_matrix))
                active_control.writeOnce(cartesian_pose)

                now_wall = time.time()
                now_mono = time.monotonic()
                capture_interval_sec = 0.0 if args.hold_capture_fps <= 0.0 else 1.0 / args.hold_capture_fps
                if now_mono - last_hold_capture_time_sec < capture_interval_sec:
                    continue
                frame_counter_before = frame_counter
                still_basename = None
                if waypoint_index not in saved_waypoint_stills:
                    still_basename = f"waypoint_{waypoint_index:03d}_{waypoint_label}"
                frame_counter = maybe_record_frame(
                    frame_buffer=frame_buffer,
                    recording_writer=recording_writer,
                    robot_state=robot_state,
                    monotonic_time_sec=now_mono,
                    wall_time_sec=now_wall,
                    waypoint_index=waypoint_index,
                    waypoint_label=waypoint_label,
                    gripper_T_camera=gripper_T_camera,
                    camera_T_gripper=camera_T_gripper,
                    depth_dir=depth_dir,
                    rgb_dir=rgb_dir,
                    still_basename=still_basename,
                    frame_counter=frame_counter,
                )
                if still_basename is not None and frame_counter > frame_counter_before:
                    saved_waypoint_stills.add(waypoint_index)
                if frame_counter > frame_counter_before:
                    last_hold_capture_time_sec = now_mono

            current_segment_start = target_matrix.copy()

        robot_state, _ = active_control.readOnce()
        final_pose = CartesianPose(matrix_to_pose_list(current_segment_start))
        final_pose.motion_finished = True
        active_control.writeOnce(final_pose)

        metadata = {
            "robot_ip": args.robot_ip,
            "waypoint_file": str(args.waypoints.resolve()),
            "video_path": str(video_path.resolve()),
            "depth_preview_video_path": str(depth_preview_video_path.resolve()),
            "rgb_dir": str(rgb_dir.resolve()),
            "depth_dir": str(depth_dir.resolve()),
            "pose_log_path": str(pose_log_path.resolve()),
            "requested_width": args.width,
            "requested_height": args.height,
            "requested_fps": args.fps,
            "stream_width": active_mode[0],
            "stream_height": active_mode[1],
            "stream_fps": active_mode[2],
            "color_intrinsics": color_intrinsics,
            "depth_intrinsics": depth_intrinsics,
            "depth_scale_m_per_unit": depth_scale,
            "frames_recorded": frame_counter,
            "run_start_wall_sec": run_start_wall_sec,
            "run_end_wall_sec": time.time(),
            "run_duration_sec": time.monotonic() - run_start_monotonic_sec,
            "gripper_T_camera": (
                {"matrix": jsonable_matrix(gripper_T_camera)} if gripper_T_camera is not None else None
            ),
            "camera_T_gripper": (
                {"matrix": jsonable_matrix(camera_T_gripper)} if camera_T_gripper is not None else None
            ),
            "waypoints": [
                {
                    "label": waypoint["label"],
                    "duration_sec": waypoint["duration_sec"],
                    "hold_sec": waypoint["hold_sec"],
                    "target_matrix": jsonable_matrix(waypoint["target_matrix"]),
                }
                for waypoint in waypoints
            ],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(f"Finished. Recorded {frame_counter} frames.")
        print(f"Saved video to {video_path}")
        print(f"Saved depth preview video to {depth_preview_video_path}")
        print(f"Saved RGB stills to {rgb_dir}")
        print(f"Saved raw depth stills to {depth_dir}")
        print(f"Saved per-frame poses to {pose_log_path}")
        print(f"Saved run metadata to {metadata_path}")

    except Exception as exc:
        raise SystemExit(f"Recording failed: {exc}") from exc
    finally:
        if recording_writer is not None:
            try:
                recording_writer.close()
            except Exception:
                pass
        try:
            frame_buffer.stop()
        except Exception:
            pass
        try:
            pipeline.stop()
        except Exception:
            pass
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
