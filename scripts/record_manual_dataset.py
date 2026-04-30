#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously record a manual eye-in-hand dataset while you move the robot by hand: "
            "RGB video, raw aligned depth frames, depth preview video, per-frame base_T_ee, "
            "and derived base_T_camera when calibration is provided."
        )
    )
    parser.add_argument("--robot-ip", default="192.168.1.11", help="Franka robot IP address.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Optional hand-eye calibration JSON containing gripper_T_camera.",
    )
    parser.add_argument("--width", type=int, default=640, help="RGB/depth stream width.")
    parser.add_argument("--height", type=int, default=480, help="RGB/depth stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Camera stream FPS.")
    parser.add_argument(
        "--capture-fps",
        type=float,
        default=5.0,
        help="Maximum frame logging rate. Lower values reduce system load.",
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
        help="Delay after starting RealSense before warmup begins.",
    )
    parser.add_argument(
        "--frame-timeout-ms",
        type=int,
        default=5000,
        help="Timeout for RealSense waits during warmup.",
    )
    parser.add_argument(
        "--warmup-retries",
        type=int,
        default=3,
        help="How many times to retry RealSense warmup before failing.",
    )
    parser.add_argument(
        "--allow-stream-fallback",
        action="store_true",
        help="Try nearby known-safe RealSense stream modes if the requested mode fails.",
    )
    parser.add_argument("--preview", action="store_true", help="Show live RGB + depth preview.")
    parser.add_argument("--video-name", default="trajectory.mp4", help="RGB video filename.")
    parser.add_argument(
        "--depth-preview-video-name",
        default="depth_preview.mp4",
        help="Colorized depth preview video filename.",
    )
    parser.add_argument("--poses-name", default="frame_poses.jsonl", help="Per-frame pose log filename.")
    parser.add_argument("--rgb-dir-name", default="rgb", help="Directory for saved RGB frames.")
    parser.add_argument("--depth-dir-name", default="depth", help="Directory for saved depth PNGs.")
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Save RGB/depth images and poses only. Skip RGB/depth video recording.",
    )
    parser.add_argument(
        "--save-every-frame",
        action="store_true",
        help="Save RGB PNGs for every logged frame in addition to RGB video.",
    )
    parser.add_argument(
        "--preview-dir-name",
        default="preview",
        help="Directory for latest preview images when GUI preview is unavailable.",
    )
    parser.add_argument(
        "--export-h264",
        action="store_true",
        help="Also export an H.264-compatible copy of the RGB video after recording.",
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
        config.enable_stream(rs.stream.depth, candidate_width, candidate_height, rs.format.z16, candidate_fps)
        config.enable_stream(rs.stream.color, candidate_width, candidate_height, rs.format.bgr8, candidate_fps)
        try:
            profile = pipeline.start(config)
            return profile, (candidate_width, candidate_height, candidate_fps)
        except RuntimeError as exc:
            errors.append(f"{candidate_width}x{candidate_height} @ {candidate_fps} FPS -> {exc}")
    raise RuntimeError("Failed to start the RealSense pipeline for all attempted stream modes.\n" + "\n".join(errors))


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
    for attempt in range(1, max(1, warmup_retries) + 1):
        try:
            for frame_index in range(1, warmup_frames + 1):
                wait_for_frames(pipeline, timeout_ms, frame_index)
            return
        except Exception as exc:
            last_error = exc
            if attempt == max(1, warmup_retries):
                break
            print(f"Camera warmup attempt {attempt}/{warmup_retries} failed: {exc}")
            time.sleep(1.0)
    raise RuntimeError(f"RealSense warmup failed: {last_error}")


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


def validate_transform_matrix(matrix: Any) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.shape != (4, 4):
        raise ValueError("Pose matrix must have shape 4x4.")
    return array


def transform_from_payload(payload: Any) -> np.ndarray:
    if isinstance(payload, dict) and "matrix" in payload:
        return validate_transform_matrix(payload["matrix"])
    return validate_transform_matrix(payload)


def load_gripper_T_camera(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    if "gripper_T_camera" not in payload:
        raise ValueError("Calibration JSON must contain gripper_T_camera.")
    return transform_from_payload(payload["gripper_T_camera"])


def pose_list_to_matrix(pose: list[float]) -> np.ndarray:
    return np.array(pose, dtype=float).reshape(4, 4, order="F")


def jsonable_matrix(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=float).tolist()


class ConsoleInputThread:
    def __init__(self) -> None:
        self.queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="manual-dataset-input", daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            try:
                raw = input().strip()
            except EOFError:
                raw = "q"
            self.queue.put(raw)
            if raw.lower() == "q":
                break


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

        self._thread = threading.Thread(target=worker, name="manual-dataset-frame-buffer", daemon=True)
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


class PreviewWindow:
    def __init__(self) -> None:
        self._window_name = "Manual Dataset Preview"
        self._enabled = False
        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            self._enabled = True
        except cv2.error:
            self._enabled = False

    def update(self, color_image: np.ndarray, depth_image: np.ndarray) -> None:
        if not self._enabled:
            return
        combined = np.hstack([color_image, depth_image_to_preview(depth_image)])
        cv2.imshow(self._window_name, combined)
        cv2.waitKey(1)

    def close(self) -> None:
        if self._enabled:
            cv2.destroyWindow(self._window_name)

    @property
    def enabled(self) -> bool:
        return self._enabled


class PreviewFileWriter:
    def __init__(self, preview_dir: Path) -> None:
        self._preview_dir = preview_dir
        self._preview_dir.mkdir(parents=True, exist_ok=True)
        self._last_write_time = float("-inf")

    def update(self, color_image: np.ndarray, depth_image: np.ndarray, now_mono: float) -> None:
        if now_mono - self._last_write_time < 0.25:
            return
        rgb_path = self._preview_dir / "latest_rgb.jpg"
        depth_path = self._preview_dir / "latest_depth.jpg"
        cv2.imwrite(str(rgb_path), color_image)
        cv2.imwrite(str(depth_path), depth_image_to_preview(depth_image))
        self._last_write_time = now_mono


class AsyncWriter:
    def __init__(self, *, video_writer: cv2.VideoWriter, depth_preview_writer: cv2.VideoWriter, pose_log_file: Any) -> None:
        self._video_writer = video_writer
        self._depth_preview_writer = depth_preview_writer
        self._pose_log_file = pose_log_file
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="manual-dataset-writer", daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            packet = self._queue.get()
            if packet is None:
                self._queue.task_done()
                break
            color_image = packet["color_image"]
            depth_image = packet["depth_image"]
            if self._video_writer is not None:
                self._video_writer.write(color_image)
            if self._depth_preview_writer is not None:
                self._depth_preview_writer.write(depth_image_to_preview(depth_image))
            rgb_path: Path | None = packet["rgb_path"]
            if rgb_path is not None:
                rgb_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(rgb_path), color_image)
            depth_path = packet["depth_path"]
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(depth_path), depth_image)
            self._pose_log_file.write(json.dumps(packet["entry"]) + "\n")
            self._pose_log_file.flush()
            self._queue.task_done()
        if self._video_writer is not None:
            self._video_writer.release()
        if self._depth_preview_writer is not None:
            self._depth_preview_writer.release()
        self._pose_log_file.close()

    def enqueue(self, packet: dict[str, Any]) -> None:
        self._queue.put(packet)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=10.0)


def candidate_ffmpeg_paths() -> list[str]:
    candidates: list[str] = []
    for path in ["/usr/bin/ffmpeg", shutil.which("ffmpeg")]:
        if path and path not in candidates:
            candidates.append(path)
    return candidates


def export_compatible_video(video_path: Path, output_path: Path) -> tuple[bool, str]:
    encoder_options = [
        ["-c:v", "libx264", "-pix_fmt", "yuv420p"],
        ["-c:v", "libopenh264", "-pix_fmt", "yuv420p"],
        ["-c:v", "mpeg4"],
    ]
    errors: list[str] = []
    for ffmpeg in candidate_ffmpeg_paths():
        for encoder_args in encoder_options:
            command = [ffmpeg, "-y", "-i", str(video_path), *encoder_args, str(output_path)]
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                return True, f"{Path(ffmpeg).name} with {' '.join(encoder_args)}"
            errors.append(f"{ffmpeg} {' '.join(encoder_args)} -> exit {result.returncode}")
    return False, " | ".join(errors) if errors else "ffmpeg not found"


def main() -> None:
    args = parse_args()
    if args.capture_fps <= 0.0:
        raise ValueError("--capture-fps must be positive.")

    from calibration.franka_utils import connect_robot, exit_with_error

    gripper_T_camera = load_gripper_T_camera(args.calibration) if args.calibration else None
    camera_T_gripper = np.linalg.inv(gripper_T_camera) if gripper_T_camera is not None else None

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / args.video_name
    depth_preview_video_path = output_dir / args.depth_preview_video_name
    pose_log_path = output_dir / args.poses_name
    rgb_dir = output_dir / args.rgb_dir_name
    depth_dir = output_dir / args.depth_dir_name
    preview_dir = output_dir / args.preview_dir_name
    metadata_path = output_dir / "run_metadata.json"

    print(f"Output dir:           {output_dir}")
    if not args.images_only:
        print(f"RGB video:            {video_path}")
        print(f"Depth preview video:  {depth_preview_video_path}")
    print(f"Pose log:             {pose_log_path}")
    print(f"RGB frames dir:       {rgb_dir}")
    print(f"Depth frames dir:     {depth_dir}")
    if args.preview:
        print(f"Preview dir fallback: {preview_dir}")
    if gripper_T_camera is not None:
        print("Camera pose logging enabled via gripper_T_camera.")
    print()
    print("Manual dataset recording")
    print("Commands:")
    print("  Enter: start recording")
    print("  p: pause/resume recording")
    print("  m: mark the next saved frame")
    print("  q: stop and save")
    if args.preview:
        print("Preview window: left RGB, right depth preview.")
    print()
    print("Move the robot manually. No Cartesian control is started by this script.")

    try:
        robot = connect_robot(args.robot_ip)
    except RuntimeError as exc:
        exit_with_error(str(exc))

    pipeline = rs.pipeline()
    align_to_color = rs.align(rs.stream.color)
    frame_buffer = LatestAlignedFrameBuffer()
    preview_window = PreviewWindow() if args.preview else None
    preview_file_writer = None
    if args.preview and preview_window is not None and not preview_window.enabled:
        preview_file_writer = PreviewFileWriter(preview_dir)
        print("OpenCV GUI preview is unavailable. Writing latest preview images to preview/.")
    writer = None
    input_thread = ConsoleInputThread()
    recording_started = False
    paused = False
    mark_next_frame = False
    frame_index = 0
    last_capture_time = float("-inf")
    color_intrinsics = None
    depth_intrinsics = None
    depth_scale = None
    active_mode = None
    run_start_wall_sec = time.time()
    run_start_monotonic_sec = time.monotonic()

    try:
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
        depth_scale = float(active_device.first_depth_sensor().get_depth_scale())

        warmup_camera(
            pipeline=pipeline,
            warmup_frames=args.warmup_frames,
            timeout_ms=args.frame_timeout_ms,
            startup_grace_sec=args.startup_grace_sec,
            warmup_retries=args.warmup_retries,
        )
        frame_buffer.start(pipeline, align_to_color)

        print("Press Enter to start recording...")
        while not recording_started:
            latest_frame = frame_buffer.get_latest()
            if preview_window is not None and latest_frame is not None:
                preview_window.update(latest_frame["color_image"], latest_frame["depth_image"])
            if preview_file_writer is not None and latest_frame is not None:
                preview_file_writer.update(latest_frame["color_image"], latest_frame["depth_image"], time.monotonic())
            try:
                command = input_thread.queue.get(timeout=0.05).strip().lower()
            except queue.Empty:
                continue
            if command == "":
                writer = AsyncWriter(
                    video_writer=(
                        None
                        if args.images_only
                        else make_video_writer(video_path, active_mode[2], active_mode[0], active_mode[1])
                    ),
                    depth_preview_writer=(
                        None
                        if args.images_only
                        else make_video_writer(depth_preview_video_path, active_mode[2], active_mode[0], active_mode[1])
                    ),
                    pose_log_file=pose_log_path.open("w", encoding="utf-8"),
                )
                recording_started = True
                run_start_wall_sec = time.time()
                run_start_monotonic_sec = time.monotonic()
                print("Recording started. Use p to pause/resume, m to mark, q to stop.")
            elif command == "q":
                print("Stopped before recording started.")
                return
            else:
                print("Press Enter to start or q to quit.")

        while True:
            latest_frame = frame_buffer.get_latest()
            if preview_window is not None and latest_frame is not None:
                preview_window.update(latest_frame["color_image"], latest_frame["depth_image"])
            if preview_file_writer is not None and latest_frame is not None:
                preview_file_writer.update(latest_frame["color_image"], latest_frame["depth_image"], time.monotonic())

            try:
                while True:
                    command = input_thread.queue.get_nowait().strip().lower()
                    if command == "q":
                        raise KeyboardInterrupt
                    if command == "p":
                        paused = not paused
                        print("Recording paused." if paused else "Recording resumed.")
                    elif command == "m":
                        mark_next_frame = True
                        print("Will mark the next saved frame.")
            except queue.Empty:
                pass

            if paused or latest_frame is None:
                time.sleep(0.01)
                continue

            now_mono = time.monotonic()
            if now_mono - last_capture_time < (1.0 / args.capture_fps):
                time.sleep(0.005)
                continue

            try:
                state = robot.read_once()
            except Exception as exc:
                raise RuntimeError(f"Failed to read Franka pose during recording: {exc}") from exc

            base_T_ee = pose_list_to_matrix(state.O_T_EE)
            entry: dict[str, Any] = {
                "frame_index": frame_index,
                "timestamp_wall_sec": time.time(),
                "timestamp_monotonic_sec": now_mono,
                "realsense_timestamp_ms": latest_frame["realsense_timestamp_ms"],
                "realsense_frame_number": latest_frame["realsense_frame_number"],
                "depth_realsense_frame_number": latest_frame["depth_realsense_frame_number"],
                "base_T_ee": {"matrix": jsonable_matrix(base_T_ee)},
                "base_T_gripper": {"matrix": jsonable_matrix(base_T_ee)},
            }
            if mark_next_frame:
                entry["marked"] = True
            should_save_rgb = args.save_every_frame or args.images_only
            rgb_path = rgb_dir / f"rgb_{frame_index:06d}.png" if should_save_rgb else None
            depth_path = depth_dir / f"depth_{frame_index:06d}.png"
            entry["depth_path"] = str(depth_path.relative_to(output_dir))
            if rgb_path is not None:
                entry["rgb_path"] = str(rgb_path.relative_to(output_dir))

            if gripper_T_camera is not None:
                base_T_camera = base_T_ee @ gripper_T_camera
                entry["base_T_camera"] = {"matrix": jsonable_matrix(base_T_camera)}
                entry["camera_T_base"] = {"matrix": jsonable_matrix(np.linalg.inv(base_T_camera))}
                if camera_T_gripper is not None:
                    entry["camera_T_gripper"] = {"matrix": jsonable_matrix(camera_T_gripper)}

            assert writer is not None
            writer.enqueue(
                {
                    "color_image": latest_frame["color_image"],
                    "depth_image": latest_frame["depth_image"],
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "entry": entry,
                }
            )
            frame_index += 1
            last_capture_time = now_mono
            mark_next_frame = False
    except KeyboardInterrupt:
        print("Stopping recording...")
    except Exception as exc:
        raise SystemExit(f"Recording failed: {exc}") from exc
    finally:
        if writer is not None:
            try:
                writer.close()
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
        if preview_window is not None:
            try:
                preview_window.close()
            except Exception:
                pass

    metadata = {
        "robot_ip": args.robot_ip,
        "output_dir": str(output_dir),
        "video_path": None if args.images_only else str(video_path.resolve()),
        "depth_preview_video_path": None if args.images_only else str(depth_preview_video_path.resolve()),
        "rgb_dir": str(rgb_dir.resolve()),
        "depth_dir": str(depth_dir.resolve()),
        "preview_dir": str(preview_dir.resolve()) if args.preview else None,
        "pose_log_path": str(pose_log_path.resolve()),
        "requested_width": args.width,
        "requested_height": args.height,
        "requested_fps": args.fps,
        "capture_fps": args.capture_fps,
        "stream_width": active_mode[0] if active_mode is not None else None,
        "stream_height": active_mode[1] if active_mode is not None else None,
        "stream_fps": active_mode[2] if active_mode is not None else None,
        "color_intrinsics": color_intrinsics,
        "depth_intrinsics": depth_intrinsics,
        "depth_scale_m_per_unit": depth_scale,
        "frames_recorded": frame_index,
        "save_every_frame": args.save_every_frame or args.images_only,
        "images_only": args.images_only,
        "run_start_wall_sec": run_start_wall_sec,
        "run_end_wall_sec": time.time(),
        "run_duration_sec": time.monotonic() - run_start_monotonic_sec,
        "gripper_T_camera": {"matrix": jsonable_matrix(gripper_T_camera)} if gripper_T_camera is not None else None,
        "camera_T_gripper": {"matrix": jsonable_matrix(camera_T_gripper)} if camera_T_gripper is not None else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Finished. Recorded {frame_index} frames.")
    if not args.images_only:
        print(f"Saved RGB video to {video_path}")
        print(f"Saved depth preview video to {depth_preview_video_path}")
    print(f"Saved raw depth frames to {depth_dir}")
    if args.save_every_frame or args.images_only:
        print(f"Saved RGB frames to {rgb_dir}")
    print(f"Saved pose log to {pose_log_path}")
    print(f"Saved metadata to {metadata_path}")
    if preview_file_writer is not None:
        print(f"Updated latest preview images in {preview_dir}")

    if args.export_h264 and not args.images_only:
        h264_path = output_dir / "trajectory_h264.mp4"
        success, detail = export_compatible_video(video_path, h264_path)
        if success:
            print(f"Wrote compatible video to {h264_path} using {detail}")
        else:
            print(f"Skipped compatible video export: {detail}")


if __name__ == "__main__":
    main()
