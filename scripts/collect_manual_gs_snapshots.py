#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import select
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibration.franka_utils import connect_robot, exit_with_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manually collect a Gaussian Splatting snapshot dataset. "
            "Move the robot by hand and press Enter to save one RGB image, one raw depth image, "
            "and the corresponding robot/camera pose."
        )
    )
    parser.add_argument("--robot-ip", default="192.168.1.11", help="Franka robot IP address.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Optional hand-eye calibration JSON containing gripper_T_camera.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Stream width.")
    parser.add_argument("--height", type=int, default=720, help="Stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Stream FPS for preview/capture.")
    parser.add_argument("--preview", action="store_true", help="Show live preview if OpenCV GUI is available.")
    parser.add_argument(
        "--allow-stream-fallback",
        action="store_true",
        help="Try nearby known-safe RealSense stream modes if the requested mode fails.",
    )
    parser.add_argument("--warmup-frames", type=int, default=30, help="Frames to discard before capture starts.")
    parser.add_argument("--startup-grace-sec", type=float, default=2.0, help="Delay after pipeline start.")
    parser.add_argument("--frame-timeout-ms", type=int, default=5000, help="RealSense wait timeout.")
    parser.add_argument("--warmup-retries", type=int, default=3, help="Warmup retry count.")
    parser.add_argument("--rgb-dir-name", default="rgb", help="Directory for RGB images.")
    parser.add_argument("--depth-dir-name", default="depth", help="Directory for raw depth PNGs.")
    parser.add_argument("--preview-dir-name", default="preview", help="Directory for fallback preview images.")
    parser.add_argument("--poses-name", default="frame_poses.jsonl", help="Per-snapshot pose log filename.")
    parser.add_argument(
        "--settle-sec",
        type=float,
        default=0.75,
        help="How long the robot pose must remain nearly unchanged before a snapshot is accepted.",
    )
    parser.add_argument(
        "--stationary-translation-mm",
        type=float,
        default=1.0,
        help="Maximum translation drift, in mm, allowed during the settle window.",
    )
    parser.add_argument(
        "--stationary-rotation-deg",
        type=float,
        default=0.5,
        help="Maximum rotation drift, in degrees, allowed during the settle window.",
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
            "  - Close other camera users.\n"
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


def rotation_angle_deg(rotation_delta: np.ndarray) -> float:
    trace = float(np.trace(rotation_delta))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def pose_distance(
    reference: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float]:
    translation_mm = float(np.linalg.norm(current[:3, 3] - reference[:3, 3]) * 1000.0)
    rotation_delta = reference[:3, :3].T @ current[:3, :3]
    angle_deg = rotation_angle_deg(rotation_delta)
    return translation_mm, angle_deg


def wait_for_stationary_pose(
    *,
    robot: Any,
    settle_sec: float,
    translation_threshold_mm: float,
    rotation_threshold_deg: float,
    poll_interval_sec: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    if settle_sec <= 0.0:
        state = robot.read_once()
        pose = pose_list_to_matrix(state.O_T_EE)
        return pose, pose

    deadline_start = time.monotonic()
    state = robot.read_once()
    reference_pose = pose_list_to_matrix(state.O_T_EE)
    stable_since = deadline_start

    while True:
        time.sleep(poll_interval_sec)
        state = robot.read_once()
        current_pose = pose_list_to_matrix(state.O_T_EE)
        translation_mm, rotation_deg = pose_distance(reference_pose, current_pose)

        if translation_mm <= translation_threshold_mm and rotation_deg <= rotation_threshold_deg:
            if time.monotonic() - stable_since >= settle_sec:
                return reference_pose, current_pose
        else:
            reference_pose = current_pose
            stable_since = time.monotonic()


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


class PreviewWindow:
    def __init__(self) -> None:
        self._window_name = "GS Snapshot Preview"
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
        cv2.imwrite(str(self._preview_dir / "latest_rgb.jpg"), color_image)
        cv2.imwrite(str(self._preview_dir / "latest_depth.jpg"), depth_image_to_preview(depth_image))
        self._last_write_time = now_mono


def main() -> None:
    args = parse_args()
    if args.settle_sec < 0.0:
        raise ValueError("--settle-sec must be non-negative.")
    if args.stationary_translation_mm < 0.0:
        raise ValueError("--stationary-translation-mm must be non-negative.")
    if args.stationary_rotation_deg < 0.0:
        raise ValueError("--stationary-rotation-deg must be non-negative.")

    gripper_T_camera = load_gripper_T_camera(args.calibration) if args.calibration else None
    camera_T_gripper = np.linalg.inv(gripper_T_camera) if gripper_T_camera is not None else None

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = output_dir / args.rgb_dir_name
    depth_dir = output_dir / args.depth_dir_name
    preview_dir = output_dir / args.preview_dir_name
    pose_log_path = output_dir / args.poses_name
    metadata_path = output_dir / "run_metadata.json"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir:      {output_dir}")
    print(f"RGB dir:         {rgb_dir}")
    print(f"Depth dir:       {depth_dir}")
    print(f"Pose log:        {pose_log_path}")
    if args.preview:
        print(f"Preview fallback: {preview_dir}")
    if gripper_T_camera is not None:
        print("Camera pose logging enabled via gripper_T_camera.")
    print()
    print("Manual GS snapshot capture")
    print("Commands:")
    print("  Enter: save one RGB + depth + pose snapshot")
    print("  q: stop and save")
    print("Move the robot manually. No Cartesian control is started by this script.")
    print(
        f"Stationary capture: settle={args.settle_sec:.2f}s "
        f"translation<={args.stationary_translation_mm:.3f} mm "
        f"rotation<={args.stationary_rotation_deg:.3f} deg"
    )

    try:
        robot = connect_robot(args.robot_ip)
    except RuntimeError as exc:
        exit_with_error(str(exc))

    pipeline = rs.pipeline()
    align_to_color = rs.align(rs.stream.color)
    preview_window = PreviewWindow() if args.preview else None
    preview_file_writer = None
    if args.preview and preview_window is not None and not preview_window.enabled:
        preview_file_writer = PreviewFileWriter(preview_dir)
        print("OpenCV GUI preview is unavailable.")
        print(f"Writing latest preview images to: {preview_dir}")
        print(f"  RGB preview:   {preview_dir / 'latest_rgb.jpg'}")
        print(f"  Depth preview: {preview_dir / 'latest_depth.jpg'}")

    color_intrinsics = None
    depth_intrinsics = None
    depth_scale = None
    active_mode = None
    snapshot_index = 0
    pose_log_file = None
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

        pose_log_file = pose_log_path.open("w", encoding="utf-8")
        latest_color = None
        latest_depth = None
        latest_meta = None
        prompt = "Press Enter to save snapshot, or q to quit: "
        print(prompt, end="", flush=True)

        while True:
            frames = pipeline.poll_for_frames()
            if frames:
                aligned_frames = align_to_color.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if color_frame and depth_frame:
                    latest_color = np.asanyarray(color_frame.get_data())
                    latest_depth = np.asanyarray(depth_frame.get_data())
                    latest_meta = {
                        "realsense_timestamp_ms": float(color_frame.get_timestamp()),
                        "realsense_frame_number": int(color_frame.get_frame_number()),
                        "depth_realsense_frame_number": int(depth_frame.get_frame_number()),
                    }
                    if args.preview:
                        if preview_window is not None:
                            preview_window.update(latest_color, latest_depth)
                        if preview_file_writer is not None:
                            preview_file_writer.update(latest_color, latest_depth, time.monotonic())

            ready_stdin, _, _ = select.select([sys.stdin], [], [], 0.03)
            if not ready_stdin:
                continue

            raw = sys.stdin.readline()
            if raw == "":
                raise RuntimeError("Interactive stdin closed while waiting for a snapshot command.")
            raw = raw.strip().lower()
            if raw == "q":
                print()
                break

            print(
                f"\nWaiting for stationary pose before snapshot {snapshot_index} "
                f"(settle {args.settle_sec:.2f}s)..."
            )
            try:
                settle_reference_pose, settle_final_pose = wait_for_stationary_pose(
                    robot=robot,
                    settle_sec=args.settle_sec,
                    translation_threshold_mm=args.stationary_translation_mm,
                    rotation_threshold_deg=args.stationary_rotation_deg,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed while waiting for a stationary Franka pose: {exc}") from exc

            translation_mm, rotation_deg = pose_distance(settle_reference_pose, settle_final_pose)
            print(
                f"Stationary pose accepted for snapshot {snapshot_index}: "
                f"translation_drift_mm={translation_mm:.3f}, rotation_drift_deg={rotation_deg:.3f}"
            )

            frames = wait_for_frames(pipeline, args.frame_timeout_ms, snapshot_index + 1)
            aligned_frames = align_to_color.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError("Failed to fetch a valid aligned RGB/depth frame for snapshot.")
            latest_color = np.asanyarray(color_frame.get_data())
            latest_depth = np.asanyarray(depth_frame.get_data())
            latest_meta = {
                "realsense_timestamp_ms": float(color_frame.get_timestamp()),
                "realsense_frame_number": int(color_frame.get_frame_number()),
                "depth_realsense_frame_number": int(depth_frame.get_frame_number()),
            }

            try:
                state = robot.read_once()
            except Exception as exc:
                raise RuntimeError(f"Failed to read Franka pose for snapshot: {exc}") from exc

            base_T_ee = pose_list_to_matrix(state.O_T_EE)
            final_translation_mm, final_rotation_deg = pose_distance(settle_final_pose, base_T_ee)
            rgb_path = rgb_dir / f"rgb_{snapshot_index:06d}.png"
            depth_path = depth_dir / f"depth_{snapshot_index:06d}.png"
            if not cv2.imwrite(str(rgb_path), latest_color):
                raise RuntimeError(f"Failed to write RGB image to {rgb_path}")
            if not cv2.imwrite(str(depth_path), latest_depth):
                raise RuntimeError(f"Failed to write depth image to {depth_path}")

            entry: dict[str, Any] = {
                "frame_index": snapshot_index,
                "timestamp_wall_sec": time.time(),
                "timestamp_monotonic_sec": time.monotonic(),
                "realsense_timestamp_ms": latest_meta["realsense_timestamp_ms"],
                "realsense_frame_number": latest_meta["realsense_frame_number"],
                "depth_realsense_frame_number": latest_meta["depth_realsense_frame_number"],
                "rgb_path": str(rgb_path.relative_to(output_dir)),
                "depth_path": str(depth_path.relative_to(output_dir)),
                "base_T_ee": {"matrix": jsonable_matrix(base_T_ee)},
                "base_T_gripper": {"matrix": jsonable_matrix(base_T_ee)},
                "stationary_capture": {
                    "settle_sec": args.settle_sec,
                    "translation_threshold_mm": args.stationary_translation_mm,
                    "rotation_threshold_deg": args.stationary_rotation_deg,
                    "settle_reference_pose": {"matrix": jsonable_matrix(settle_reference_pose)},
                    "settle_final_pose": {"matrix": jsonable_matrix(settle_final_pose)},
                    "post_frame_pose_delta_mm": final_translation_mm,
                    "post_frame_pose_delta_deg": final_rotation_deg,
                },
            }
            if gripper_T_camera is not None:
                base_T_camera = base_T_ee @ gripper_T_camera
                entry["base_T_camera"] = {"matrix": jsonable_matrix(base_T_camera)}
                entry["camera_T_base"] = {"matrix": jsonable_matrix(np.linalg.inv(base_T_camera))}
                if camera_T_gripper is not None:
                    entry["camera_T_gripper"] = {"matrix": jsonable_matrix(camera_T_gripper)}

            pose_log_file.write(json.dumps(entry) + "\n")
            pose_log_file.flush()
            print(f"Saved snapshot {snapshot_index}: {rgb_path.name}, {depth_path.name}")
            snapshot_index += 1
            print(prompt, end="", flush=True)
    except Exception as exc:
        raise SystemExit(f"Snapshot collection failed: {exc}") from exc
    finally:
        if pose_log_file is not None:
            pose_log_file.close()
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
        "rgb_dir": str(rgb_dir.resolve()),
        "depth_dir": str(depth_dir.resolve()),
        "preview_dir": str(preview_dir.resolve()) if args.preview else None,
        "pose_log_path": str(pose_log_path.resolve()),
        "requested_width": args.width,
        "requested_height": args.height,
        "requested_fps": args.fps,
        "stream_width": active_mode[0] if active_mode is not None else None,
        "stream_height": active_mode[1] if active_mode is not None else None,
        "stream_fps": active_mode[2] if active_mode is not None else None,
        "color_intrinsics": color_intrinsics,
        "depth_intrinsics": depth_intrinsics,
        "depth_scale_m_per_unit": depth_scale,
        "frames_recorded": snapshot_index,
        "run_start_wall_sec": run_start_wall_sec,
        "run_end_wall_sec": time.time(),
        "run_duration_sec": time.monotonic() - run_start_monotonic_sec,
        "gripper_T_camera": {"matrix": jsonable_matrix(gripper_T_camera)} if gripper_T_camera is not None else None,
        "camera_T_gripper": {"matrix": jsonable_matrix(camera_T_gripper)} if camera_T_gripper is not None else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Finished. Saved {snapshot_index} snapshots.")
    print(f"RGB images:   {rgb_dir}")
    print(f"Depth images: {depth_dir}")
    print(f"Pose log:     {pose_log_path}")
    print(f"Metadata:     {metadata_path}")
    if preview_file_writer is not None:
        print(f"Latest preview images: {preview_dir}")


if __name__ == "__main__":
    main()
