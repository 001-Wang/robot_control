#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image
import pyrealsense2 as rs

from franka_utils import connect_robot, exit_with_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect eye-in-hand calibration samples: one RealSense RGB image plus one "
            "synchronized base_T_gripper pose per capture."
        )
    )
    parser.add_argument("--robot-ip", help="Franka robot IP address.")
    parser.add_argument(
        "--pose-file",
        type=Path,
        help=(
            "JSON file containing the current base_T_gripper pose. "
            "Supported formats: {'matrix': [[...]]}, "
            "{'base_T_gripper': {'matrix': [[...]]}}, or "
            "{'translation': [...], 'quaternion_xyzw': [...]}."
        ),
    )
    parser.add_argument(
        "--manual-pose",
        action="store_true",
        help="Prompt for a pose at each capture instead of reading it from the robot.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Output dataset JSON path. Created if it does not exist.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory where captured RGB images are stored.",
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
        "--warmup-frames",
        type=int,
        default=30,
        help="Frames to discard before the first capture.",
    )
    parser.add_argument(
        "--frame-timeout-ms",
        type=int,
        default=5000,
        help="Timeout for each RealSense frame wait in milliseconds.",
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
        "--prefix",
        default="view",
        help="Filename prefix for saved images and sample ids.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a live preview window and capture with keyboard input.",
    )
    args = parser.parse_args()

    pose_sources = [bool(args.robot_ip), bool(args.pose_file), bool(args.manual_pose)]
    if sum(pose_sources) != 1:
        parser.error("Provide exactly one pose source: --robot-ip, --pose-file, or --manual-pose.")
    return args


def default_dataset() -> dict[str, Any]:
    return {
        "handeye_method": "tsai",
        "camera": {
            "camera_matrix": [
                [604.4125366210938, 0.0, 323.9523620605469],
                [0.0, 603.8748779296875, 243.05894470214844],
                [0.0, 0.0, 1.0],
            ],
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "target": {
            "type": "charuco",
            "dictionary": "4X4_50",
            "squares_x": 7,
            "squares_y": 10,
            "square_length_m": 0.025,
            "marker_length_m": 0.018,
        },
        "samples": [],
    }


def load_or_create_dataset(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return default_dataset()


def save_dataset(path: Path, dataset: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, indent=2))


def next_index(samples: list[dict[str, Any]], prefix: str) -> int:
    indices: list[int] = []
    for sample in samples:
        sample_id = sample.get("id", "")
        if sample_id.startswith(f"{prefix}_"):
            suffix = sample_id[len(prefix) + 1 :]
            if suffix.isdigit():
                indices.append(int(suffix))
    return max(indices, default=-1) + 1


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


def print_realsense_diagnostics() -> None:
    print("RealSense diagnostics")
    try:
        context = rs.context()
        devices = list(context.query_devices())
    except Exception as exc:
        print(f"  Failed to create RealSense context: {exc}")
        return

    print(f"  Detected devices: {len(devices)}")
    for index, device in enumerate(devices):
        name = safe_camera_info(device, rs.camera_info.name, "<unknown>")
        serial = safe_camera_info(device, rs.camera_info.serial_number, "<unknown>")
        print(f"  Device {index}: {name} | serial={serial}")
        try:
            sensors = list(device.query_sensors())
        except Exception as exc:
            print(f"    Failed to query sensors: {exc}")
            continue
        for sensor in sensors:
            sensor_name = safe_camera_info(sensor, rs.camera_info.name, "<unknown>")
            print(f"    - {sensor_name}")


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


def capture_rgb_image(pipeline: rs.pipeline, timeout_ms: int) -> np.ndarray:
    frames = wait_for_frames(pipeline, timeout_ms, 1)
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError("No RGB frame received from RealSense.")
    return np.asanyarray(color_frame.get_data())


def save_rgb_image(image_bgr: np.ndarray, image_path: Path) -> None:
    image_rgb = image_bgr[:, :, ::-1]
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(image_path)


def read_base_T_ee(robot: Any) -> list[list[float]]:
    state = robot.read_once()
    return np.array(state.O_T_EE, dtype=float).reshape(4, 4, order="F").tolist()


def quaternion_xyzw_to_matrix(quaternion_xyzw: list[float]) -> np.ndarray:
    if len(quaternion_xyzw) != 4:
        raise ValueError("Quaternion must have exactly 4 values: qx qy qz qw.")
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


def validate_transform_matrix(matrix: Any) -> list[list[float]]:
    array = np.asarray(matrix, dtype=float)
    if array.shape != (4, 4):
        raise ValueError("Pose matrix must have shape 4x4.")
    return array.tolist()


def parse_pose_payload(payload: Any) -> list[list[float]]:
    if isinstance(payload, dict):
        if "base_T_gripper" in payload:
            return parse_pose_payload(payload["base_T_gripper"])
        if "matrix" in payload:
            return validate_transform_matrix(payload["matrix"])
        if "translation" in payload and "quaternion_xyzw" in payload:
            translation = np.asarray(payload["translation"], dtype=float)
            if translation.shape != (3,):
                raise ValueError("Translation must have exactly 3 values.")
            rotation = quaternion_xyzw_to_matrix(payload["quaternion_xyzw"])
            transform = np.eye(4, dtype=float)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation
            return transform.tolist()
    raise ValueError(
        "Pose JSON must contain either 'matrix', "
        "'base_T_gripper': {'matrix': ...}, or translation + quaternion_xyzw."
    )


def read_pose_file(path: Path) -> list[list[float]]:
    return parse_pose_payload(json.loads(path.read_text()))


def prompt_manual_pose() -> list[list[float]]:
    print("Enter pose as either 16 row-major matrix values or 7 values: tx ty tz qx qy qz qw.")
    while True:
        raw = input("pose> ").strip()
        values = raw.replace(",", " ").split()
        if len(values) == 16:
            matrix = np.asarray([float(value) for value in values], dtype=float).reshape(4, 4)
            return validate_transform_matrix(matrix)
        if len(values) == 7:
            translation = [float(value) for value in values[:3]]
            quaternion_xyzw = [float(value) for value in values[3:]]
            return parse_pose_payload(
                {"translation": translation, "quaternion_xyzw": quaternion_xyzw}
            )
        print("Expected 16 matrix values or 7 pose values. Try again.")


def capture_sample(
    *,
    dataset: dict[str, Any],
    pipeline: rs.pipeline,
    read_pose: Callable[[], list[list[float]]],
    image_dir: Path,
    dataset_path: Path,
    prefix: str,
    sample_index: int,
    frame_timeout_ms: int,
) -> int:
    sample_id = f"{prefix}_{sample_index:03d}"
    image_filename = f"{sample_id}.png"
    image_path = image_dir / image_filename

    image_bgr = capture_rgb_image(pipeline, frame_timeout_ms)
    base_T_gripper = read_pose()
    save_rgb_image(image_bgr, image_path)

    sample = {
        "id": sample_id,
        "image": str(image_path.resolve().relative_to(dataset_path.resolve().parent)),
        "base_T_gripper": {"matrix": base_T_gripper},
    }
    dataset["samples"].append(sample)
    save_dataset(dataset_path, dataset)

    print(f"Captured {sample_id}")
    print(f"  image: {image_path}")
    print("  pose : saved as base_T_gripper.matrix")
    return sample_index + 1


def preview_loop_opencv(
    *,
    dataset: dict[str, Any],
    pipeline: rs.pipeline,
    read_pose: Callable[[], list[list[float]]],
    image_dir: Path,
    dataset_path: Path,
    prefix: str,
    sample_index: int,
    frame_timeout_ms: int,
) -> None:
    print("Preview mode")
    print("Controls: c or space=capture, q=quit")

    while True:
        frames = wait_for_frames(pipeline, frame_timeout_ms, 1)
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image_bgr = np.asanyarray(color_frame.get_data())
        overlay = image_bgr.copy()
        cv2.putText(
            overlay,
            f"samples: {len(dataset['samples'])}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "c/space: capture   q: quit",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Hand-Eye Collector", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in {ord("c"), 32}:
            sample_index = capture_sample(
                dataset=dataset,
                pipeline=pipeline,
                read_pose=read_pose,
                image_dir=image_dir,
                dataset_path=dataset_path,
                prefix=prefix,
                sample_index=sample_index,
                frame_timeout_ms=frame_timeout_ms,
            )

    cv2.destroyAllWindows()


def preview_loop_matplotlib(
    *,
    dataset: dict[str, Any],
    pipeline: rs.pipeline,
    read_pose: Callable[[], list[list[float]]],
    image_dir: Path,
    dataset_path: Path,
    prefix: str,
    sample_index: int,
    frame_timeout_ms: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "OpenCV preview is unavailable, and Matplotlib could not be imported for the "
            f"fallback preview path. Original Matplotlib error: {exc}"
        ) from exc

    print("OpenCV GUI unavailable; using Matplotlib preview.")
    print("Press c in the figure window to capture, q to quit.")

    plt.ion()
    figure, axis = plt.subplots()
    axis.set_title("Hand-Eye Collector")
    state = {"capture": False, "quit": False}

    def on_key(event: Any) -> None:
        if event.key == "c":
            state["capture"] = True
        elif event.key == "q":
            state["quit"] = True

    figure.canvas.mpl_connect("key_press_event", on_key)
    artist = None

    while not state["quit"]:
        frames = wait_for_frames(pipeline, frame_timeout_ms, 1)
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image_bgr = np.asanyarray(color_frame.get_data())
        image_rgb = image_bgr[:, :, ::-1]
        if artist is None:
            artist = axis.imshow(image_rgb)
            axis.axis("off")
        else:
            artist.set_data(image_rgb)
        axis.set_xlabel(f"samples: {len(dataset['samples'])} | c=capture q=quit")
        figure.canvas.draw_idle()
        plt.pause(0.001)

        if state["capture"]:
            sample_index = capture_sample(
                dataset=dataset,
                pipeline=pipeline,
                read_pose=read_pose,
                image_dir=image_dir,
                dataset_path=dataset_path,
                prefix=prefix,
                sample_index=sample_index,
                frame_timeout_ms=frame_timeout_ms,
            )
            state["capture"] = False

    plt.close(figure)


def preview_loop(**kwargs: Any) -> None:
    try:
        preview_loop_opencv(**kwargs)
    except cv2.error:
        preview_loop_matplotlib(**kwargs)


def main() -> None:
    args = parse_args()
    dataset = load_or_create_dataset(args.dataset)
    args.image_dir.mkdir(parents=True, exist_ok=True)

    if args.robot_ip:
        try:
            robot = connect_robot(args.robot_ip)
        except RuntimeError as exc:
            exit_with_error(str(exc))
        read_pose = lambda: read_base_T_ee(robot)
    elif args.pose_file:
        read_pose = lambda: read_pose_file(args.pose_file)
    else:
        read_pose = prompt_manual_pose

    print_realsense_diagnostics()
    print(
        "Requested stream: "
        f"color {args.width}x{args.height} @ {args.fps} FPS, "
        f"warmup_frames={args.warmup_frames}, timeout_ms={args.frame_timeout_ms}"
    )

    pipeline = rs.pipeline()
    try:
        profile, active_mode = start_pipeline_with_mode(
            pipeline=pipeline,
            width=args.width,
            height=args.height,
            fps=args.fps,
            allow_stream_fallback=args.allow_stream_fallback,
        )
    except RuntimeError as exc:
        exit_with_error(
            "Failed to start the RealSense pipeline.\n"
            "Checks:\n"
            "  - Another process may already own the camera.\n"
            "  - The requested color stream may be unsupported by this device.\n"
            "  - The OS may not have proper access to the camera.\n"
            f"Original error: {exc}"
        )

    active_device = profile.get_device()
    active_name = safe_camera_info(active_device, rs.camera_info.name, "<unknown>")
    active_serial = safe_camera_info(active_device, rs.camera_info.serial_number, "<unknown>")
    print(
        f"Streaming from: {active_name} | serial={active_serial} | "
        f"mode={active_mode[0]}x{active_mode[1]} @ {active_mode[2]} FPS"
    )

    try:
        warmup_camera(
            pipeline=pipeline,
            warmup_frames=args.warmup_frames,
            timeout_ms=args.frame_timeout_ms,
            startup_grace_sec=args.startup_grace_sec,
            warmup_retries=args.warmup_retries,
        )

        sample_index = next_index(dataset["samples"], args.prefix)
        if args.preview:
            preview_loop(
                dataset=dataset,
                pipeline=pipeline,
                read_pose=read_pose,
                image_dir=args.image_dir,
                dataset_path=args.dataset,
                prefix=args.prefix,
                sample_index=sample_index,
                frame_timeout_ms=args.frame_timeout_ms,
            )
        else:
            print("Manual capture mode")
            print("Press Enter to capture one sample.")
            print("Type q then Enter to quit.")
            while True:
                command = input("> ").strip().lower()
                if command in {"q", "quit", "exit"}:
                    break
                if command not in {"", "c", "capture"}:
                    print("Press Enter to capture, or type q to quit.")
                    continue
                sample_index = capture_sample(
                    dataset=dataset,
                    pipeline=pipeline,
                    read_pose=read_pose,
                    image_dir=args.image_dir,
                    dataset_path=args.dataset,
                    prefix=args.prefix,
                    sample_index=sample_index,
                    frame_timeout_ms=args.frame_timeout_ms,
                )
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
