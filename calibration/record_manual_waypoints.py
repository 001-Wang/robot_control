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

from franka_utils import connect_robot, exit_with_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively record Cartesian waypoints from the robot's current base_T_ee "
            "while you manually move the arm. The output JSON is compatible with "
            "scripts/record_waypoint_trajectory.py."
        )
    )
    parser.add_argument(
        "--robot-ip",
        default="192.168.1.11",
        help="Franka robot IP address.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/waypoints.manual.json"),
        help="Output JSON file for recorded waypoints.",
    )
    parser.add_argument(
        "--default-duration-sec",
        type=float,
        default=4.0,
        help="Default motion duration assigned to each recorded waypoint.",
    )
    parser.add_argument(
        "--default-hold-sec",
        type=float,
        default=1.0,
        help="Default hold duration assigned to each recorded waypoint.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a live RealSense preview window while recording waypoints.",
    )
    parser.add_argument("--width", type=int, default=640, help="Preview stream width.")
    parser.add_argument("--height", type=int, default=480, help="Preview stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Preview stream FPS.")
    return parser.parse_args()


def read_base_T_ee(robot: object) -> np.ndarray:
    state = robot.read_once()
    return np.array(state.O_T_EE, dtype=float).reshape(4, 4, order="F")


def quaternion_xyzw_from_matrix(rotation: np.ndarray) -> list[float]:
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
    return quaternion.tolist()


def waypoint_payload(
    *,
    label: str,
    duration_sec: float,
    hold_sec: float,
    base_T_ee: np.ndarray,
) -> dict[str, Any]:
    return {
        "label": label,
        "duration_sec": duration_sec,
        "hold_sec": hold_sec,
        "pose": {
            "translation": base_T_ee[:3, 3].astype(float).tolist(),
            "quaternion_xyzw": quaternion_xyzw_from_matrix(base_T_ee[:3, :3]),
            "matrix": base_T_ee.astype(float).tolist(),
        },
    }


def save_waypoints(path: Path, robot_ip: str, waypoints: list[dict[str, Any]]) -> None:
    payload = {
        "robot_ip": robot_ip,
        "waypoints": waypoints,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class ConsoleInputThread:
    def __init__(self) -> None:
        self.queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="waypoint-console-input", daemon=True)
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


class RealsensePreview:
    def __init__(self, width: int, height: int, fps: int) -> None:
        self._pipeline = rs.pipeline()
        self._align = rs.align(rs.stream.color)
        self._running = False
        self._window_name = "RealSense Preview"
        self._config = rs.config()
        self._config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    def start(self) -> None:
        self._pipeline.start(self._config)
        self._running = True
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

    def update(self) -> None:
        if not self._running:
            return
        frames = self._pipeline.poll_for_frames()
        if not frames:
            cv2.waitKey(1)
            return

        aligned_frames = self._align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            cv2.waitKey(1)
            return

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_preview = depth_image_to_preview(depth_image)
        combined = np.hstack([color_image, depth_preview])
        cv2.imshow(self._window_name, combined)
        cv2.waitKey(1)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self._pipeline.stop()
        except Exception:
            pass
        cv2.destroyWindow(self._window_name)


def main() -> None:
    args = parse_args()
    if args.default_duration_sec <= 0.0:
        raise ValueError("--default-duration-sec must be positive.")
    if args.default_hold_sec < 0.0:
        raise ValueError("--default-hold-sec must be non-negative.")

    try:
        robot = connect_robot(args.robot_ip)
    except RuntimeError as exc:
        exit_with_error(str(exc))

    preview = None
    if args.preview:
        preview = RealsensePreview(args.width, args.height, args.fps)
        preview.start()

    np.set_printoptions(precision=6, suppress=True)
    print("Interactive waypoint capture")
    print("Manually move the robot, then press Enter to record the current base_T_ee.")
    print("Commands:")
    print("  Enter: record with default timing")
    print("  <label>: record with a custom label")
    print("  <label> <duration_sec> <hold_sec>: record with custom timing")
    print("  q: save and quit")
    if args.preview:
        print("Live RealSense preview is open. Left: RGB, Right: depth preview.")
    print()

    waypoints: list[dict[str, Any]] = []
    waypoint_index = 0
    input_thread = ConsoleInputThread()
    print(f"[W{waypoint_index:03d}] Enter to record, custom label/timing, or q:")

    try:
        while True:
            if preview is not None:
                preview.update()
            try:
                raw = input_thread.queue.get(timeout=0.05)
            except queue.Empty:
                continue

            raw = raw.strip()
            if raw.lower() == "q":
                break

            label = f"wp_{waypoint_index:03d}"
            duration_sec = args.default_duration_sec
            hold_sec = args.default_hold_sec

            if raw:
                parts = raw.split()
                if len(parts) == 1:
                    label = parts[0]
                elif len(parts) == 3:
                    label = parts[0]
                    duration_sec = float(parts[1])
                    hold_sec = float(parts[2])
                else:
                    print("Expected: Enter, '<label>', or '<label> <duration_sec> <hold_sec>'")
                    print()
                    print(f"[W{waypoint_index:03d}] Enter to record, custom label/timing, or q:")
                    continue

            if duration_sec <= 0.0 or hold_sec < 0.0:
                print("duration_sec must be positive and hold_sec must be non-negative.")
                print()
                print(f"[W{waypoint_index:03d}] Enter to record, custom label/timing, or q:")
                continue

            try:
                base_T_ee = read_base_T_ee(robot)
            except RuntimeError as exc:
                if waypoints:
                    save_waypoints(args.output, args.robot_ip, waypoints)
                    print(f"Saved partial waypoint file to {args.output}")
                exit_with_error(f"Failed to read robot pose: {exc}")

            waypoint = waypoint_payload(
                label=label,
                duration_sec=duration_sec,
                hold_sec=hold_sec,
                base_T_ee=base_T_ee,
            )
            waypoints.append(waypoint)

            print(f"Recorded {label}: xyz={waypoint['pose']['translation']}")
            print(base_T_ee)
            print()
            waypoint_index += 1
            print(f"[W{waypoint_index:03d}] Enter to record, custom label/timing, or q:")
    finally:
        if preview is not None:
            preview.stop()

    if not waypoints:
        print("No waypoints recorded. Nothing saved.")
        return

    save_waypoints(args.output, args.robot_ip, waypoints)
    print(f"Saved {len(waypoints)} waypoints to {args.output}")


if __name__ == "__main__":
    main()
