#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pyrealsense2 as rs
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one RGB frame from an Intel RealSense camera."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("realsense_test.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Color stream width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Color stream height.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Color stream frame rate.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Number of frames to discard before saving one image.",
    )
    parser.add_argument(
        "--frame-timeout-ms",
        type=int,
        default=5000,
        help="Timeout for each wait_for_frames call in milliseconds.",
    )
    return parser.parse_args()


def safe_camera_info(device_or_sensor: Any, info: Any, fallback: str) -> str:
    try:
        return device_or_sensor.get_info(info)
    except Exception:
        return fallback


def print_realsense_diagnostics() -> None:
    print("RealSense diagnostics")
    try:
        context = rs.context()
        devices = list(context.query_devices())
    except Exception as exc:
        print(f"  Failed to create RealSense context: {exc}")
        print("  This usually means the OS cannot expose the camera to librealsense.")
        return

    print(f"  Detected devices: {len(devices)}")
    if not devices:
        print("  No RealSense device detected.")
        print("  Checks:")
        print("    - Reconnect the camera and use a direct USB 3 port.")
        print("    - Close realsense-viewer, ROS nodes, and other camera users.")
        print("    - Confirm the OS sees the USB device.")
        return

    for index, device in enumerate(devices):
        name = safe_camera_info(device, rs.camera_info.name, "<unknown>")
        serial = safe_camera_info(device, rs.camera_info.serial_number, "<unknown>")
        product_line = safe_camera_info(device, rs.camera_info.product_line, "<unknown>")
        print(f"  Device {index}: {name} | serial={serial} | product_line={product_line}")
        try:
            sensors = list(device.query_sensors())
        except Exception as exc:
            print(f"    Failed to query sensors: {exc}")
            continue
        print(f"    Sensors: {len(sensors)}")
        for sensor_index, sensor in enumerate(sensors):
            sensor_name = safe_camera_info(sensor, rs.camera_info.name, f"<sensor {sensor_index}>")
            print(f"      - {sensor_name}")


def wait_for_color_frame(
    pipeline: rs.pipeline,
    timeout_ms: int,
    frame_index: int,
) -> rs.composite_frame:
    try:
        return pipeline.wait_for_frames(timeout_ms=timeout_ms)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Timed out waiting for frame {frame_index} after {timeout_ms} ms.\n"
            "RealSense checks:\n"
            "  - Confirm the camera is connected and visible to librealsense.\n"
            "  - Try a direct USB 3 port instead of a hub.\n"
            "  - Close realsense-viewer, ROS, and other camera users.\n"
            "  - Try a lower stream load such as --width 424 --height 240 --fps 15.\n"
            "  - If this machine is remote/containerized, confirm USB device passthrough.\n"
            f"Original error: {exc}"
        ) from exc


def main() -> None:
    args = parse_args()

    print_realsense_diagnostics()
    print(
        "Requested stream: "
        f"color {args.width}x{args.height} @ {args.fps} FPS, "
        f"warmup_frames={args.warmup_frames}, timeout_ms={args.frame_timeout_ms}"
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to start the RealSense pipeline.\n"
            "Checks:\n"
            "  - Another process may already own the camera.\n"
            "  - The requested color stream may be unsupported by this device.\n"
            "  - The OS may not have proper USB/udev access to the camera.\n"
            f"Original error: {exc}"
        ) from exc

    active_device = profile.get_device()
    active_name = safe_camera_info(active_device, rs.camera_info.name, "<unknown>")
    active_serial = safe_camera_info(active_device, rs.camera_info.serial_number, "<unknown>")
    print(f"Streaming from: {active_name} | serial={active_serial}")

    try:
        for frame_index in range(1, args.warmup_frames + 1):
            wait_for_color_frame(pipeline, args.frame_timeout_ms, frame_index)

        frames = wait_for_color_frame(
            pipeline,
            args.frame_timeout_ms,
            args.warmup_frames + 1,
        )
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(
                "Pipeline returned frames, but no color frame was present.\n"
                "The device may not expose the requested color stream in the current mode."
            )

        image_bgr = np.asanyarray(color_frame.get_data())
        image_rgb = image_bgr[:, :, ::-1]

        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image_rgb).save(args.output)
        print(f"Saved {args.output}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
