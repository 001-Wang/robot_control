#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a recorded known-pose dataset into a Nerfstudio/Instant-NGP style "
            "transforms.json using camera-to-world matrices in an OpenGL-style camera frame."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Dataset directory containing run_metadata.json and frame_poses.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output transforms.json path. Defaults to <run-dir>/transforms_known_poses.json.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Root directory for relative file_path entries. Defaults to --run-dir. "
            "Paths in transforms.json are written relative to this root."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if not entries:
        raise ValueError(f"No entries found in {path}")
    return entries


def matrix_from_entry(entry: dict[str, Any], key: str) -> np.ndarray:
    matrix = np.asarray(entry[key]["matrix"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{key} must be 4x4, got {matrix.shape}")
    return matrix


def image_relative_path(image_root: Path, run_dir: Path, stored_path: str) -> str:
    path = Path(stored_path)
    if not path.is_absolute():
        path = (run_dir / path).resolve()
    relative = path.relative_to(image_root.resolve())
    return relative.as_posix()


def opencv_camera_to_opengl_camera() -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[1, 1] = -1.0
    transform[2, 2] = -1.0
    return transform


def export_payload(
    *,
    metadata: dict[str, Any],
    entries: list[dict[str, Any]],
    run_dir: Path,
    image_root: Path,
) -> dict[str, Any]:
    intr = metadata["color_intrinsics"]
    width = int(intr["width"])
    height = int(intr["height"])
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["ppx"])
    cy = float(intr["ppy"])
    coeffs = [float(value) for value in intr.get("coeffs", [])]

    camera_convert = opencv_camera_to_opengl_camera()
    frames: list[dict[str, Any]] = []
    for entry in entries:
        if "base_T_camera" in entry:
            world_T_camera_cv = matrix_from_entry(entry, "base_T_camera")
        elif "camera_T_base" in entry:
            world_T_camera_cv = np.linalg.inv(matrix_from_entry(entry, "camera_T_base"))
        else:
            raise ValueError("Each pose entry must contain base_T_camera or camera_T_base.")

        world_T_camera_gl = world_T_camera_cv @ camera_convert

        if "rgb_path" not in entry:
            raise ValueError("Each pose entry must contain rgb_path.")

        frame_payload: dict[str, Any] = {
            "file_path": image_relative_path(image_root, run_dir, entry["rgb_path"]),
            "transform_matrix": world_T_camera_gl.tolist(),
        }
        if "depth_path" in entry:
            frame_payload["depth_file_path"] = image_relative_path(image_root, run_dir, entry["depth_path"])
        if "realsense_timestamp_ms" in entry:
            frame_payload["realsense_timestamp_ms"] = float(entry["realsense_timestamp_ms"])
        if "frame_index" in entry:
            frame_payload["frame_index"] = int(entry["frame_index"])
        frames.append(frame_payload)

    camera_angle_x = 2.0 * np.arctan(width / (2.0 * fx))
    camera_angle_y = 2.0 * np.arctan(height / (2.0 * fy))
    payload: dict[str, Any] = {
        "camera_model": "OPENCV" if any(abs(value) > 1e-12 for value in coeffs) else "PINHOLE",
        "w": width,
        "h": height,
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "camera_angle_x": float(camera_angle_x),
        "camera_angle_y": float(camera_angle_y),
        "k1": coeffs[0] if len(coeffs) > 0 else 0.0,
        "k2": coeffs[1] if len(coeffs) > 1 else 0.0,
        "p1": coeffs[2] if len(coeffs) > 2 else 0.0,
        "p2": coeffs[3] if len(coeffs) > 3 else 0.0,
        "frames": frames,
        "pose_convention": {
            "input": "camera optical frame from base_T_camera / camera_T_base",
            "output": "camera-to-world matrix with OpenGL-style camera axes (+x right, +y up, +z back)",
        },
    }
    return payload


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metadata_path = run_dir / "run_metadata.json"
    poses_path = run_dir / "frame_poses.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing pose log: {poses_path}")

    image_root = args.image_root.resolve() if args.image_root is not None else run_dir
    output_path = (
        args.output.resolve()
        if args.output is not None
        else run_dir / "transforms_known_poses.json"
    )

    metadata = load_json(metadata_path)
    entries = load_jsonl(poses_path)
    payload = export_payload(metadata=metadata, entries=entries, run_dir=run_dir, image_root=image_root)

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote transforms.json export to {output_path}")
    print(f"Frames: {len(payload['frames'])}")
    print(f"Image root: {image_root}")


if __name__ == "__main__":
    main()
