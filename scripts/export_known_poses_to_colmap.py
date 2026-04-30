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
            "Export a recorded dataset with known camera poses into a COLMAP-style "
            "text sparse model containing cameras.txt, images.txt, and points3D.txt."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Dataset directory containing run_metadata.json and frame_poses.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output sparse model directory. Defaults to <run-dir>/colmap_known_poses/sparse/0.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Root directory COLMAP should treat as the image folder. "
            "Image names in images.txt are written relative to this root. "
            "Defaults to --run-dir."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_pose_entries(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    if not entries:
        raise ValueError(f"No pose entries found in {path}")
    return entries


def matrix_from_entry(entry: dict[str, Any], key: str) -> np.ndarray:
    matrix = np.asarray(entry[key]["matrix"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{key} must be 4x4, got {matrix.shape}")
    return matrix


def rotation_matrix_to_colmap_qvec(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
    qvec /= np.linalg.norm(qvec)
    return qvec


def camera_model_and_params(metadata: dict[str, Any]) -> tuple[str, list[float], int, int]:
    intr = metadata["color_intrinsics"]
    width = int(intr["width"])
    height = int(intr["height"])
    coeffs = [float(value) for value in intr.get("coeffs", [])]
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["ppx"])
    cy = float(intr["ppy"])

    if any(abs(value) > 1e-12 for value in coeffs):
        model = "OPENCV"
        params = [fx, fy, cx, cy, *coeffs[:4]]
    else:
        model = "PINHOLE"
        params = [fx, fy, cx, cy]
    return model, params, width, height


def relative_image_name(image_root: Path, absolute_or_relative: str, run_dir: Path) -> str:
    image_path = Path(absolute_or_relative)
    if not image_path.is_absolute():
        image_path = (run_dir / image_path).resolve()
    return str(image_path.relative_to(image_root.resolve()))


def write_cameras_txt(path: Path, metadata: dict[str, Any]) -> None:
    model, params, width, height = camera_model_and_params(metadata)
    params_str = " ".join(f"{value:.16g}" for value in params)
    content = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        "# Number of cameras: 1",
        f"1 {model} {width} {height} {params_str}",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def write_images_txt(path: Path, entries: list[dict[str, Any]], run_dir: Path, image_root: Path) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(entries)}, mean observations per image: 0",
    ]
    for image_id, entry in enumerate(entries, start=1):
        if "camera_T_base" in entry:
            world_to_camera = matrix_from_entry(entry, "camera_T_base")
        elif "base_T_camera" in entry:
            world_to_camera = np.linalg.inv(matrix_from_entry(entry, "base_T_camera"))
        else:
            raise ValueError("Each pose entry must contain camera_T_base or base_T_camera.")

        qvec = rotation_matrix_to_colmap_qvec(world_to_camera[:3, :3])
        tvec = world_to_camera[:3, 3]

        image_name_key = "rgb_path" if "rgb_path" in entry else "image_path"
        if image_name_key not in entry:
            raise ValueError("Each pose entry must contain rgb_path or image_path.")
        image_name = relative_image_name(image_root, entry[image_name_key], run_dir)

        lines.append(
            f"{image_id} "
            f"{qvec[0]:.16g} {qvec[1]:.16g} {qvec[2]:.16g} {qvec[3]:.16g} "
            f"{tvec[0]:.16g} {tvec[1]:.16g} {tvec[2]:.16g} "
            f"1 {image_name}"
        )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_points3d_txt(path: Path) -> None:
    content = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        "# Number of points: 0, mean track length: 0",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metadata_path = run_dir / "run_metadata.json"
    poses_path = run_dir / "frame_poses.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing pose log: {poses_path}")

    metadata = load_json(metadata_path)
    entries = load_pose_entries(poses_path)

    image_root = args.image_root.resolve() if args.image_root is not None else run_dir
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "colmap_known_poses" / "sparse" / "0"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    write_cameras_txt(output_dir / "cameras.txt", metadata)
    write_images_txt(output_dir / "images.txt", entries, run_dir, image_root)
    write_points3d_txt(output_dir / "points3D.txt")

    summary = {
        "run_dir": str(run_dir),
        "image_root": str(image_root),
        "output_dir": str(output_dir),
        "num_images": len(entries),
        "camera_model": camera_model_and_params(metadata)[0],
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote COLMAP-style known-pose model to {output_dir}")
    print(f"Images: {len(entries)}")
    print(f"Image root: {image_root}")


if __name__ == "__main__":
    main()
