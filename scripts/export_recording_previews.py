#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create viewer-friendly artifacts from a recorded trajectory directory: "
            "8-bit colorized depth previews and an H.264 MP4 derived from trajectory.mp4."
        )
    )
    parser.add_argument(
        "recording_dir",
        type=Path,
        help="Path to a run directory such as runs/traj_001.",
    )
    parser.add_argument(
        "--depth-dir-name",
        default="depth",
        help="Raw depth image directory inside the recording directory.",
    )
    parser.add_argument(
        "--depth-preview-dir-name",
        default="depth_preview",
        help="Output directory for colorized 8-bit depth previews.",
    )
    parser.add_argument(
        "--video-name",
        default="trajectory.mp4",
        help="Input video filename inside the recording directory.",
    )
    parser.add_argument(
        "--video-preview-name",
        default="trajectory_h264.mp4",
        help="Output H.264 video filename inside the recording directory.",
    )
    return parser.parse_args()


def build_depth_preview(depth_image: np.ndarray) -> np.ndarray:
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


def export_depth_previews(depth_dir: Path, preview_dir: Path) -> int:
    preview_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for depth_path in sorted(depth_dir.glob("*.png")):
        depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            raise RuntimeError(f"Failed to read depth image: {depth_path}")
        if depth_image.ndim != 2 or depth_image.dtype != np.uint16:
            raise RuntimeError(f"Expected 16-bit single-channel depth PNG: {depth_path}")

        preview = build_depth_preview(depth_image)
        preview_path = preview_dir / depth_path.name
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"Failed to write depth preview: {preview_path}")
        count += 1
    return count


def candidate_ffmpeg_paths() -> list[str]:
    candidates: list[str] = []
    for path in ["/usr/bin/ffmpeg", shutil.which("ffmpeg")]:
        if path and path not in candidates:
            candidates.append(path)
    return candidates


def export_compatible_video(video_path: Path, output_path: Path) -> tuple[bool, str]:
    ffmpeg_paths = candidate_ffmpeg_paths()
    if not ffmpeg_paths:
        return False, "ffmpeg not found"

    encoder_options = [
        ["-c:v", "libx264", "-pix_fmt", "yuv420p"],
        ["-c:v", "libopenh264", "-pix_fmt", "yuv420p"],
        ["-c:v", "mpeg4"],
    ]
    errors: list[str] = []

    for ffmpeg in ffmpeg_paths:
        for encoder_args in encoder_options:
            command = [ffmpeg, "-y", "-i", str(video_path), *encoder_args, str(output_path)]
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode == 0:
                return True, f"{Path(ffmpeg).name} with {' '.join(encoder_args)}"
            errors.append(
                f"{ffmpeg} {' '.join(encoder_args)} -> exit {result.returncode}: "
                f"{result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'unknown error'}"
            )

    return False, " | ".join(errors)


def main() -> None:
    args = parse_args()
    recording_dir = args.recording_dir.resolve()
    depth_dir = recording_dir / args.depth_dir_name
    preview_dir = recording_dir / args.depth_preview_dir_name
    video_path = recording_dir / args.video_name
    video_preview_path = recording_dir / args.video_preview_name

    if not depth_dir.is_dir():
        raise SystemExit(f"Depth directory not found: {depth_dir}")
    if not video_path.is_file():
        raise SystemExit(f"Video file not found: {video_path}")

    depth_count = export_depth_previews(depth_dir, preview_dir)
    print(f"Wrote {depth_count} depth previews to {preview_dir}")

    success, detail = export_compatible_video(video_path, video_preview_path)
    if success:
        print(f"Wrote compatible video to {video_preview_path} using {detail}")
    else:
        print(f"Skipped compatible video export: {detail}")


if __name__ == "__main__":
    main()
