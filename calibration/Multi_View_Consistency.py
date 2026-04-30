#!/usr/bin/env python3
"""
Evaluate hand-eye / pose consistency by repeatedly observing the SAME fixed
point on the table from multiple robot viewpoints, then checking whether the
reconstructed base-frame 3D positions cluster tightly.

Workflow:
1. Move robot to a viewpoint and keep still.
2. Run this script and click the SAME physical point in the RGB image.
3. It reads aligned depth at the clicked pixel.
4. It back-projects the point into camera coordinates.
5. It transforms the point into base coordinates using:
       base_T_camera = base_T_gripper @ gripper_T_camera
6. It appends the base-frame point to a dataset JSON.
7. Repeat across many viewpoints.
8. Use --analyze to report scatter statistics.

What to conclude:
- Tight cluster (e.g. <= 1 cm std / radius): hand-eye + pose chain is likely OK.
- Several cm scatter: hand-eye / sync / TCP definition / depth usage has issues.

IMPORTANT:
- Use the SAME physical point every time.
- Prefer a high-contrast point with valid depth.
- Keep the robot still before capture.
- Use aligned depth to RGB.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception as exc:
    rs = None
    print(f"[WARN] pyrealsense2 import failed: {exc}")


# -----------------------------------------------------------------------------
# Replace with your actual Franka interface
# -----------------------------------------------------------------------------
class RobotInterface:
    def __init__(self, robot_ip: str):
        self.robot_ip = robot_ip
        self._connected = False

    def connect(self) -> None:
        print(f"[INFO] Placeholder robot connect to {self.robot_ip}")
        self._connected = True

    def read_base_T_gripper(self) -> np.ndarray:
        raise NotImplementedError(
            "Replace read_base_T_gripper() with your real Franka pose reader."
        )


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def load_matrix_from_json(path: Path, key: Optional[str] = None) -> np.ndarray:
    data = json.loads(path.read_text())
    if key is not None:
        data = data[key]
    if isinstance(data, dict) and "matrix" in data:
        data = data["matrix"]
    mat = np.array(data, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix in {path}, got shape {mat.shape}")
    return mat


def make_homogeneous_point(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z, 1.0], dtype=np.float64)


def backproject_pixel_to_camera(
    u: int,
    v: int,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return make_homogeneous_point(x, y, z)


def get_robust_depth(depth_m: np.ndarray, u: int, v: int, radius: int = 2) -> float:
    h, w = depth_m.shape
    u0 = max(0, u - radius)
    u1 = min(w, u + radius + 1)
    v0 = max(0, v - radius)
    v1 = min(h, v + radius + 1)
    patch = depth_m[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 5.0)]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def rotation_angle_deg(R: np.ndarray) -> float:
    trace_val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace_val)))


# -----------------------------------------------------------------------------
# RealSense
# -----------------------------------------------------------------------------
@dataclass
class FrameData:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float


def start_realsense_pipeline(width: int, height: int, fps: int):
    if rs is None:
        raise RuntimeError("pyrealsense2 is not available.")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    for _ in range(15):
        pipeline.wait_for_frames()

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()

    intrinsics = {
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "depth_scale": float(depth_scale),
    }
    return pipeline, align, intrinsics


def capture_aligned_rgbd(pipeline, align, intrinsics) -> FrameData:
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        raise RuntimeError("Failed to get aligned color/depth frames.")

    color = np.asanyarray(color_frame.get_data())
    depth_raw = np.asanyarray(depth_frame.get_data())
    depth_m = depth_raw.astype(np.float32) * intrinsics["depth_scale"]

    return FrameData(
        color_bgr=color,
        depth_m=depth_m,
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["cx"],
        cy=intrinsics["cy"],
    )


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
class ClickSelector:
    def __init__(self, image_bgr: np.ndarray):
        self.image_bgr = image_bgr.copy()
        self.clicked: Optional[Tuple[int, int]] = None
        self.window_name = "Click SAME fixed physical point (q/ESC to quit)"

    def _callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked = (x, y)
            vis = self.image_bgr.copy()
            cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                vis,
                f"({x}, {y})",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(self.window_name, vis)

    def run(self) -> Optional[Tuple[int, int]]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, self.image_bgr)
        cv2.setMouseCallback(self.window_name, self._callback)

        while True:
            key = cv2.waitKey(20) & 0xFF
            if self.clicked is not None:
                break
            if key in (27, ord("q")):
                break

        cv2.destroyWindow(self.window_name)
        return self.clicked


# -----------------------------------------------------------------------------
# Dataset IO
# -----------------------------------------------------------------------------
def load_dataset(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "version": 1,
        "records": [],
    }


def save_dataset(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))
    print(f"[INFO] Saved dataset: {path}")


def append_record(dataset: dict, record: dict) -> None:
    dataset.setdefault("records", []).append(record)


# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
def analyze_records(records: list[dict]) -> int:
    if len(records) < 2:
        print("[ERROR] Need at least 2 records to analyze.")
        return 1

    pts = np.array([r["point_base_xyz_m"] for r in records], dtype=np.float64)
    centroid = pts.mean(axis=0)
    diffs = pts - centroid[None, :]
    radial = np.linalg.norm(diffs, axis=1)

    std_xyz = pts.std(axis=0)
    rms_xyz = np.sqrt(np.mean(diffs**2, axis=0))
    mean_radius = float(np.mean(radial))
    median_radius = float(np.median(radial))
    max_radius = float(np.max(radial))
    p90_radius = float(np.percentile(radial, 90))

    print("=" * 80)
    print("Fixed-point multi-view consistency analysis")
    print("=" * 80)
    print(f"Number of samples: {len(records)}")
    print(f"Centroid [m]: {centroid.tolist()}")
    print()
    print(f"Std XYZ [m]: {std_xyz.tolist()}")
    print(f"RMS XYZ [m]: {rms_xyz.tolist()}")
    print()
    print(f"Mean radius from centroid [m]:   {mean_radius:.6f}")
    print(f"Median radius from centroid [m]: {median_radius:.6f}")
    print(f"P90 radius from centroid [m]:    {p90_radius:.6f}")
    print(f"Max radius from centroid [m]:    {max_radius:.6f}")
    print()

    print("Per-sample error from centroid:")
    for i, (rec, r) in enumerate(zip(records, radial)):
        print(
            f"  [{i:02d}] label={rec.get('label', ''):>10s} "
            f"pixel={tuple(rec['pixel_uv'])} depth={rec['depth_m']:.4f} m "
            f"radius={r:.6f} m"
        )

    print()
    print("Rule of thumb:")
    print("  <= 0.01 m : very good")
    print("  0.01-0.02 : usable but watch closely")
    print("  0.02-0.05 : suspicious")
    print("  > 0.05 m  : likely calibration / sync / frame-definition issue")
    print("=" * 80)
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--robot-ip", type=str, default=None, help="Robot IP address")
    p.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Path to JSON containing gripper_T_camera as 4x4 matrix, or top-level matrix JSON.",
    )
    p.add_argument(
        "--calibration-key",
        type=str,
        default=None,
        help="Optional JSON key for the 4x4 gripper_T_camera matrix.",
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("fixed_point_consistency.json"),
        help="Path to dataset JSON for appended samples and analysis.",
    )
    p.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional label for this sample, e.g. view_01.",
    )
    p.add_argument(
        "--analyze",
        action="store_true",
        help="Only analyze the dataset JSON; do not capture a new sample.",
    )
    p.add_argument(
        "--use-manual-base-T-gripper",
        type=Path,
        default=None,
        help="Optional JSON file containing a 4x4 base_T_gripper for offline testing.",
    )
    return p


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    args = build_argparser().parse_args()

    dataset = load_dataset(args.dataset)
    if args.analyze:
        return analyze_records(dataset.get("records", []))

    gripper_T_camera = load_matrix_from_json(args.calibration, args.calibration_key)
    print("[INFO] Loaded gripper_T_camera:")
    print(np.array2string(gripper_T_camera, precision=6, suppress_small=True))

    pipeline, align, intrinsics = start_realsense_pipeline(args.width, args.height, args.fps)
    robot: Optional[RobotInterface] = None

    try:
        frame = capture_aligned_rgbd(pipeline, align, intrinsics)
        selector = ClickSelector(frame.color_bgr)
        clicked = selector.run()
        if clicked is None:
            print("[INFO] No point selected. Exiting.")
            return 0

        u, v = clicked
        depth_m = get_robust_depth(frame.depth_m, u, v, radius=2)
        if not np.isfinite(depth_m):
            print("[ERROR] No valid depth near clicked pixel.")
            return 1

        P_camera = backproject_pixel_to_camera(
            u=u,
            v=v,
            depth_m=depth_m,
            fx=frame.fx,
            fy=frame.fy,
            cx=frame.cx,
            cy=frame.cy,
        )

        if args.use_manual_base_T_gripper is not None:
            base_T_gripper = load_matrix_from_json(args.use_manual_base_T_gripper)
            print("[INFO] Using offline base_T_gripper from file.")
        else:
            if args.robot_ip is None:
                print("[ERROR] Need either --robot-ip or --use-manual-base-T-gripper")
                return 1
            robot = RobotInterface(args.robot_ip)
            robot.connect()
            base_T_gripper = robot.read_base_T_gripper()

        base_T_camera = base_T_gripper @ gripper_T_camera
        P_base = base_T_camera @ P_camera
        xyz_base = P_base[:3]

        record = {
            "label": args.label,
            "pixel_uv": [int(u), int(v)],
            "depth_m": float(depth_m),
            "point_camera_xyz_m": [float(x) for x in P_camera[:3]],
            "point_base_xyz_m": [float(x) for x in xyz_base],
            "base_T_gripper": base_T_gripper.tolist(),
            "gripper_T_camera": gripper_T_camera.tolist(),
            "intrinsics": {
                "fx": frame.fx,
                "fy": frame.fy,
                "cx": frame.cx,
                "cy": frame.cy,
            },
        }
        append_record(dataset, record)
        save_dataset(args.dataset, dataset)

        np.set_printoptions(precision=6, suppress=True)
        print()
        print("[RESULT] Sample appended")
        print(f"  label:      {args.label}")
        print(f"  pixel:      {(u, v)}")
        print(f"  depth [m]:  {depth_m:.6f}")
        print(f"  P_camera:   {P_camera[:3].tolist()}")
        print(f"  P_base [m]: {xyz_base.tolist()}")
        print(f"  total samples in dataset: {len(dataset['records'])}")

        if len(dataset["records"]) >= 2:
            print()
            print("[INFO] Running immediate analysis...")
            return analyze_records(dataset["records"])

        return 0

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
