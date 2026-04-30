#!/usr/bin/env python3
"""
Click a pixel in the current RGB image, turn it into a 3D target in the robot
base frame, and optionally move the Franka to a hover pose above it.

This is intended as a practical hand-eye calibration check:
1. Capture one aligned RGB/depth frame from RealSense.
2. Let the user click a pixel in the RGB image.
3. Convert the click into a camera-frame 3D target using either:
   - aligned depth at that pixel, or
   - ray / table-plane intersection in the robot base frame.
4. Transform the target into base coordinates using:
       base_T_camera = base_T_gripper @ gripper_T_camera
5. Optionally move the robot to a hover pose above the target while keeping the
   current EE orientation fixed.

Safety:
- Always start with --dry-run.
- Verify the reported base-frame point before enabling motion.
- Prefer --target-source plane when validating hand-eye against a known table.
- The script only moves to a hover pose; it does not descend automatically.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception as exc:
    rs = None
    print(f"[WARN] pyrealsense2 import failed: {exc}")

try:
    from pylibfranka import CartesianPose, ControllerMode
except Exception as exc:
    CartesianPose = None
    ControllerMode = None
    print(f"[WARN] pylibfranka import failed: {exc}")

from franka_utils import connect_robot, exit_with_error


def validate_transform_matrix(matrix: Any) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got shape {array.shape}.")
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
        dtype=np.float64,
    )


def transform_from_payload(payload: Any) -> np.ndarray:
    if isinstance(payload, dict):
        if "matrix" in payload:
            return validate_transform_matrix(payload["matrix"])
        if "translation" in payload and "quaternion_xyzw" in payload:
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = quaternion_xyzw_to_matrix(payload["quaternion_xyzw"])
            transform[:3, 3] = np.asarray(payload["translation"], dtype=np.float64).reshape(3)
            return transform
    return validate_transform_matrix(payload)


def resolve_nested_key(payload: Any, dotted_key: str) -> Any:
    current = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Key path '{dotted_key}' not found in JSON payload.")
        current = current[part]
    return current


def load_gripper_T_camera(path: Path, key: Optional[str] = None) -> np.ndarray:
    payload = json.loads(path.read_text())
    if key is not None:
        return transform_from_payload(resolve_nested_key(payload, key))
    if isinstance(payload, dict) and "gripper_T_camera" in payload:
        return transform_from_payload(payload["gripper_T_camera"])
    return transform_from_payload(payload)


def load_table_plane(path: Path) -> tuple[np.ndarray, float]:
    payload = json.loads(path.read_text())
    plane_payload = payload["plane"] if isinstance(payload, dict) and "plane" in payload else payload
    normal = np.asarray(plane_payload["normal"], dtype=np.float64).reshape(3)
    offset = float(plane_payload["offset"])
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0.0:
        raise ValueError("Plane normal must be non-zero.")
    return normal / normal_norm, offset / normal_norm


def save_waypoint_json(path: Path, xyz: np.ndarray, name: str = "clicked_target") -> None:
    payload = {
        "waypoints": [
            {
                "name": name,
                "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                "duration": 4.0,
                "hold": 1.0,
            }
        ]
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"[INFO] Wrote waypoint JSON to: {path}")


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
    return make_homogeneous_point(x, y, depth_m)


def camera_ray_direction(
    u: int,
    v: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def intersect_camera_ray_with_plane(
    *,
    base_T_camera: np.ndarray,
    u: int,
    v: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    plane_normal: np.ndarray,
    plane_offset: float,
) -> np.ndarray:
    ray_camera = camera_ray_direction(u, v, fx, fy, cx, cy)
    ray_base = base_T_camera[:3, :3] @ ray_camera
    origin_base = base_T_camera[:3, 3]
    denominator = float(plane_normal @ ray_base)
    if abs(denominator) < 1e-8:
        raise RuntimeError("Clicked camera ray is nearly parallel to the plane.")
    distance_along_ray = -float(plane_normal @ origin_base + plane_offset) / denominator
    if distance_along_ray <= 0.0:
        raise RuntimeError(
            "Plane intersection lies behind the camera. Check the hand-eye transform and plane."
        )
    return origin_base + distance_along_ray * ray_base


def pose_with_same_orientation_and_new_translation(base_T_current: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    target = base_T_current.copy()
    target[:3, 3] = np.asarray(xyz, dtype=np.float64).reshape(3)
    return target


def pose_with_fixed_gripper_orientation_and_camera_at_translation(
    base_T_gripper_current: np.ndarray,
    gripper_T_camera: np.ndarray,
    camera_xyz_in_base: np.ndarray,
) -> np.ndarray:
    target = base_T_gripper_current.copy()
    gripper_rotation = target[:3, :3]
    camera_offset_in_gripper = gripper_T_camera[:3, 3]
    target[:3, 3] = (
        np.asarray(camera_xyz_in_base, dtype=np.float64).reshape(3)
        - gripper_rotation @ camera_offset_in_gripper
    )
    return target


def matrix_to_pose_list(matrix: np.ndarray) -> list[float]:
    return np.asarray(matrix, dtype=np.float64).reshape(16, order="F").tolist()


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
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
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


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
    return alpha * alpha * alpha * (10.0 + alpha * (-15.0 + 6.0 * alpha))


def apply_default_collision_behavior(robot: object) -> None:
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


class FrankaRobotInterface:
    def __init__(self, robot_ip: str):
        if CartesianPose is None or ControllerMode is None:
            raise RuntimeError("pylibfranka is required for Franka motion.")
        self.robot_ip = robot_ip
        self.robot = None

    def connect(self) -> None:
        self.robot = connect_robot(self.robot_ip)
        apply_default_collision_behavior(self.robot)

    def read_base_T_gripper(self) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError("Robot is not connected.")
        state = self.robot.read_once()
        return np.array(state.O_T_EE, dtype=np.float64).reshape(4, 4, order="F")

    def move_to_pose(self, base_T_target: np.ndarray, duration: float = 4.0) -> None:
        if self.robot is None:
            raise RuntimeError("Robot is not connected.")
        if duration <= 0.0:
            raise ValueError("Motion duration must be positive.")

        try:
            active_control = self.robot.start_cartesian_pose_control(ControllerMode.JointImpedance)
        except Exception as exc:
            message = str(exc)
            if "cannot start at singular pose" in message:
                raise RuntimeError(
                    "Failed to start Cartesian pose control because the robot is near a singular pose."
                ) from exc
            raise

        robot_state, motion_dt = active_control.readOnce()
        start_pose = np.array(robot_state.O_T_EE, dtype=np.float64).reshape(4, 4, order="F")
        start_translation = start_pose[:3, 3].copy()
        start_quaternion = matrix_to_quaternion_xyzw(start_pose)
        target_pose = np.asarray(base_T_target, dtype=np.float64).copy()
        target_rotation = target_pose[:3, :3].copy()
        target_translation = target_pose[:3, 3].copy()
        target_quaternion = matrix_to_quaternion_xyzw(target_pose)

        elapsed = 0.0
        motion_finished = False
        while not motion_finished:
            dt = motion_dt.to_sec()
            elapsed += dt
            alpha = quintic_time_scaling(min(elapsed / duration, 1.0))
            commanded_pose = np.eye(4, dtype=np.float64)
            commanded_quaternion = slerp_xyzw(start_quaternion, target_quaternion, alpha)
            commanded_pose[:3, :3] = quaternion_xyzw_to_matrix(commanded_quaternion.tolist())
            commanded_pose[:3, 3] = (1.0 - alpha) * start_translation + alpha * target_translation
            command = CartesianPose(matrix_to_pose_list(commanded_pose))
            if elapsed >= duration:
                command.motion_finished = True
                motion_finished = True
            active_control.writeOnce(command)
            if not motion_finished:
                robot_state, motion_dt = active_control.readOnce()


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
    depth_scale = float(depth_sensor.get_depth_scale())
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()

    intrinsics = {
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "depth_scale": depth_scale,
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


class ClickSelector:
    def __init__(self, image_bgr: np.ndarray):
        self.image_bgr = image_bgr.copy()
        self.clicked: Optional[Tuple[int, int]] = None
        self.window_name = "Click target pixel (press q or ESC to quit)"

    def _callback(self, event, x, y, flags, param):
        del flags, param
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


def get_robust_depth(depth_m: np.ndarray, u: int, v: int, radius: int = 2) -> float:
    height, width = depth_m.shape
    u0 = max(0, u - radius)
    u1 = min(width, u + radius + 1)
    v0 = max(0, v - radius)
    v1 = min(height, v + radius + 1)
    patch = depth_m[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 5.0)]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def signed_distance_to_plane(point: np.ndarray, plane_normal: np.ndarray, plane_offset: float) -> float:
    return float(plane_normal @ point + plane_offset)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", type=str, default=None, help="Robot IP address.")
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Path to a 4x4 transform JSON or calibration JSON containing gripper_T_camera.",
    )
    parser.add_argument(
        "--calibration-key",
        type=str,
        default=None,
        help="Optional dotted key path to the transform, for example gripper_T_camera.matrix.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--target-source",
        choices=["depth", "plane"],
        default="depth",
        help=(
            "How to compute the motion target. "
            "'depth' uses aligned depth at the clicked pixel. "
            "'plane' uses only the clicked RGB pixel plus the calibrated camera pose "
            "and the known table plane. In 'plane' mode, depth is logged only as a check."
        ),
    )
    parser.add_argument(
        "--plane",
        type=Path,
        default=Path("table_plane.json"),
        help="Plane JSON used when --target-source plane.",
    )
    parser.add_argument(
        "--z-offset",
        type=float,
        default=0.10,
        help="How far above the clicked point to place the EE hover target, in meters.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not send motion to the robot. Only print and save the target.",
    )
    parser.add_argument(
        "--motion-duration",
        type=float,
        default=4.0,
        help="Hover motion duration in seconds.",
    )
    parser.add_argument(
        "--output-waypoint",
        type=Path,
        default=Path("clicked_waypoint.json"),
        help="Where to save the generated waypoint JSON.",
    )
    parser.add_argument(
        "--use-manual-base-T-gripper",
        type=Path,
        default=None,
        help="Optional JSON file containing a 4x4 base_T_gripper for offline testing.",
    )
    parser.add_argument(
        "--skip-motion-confirmation",
        action="store_true",
        help="Skip the interactive confirmation before moving the robot.",
    )
    parser.add_argument(
        "--hover-frame",
        choices=["gripper", "camera"],
        default="gripper",
        help=(
            "Which frame should be placed above the clicked point. "
            "'gripper' moves the EE origin there; 'camera' keeps the current "
            "gripper orientation but solves for a gripper pose whose camera "
            "origin lands above the clicked point."
        ),
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()

    try:
        gripper_T_camera = load_gripper_T_camera(args.calibration, args.calibration_key)
    except Exception as exc:
        exit_with_error(f"Failed to load calibration from {args.calibration}: {exc}")

    print("[INFO] Loaded gripper_T_camera:")
    print(np.array2string(gripper_T_camera, precision=6, suppress_small=True))

    plane_normal = None
    plane_offset = None
    if args.target_source == "plane":
        try:
            plane_normal, plane_offset = load_table_plane(args.plane)
        except Exception as exc:
            exit_with_error(f"Failed to load plane from {args.plane}: {exc}")
        print(f"[INFO] Loaded plane normal={plane_normal.tolist()} offset={plane_offset:.9f}")

    pipeline = None
    robot: Optional[FrankaRobotInterface] = None
    try:
        pipeline, align, intrinsics = start_realsense_pipeline(args.width, args.height, args.fps)
        print("[INFO] Color intrinsics:")
        print(json.dumps({k: intrinsics[k] for k in ["fx", "fy", "cx", "cy"]}, indent=2))

        frame = capture_aligned_rgbd(pipeline, align, intrinsics)

        selector = ClickSelector(frame.color_bgr)
        clicked = selector.run()
        if clicked is None:
            print("[INFO] No point selected. Exiting.")
            return 0

        u, v = clicked
        print(f"[INFO] Clicked pixel: u={u}, v={v}")

        if args.use_manual_base_T_gripper is not None:
            base_T_gripper = transform_from_payload(json.loads(args.use_manual_base_T_gripper.read_text()))
            print("[INFO] Using offline base_T_gripper from file:")
            print(np.array2string(base_T_gripper, precision=6, suppress_small=True))
        else:
            if args.robot_ip is None:
                exit_with_error("Need either --robot-ip or --use-manual-base-T-gripper.")
            robot = FrankaRobotInterface(args.robot_ip)
            robot.connect()
            base_T_gripper = robot.read_base_T_gripper()
            print("[INFO] Current base_T_gripper:")
            print(np.array2string(base_T_gripper, precision=6, suppress_small=True))

        base_T_camera = base_T_gripper @ gripper_T_camera
        print("[INFO] base_T_camera:")
        print(np.array2string(base_T_camera, precision=6, suppress_small=True))

        depth_m = get_robust_depth(frame.depth_m, u, v, radius=2)
        if np.isfinite(depth_m):
            point_camera_from_depth = backproject_pixel_to_camera(
                u=u,
                v=v,
                depth_m=depth_m,
                fx=frame.fx,
                fy=frame.fy,
                cx=frame.cx,
                cy=frame.cy,
            )
            point_base_from_depth = (base_T_camera @ point_camera_from_depth)[:3]
            print(f"[INFO] Robust depth: {depth_m:.4f} m")
            print(f"[INFO] Depth-based target in base frame [m]: {point_base_from_depth}")
        else:
            point_base_from_depth = None
            print("[WARN] No valid depth near clicked pixel.")

        if args.target_source == "plane":
            assert plane_normal is not None and plane_offset is not None
            print(
                "[INFO] Motion target source: plane-only "
                "(RGB click + calibrated camera pose + table plane)."
            )
            xyz_base = intersect_camera_ray_with_plane(
                base_T_camera=base_T_camera,
                u=u,
                v=v,
                fx=frame.fx,
                fy=frame.fy,
                cx=frame.cx,
                cy=frame.cy,
                plane_normal=plane_normal,
                plane_offset=plane_offset,
            )
            plane_distance = signed_distance_to_plane(xyz_base, plane_normal, plane_offset)
            print(f"[RESULT] Plane-intersection target in base frame [m]: {xyz_base}")
            print(f"[INFO] Plane residual at target: {plane_distance:.9f} m")
            if point_base_from_depth is not None:
                print("[INFO] Depth is used for logging only in plane mode; it does not affect motion.")
                print(
                    "[INFO] Depth-vs-plane delta [m]: "
                    f"{(point_base_from_depth - xyz_base).tolist()}"
                )
        else:
            print("[INFO] Motion target source: aligned depth.")
            if point_base_from_depth is None:
                exit_with_error("No valid depth near the clicked pixel, and --target-source depth was selected.")
            xyz_base = point_base_from_depth
            print(f"[RESULT] Depth target in base frame [m]: {xyz_base}")
            if plane_normal is not None and plane_offset is not None:
                print(
                    "[INFO] Signed depth-target distance to plane: "
                    f"{signed_distance_to_plane(xyz_base, plane_normal, plane_offset):.6f} m"
                )

        save_waypoint_json(args.output_waypoint, xyz_base, name="clicked_target")

        hover_xyz = xyz_base.copy()
        hover_xyz[2] += args.z_offset
        print(f"[INFO] Hover target [m]: {hover_xyz}")

        if args.dry_run:
            print("[INFO] Dry run enabled. Not moving robot.")
            return 0

        if robot is None:
            exit_with_error("Robot interface is unavailable for motion.")

        if not args.skip_motion_confirmation:
            print("WARNING: This will move the robot to the hover target above the clicked point.")
            print("Verify the printed target first and keep the stop button at hand.")
            try:
                response = input("Type 'move' to continue: ").strip().lower()
            except EOFError:
                print(
                    "[INFO] No interactive stdin is available for motion confirmation. "
                    "Re-run with --skip-motion-confirmation if you want to execute the move."
                )
                return 1
            if response != "move":
                print("[INFO] Motion cancelled by user.")
                return 0

        if args.hover_frame == "camera":
            base_T_target = pose_with_fixed_gripper_orientation_and_camera_at_translation(
                base_T_gripper_current=base_T_gripper,
                gripper_T_camera=gripper_T_camera,
                camera_xyz_in_base=hover_xyz,
            )
            print("[INFO] Hover frame: camera")
            print(
                "[INFO] Commanding gripper pose so the calibrated camera origin "
                "hovers above the clicked point."
            )
        else:
            base_T_target = pose_with_same_orientation_and_new_translation(base_T_gripper, hover_xyz)
            print("[INFO] Hover frame: gripper")
            print("[INFO] Commanding the gripper/EE origin to hover above the clicked point.")

        print("[INFO] Commanded base_T_target:")
        print(np.array2string(base_T_target, precision=6, suppress_small=True))
        print("[INFO] Sending hover target pose to robot...")
        start_time = time.time()
        robot.move_to_pose(base_T_target, duration=args.motion_duration)
        print(f"[INFO] Motion command completed in {time.time() - start_time:.2f} s.")
        print("[INFO] Verify XY alignment before any manual descent.")
        return 0

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        return 130
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
