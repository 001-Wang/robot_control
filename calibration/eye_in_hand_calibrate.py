#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate an eye-in-hand camera mounted on a robot gripper and export "
            "camera poses in the robot base frame for each view."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset JSON path.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write debug images with detected ChArUco features.",
    )
    return parser.parse_args()


@dataclass
class Pose:
    rotation: np.ndarray
    translation: np.ndarray

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation.reshape(3)
        return matrix

    def inverse(self) -> "Pose":
        rotation_inv = self.rotation.T
        translation_inv = -(rotation_inv @ self.translation.reshape(3, 1))
        return Pose(rotation_inv, translation_inv.reshape(3))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        raise ValueError("Quaternion has zero norm.")
    return quaternion / norm


def quaternion_to_rotation_matrix(quaternion_xyzw: list[float]) -> np.ndarray:
    x, y, z, w = normalize_quaternion(np.asarray(quaternion_xyzw, dtype=np.float64))
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation[2, 1] - rotation[1, 2]) * s
        y = (rotation[0, 2] - rotation[2, 0]) * s
        z = (rotation[1, 0] - rotation[0, 1]) * s
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
    return normalize_quaternion(np.array([x, y, z, w], dtype=np.float64))


def parse_pose(raw_pose: dict[str, Any]) -> Pose:
    if "matrix" in raw_pose:
        matrix = np.asarray(raw_pose["matrix"], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("Pose matrix must be 4x4.")
        return Pose(matrix[:3, :3], matrix[:3, 3])

    position = np.asarray(raw_pose["translation"], dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("Pose translation must have 3 elements.")

    if "quaternion_xyzw" in raw_pose:
        rotation = quaternion_to_rotation_matrix(raw_pose["quaternion_xyzw"])
    elif "rotation_matrix" in raw_pose:
        rotation = np.asarray(raw_pose["rotation_matrix"], dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("Pose rotation_matrix must be 3x3.")
    else:
        raise ValueError("Pose needs either quaternion_xyzw or rotation_matrix.")
    return Pose(rotation, position)


def to_serializable_pose(pose: Pose) -> dict[str, Any]:
    quaternion = rotation_matrix_to_quaternion(pose.rotation)
    return {
        "translation": pose.translation.reshape(3).tolist(),
        "quaternion_xyzw": quaternion.tolist(),
        "matrix": pose.as_matrix().tolist(),
    }


def aruco_dictionary(name: str) -> Any:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV aruco module is unavailable. Install an OpenCV build with contrib modules."
        )
    key = f"DICT_{name}"
    if not hasattr(cv2.aruco, key):
        raise ValueError(f"Unsupported ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, key))


def build_charuco_board(board_cfg: dict[str, Any]) -> Any:
    dictionary = aruco_dictionary(board_cfg["dictionary"])
    return cv2.aruco.CharucoBoard(
        (int(board_cfg["squares_x"]), int(board_cfg["squares_y"])),
        float(board_cfg["square_length_m"]),
        float(board_cfg["marker_length_m"]),
        dictionary,
    )


def detect_charuco_pose(
    image_path: Path,
    board: Any,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    visualize: bool,
) -> Pose:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = cv2.aruco.ArucoDetector(board.getDictionary())
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    if marker_ids is None or len(marker_ids) == 0:
        raise RuntimeError(f"No ArUco markers detected: {image_path}")

    detected, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
    )
    if detected is None or int(detected) < 4 or charuco_ids is None:
        raise RuntimeError(f"Not enough ChArUco corners detected: {image_path}")

    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charucoCorners=charuco_corners,
        charucoIds=charuco_ids,
        board=board,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        rvec=None,
        tvec=None,
    )
    if not success:
        raise RuntimeError(f"ChArUco pose estimation failed: {image_path}")

    rotation, _ = cv2.Rodrigues(rvec)

    if visualize:
        debug = image.copy()
        cv2.aruco.drawDetectedMarkers(debug, marker_corners, marker_ids)
        cv2.aruco.drawDetectedCornersCharuco(debug, charuco_corners, charuco_ids)
        cv2.drawFrameAxes(debug, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
        debug_path = image_path.with_name(f"{image_path.stem}_corners{image_path.suffix}")
        cv2.imwrite(str(debug_path), debug)

    return Pose(rotation, tvec.reshape(3))


def handeye_method(name: str) -> int:
    methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    if name not in methods:
        raise ValueError(f"Unsupported hand-eye method: {name}")
    return methods[name]


def main() -> None:
    args = parse_args()
    dataset = load_json(args.dataset)
    dataset_dir = args.dataset.resolve().parent

    intrinsics = dataset["camera"]
    camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(intrinsics.get("dist_coeffs", []), dtype=np.float64)
    if camera_matrix.shape != (3, 3):
        raise ValueError("camera.camera_matrix must be 3x3.")
    if dist_coeffs.ndim == 1:
        dist_coeffs = dist_coeffs.reshape(-1, 1)

    target = dataset["target"]
    if target.get("type", "charuco") != "charuco":
        raise ValueError("This script currently expects target.type == 'charuco'.")
    charuco_board = build_charuco_board(target)

    method = handeye_method(dataset.get("handeye_method", "tsai"))
    samples = dataset["samples"]
    if len(samples) < 3:
        raise ValueError("At least 3 samples are required; 10-20 diverse poses are recommended.")

    rotations_base_gripper: list[np.ndarray] = []
    translations_base_gripper: list[np.ndarray] = []
    rotations_target_camera: list[np.ndarray] = []
    translations_target_camera: list[np.ndarray] = []
    valid_samples: list[dict[str, Any]] = []

    for index, sample in enumerate(samples):
        sample_id = sample.get("id", f"sample_{index:03d}")
        image_path = (dataset_dir / sample["image"]).resolve()
        base_to_gripper = parse_pose(sample["base_T_gripper"])
        target_to_camera = detect_charuco_pose(
            image_path=image_path,
            board=charuco_board,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            visualize=args.visualize,
        )

        rotations_base_gripper.append(base_to_gripper.rotation)
        translations_base_gripper.append(base_to_gripper.translation.reshape(3, 1))
        rotations_target_camera.append(target_to_camera.rotation)
        translations_target_camera.append(target_to_camera.translation.reshape(3, 1))
        valid_samples.append({"id": sample_id, "image": str(image_path)})

    rotation_gripper_camera, translation_gripper_camera = cv2.calibrateHandEye(
        rotations_base_gripper,
        translations_base_gripper,
        rotations_target_camera,
        translations_target_camera,
        method=method,
    )

    gripper_to_camera = Pose(
        rotation=np.asarray(rotation_gripper_camera, dtype=np.float64),
        translation=np.asarray(translation_gripper_camera, dtype=np.float64).reshape(3),
    )

    per_view = []
    for sample_meta, rotation_base_gripper, translation_base_gripper, rotation_target_camera, translation_target_camera in zip(
        valid_samples,
        rotations_base_gripper,
        translations_base_gripper,
        rotations_target_camera,
        translations_target_camera,
    ):
        base_to_gripper = Pose(rotation_base_gripper, translation_base_gripper.reshape(3))
        base_to_camera = Pose(
            rotation=base_to_gripper.rotation @ gripper_to_camera.rotation,
            translation=(
                base_to_gripper.rotation @ gripper_to_camera.translation.reshape(3, 1)
                + base_to_gripper.translation.reshape(3, 1)
            ).reshape(3),
        )
        target_to_camera = Pose(rotation_target_camera, translation_target_camera.reshape(3))
        per_view.append(
            {
                "id": sample_meta["id"],
                "image": sample_meta["image"],
                "base_T_camera": to_serializable_pose(base_to_camera),
                "camera_T_base": to_serializable_pose(base_to_camera.inverse()),
                "camera_T_target": to_serializable_pose(target_to_camera.inverse()),
                "target_T_camera": to_serializable_pose(target_to_camera),
            }
        )

    output = {
        "frame_convention": {
            "base": "robot base frame",
            "gripper": "robot end-effector / gripper frame",
            "camera": "camera optical frame",
            "target": "calibration target frame",
        },
        "handeye_method": dataset.get("handeye_method", "tsai"),
        "target_type": "charuco",
        "gripper_T_camera": to_serializable_pose(gripper_to_camera),
        "camera_T_gripper": to_serializable_pose(gripper_to_camera.inverse()),
        "views": per_view,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"Wrote calibration results to {args.output}")


if __name__ == "__main__":
    main()
