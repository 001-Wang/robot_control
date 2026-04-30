#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a sparse point cloud from a known-pose dataset by matching SIFT "
            "features across nearby views and triangulating with the recorded camera poses."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Dataset directory.")
    parser.add_argument(
        "--output-ply",
        type=Path,
        default=None,
        help="Output PLY path. Defaults to <run-dir>/sparse_known_pose.ply.",
    )
    parser.add_argument(
        "--output-stats",
        type=Path,
        default=None,
        help="Output JSON stats path. Defaults to <run-dir>/sparse_known_pose_stats.json.",
    )
    parser.add_argument("--max-features", type=int, default=4000, help="Max SIFT features per image.")
    parser.add_argument("--max-pair-gap", type=int, default=2, help="Maximum frame index gap to match.")
    parser.add_argument("--ratio-test", type=float, default=0.75, help="Lowe ratio test threshold.")
    parser.add_argument("--min-matches", type=int, default=40, help="Minimum descriptor matches to keep a pair.")
    parser.add_argument(
        "--min-baseline-m",
        type=float,
        default=0.02,
        help="Minimum baseline between camera centers to keep a pair.",
    )
    parser.add_argument(
        "--min-triangulation-angle-deg",
        type=float,
        default=1.0,
        help="Minimum triangulation angle to keep a 3D point.",
    )
    parser.add_argument(
        "--max-reproj-error-px",
        type=float,
        default=3.0,
        help="Maximum reprojection error per observation to keep a 3D point.",
    )
    parser.add_argument(
        "--dedup-voxel-size-m",
        type=float,
        default=0.003,
        help="Voxel size used to merge nearby triangulated points.",
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
    return entries


def matrix_from_entry(entry: dict[str, Any], key: str) -> np.ndarray:
    matrix = np.asarray(entry[key]["matrix"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{key} must be 4x4, got {matrix.shape}")
    return matrix


def intrinsics_matrix(metadata: dict[str, Any]) -> np.ndarray:
    intr = metadata["color_intrinsics"]
    return np.array(
        [
            [float(intr["fx"]), 0.0, float(intr["ppx"])],
            [0.0, float(intr["fy"]), float(intr["ppy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def projection_matrix(camera_T_world: np.ndarray, K: np.ndarray) -> np.ndarray:
    return K @ camera_T_world[:3, :]


def camera_center_from_world_T_camera(world_T_camera: np.ndarray) -> np.ndarray:
    return world_T_camera[:3, 3]


def triangulation_angle_deg(c1: np.ndarray, c2: np.ndarray, point: np.ndarray) -> float:
    ray1 = point - c1
    ray2 = point - c2
    norm1 = np.linalg.norm(ray1)
    norm2 = np.linalg.norm(ray2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    cosine = np.clip(np.dot(ray1, ray2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def reproject(world_point: np.ndarray, camera_T_world: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, float]:
    camera_point = camera_T_world[:3, :3] @ world_point + camera_T_world[:3, 3]
    if camera_point[2] <= 0.0:
        return np.zeros(2, dtype=np.float64), float("inf")
    image_point = K @ camera_point
    uv = image_point[:2] / image_point[2]
    return uv, float(camera_point[2])


def write_ascii_ply(path: Path, points_xyz: np.ndarray, colors_rgb: np.ndarray) -> None:
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points_xyz)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(header) + "\n")
        for xyz, rgb in zip(points_xyz, colors_rgb):
            handle.write(
                f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )


def deduplicate_points(
    points_xyz: list[np.ndarray],
    colors_rgb: list[np.ndarray],
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not points_xyz:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

    if voxel_size_m <= 0.0:
        return np.asarray(points_xyz, dtype=np.float64), np.asarray(colors_rgb, dtype=np.uint8)

    buckets: dict[tuple[int, int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    for xyz, rgb in zip(points_xyz, colors_rgb):
        key = tuple(np.floor(xyz / voxel_size_m).astype(int).tolist())
        buckets.setdefault(key, []).append((xyz, rgb))

    merged_xyz: list[np.ndarray] = []
    merged_rgb: list[np.ndarray] = []
    for values in buckets.values():
        xyzs = np.stack([value[0] for value in values], axis=0)
        rgbs = np.stack([value[1] for value in values], axis=0)
        merged_xyz.append(np.mean(xyzs, axis=0))
        merged_rgb.append(np.mean(rgbs, axis=0).astype(np.uint8))
    return np.asarray(merged_xyz, dtype=np.float64), np.asarray(merged_rgb, dtype=np.uint8)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metadata = load_json(run_dir / "run_metadata.json")
    entries = load_jsonl(run_dir / "frame_poses.jsonl")
    if not entries:
        raise ValueError("No pose entries found.")

    output_ply = args.output_ply.resolve() if args.output_ply is not None else run_dir / "sparse_known_pose.ply"
    output_stats = (
        args.output_stats.resolve()
        if args.output_stats is not None
        else run_dir / "sparse_known_pose_stats.json"
    )

    K = intrinsics_matrix(metadata)
    image_dir = run_dir
    detector = cv2.SIFT_create(nfeatures=args.max_features)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    frames: list[dict[str, Any]] = []
    for entry in entries:
        image_path = image_dir / entry["rgb_path"]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = detector.detectAndCompute(image_gray, None)
        if "camera_T_base" in entry:
            camera_T_world = matrix_from_entry(entry, "camera_T_base")
            world_T_camera = np.linalg.inv(camera_T_world)
        elif "base_T_camera" in entry:
            world_T_camera = matrix_from_entry(entry, "base_T_camera")
            camera_T_world = np.linalg.inv(world_T_camera)
        else:
            raise ValueError("Each frame entry must contain camera_T_base or base_T_camera.")
        frames.append(
            {
                "entry": entry,
                "image_path": image_path,
                "image_bgr": image_bgr,
                "keypoints": keypoints,
                "descriptors": descriptors,
                "camera_T_world": camera_T_world,
                "world_T_camera": world_T_camera,
                "projection": projection_matrix(camera_T_world, K),
                "camera_center": camera_center_from_world_T_camera(world_T_camera),
            }
        )

    all_points_xyz: list[np.ndarray] = []
    all_colors_rgb: list[np.ndarray] = []
    pair_stats: list[dict[str, Any]] = []

    for i, frame_i in enumerate(frames):
        desc_i = frame_i["descriptors"]
        if desc_i is None or len(frame_i["keypoints"]) < args.min_matches:
            continue
        for j in range(i + 1, min(len(frames), i + 1 + args.max_pair_gap)):
            frame_j = frames[j]
            desc_j = frame_j["descriptors"]
            if desc_j is None or len(frame_j["keypoints"]) < args.min_matches:
                continue

            baseline = float(np.linalg.norm(frame_i["camera_center"] - frame_j["camera_center"]))
            if baseline < args.min_baseline_m:
                pair_stats.append(
                    {
                        "i": i,
                        "j": j,
                        "baseline_m": baseline,
                        "status": "skipped_baseline",
                    }
                )
                continue

            knn_matches = matcher.knnMatch(desc_i, desc_j, k=2)
            good_matches = []
            for pair in knn_matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < args.ratio_test * n.distance:
                    good_matches.append(m)
            if len(good_matches) < args.min_matches:
                pair_stats.append(
                    {
                        "i": i,
                        "j": j,
                        "baseline_m": baseline,
                        "raw_matches": len(good_matches),
                        "status": "too_few_matches",
                    }
                )
                continue

            pts_i = np.float32([frame_i["keypoints"][m.queryIdx].pt for m in good_matches])
            pts_j = np.float32([frame_j["keypoints"][m.trainIdx].pt for m in good_matches])

            F, inlier_mask = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC, 1.5, 0.99)
            if F is None or inlier_mask is None:
                pair_stats.append(
                    {"i": i, "j": j, "baseline_m": baseline, "raw_matches": len(good_matches), "status": "no_F"}
                )
                continue
            inlier_mask = inlier_mask.ravel().astype(bool)
            pts_i_in = pts_i[inlier_mask]
            pts_j_in = pts_j[inlier_mask]
            matches_in = [match for match, keep in zip(good_matches, inlier_mask) if keep]
            if len(matches_in) < args.min_matches:
                pair_stats.append(
                    {
                        "i": i,
                        "j": j,
                        "baseline_m": baseline,
                        "raw_matches": len(good_matches),
                        "inlier_matches": len(matches_in),
                        "status": "too_few_inliers",
                    }
                )
                continue

            homogeneous = cv2.triangulatePoints(
                frame_i["projection"],
                frame_j["projection"],
                pts_i_in.T,
                pts_j_in.T,
            )
            points_world = (homogeneous[:3] / homogeneous[3]).T

            kept = 0
            for point_world, pt_i, pt_j, match in zip(points_world, pts_i_in, pts_j_in, matches_in):
                if not np.all(np.isfinite(point_world)):
                    continue

                reproj_i, depth_i = reproject(point_world, frame_i["camera_T_world"], K)
                reproj_j, depth_j = reproject(point_world, frame_j["camera_T_world"], K)
                if depth_i <= 0.0 or depth_j <= 0.0:
                    continue

                err_i = float(np.linalg.norm(reproj_i - pt_i))
                err_j = float(np.linalg.norm(reproj_j - pt_j))
                if err_i > args.max_reproj_error_px or err_j > args.max_reproj_error_px:
                    continue

                angle_deg = triangulation_angle_deg(frame_i["camera_center"], frame_j["camera_center"], point_world)
                if angle_deg < args.min_triangulation_angle_deg:
                    continue

                u, v = frame_i["keypoints"][match.queryIdx].pt
                u_i = int(np.clip(round(u), 0, frame_i["image_bgr"].shape[1] - 1))
                v_i = int(np.clip(round(v), 0, frame_i["image_bgr"].shape[0] - 1))
                color_bgr = frame_i["image_bgr"][v_i, u_i]
                color_rgb = np.array([color_bgr[2], color_bgr[1], color_bgr[0]], dtype=np.uint8)

                all_points_xyz.append(point_world.astype(np.float64))
                all_colors_rgb.append(color_rgb)
                kept += 1

            pair_stats.append(
                {
                    "i": i,
                    "j": j,
                    "image_i": frame_i["entry"]["rgb_path"],
                    "image_j": frame_j["entry"]["rgb_path"],
                    "baseline_m": baseline,
                    "raw_matches": len(good_matches),
                    "inlier_matches": len(matches_in),
                    "kept_points": kept,
                    "status": "ok",
                }
            )

    merged_points_xyz, merged_colors_rgb = deduplicate_points(
        all_points_xyz,
        all_colors_rgb,
        voxel_size_m=args.dedup_voxel_size_m,
    )
    write_ascii_ply(output_ply, merged_points_xyz, merged_colors_rgb)

    stats = {
        "run_dir": str(run_dir),
        "num_frames": len(frames),
        "raw_triangulated_points": len(all_points_xyz),
        "deduplicated_points": int(len(merged_points_xyz)),
        "parameters": {
            "max_features": args.max_features,
            "max_pair_gap": args.max_pair_gap,
            "ratio_test": args.ratio_test,
            "min_matches": args.min_matches,
            "min_baseline_m": args.min_baseline_m,
            "min_triangulation_angle_deg": args.min_triangulation_angle_deg,
            "max_reproj_error_px": args.max_reproj_error_px,
            "dedup_voxel_size_m": args.dedup_voxel_size_m,
        },
        "pairs": pair_stats,
        "output_ply": str(output_ply),
    }
    output_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Wrote sparse point cloud to {output_ply}")
    print(f"Raw triangulated points: {len(all_points_xyz)}")
    print(f"Deduplicated points: {len(merged_points_xyz)}")
    print(f"Wrote stats to {output_stats}")


if __name__ == "__main__":
    main()
