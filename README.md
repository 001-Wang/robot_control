# Eye-in-Hand Camera Calibration

This repo now contains a minimal hand-eye calibration workflow for a camera mounted on the robot gripper.

## Goal

Given:

- One image per robot pose.
- The robot pose `base_T_gripper` for the same timestamp.
- Known camera intrinsics.
- A fixed checkerboard target in the scene.

The script solves:

- `gripper_T_camera`: the rigid transform from gripper frame to camera frame.
- `base_T_camera` for every captured view.

## Install

Create the environment from [environment-franka.yml](/home/zuoxu/project/geometric_robot_dgs/environment-franka.yml). The calibration pipeline needs an OpenCV build with the `aruco` module because it uses ChArUco detection plus `calibrateHandEye`.

## Dataset Format

Use [data/calibration_dataset.example.json](/home/zuoxu/project/geometric_robot_dgs/data/calibration_dataset.example.json) as the starting point, or create a JSON file like this:

```json
{
  "handeye_method": "tsai",
  "camera": {
    "camera_matrix": [
      [615.0, 0.0, 320.0],
      [0.0, 615.0, 240.0],
      [0.0, 0.0, 1.0]
    ],
    "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]
  },
  "target": {
    "type": "charuco",
    "dictionary": "4X4_50",
    "squares_x": 7,
    "squares_y": 10,
    "square_length_m": 0.024,
    "marker_length_m": 0.018
  },
  "samples": [
    {
      "id": "view_000",
      "image": "data/view_000.png",
      "base_T_gripper": {
        "translation": [0.52, -0.11, 0.34],
        "quaternion_xyzw": [0.0, 0.7071, 0.0, 0.7071]
      }
    }
  ]
}
```

Notes:

- For a ChArUco board, `dictionary`, `squares_x`, `squares_y`, `square_length_m`, and `marker_length_m` must match the printed board exactly.
- `base_T_gripper` must be synchronized with the image.
- Robot poses can also be given as a `4x4` homogeneous matrix.

## Run

```bash
python scripts/eye_in_hand_calibrate.py \
  --dataset data/calibration_dataset.json \
  --output data/calibration_result.json
```

Optional:

```bash
python scripts/eye_in_hand_calibrate.py \
  --dataset data/calibration_dataset.json \
  --output data/calibration_result.json \
  --visualize
```

`--visualize` writes debug images with detected ChArUco features next to the original images.

## Recreate The Board

Generate a printable ChArUco board that matches the current dataset target:

```bash
python scripts/create_charuco_board.py \
  --dataset data/calibration_dataset.json \
  --paper a4 \
  --output data/charuco_board_a4.pdf
```

Print the PDF at `100%` scale with scaling disabled. The printed `square_length` and
`marker_length` must match the values in the dataset JSON exactly.

## Output

The result JSON contains:

- `gripper_T_camera`
- `camera_T_gripper`
- `views[*].base_T_camera`
- `views[*].camera_T_base`
- `views[*].target_T_camera`
- `views[*].camera_T_target`

This gives you exactly what you asked for: one image per view and the corresponding camera pose in the manipulator base frame.

## Data Collection Guidance

- Use at least 10 to 20 views.
- Move the wrist through diverse rotations, not just translations.
- Keep the ChArUco board fully visible and sharply focused.
- Avoid collecting all poses from one side of the board.
- Calibrate camera intrinsics first; hand-eye results will be poor if intrinsics are wrong.

## Alternative Pose Sources

If `pylibfranka` cannot connect on your workstation, the collector can capture images
without Franka access and read poses from another source instead.

Manual pose entry:

```bash
python scripts/collect_handeye_dataset.py \
  --manual-pose \
  --dataset data/calibration_dataset.json \
  --image-dir data/images \
  --preview
```

At each capture, paste either:

- `16` row-major values for a `4x4` transform matrix, or
- `7` values: `tx ty tz qx qy qz qw`

External pose file:

```bash
python scripts/collect_handeye_dataset.py \
  --pose-file data/current_pose.json \
  --dataset data/calibration_dataset.json \
  --image-dir data/images \
  --preview
```

The JSON file may contain either:

```json
{
  "matrix": [[...], [...], [...], [...]]
}
```

or:

```json
{
  "translation": [0.52, -0.11, 0.34],
  "quaternion_xyzw": [0.0, 0.7071, 0.0, 0.7071]
}
```

## Frame Convention

The script assumes an eye-in-hand setup:

- The calibration board is static in the world.
- The camera is rigidly attached to the gripper.
- `base_T_gripper` means gripper pose expressed in robot base coordinates.

If your robot API returns `gripper_T_base` instead, invert it before writing the dataset.
# robot_control
