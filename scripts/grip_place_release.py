#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
from pylibfranka import CartesianPose, ControllerMode, Gripper, RealtimeConfig, Robot


def smoothstep5(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha**2)


def apply_collision_behavior(robot: Robot) -> None:
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


def xyz_from_pose(pose: list[float]) -> str:
    return f"[{pose[12]:.6f}, {pose[13]:.6f}, {pose[14]:.6f}]"


def translated_pose(
    pose: list[float],
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
) -> list[float]:
    result = pose.copy()
    result[12] += dx
    result[13] += dy
    result[14] += dz
    return result


def run_cartesian_segments(
    robot: Robot,
    waypoints: list[tuple[str, list[float]]],
    duration_sec: float,
) -> None:
    if len(waypoints) < 2:
        raise ValueError("At least two waypoints are required.")

    active_control = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)
    robot_state, period = active_control.readOnce()

    for waypoint_index in range(1, len(waypoints)):
        segment_start = waypoints[waypoint_index - 1][1]
        label, segment_target = waypoints[waypoint_index]
        print(f"Moving {label}: {xyz_from_pose(segment_start)} -> {xyz_from_pose(segment_target)}")

        elapsed = 0.0
        segment_done = False
        final_segment = waypoint_index == len(waypoints) - 1
        while not segment_done:
            elapsed += period.to_sec()
            alpha = smoothstep5(min(elapsed / duration_sec, 1.0))
            command_pose = segment_start.copy()
            command_pose[12] = (1.0 - alpha) * segment_start[12] + alpha * segment_target[12]
            command_pose[13] = (1.0 - alpha) * segment_start[13] + alpha * segment_target[13]
            command_pose[14] = (1.0 - alpha) * segment_start[14] + alpha * segment_target[14]

            command = CartesianPose(command_pose)
            if elapsed >= duration_sec:
                segment_done = True
                if final_segment:
                    command.motion_finished = True

            active_control.writeOnce(command)
            if not (segment_done and final_segment):
                robot_state, period = active_control.readOnce()


def grasp_with_options(gripper: Gripper, args: argparse.Namespace) -> bool:
    if args.force_detect_grasp:
        grasp_width = args.grasp_width
        epsilon_inner = min(args.grasp_width, args.force_detect_min_width)
        epsilon_outer = max(0.0, args.force_detect_max_width - args.grasp_width)
        print(
            "Force-detect grasp: "
            f"target_width={grasp_width:.4f} m, "
            f"closing with force={args.gripper_force:.1f} N, "
            f"accepting object width from {args.force_detect_min_width:.4f} m "
            f"to {args.force_detect_max_width:.4f} m"
        )
    else:
        grasp_width = args.grasp_width
        epsilon_inner = args.grasp_epsilon_inner
        epsilon_outer = args.grasp_epsilon_outer
        print(
            "Width-checked grasp: "
            f"width={grasp_width:.4f} m, "
            f"epsilon_inner={epsilon_inner:.4f} m, "
            f"epsilon_outer={epsilon_outer:.4f} m"
        )

    try:
        grasp_ok = gripper.grasp(
            grasp_width,
            args.gripper_speed,
            args.gripper_force,
            epsilon_inner,
            epsilon_outer,
        )
    except TypeError:
        if args.force_detect_grasp or epsilon_inner != 0.005 or epsilon_outer != 0.005:
            print("This pylibfranka build does not expose grasp epsilon arguments.")
        grasp_ok = gripper.grasp(grasp_width, args.gripper_speed, args.gripper_force)

    if grasp_ok or not args.force_detect_grasp:
        return grasp_ok

    gripper_state = gripper.read_once()
    width = float(gripper_state.width)
    contact_by_width = args.force_detect_min_width <= width <= args.force_detect_max_width
    print(
        "Force-detect fallback: "
        f"gripper width after grasp={width:.6f} m, "
        f"is_grasped={gripper_state.is_grasped}"
    )
    if contact_by_width:
        print("Accepting grasp because fingers stopped above closed width.")
        return True
    return False


def gripper_command_summary(args: argparse.Namespace) -> str:
    if args.gripper_command == "move":
        return (
            "Gripper move command: "
            f"target_width={args.grasp_width:.4f} m, speed={args.gripper_speed:.3f} m/s"
        )
    if args.force_detect_grasp:
        return (
            "Force-detect grasp command: "
            f"speed={args.gripper_speed:.3f} m/s, force={args.gripper_force:.1f} N, "
            f"accepted_width=[{args.force_detect_min_width:.4f}, "
            f"{args.force_detect_max_width:.4f}] m"
        )
    return (
        "Grasp command: "
        f"width={args.grasp_width:.4f} m, speed={args.gripper_speed:.3f} m/s, "
        f"force={args.gripper_force:.1f} N"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pick an object, translate it, lower it, then release it."
    )
    parser.add_argument("--robot-ip", "--ip", default="192.168.1.11", help="Franka robot IP address.")
    parser.add_argument(
        "--gripper-command",
        choices=["grasp", "move"],
        default="grasp",
        help="Use force grasping, or only move the fingers to a target width.",
    )
    parser.add_argument("--grasp-width", type=float, default=0.005, help="Object grasp width in meters.")
    parser.add_argument(
        "--grasp-epsilon-inner",
        type=float,
        default=0.005,
        help="Allowed final width below --grasp-width for Franka grasp success.",
    )
    parser.add_argument(
        "--grasp-epsilon-outer",
        type=float,
        default=0.005,
        help="Allowed final width above --grasp-width for Franka grasp success.",
    )
    parser.add_argument(
        "--force-detect-grasp",
        action="store_true",
        help="Close with force and accept any detected object width up to --force-detect-max-width.",
    )
    parser.add_argument(
        "--force-detect-max-width",
        type=float,
        default=0.08,
        help="Maximum accepted object width when using --force-detect-grasp.",
    )
    parser.add_argument(
        "--force-detect-min-width",
        type=float,
        default=0.002,
        help="Minimum final gripper width treated as contact in force-detect fallback.",
    )
    parser.add_argument("--gripper-speed", type=float, default=0.1, help="Gripper speed in m/s.")
    parser.add_argument("--gripper-force", type=float, default=60.0, help="Gripper force in N.")
    parser.add_argument("--homing", action="store_true", help="Home the gripper before grasping.")
    parser.add_argument("--lift-m", type=float, default=0.10, help="Straight-up lift distance in meters.")
    parser.add_argument("--right-m", type=float, default=0.20, help="Rightward move distance in meters.")
    parser.add_argument("--down-m", type=float, default=0.10, help="Downward move distance in meters.")
    parser.add_argument(
        "--right-axis",
        choices=["x", "y"],
        default="y",
        help="Base-frame horizontal axis used for the rightward move.",
    )
    parser.add_argument(
        "--right-sign",
        type=float,
        default=-1.0,
        choices=[-1.0, 1.0],
        help="Direction sign on --right-axis. Default -1 means base -Y.",
    )
    parser.add_argument("--release-width", type=float, default=0.08, help="Gripper opening width for release.")
    parser.add_argument("--duration-sec", type=float, default=4.0, help="Motion duration per segment in seconds.")
    parser.add_argument("--yes", action="store_true", help="Skip interactive safety confirmation.")
    args = parser.parse_args()

    if args.duration_sec <= 0.0:
        raise ValueError("--duration-sec must be positive.")
    if args.lift_m <= 0.0:
        raise ValueError("--lift-m must be positive for an upward lift.")
    if args.right_m <= 0.0:
        raise ValueError("--right-m must be positive.")
    if args.down_m <= 0.0:
        raise ValueError("--down-m must be positive.")
    if args.grasp_epsilon_inner < 0.0 or args.grasp_epsilon_outer < 0.0:
        raise ValueError("Grasp epsilon values must be non-negative.")
    if args.force_detect_max_width <= 0.0:
        raise ValueError("--force-detect-max-width must be positive.")
    if args.force_detect_min_width < 0.0:
        raise ValueError("--force-detect-min-width must be non-negative.")
    if args.force_detect_min_width >= args.force_detect_max_width:
        raise ValueError("--force-detect-min-width must be smaller than --force-detect-max-width.")

    print("WARNING: this will grasp with the gripper and move the robot.")
    print(f"Robot IP: {args.robot_ip}")
    print(gripper_command_summary(args))
    right_dx = args.right_sign * args.right_m if args.right_axis == "x" else 0.0
    right_dy = args.right_sign * args.right_m if args.right_axis == "y" else 0.0
    print(f"Lift command: +{args.lift_m:.3f} m in base Z over {args.duration_sec:.2f} s")
    print(
        "Place move command: "
        f"right vector=[{right_dx:.3f}, {right_dy:.3f}, 0.000] m, "
        f"then down {args.down_m:.3f} m, then release to {args.release_width:.3f} m"
    )
    if not args.yes:
        try:
            input("Press Enter only if the workspace is clear and the user stop is ready...")
        except EOFError:
            print(
                "No interactive stdin is available. Re-run with --yes only after "
                "confirming the workspace is clear and the user stop is ready."
            )
            return 1

    gripper = Gripper(args.robot_ip)
    if args.homing:
        print("Homing gripper...")
        gripper.homing()

    gripper_state = gripper.read_once()
    print(f"Current gripper width: {gripper_state.width:.6f} m")

    if args.gripper_command == "grasp":
        print("Grasping...")
        if not grasp_with_options(gripper, args):
            print(
                "Failed to grasp object; robot lift was not started. "
                "Set --grasp-width close to the real object width, or use "
                "--force-detect-min-width/--force-detect-max-width to tune contact detection."
            )
            return 1

        gripper_state = gripper.read_once()
        print(f"Gripper is_grasped: {gripper_state.is_grasped}")
        if not gripper_state.is_grasped and not args.force_detect_grasp:
            print("Object is not reported as grasped; robot lift was not started.")
            return 1
    else:
        print("Moving gripper fingers...")
        if not gripper.move(args.grasp_width, args.gripper_speed):
            print("Failed to move gripper fingers; robot lift was not started.")
            return 1
        gripper_state = gripper.read_once()
        print(f"Gripper width after move: {gripper_state.width:.6f} m")

    robot = Robot(args.robot_ip, RealtimeConfig.kIgnore)
    apply_collision_behavior(robot)
    start_pose = list(robot.read_once().O_T_EE)
    lifted_pose = translated_pose(start_pose, dz=args.lift_m)
    shifted_pose = translated_pose(lifted_pose, dx=right_dx, dy=right_dy)
    lowered_pose = translated_pose(shifted_pose, dz=-args.down_m)
    run_cartesian_segments(
        robot,
        [
            ("start", start_pose),
            ("up", lifted_pose),
            ("right", shifted_pose),
            ("down", lowered_pose),
        ],
        args.duration_sec,
    )

    print("Releasing object...")
    if not gripper.move(args.release_width, args.gripper_speed):
        print("Robot place motion finished, but gripper release command failed.")
        return 1
    print("Finished pick-place-release motion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
