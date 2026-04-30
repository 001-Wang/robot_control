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


def lift_straight_up(robot: Robot, lift_m: float, duration_sec: float) -> None:
    active_control = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)
    robot_state, period = active_control.readOnce()
    start_pose = list(robot_state.O_T_EE)
    target_pose = start_pose.copy()
    target_pose[14] += lift_m

    print(f"Start xyz:  [{start_pose[12]:.6f}, {start_pose[13]:.6f}, {start_pose[14]:.6f}]")
    print(f"Target xyz: [{target_pose[12]:.6f}, {target_pose[13]:.6f}, {target_pose[14]:.6f}]")

    elapsed = 0.0
    motion_finished = False
    while not motion_finished:
        elapsed += period.to_sec()
        alpha = smoothstep5(min(elapsed / duration_sec, 1.0))
        command_pose = start_pose.copy()
        command_pose[12] = (1.0 - alpha) * start_pose[12] + alpha * target_pose[12]
        command_pose[13] = (1.0 - alpha) * start_pose[13] + alpha * target_pose[13]
        command_pose[14] = (1.0 - alpha) * start_pose[14] + alpha * target_pose[14]

        command = CartesianPose(command_pose)
        if elapsed >= duration_sec:
            command.motion_finished = True
            motion_finished = True

        active_control.writeOnce(command)
        if not motion_finished:
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
        return gripper.grasp(
            grasp_width,
            args.gripper_speed,
            args.gripper_force,
            epsilon_inner,
            epsilon_outer,
        )
    except TypeError:
        if args.force_detect_grasp or epsilon_inner != 0.005 or epsilon_outer != 0.005:
            print("This pylibfranka build does not expose grasp epsilon arguments.")
        return gripper.grasp(grasp_width, args.gripper_speed, args.gripper_force)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Grasp with the Franka gripper, then move the EE straight up."
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
    parser.add_argument("--gripper-speed", type=float, default=0.1, help="Gripper speed in m/s.")
    parser.add_argument("--gripper-force", type=float, default=60.0, help="Gripper force in N.")
    parser.add_argument("--homing", action="store_true", help="Home the gripper before grasping.")
    parser.add_argument("--lift-m", type=float, default=0.10, help="Straight-up lift distance in meters.")
    parser.add_argument("--duration-sec", type=float, default=4.0, help="Lift motion duration in seconds.")
    parser.add_argument("--yes", action="store_true", help="Skip interactive safety confirmation.")
    args = parser.parse_args()

    if args.duration_sec <= 0.0:
        raise ValueError("--duration-sec must be positive.")
    if args.lift_m == 0.0:
        raise ValueError("--lift-m must be non-zero.")
    if args.grasp_epsilon_inner < 0.0 or args.grasp_epsilon_outer < 0.0:
        raise ValueError("Grasp epsilon values must be non-negative.")
    if args.force_detect_max_width <= 0.0:
        raise ValueError("--force-detect-max-width must be positive.")

    print("WARNING: this will grasp with the gripper and move the robot.")
    print(f"Robot IP: {args.robot_ip}")
    if args.gripper_command == "grasp":
        print(
            "Grasp command: "
            f"width={args.grasp_width:.4f} m, speed={args.gripper_speed:.3f} m/s, "
            f"force={args.gripper_force:.1f} N"
        )
    else:
        print(
            "Gripper move command: "
            f"target_width={args.grasp_width:.4f} m, speed={args.gripper_speed:.3f} m/s"
        )
    print(f"Z move command: {args.lift_m:+.3f} m in base Z over {args.duration_sec:.2f} s")
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
                "--force-detect-grasp to judge contact by force over a wider width range."
            )
            return 1

        gripper_state = gripper.read_once()
        print(f"Gripper is_grasped: {gripper_state.is_grasped}")
        if not gripper_state.is_grasped:
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
    lift_straight_up(robot, args.lift_m, args.duration_sec)
    print("Finished grasp-and-lift motion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
