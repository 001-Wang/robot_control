#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from franka_utils import connect_robot, exit_with_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read one Franka end-effector pose from the robot state."
    )
    parser.add_argument(
        "--robot-ip",
        default="192.168.1.11",
        help="Franka robot IP address.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        robot = connect_robot(args.robot_ip)
    except RuntimeError as exc:
        exit_with_error(str(exc))

    state = robot.read_once()
    base_T_ee = np.array(state.O_T_EE, dtype=float).reshape(4, 4, order="F")

    np.set_printoptions(precision=6, suppress=True)
    print("O_T_EE (base_T_ee):")
    print(base_T_ee)


if __name__ == "__main__":
    main()
