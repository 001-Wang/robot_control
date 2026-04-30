from __future__ import annotations

import sys
from typing import NoReturn

import pylibfranka


def connect_robot(robot_ip: str) -> pylibfranka.Robot:
    try:
        return pylibfranka.Robot(robot_ip)
    except Exception as exc:
        message = str(exc)
        if "unable to set realtime scheduling" in message:
            try:
                return pylibfranka.Robot(robot_ip, pylibfranka.RealtimeConfig.kIgnore)
            except Exception as retry_exc:
                raise RuntimeError(
                    "Failed to connect to the Franka robot because realtime scheduling is "
                    "not permitted on this workstation, and retrying with "
                    "`RealtimeConfig.kIgnore` also failed.\n"
                    f"Robot IP: {robot_ip}\n"
                    f"Original error: {message}\n"
                    f"Retry error: {retry_exc}"
                ) from retry_exc

        if "UDP receive: Timeout" in message:
            raise RuntimeError(
                "Failed to connect to the Franka robot because the UDP handshake timed out.\n"
                f"Robot IP: {robot_ip}\n"
                "Checks:\n"
                "  - Verify the workstation is on the robot subnet and can reach the robot IP.\n"
                "  - Confirm the Franka Control Interface (FCI) is enabled for this workstation.\n"
                "  - Make sure no firewall is blocking UDP traffic.\n"
                "  - Ensure no other process currently holds the robot connection.\n"
                "If this workstation cannot access FCI, use the dataset tools with --pose-file or "
                "--manual-pose instead."
            ) from exc

        raise RuntimeError(
            f"Failed to connect to the Franka robot at {robot_ip}: {message}"
        ) from exc


def exit_with_error(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(1)
