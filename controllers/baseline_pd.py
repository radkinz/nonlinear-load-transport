import numpy as np


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class BaselinePDController:
    """
    Simple PD trajectory tracker.

    This controller does not account for payload motion.
    It represents the rigid-payload baseline.
    """

    def __init__(
        self,
        kp_pos=40.0,
        kd_pos=35.0,
        kp_yaw=25.0,
        kd_yaw=6.0,
        max_force=120.0,
        max_torque=120.0,
    ):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw
        self.max_force = max_force
        self.max_torque = max_torque

    def compute_control(self, data, desired_state):
        x, y, yaw = data.qpos[0], data.qpos[1], data.qpos[2]
        vx, vy, yaw_rate = data.qvel[0], data.qvel[1], data.qvel[2]

        x_des = desired_state["x"]
        y_des = desired_state["y"]
        yaw_des = desired_state["yaw"]

        vx_des = desired_state["vx"]
        vy_des = desired_state["vy"]
        yaw_rate_des = desired_state["yaw_rate"]

        ux = self.kp_pos * (x_des - x) + self.kd_pos * (vx_des - vx)
        uy = self.kp_pos * (y_des - y) + self.kd_pos * (vy_des - vy)

        yaw_error = wrap_angle(yaw_des - yaw)
        uyaw = self.kp_yaw * yaw_error + self.kd_yaw * (yaw_rate_des - yaw_rate)

        ux = np.clip(ux, -self.max_force, self.max_force)
        uy = np.clip(uy, -self.max_force, self.max_force)
        uyaw = np.clip(uyaw, -self.max_torque, self.max_torque)

        return np.array([ux, uy, uyaw])