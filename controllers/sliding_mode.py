import numpy as np

from controllers.baseline_pd import wrap_angle


class SlidingModeController:
    """
    Sliding-mode trajectory tracker for the mobile base.

    This controller treats the shifting payload as an unknown disturbance
    and adds a robust correction term based on the sliding surface:

        s = velocity_error + lambda * position_error

    A tanh term is used instead of sign(s) to reduce chattering.
    """

    def __init__(
        self,
        kp_pos=30.0,
        kd_pos=20.0,
        lambda_pos=2.0,
        k_robust=25.0,
        eps=0.05,
        kp_yaw=25.0,
        kd_yaw=6.0,
        lambda_yaw=2.0,
        k_yaw_robust=8.0,
        yaw_eps=0.05,
        max_force=120.0,
        max_torque=120.0,
    ):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos

        self.lambda_pos = lambda_pos
        self.k_robust = k_robust
        self.eps = eps

        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw
        self.lambda_yaw = lambda_yaw
        self.k_yaw_robust = k_yaw_robust
        self.yaw_eps = yaw_eps

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

        # Position and velocity errors
        ex = x_des - x
        ey = y_des - y

        evx = vx_des - vx
        evy = vy_des - vy

        # Sliding surfaces
        sx = evx + self.lambda_pos * ex
        sy = evy + self.lambda_pos * ey

        # Baseline PD term
        ux_pd = self.kp_pos * ex + self.kd_pos * evx
        uy_pd = self.kp_pos * ey + self.kd_pos * evy

        # Robust sliding-mode term
        ux_robust = self.k_robust * np.tanh(sx / self.eps)
        uy_robust = self.k_robust * np.tanh(sy / self.eps)

        ux = ux_pd + ux_robust
        uy = uy_pd + uy_robust

        # Yaw control
        yaw_error = wrap_angle(yaw_des - yaw)
        yaw_rate_error = yaw_rate_des - yaw_rate

        s_yaw = yaw_rate_error + self.lambda_yaw * yaw_error

        uyaw_pd = self.kp_yaw * yaw_error + self.kd_yaw * yaw_rate_error
        uyaw_robust = self.k_yaw_robust * np.tanh(s_yaw / self.yaw_eps)

        uyaw = uyaw_pd + uyaw_robust

        ux = np.clip(ux, -self.max_force, self.max_force)
        uy = np.clip(uy, -self.max_force, self.max_force)
        uyaw = np.clip(uyaw, -self.max_torque, self.max_torque)

        return np.array([ux, uy, uyaw])