import numpy as np

from controllers.baseline_pd import wrap_angle


class AdaptivePDController:
    """
    Adaptive PD controller with online disturbance compensation.

    The controller starts with standard PD tracking, then learns an
    additive disturbance estimate in x and y. This is useful for the
    sliding-payload case because the shifting load creates unknown,
    time-varying forces on the base.

    Control form:
        u = u_PD + d_hat

    Adaptation law:
        d_hat_dot = gamma * s - sigma * d_hat

    where:
        s = velocity_error + lambda * position_error

    The leakage term sigma prevents the disturbance estimate from
    growing without bound.
    """

    def __init__(
        self,
        kp_pos=35.0,
        kd_pos=25.0,
        kp_yaw=25.0,
        kd_yaw=6.0,
        lambda_pos=2.0,
        gamma=15.0,
        sigma=1.0,
        max_disturbance_estimate=60.0,
        max_force=120.0,
        max_torque=120.0,
        dt=0.002,
    ):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw

        self.lambda_pos = lambda_pos
        self.gamma = gamma
        self.sigma = sigma

        self.max_disturbance_estimate = max_disturbance_estimate
        self.max_force = max_force
        self.max_torque = max_torque
        self.dt = dt

        # Learned disturbance estimates
        self.d_hat_x = 0.0
        self.d_hat_y = 0.0

    def reset(self):
        self.d_hat_x = 0.0
        self.d_hat_y = 0.0

    def compute_control(self, data, desired_state):
        x, y, yaw = data.qpos[0], data.qpos[1], data.qpos[2]
        vx, vy, yaw_rate = data.qvel[0], data.qvel[1], data.qvel[2]

        x_des = desired_state["x"]
        y_des = desired_state["y"]
        yaw_des = desired_state["yaw"]

        vx_des = desired_state["vx"]
        vy_des = desired_state["vy"]
        yaw_rate_des = desired_state["yaw_rate"]

        # Tracking errors
        ex = x_des - x
        ey = y_des - y

        evx = vx_des - vx
        evy = vy_des - vy

        # Sliding/filter errors for adaptation
        sx = evx + self.lambda_pos * ex
        sy = evy + self.lambda_pos * ey

        # Adaptive disturbance estimate update
        d_hat_x_dot = self.gamma * sx - self.sigma * self.d_hat_x
        d_hat_y_dot = self.gamma * sy - self.sigma * self.d_hat_y

        self.d_hat_x += self.dt * d_hat_x_dot
        self.d_hat_y += self.dt * d_hat_y_dot

        self.d_hat_x = np.clip(
            self.d_hat_x,
            -self.max_disturbance_estimate,
            self.max_disturbance_estimate,
        )
        self.d_hat_y = np.clip(
            self.d_hat_y,
            -self.max_disturbance_estimate,
            self.max_disturbance_estimate,
        )

        # Baseline PD
        ux_pd = self.kp_pos * ex + self.kd_pos * evx
        uy_pd = self.kp_pos * ey + self.kd_pos * evy

        # Adaptive compensation
        ux = ux_pd + self.d_hat_x
        uy = uy_pd + self.d_hat_y

        # Yaw is kept as ordinary PD
        yaw_error = wrap_angle(yaw_des - yaw)
        yaw_rate_error = yaw_rate_des - yaw_rate
        uyaw = self.kp_yaw * yaw_error + self.kd_yaw * yaw_rate_error

        ux = np.clip(ux, -self.max_force, self.max_force)
        uy = np.clip(uy, -self.max_force, self.max_force)
        uyaw = np.clip(uyaw, -self.max_torque, self.max_torque)

        return np.array([ux, uy, uyaw])