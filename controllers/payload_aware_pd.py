import numpy as np

from controllers.baseline_pd import wrap_angle


class PayloadAwarePDController:
    """
    PD trajectory tracker with payload-motion feedback.

    The baseline PD controller only tracks the mobile base.
    This controller adds feedback from the payload displacement
    relative to the base, so it can react when the internal load shifts.

    It assumes the payload has a sliding joint named payload_slide_x.
    """

    def __init__(
        self,
        kp_pos=40.0,
        kd_pos=35.0,
        kp_yaw=25.0,
        kd_yaw=6.0,
        kp_payload=80.0,
        kd_payload=20.0,
        max_force=120.0,
        max_torque=120.0,
    ):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw

        # Payload compensation gains
        self.kp_payload = kp_payload
        self.kd_payload = kd_payload

        self.max_force = max_force
        self.max_torque = max_torque

        self.payload_joint_id = None
        self.payload_qpos_addr = None
        self.payload_qvel_addr = None

    def _initialize_payload_joint(self, model):
        """
        Finds the payload sliding joint once and stores its qpos/qvel indices.
        """
        if self.payload_joint_id is not None:
            return

        joint_id = None

        for j in range(model.njnt):
            name = model.joint(j).name
            if name == "payload_slide_x":
                joint_id = j
                break

        if joint_id is None:
            raise ValueError(
                "Could not find joint named 'payload_slide_x'. "
                "Make sure this controller is used with the sliding-payload model."
            )

        self.payload_joint_id = joint_id
        self.payload_qpos_addr = model.jnt_qposadr[joint_id]
        self.payload_qvel_addr = model.jnt_dofadr[joint_id]

    def compute_control(self, model, data, desired_state):
        self._initialize_payload_joint(model)

        x, y, yaw = data.qpos[0], data.qpos[1], data.qpos[2]
        vx, vy, yaw_rate = data.qvel[0], data.qvel[1], data.qvel[2]

        x_des = desired_state["x"]
        y_des = desired_state["y"]
        yaw_des = desired_state["yaw"]

        vx_des = desired_state["vx"]
        vy_des = desired_state["vy"]
        yaw_rate_des = desired_state["yaw_rate"]

        # Standard base PD control
        ux = self.kp_pos * (x_des - x) + self.kd_pos * (vx_des - vx)
        uy = self.kp_pos * (y_des - y) + self.kd_pos * (vy_des - vy)

        yaw_error = wrap_angle(yaw_des - yaw)
        uyaw = self.kp_yaw * yaw_error + self.kd_yaw * (yaw_rate_des - yaw_rate)

        # Payload state from the sliding joint.
        # payload_disp_x > 0 means payload has shifted forward relative to base.
        payload_disp_x = data.qpos[self.payload_qpos_addr]
        payload_vel_x = data.qvel[self.payload_qvel_addr]

        # Compensation term.
        # If payload shifts forward, reduce forward control / increase braking.
        # If payload shifts backward, add forward control.
        payload_comp = (
            -self.kp_payload * payload_disp_x
            -self.kd_payload * payload_vel_x
        )

        ux += payload_comp

        ux = np.clip(ux, -self.max_force, self.max_force)
        uy = np.clip(uy, -self.max_force, self.max_force)
        uyaw = np.clip(uyaw, -self.max_torque, self.max_torque)

        return np.array([ux, uy, uyaw])