from pathlib import Path
import csv
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from controllers.baseline_pd import BaselinePDController, wrap_angle
from trajectories.aggressive_braking import AggressiveBrakingTrajectory


LOG_DIR = ROOT / "logs"

EXPERIMENTS = [
    {
        "name": "sliding_payload",
        "model_path": ROOT / "models" / "scene.xml",
        "log_path": LOG_DIR / "braking_baseline_free_payload.csv",
    },
    {
        "name": "fixed_payload",
        "model_path": ROOT / "models" / "scene_fixed.xml",
        "log_path": LOG_DIR / "braking_baseline_fixed_payload.csv",
    },
]


def get_body_id(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    if body_id == -1:
        raise ValueError(f"Body '{body_name}' not found.")

    return body_id


def get_payload_position_in_base_frame(data, base_body_id, payload_body_id):
    base_pos = data.xpos[base_body_id].copy()
    payload_pos = data.xpos[payload_body_id].copy()

    base_rot = data.xmat[base_body_id].reshape(3, 3)

    payload_rel_world = payload_pos - base_pos
    payload_rel_base = base_rot.T @ payload_rel_world

    return payload_rel_base


def run_single_experiment(model_path, log_path, experiment_name):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    controller = BaselinePDController()
    trajectory = AggressiveBrakingTrajectory(speed=0.8, brake_time=2.0)

    duration = 8.0
    steps = int(duration / model.opt.timestep)

    mujoco.mj_forward(model, data)

    base_body_id = get_body_id(model, "mobile_base")
    payload_body_id = get_body_id(model, "payload")

    base_pos0 = data.xpos[base_body_id].copy()
    payload_pos0 = data.xpos[payload_body_id].copy()

    payload_rel_world0 = payload_pos0 - base_pos0
    payload_distance0 = np.linalg.norm(payload_rel_world0)

    payload_rel_base0 = get_payload_position_in_base_frame(
        data, base_body_id, payload_body_id
    )

    rows = []

    for _ in range(steps):
        t = data.time

        desired = trajectory.get_desired_state(t)
        ctrl = controller.compute_control(data, desired)

        data.ctrl[:] = ctrl

        # No external disturbance in this experiment.
        data.xfrc_applied[:, :] = 0.0
        disturbance_fx = 0.0

        mujoco.mj_step(model, data)

        base_pos = data.xpos[base_body_id].copy()
        payload_pos = data.xpos[payload_body_id].copy()

        payload_rel_world = payload_pos - base_pos
        payload_disp_world = payload_rel_world - payload_rel_world0

        payload_distance = np.linalg.norm(payload_rel_world)
        payload_distance_change = payload_distance - payload_distance0

        payload_rel_base = get_payload_position_in_base_frame(
            data, base_body_id, payload_body_id
        )
        payload_disp_base = payload_rel_base - payload_rel_base0

        x, y, yaw = data.qpos[0], data.qpos[1], data.qpos[2]
        vx, vy, yaw_rate = data.qvel[0], data.qvel[1], data.qvel[2]

        x_error = desired["x"] - x
        y_error = desired["y"] - y

        tracking_error = np.sqrt(x_error**2 + y_error**2)
        yaw_error = wrap_angle(desired["yaw"] - yaw)

        ctrl_x = ctrl[0]
        ctrl_y = ctrl[1]
        ctrl_yaw = ctrl[2]

        control_effort = np.sqrt(ctrl_x**2 + ctrl_y**2)
        total_control_effort = np.sqrt(ctrl_x**2 + ctrl_y**2 + ctrl_yaw**2)

        ctrl_x_saturated = abs(ctrl_x) >= 149.0
        ctrl_y_saturated = abs(ctrl_y) >= 149.0

        rows.append({
            "experiment": experiment_name,
            "time": t,

            "base_x": x,
            "base_y": y,
            "base_yaw": yaw,
            "base_vx": vx,
            "base_vy": vy,
            "base_yaw_rate": yaw_rate,

            "payload_x": payload_pos[0],
            "payload_y": payload_pos[1],
            "payload_z": payload_pos[2],

            "payload_rel_world_x": payload_rel_world[0],
            "payload_rel_world_y": payload_rel_world[1],
            "payload_rel_world_z": payload_rel_world[2],

            "payload_disp_world_x": payload_disp_world[0],
            "payload_disp_world_y": payload_disp_world[1],
            "payload_disp_world_z": payload_disp_world[2],

            "payload_rel_base_x": payload_rel_base[0],
            "payload_rel_base_y": payload_rel_base[1],
            "payload_rel_base_z": payload_rel_base[2],

            "payload_disp_base_x": payload_disp_base[0],
            "payload_disp_base_y": payload_disp_base[1],
            "payload_disp_base_z": payload_disp_base[2],

            "payload_distance": payload_distance,
            "payload_distance_change": payload_distance_change,

            "x_des": desired["x"],
            "y_des": desired["y"],
            "yaw_des": desired["yaw"],

            "x_error": x_error,
            "y_error": y_error,
            "tracking_error": tracking_error,
            "yaw_error": yaw_error,

            "ctrl_x": ctrl_x,
            "ctrl_y": ctrl_y,
            "ctrl_yaw": ctrl_yaw,
            "control_effort": control_effort,
            "total_control_effort": total_control_effort,

            "ctrl_x_saturated": ctrl_x_saturated,
            "ctrl_y_saturated": ctrl_y_saturated,

            "disturbance_fx": disturbance_fx,
        })

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {experiment_name} log to: {log_path}")


def main():
    LOG_DIR.mkdir(exist_ok=True)

    for exp in EXPERIMENTS:
        run_single_experiment(
            model_path=exp["model_path"],
            log_path=exp["log_path"],
            experiment_name=exp["name"],
        )


if __name__ == "__main__":
    main()