from pathlib import Path
import csv
import sys
import copy

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.payload_config import CONFIG
from utils.generate_payload_models import generate_all_models

from controllers.baseline_pd import BaselinePDController, wrap_angle
from trajectories.aggressive_braking import AggressiveBrakingTrajectory


LOG_DIR = ROOT / "logs" / "baseline_payload_comparison"
LOG_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    {
        "name": "sliding_payload",
        "model_path": ROOT / "models" / "scene.xml",
        "log_path": LOG_DIR / "baseline_sliding_payload.csv",
    },
    {
        "name": "fixed_payload",
        "model_path": ROOT / "models" / "scene_fixed.xml",
        "log_path": LOG_DIR / "baseline_fixed_payload.csv",
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


def compute_settling_time(times, errors, threshold=0.05):
    times = np.asarray(times)
    errors = np.asarray(errors)

    for i in range(len(errors)):
        if np.all(errors[i:] < threshold):
            return times[i]

    return np.nan


def run_single_experiment(model_path, log_path, experiment_name):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    controller = BaselinePDController(
        kp_pos=40.0,
        kd_pos=35.0,
        kp_yaw=25.0,
        kd_yaw=6.0,
        max_force=150.0,
        max_torque=60.0,
    )

    trajectory = AggressiveBrakingTrajectory(
        speed=0.8,
        brake_time=2.0,
    )

    duration = 15.0
    steps = int(duration / model.opt.timestep)

    mujoco.mj_forward(model, data)

    base_body_id = get_body_id(model, "mobile_base")
    payload_body_id = get_body_id(model, "payload")

    base_pos0 = data.xpos[base_body_id].copy()
    payload_pos0 = data.xpos[payload_body_id].copy()

    payload_rel_world0 = payload_pos0 - base_pos0
    payload_distance0 = np.linalg.norm(payload_rel_world0)

    payload_rel_base0 = get_payload_position_in_base_frame(
        data,
        base_body_id,
        payload_body_id,
    )

    rows = []

    for _ in range(steps):
        t = data.time

        desired = trajectory.get_desired_state(t)
        ctrl = controller.compute_control(data, desired)

        data.ctrl[:] = ctrl
        data.xfrc_applied[:, :] = 0.0

        mujoco.mj_step(model, data)

        base_pos = data.xpos[base_body_id].copy()
        payload_pos = data.xpos[payload_body_id].copy()

        payload_rel_world = payload_pos - base_pos
        payload_disp_world = payload_rel_world - payload_rel_world0

        payload_distance = np.linalg.norm(payload_rel_world)
        payload_distance_change = payload_distance - payload_distance0

        payload_rel_base = get_payload_position_in_base_frame(
            data,
            base_body_id,
            payload_body_id,
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
        })

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    tracking_errors = np.array([r["tracking_error"] for r in rows])
    control_efforts = np.array([r["control_effort"] for r in rows])
    payload_disps = np.array([r["payload_disp_base_x"] for r in rows])
    times = np.array([r["time"] for r in rows])

    rms_error = float(np.sqrt(np.mean(tracking_errors**2)))
    max_error = float(np.max(tracking_errors))
    max_payload_disp = float(np.max(np.abs(payload_disps)))
    mean_control_effort = float(np.mean(control_efforts))
    settling_time = float(compute_settling_time(times, tracking_errors, threshold=0.05))

    print(
        f"{experiment_name} | "
        f"RMS error={rms_error:.4f} | "
        f"max error={max_error:.4f} | "
        f"max payload disp={max_payload_disp:.4f} | "
        f"mean effort={mean_control_effort:.2f} | "
        f"settling={settling_time:.2f}"
    )


def main():
    config = copy.deepcopy(CONFIG)

    # Baseline demonstration parameters.
    # Keep these consistent with the rest of the paper unless you explicitly
    # state otherwise.
    config["payload_mass"] = 20.0
    config["slide_range_min"] = -0.15
    config["slide_range_max"] = 0.15

    generate_all_models(config)

    for exp in EXPERIMENTS:
        run_single_experiment(
            model_path=exp["model_path"],
            log_path=exp["log_path"],
            experiment_name=exp["name"],
        )


if __name__ == "__main__":
    main()