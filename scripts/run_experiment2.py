from pathlib import Path
import csv
import sys

import mujoco
import numpy as np

# Allow imports from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from controllers.baseline_pd import BaselinePDController, wrap_angle
from trajectories.aggressive_braking import AggressiveBrakingTrajectory


LOG_DIR = ROOT / "logs"

EXPERIMENTS = [
    {
        "name": "free_payload",
        "model_path": ROOT / "models" / "scene.xml",
        "log_path": LOG_DIR / "braking_free_payload.csv",
    },
    {
        "name": "fixed_payload",
        "model_path": ROOT / "models" / "scene_fixed.xml",
        "log_path": LOG_DIR / "braking_fixed_payload.csv",
    },
]


def get_body_position(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    if body_id == -1:
        raise ValueError(f"Body '{body_name}' not found in model.")

    return data.xpos[body_id].copy()


def run_single_experiment(model_path, log_path, experiment_name):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    controller = BaselinePDController()
    trajectory = AggressiveBrakingTrajectory()

    duration = 8.0
    dt = model.opt.timestep
    steps = int(duration / dt)

    rows = []

    for _ in range(steps):
        t = data.time

        desired = trajectory.get_desired_state(t)
        ctrl = controller.compute_control(data, desired)
        data.ctrl[:] = ctrl

        mujoco.mj_step(model, data)

        base_pos = get_body_position(model, data, "mobile_base")
        payload_pos = get_body_position(model, data, "payload")
        payload_rel = payload_pos - base_pos

        x, y, yaw = data.qpos[0], data.qpos[1], data.qpos[2]
        vx, vy, yaw_rate = data.qvel[0], data.qvel[1], data.qvel[2]

        tracking_error = np.sqrt((desired["x"] - x) ** 2 + (desired["y"] - y) ** 2)
        yaw_error = wrap_angle(desired["yaw"] - yaw)

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

            "payload_rel_x": payload_rel[0],
            "payload_rel_y": payload_rel[1],
            "payload_rel_z": payload_rel[2],

            "x_des": desired["x"],
            "y_des": desired["y"],
            "yaw_des": desired["yaw"],

            "tracking_error": tracking_error,
            "yaw_error": yaw_error,

            "ctrl_x": ctrl[0],
            "ctrl_y": ctrl[1],
            "ctrl_yaw": ctrl[2],
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