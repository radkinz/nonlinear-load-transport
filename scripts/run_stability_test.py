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
from controllers.payload_aware_pd import PayloadAwarePDController
from controllers.sliding_mode import SlidingModeController
from controllers.adaptive_pd import AdaptivePDController
from trajectories.aggressive_braking import AggressiveBrakingTrajectory


LOG_DIR = ROOT / "logs" / "stability_test"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = LOG_DIR / "initial_offset_stability_summary.csv"

MODEL_PATH = ROOT / "models" / "scene.xml"


INITIAL_PAYLOAD_OFFSETS = [-0.15, -0.10, 0.00, 0.10, 0.15]


CONTROLLERS = [
    {
        "name": "baseline_pd",
        "controller_factory": lambda: BaselinePDController(
            kp_pos=40.0,
            kd_pos=35.0,
            kp_yaw=25.0,
            kd_yaw=6.0,
            max_force=150.0,
            max_torque=60.0,
        ),
        "requires_model": False,
    },
    {
        "name": "payload_aware_pd",
        "controller_factory": lambda: PayloadAwarePDController(
            kp_pos=38.0,
            kd_pos=34.0,
            kp_payload=55.0,
            kd_payload=20.0,
            kp_yaw=25.0,
            kd_yaw=6.0,
            max_force=150.0,
            max_torque=60.0,
        ),
        "requires_model": True,
    },
    {
        "name": "sliding_mode",
        "controller_factory": lambda: SlidingModeController(
            kp_pos=30.0,
            kd_pos=20.0,
            lambda_pos=2.5,
            k_robust=80.0,
            eps=0.08,
            kp_yaw=25.0,
            kd_yaw=6.0,
            max_force=150.0,
            max_torque=60.0,
        ),
        "requires_model": False,
    },
    {
        "name": "adaptive_pd",
        "controller_factory": lambda: AdaptivePDController(
            kp_pos=45.0,
            kd_pos=40.0,
            lambda_pos=3.0,
            gamma=2.0,
            sigma=8.0,
            max_disturbance_estimate=15.0,
            kp_yaw=25.0,
            kd_yaw=6.0,
            max_force=150.0,
            max_torque=60.0,
        ),
        "requires_model": False,
    },
]


def get_body_id(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body '{body_name}' not found.")
    return body_id


def get_joint_qpos_addr(model, joint_name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id == -1:
        raise ValueError(f"Joint '{joint_name}' not found.")
    return model.jnt_qposadr[joint_id]


def get_payload_position_in_base_frame(data, base_body_id, payload_body_id):
    base_pos = data.xpos[base_body_id].copy()
    payload_pos = data.xpos[payload_body_id].copy()
    base_rot = data.xmat[base_body_id].reshape(3, 3)

    payload_rel_world = payload_pos - base_pos
    payload_rel_base = base_rot.T @ payload_rel_world

    return payload_rel_base


def compute_control(controller, requires_model, model, data, desired):
    if requires_model:
        return controller.compute_control(model, data, desired)
    return controller.compute_control(data, desired)


def compute_settling_time(times, errors, threshold=0.05):
    times = np.asarray(times)
    errors = np.asarray(errors)

    for i in range(len(errors)):
        if np.all(errors[i:] < threshold):
            return times[i]

    return np.nan


def run_trial(initial_offset, controller_entry):
    controller_name = controller_entry["name"]
    controller = controller_entry["controller_factory"]()

    if hasattr(controller, "reset"):
        controller.reset()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    payload_slide_qpos_addr = get_joint_qpos_addr(model, "payload_slide_x")
    data.qpos[payload_slide_qpos_addr] = initial_offset

    trajectory = AggressiveBrakingTrajectory(
        speed=0.8,
        brake_time=2.0,
    )

    duration = 30.0
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

        ctrl = compute_control(
            controller,
            controller_entry["requires_model"],
            model,
            data,
            desired,
        )

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

        d_hat_x = getattr(controller, "d_hat_x", 0.0)
        d_hat_y = getattr(controller, "d_hat_y", 0.0)

        rows.append({
            "initial_payload_offset": initial_offset,
            "controller": controller_name,
            "time": t,

            "base_x": x,
            "base_y": y,
            "base_yaw": yaw,
            "base_vx": vx,
            "base_vy": vy,
            "base_yaw_rate": yaw_rate,

            "tracking_error": tracking_error,
            "x_error": x_error,
            "y_error": y_error,
            "yaw_error": yaw_error,

            "ctrl_x": ctrl_x,
            "ctrl_y": ctrl_y,
            "ctrl_yaw": ctrl_yaw,
            "control_effort": control_effort,
            "total_control_effort": total_control_effort,

            "payload_disp_base_x": payload_disp_base[0],
            "payload_disp_base_y": payload_disp_base[1],
            "payload_distance_change": payload_distance_change,

            "d_hat_x": d_hat_x,
            "d_hat_y": d_hat_y,
        })

    safe_offset = str(initial_offset).replace("-", "neg_").replace(".", "p")
    log_path = LOG_DIR / f"{controller_name}_offset_{safe_offset}.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    times = np.array([r["time"] for r in rows])
    tracking_errors = np.array([r["tracking_error"] for r in rows])
    control_efforts = np.array([r["control_effort"] for r in rows])
    payload_disps = np.array([r["payload_disp_base_x"] for r in rows])

    final_tracking_error = float(tracking_errors[-1])
    final_payload_disp = float(payload_disps[-1])

    bounded_tracking_error = bool(np.all(tracking_errors < 2.0))
    bounded_payload_motion = bool(np.all(np.abs(payload_disps) < 0.30))

    summary = {
        "initial_payload_offset": initial_offset,
        "controller": controller_name,
        "rms_tracking_error": float(np.sqrt(np.mean(tracking_errors**2))),
        "max_tracking_error": float(np.max(tracking_errors)),
        "final_tracking_error": final_tracking_error,
        "mean_control_effort": float(np.mean(control_efforts)),
        "max_control_effort": float(np.max(control_efforts)),
        "max_payload_disp_base_x": float(np.max(np.abs(payload_disps))),
        "final_payload_disp_base_x": final_payload_disp,
        "settling_time_0p05": float(
            compute_settling_time(times, tracking_errors, threshold=0.05)
        ),
        "bounded_tracking_error": bounded_tracking_error,
        "bounded_payload_motion": bounded_payload_motion,
        "trial_log": str(log_path),
    }

    print(
        f"offset={initial_offset:+.2f} m | "
        f"controller={controller_name} | "
        f"final error={final_tracking_error:.4f} | "
        f"RMS={summary['rms_tracking_error']:.4f}"
    )

    return summary


def main():
    config = copy.deepcopy(CONFIG)

    config["payload_mass"] = 20.0
    config["slide_range_min"] = -0.15
    config["slide_range_max"] = 0.15

    generate_all_models(config)

    summaries = []

    for initial_offset in INITIAL_PAYLOAD_OFFSETS:
        for controller_entry in CONTROLLERS:
            summary = run_trial(initial_offset, controller_entry)
            summaries.append(summary)

    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nSaved stability summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()