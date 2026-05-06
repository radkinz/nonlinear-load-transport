from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from controllers.baseline_pd import BaselinePDController
from trajectories.aggressive_braking import AggressiveBrakingTrajectory


# Choose model
USE_FIXED_PAYLOAD = False

if USE_FIXED_PAYLOAD:
    MODEL_PATH = ROOT / "models" / "scene_fixed.xml"
else:
    MODEL_PATH = ROOT / "models" / "scene.xml"


def get_body_position(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    controller = BaselinePDController()
    trajectory = AggressiveBrakingTrajectory()

    duration = 8.0

    print(f"Launching MuJoCo viewer with model: {MODEL_PATH}")

    # Initial payload displacement reference
    mujoco.mj_forward(model, data)
    base_pos0 = get_body_position(model, data, "mobile_base")
    payload_pos0 = get_body_position(model, data, "payload")
    payload_rel0 = payload_pos0 - base_pos0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_wall = time.time()

        while viewer.is_running():
            sim_time = data.time

            if sim_time > duration:
                break

            desired = trajectory.get_desired_state(sim_time)

            ctrl = controller.compute_control(data, desired)
            data.ctrl[:] = ctrl

            mujoco.mj_step(model, data)

            base_pos = get_body_position(model, data, "mobile_base")
            payload_pos = get_body_position(model, data, "payload")
            payload_rel = payload_pos - base_pos
            payload_disp = payload_rel - payload_rel0

            if int(sim_time * 100) % 20 == 0:
                print(
                    f"time={sim_time:.2f}  "
                    f"payload_dx={payload_disp[0]:.4f}"
                )

            viewer.sync()

            elapsed_wall = time.time() - start_wall
            sleep_time = sim_time - elapsed_wall
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("Experiment finished.")


if __name__ == "__main__":
    main()