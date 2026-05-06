from pathlib import Path
import sys
import time
import argparse

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from controllers.baseline_pd import BaselinePDController
from trajectories.aggressive_braking import AggressiveBrakingTrajectory


EXPERIMENTS = {
    "sliding": ROOT / "models" / "scene.xml",
    "fixed": ROOT / "models" / "scene_fixed.xml",
}


def run_viewer(model_path, experiment_name):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    controller = BaselinePDController()
    trajectory = AggressiveBrakingTrajectory(speed=0.8, brake_time=2.0)

    duration = 8.0

    print(f"Viewing experiment: {experiment_name}")
    print(f"Model: {model_path}")
    print("Close the MuJoCo window to stop.")

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_wall_time = time.time()

        while viewer.is_running() and data.time < duration:
            step_start = time.time()

            t = data.time

            desired = trajectory.get_desired_state(t)
            ctrl = controller.compute_control(data, desired)

            data.ctrl[:] = ctrl

            # No external disturbance in this experiment.
            data.xfrc_applied[:, :] = 0.0

            mujoco.mj_step(model, data)

            viewer.sync()

            # Run close to real time.
            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("Experiment finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["sliding", "fixed"],
        default="sliding",
        help="Which payload configuration to view.",
    )

    args = parser.parse_args()

    model_path = EXPERIMENTS[args.experiment]
    run_viewer(model_path, args.experiment)


if __name__ == "__main__":
    main()