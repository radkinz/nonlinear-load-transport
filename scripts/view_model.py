import time
from pathlib import Path

import mujoco
import mujoco.viewer


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "scene.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

BASE_BODY_NAME = "mobile_base"
base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY_NAME)

if base_body_id == -1:
    raise ValueError(f"Body '{BASE_BODY_NAME}' not found.")

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    while viewer.is_running():
        t = time.time() - start

        if t < 2.0:
            data.ctrl[0] = 120.0
            data.ctrl[1] = 0.0
            data.ctrl[2] = 0.0
        elif t < 3.0:
            data.ctrl[0] = -150.0
            data.ctrl[1] = 0.0
            data.ctrl[2] = 0.0
        else:
            data.ctrl[:] = 0.0

        mujoco.mj_step(model, data)

        base_pos = data.xpos[base_body_id].copy()

        # Camera follows the robot
        viewer.cam.lookat[:] = base_pos
        viewer.cam.lookat[2] = 0.45
        viewer.cam.distance = 2.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

        # Move first light to follow above the robot, if model has lights
        if model.nlight > 0:
            model.light_pos[0] = base_pos + [0.0, 0.0, 3.0]

        viewer.sync()
        time.sleep(model.opt.timestep)