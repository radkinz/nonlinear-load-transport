import time
from pathlib import Path

import mujoco
import mujoco.viewer


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "scene.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    while viewer.is_running():
        t = time.time() - start

        # Simple test motion: accelerate forward, then turn
        if t < 1.0:
            data.ctrl[0] = 80.0
            data.ctrl[1] = 0.0
            data.ctrl[2] = 0.0
        elif t < 2.0:
            data.ctrl[0] = -80.0
            data.ctrl[1] = 0.0
            data.ctrl[2] = 0.0
        elif t < 4.0:
            data.ctrl[0] = 0.0
            data.ctrl[1] = 0.0
            data.ctrl[2] = 25.0
        else:
            data.ctrl[:] = 0.0

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)