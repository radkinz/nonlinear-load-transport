from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

FREE_LOG = ROOT / "logs" / "s_curve_free_payload.csv"
FIXED_LOG = ROOT / "logs" / "s_curve_fixed_payload.csv"

PLOT_DIR = ROOT / "plots" / "s_curve_comparison"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

free_df = pd.read_csv(FREE_LOG)
fixed_df = pd.read_csv(FIXED_LOG)


# ---------------------------------------------------
# 1. Top-down XY trajectory
# ---------------------------------------------------

plt.figure()

plt.plot(
    free_df["x_des"],
    free_df["y_des"],
    label="Desired",
    linewidth=3,
)

plt.plot(
    free_df["base_x"],
    free_df["base_y"],
    label="Free payload",
)

plt.plot(
    fixed_df["base_x"],
    fixed_df["base_y"],
    label="Fixed payload",
)

plt.xlabel("x Position [m]")
plt.ylabel("y Position [m]")
plt.title("S-Curve Trajectory Tracking")
plt.axis("equal")
plt.legend()
plt.grid(True)

plt.savefig(
    PLOT_DIR / "s_curve_xy_tracking.png",
    dpi=200,
    bbox_inches="tight",
)


# ---------------------------------------------------
# 2. Lateral tracking over time
# ---------------------------------------------------

plt.figure()

plt.plot(
    free_df["time"],
    free_df["y_des"],
    label="Desired y",
    linewidth=3,
)

plt.plot(
    free_df["time"],
    free_df["base_y"],
    label="Free payload",
)

plt.plot(
    fixed_df["time"],
    fixed_df["base_y"],
    label="Fixed payload",
)

plt.xlabel("Time [s]")
plt.ylabel("y Position [m]")
plt.title("S-Curve Lateral Tracking")
plt.legend()
plt.grid(True)

plt.savefig(
    PLOT_DIR / "s_curve_lateral_tracking.png",
    dpi=200,
    bbox_inches="tight",
)


# ---------------------------------------------------
# 3. Tracking error comparison
# ---------------------------------------------------

plt.figure()

plt.plot(
    free_df["time"],
    free_df["tracking_error"],
    label="Free payload",
)

plt.plot(
    fixed_df["time"],
    fixed_df["tracking_error"],
    label="Fixed payload",
)

plt.xlabel("Time [s]")
plt.ylabel("Tracking Error [m]")
plt.title("S-Curve Tracking Error")
plt.legend()
plt.grid(True)

plt.savefig(
    PLOT_DIR / "s_curve_tracking_error.png",
    dpi=200,
    bbox_inches="tight",
)


# ---------------------------------------------------
# 4. Payload lateral displacement
# ---------------------------------------------------

free_payload_dy = free_df["payload_disp_y"]

# For the fixed payload, show ideal rigid baseline
fixed_payload_dy = np.zeros(len(fixed_df))

plt.figure()

plt.plot(
    free_df["time"],
    free_payload_dy,
    label="Free payload",
)

plt.plot(
    fixed_df["time"],
    fixed_payload_dy,
    label="Fixed payload",
)

plt.axhline(0.0, linestyle="--", linewidth=1)

plt.xlabel("Time [s]")
plt.ylabel("Payload y Displacement [m]")
plt.title("Payload Lateral Displacement During S-Curve")
plt.legend()
plt.grid(True)

plt.savefig(
    PLOT_DIR / "s_curve_payload_lateral_displacement.png",
    dpi=200,
    bbox_inches="tight",
)


# ---------------------------------------------------
# 5. Payload longitudinal + lateral displacement
# ---------------------------------------------------

plt.figure()

plt.plot(
    free_df["time"],
    free_df["payload_disp_x"],
    label="Payload x displacement",
)

plt.plot(
    free_df["time"],
    free_df["payload_disp_y"],
    label="Payload y displacement",
)

plt.xlabel("Time [s]")
plt.ylabel("Payload Displacement [m]")
plt.title("Free Payload Motion During S-Curve")
plt.legend()
plt.grid(True)

plt.savefig(
    PLOT_DIR / "s_curve_free_payload_motion.png",
    dpi=200,
    bbox_inches="tight",
)


# ---------------------------------------------------
# 6. Yaw error comparison
# ---------------------------------------------------

plt.figure()

plt.plot(
    free_df["time"],
    free_df["yaw_error"],
    label="Free payload",
)

plt.plot(
    fixed_df["time"],
    fixed_df["yaw_error"],
    label="Fixed payload",
)

plt.xlabel("Time [s]")
plt.ylabel("Yaw Error [rad]")
plt.title("S-Curve Yaw Error")
plt.legend()
plt.grid(True)

plt.savefig(
    PLOT_DIR / "s_curve_yaw_error.png",
    dpi=200,
    bbox_inches="tight",
)


# ---------------------------------------------------
# 7. Lateral control effort
# ---------------------------------------------------

plt.figure()

plt.plot(
    free_df["time"],
    free_df["ctrl_y"],
    label="Free payload",
)

plt.plot(
    fixed_df["time"],
    fixed_df["ctrl_y"],
    label="Fixed payload",
)

plt.xlabel("Time [s]")
plt.ylabel("Control y Input")
plt.title("S-Curve Lateral Control Effort")
plt.legend()
plt.grid(True)

plt.savefig(
    PLOT_DIR / "s_curve_lateral_control_effort.png",
    dpi=200,
    bbox_inches="tight",
)


# ---------------------------------------------------
# 8. Yaw control effort
# ---------------------------------------------------

plt.figure()

plt.plot(
    free_df["time"],
    free_df["ctrl_yaw"],
    label="Free payload",
)

plt.plot(
    fixed_df["time"],
    fixed_df["ctrl_yaw"],
    label="Fixed payload",
)

plt.xlabel("Time [s]")
plt.ylabel("Yaw Control Input")
plt.title("S-Curve Yaw Control Effort")
plt.legend()
plt.grid(True)

plt.savefig(
    PLOT_DIR / "s_curve_yaw_control_effort.png",
    dpi=200,
    bbox_inches="tight",
)


print(f"Saved S-curve plots to: {PLOT_DIR}")