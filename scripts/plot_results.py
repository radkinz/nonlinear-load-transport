from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

FREE_LOG = ROOT / "logs" / "braking_free_payload.csv"
FIXED_LOG = ROOT / "logs" / "braking_fixed_payload.csv"

PLOT_DIR = ROOT / "plots" / "braking_comparison"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

free_df = pd.read_csv(FREE_LOG)
fixed_df = pd.read_csv(FIXED_LOG)

# ---------------------------------------------------
# 1. X position tracking
# ---------------------------------------------------

plt.figure()

plt.plot(
    free_df["time"],
    free_df["x_des"],
    label="Desired x",
    linewidth=3,
)

plt.plot(
    free_df["time"],
    free_df["base_x"],
    label="Free payload",
)

plt.plot(
    fixed_df["time"],
    fixed_df["base_x"],
    label="Fixed payload",
)

plt.xlabel("Time [s]")
plt.ylabel("x Position [m]")
plt.title("Braking Experiment: Longitudinal Tracking")
plt.legend()
plt.grid(True)

plt.savefig(
    PLOT_DIR / "x_tracking_comparison.png",
    dpi=200,
    bbox_inches="tight",
)

# ---------------------------------------------------
# 2. Tracking error comparison
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
plt.title("Tracking Error Comparison")
plt.legend()
plt.grid(True)

plt.savefig(
    PLOT_DIR / "tracking_error_comparison.png",
    dpi=200,
    bbox_inches="tight",
)

# ---------------------------------------------------
# 3. Payload relative motion
# ---------------------------------------------------

plt.figure()

plt.plot(
    free_df["time"],
    free_df["payload_rel_x"],
    label="Free payload",
)

plt.plot(
    fixed_df["time"],
    fixed_df["payload_rel_x"],
    label="Fixed payload",
)

plt.xlabel("Time [s]")
plt.ylabel("Relative Payload x Position [m]")
plt.title("Payload Motion During Braking")
plt.legend()
plt.grid(True)

plt.savefig(
    PLOT_DIR / "payload_motion_comparison.png",
    dpi=200,
    bbox_inches="tight",
)

# ---------------------------------------------------
# 4. Control effort comparison
# ---------------------------------------------------

plt.figure()

plt.plot(
    free_df["time"],
    free_df["ctrl_x"],
    label="Free payload",
)

plt.plot(
    fixed_df["time"],
    fixed_df["ctrl_x"],
    label="Fixed payload",
)

plt.xlabel("Time [s]")
plt.ylabel("Control Input")
plt.title("Longitudinal Control Effort")
plt.legend()
plt.grid(True)

plt.savefig(
    PLOT_DIR / "control_effort_comparison.png",
    dpi=200,
    bbox_inches="tight",
)

print(f"Saved plots to: {PLOT_DIR}")

# ---------------------------------------------------
# 3. Payload displacement from initial position
# ---------------------------------------------------

free_payload_dx = free_df["payload_rel_x"] - free_df["payload_rel_x"].iloc[0]
fixed_payload_dx = np.zeros(len(fixed_df))

plt.figure()

plt.plot(
    free_df["time"],
    free_payload_dx,
    label="Free payload",
)

plt.plot(
    fixed_df["time"],
    fixed_payload_dx,
    label="Fixed payload",
)

plt.axhline(0.0, linestyle="--", linewidth=1)

plt.xlabel("Time [s]")
plt.ylabel("Payload x Displacement [m]")
plt.title("Payload Displacement During Braking")
plt.legend()
plt.grid(True)

plt.savefig(
    PLOT_DIR / "payload_displacement_comparison.png",
    dpi=200,
    bbox_inches="tight",
)