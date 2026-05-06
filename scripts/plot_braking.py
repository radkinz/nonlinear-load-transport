from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

FREE_LOG = ROOT / "logs" / "braking_baseline_free_payload.csv"
FIXED_LOG = ROOT / "logs" / "braking_baseline_fixed_payload.csv"

PLOT_DIR = ROOT / "plots" / "braking_baseline_compare"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

free_df = pd.read_csv(FREE_LOG)
fixed_df = pd.read_csv(FIXED_LOG)

BRAKE_START = 1.5
BRAKE_END = 3.0


def add_braking_region():
    plt.axvspan(
        BRAKE_START,
        BRAKE_END,
        alpha=0.2,
        label="Braking window",
    )


def save_plot(filename):
    plt.legend()
    plt.grid(True)
    plt.savefig(
        PLOT_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------
# Longitudinal Tracking
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    free_df["time"],
    free_df["x_des"],
    label="Desired x",
    linewidth=4,
)

plt.plot(
    free_df["time"],
    free_df["base_x"],
    label="Sliding payload",
    linewidth=2,
)

plt.plot(
    fixed_df["time"],
    fixed_df["base_x"],
    label="Fixed payload",
    linewidth=2,
)

add_braking_region()

plt.xlabel("Time [s]")
plt.ylabel("x Position [m]")
plt.title("Longitudinal Tracking During Braking")
save_plot("x_tracking_compare.png")


# ---------------------------------------------------
# Tracking Error
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    free_df["time"],
    free_df["tracking_error"],
    label="Sliding payload",
    linewidth=2,
)

plt.plot(
    fixed_df["time"],
    fixed_df["tracking_error"],
    label="Fixed payload",
    linewidth=2,
)

add_braking_region()

plt.xlabel("Time [s]")
plt.ylabel("Tracking Error [m]")
plt.title("Tracking Error During Braking")
save_plot("tracking_error_compare.png")


# ---------------------------------------------------
# Zoomed Tracking Error Around Braking
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    free_df["time"],
    free_df["tracking_error"],
    label="Sliding payload",
    linewidth=2,
)

plt.plot(
    fixed_df["time"],
    fixed_df["tracking_error"],
    label="Fixed payload",
    linewidth=2,
)

add_braking_region()

plt.xlim(1.0, 4.0)

plt.xlabel("Time [s]")
plt.ylabel("Tracking Error [m]")
plt.title("Tracking Error Around Braking")
save_plot("tracking_error_zoomed.png")


# ---------------------------------------------------
# X Error
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    free_df["time"],
    free_df["x_error"],
    label="Sliding payload",
    linewidth=2,
)

plt.plot(
    fixed_df["time"],
    fixed_df["x_error"],
    label="Fixed payload",
    linewidth=2,
)

add_braking_region()

plt.xlabel("Time [s]")
plt.ylabel("x Error [m]")
plt.title("Longitudinal Tracking Error")
save_plot("x_error_compare.png")


# ---------------------------------------------------
# Payload Motion in Base Frame
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    free_df["time"],
    free_df["payload_disp_base_x"],
    label="Sliding payload",
    linewidth=2,
)

plt.plot(
    fixed_df["time"],
    fixed_df["payload_disp_base_x"],
    label="Fixed payload",
    linewidth=2,
)

add_braking_region()

plt.axhline(
    0.0,
    linestyle="--",
    linewidth=1,
)

plt.xlabel("Time [s]")
plt.ylabel("Payload Displacement in Base Frame [m]")
plt.title("Internal Payload Motion")
save_plot("payload_disp_base_x_compare.png")


# ---------------------------------------------------
# Payload Distance Change
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    free_df["time"],
    free_df["payload_distance_change"],
    label="Sliding payload",
    linewidth=2,
)

plt.plot(
    fixed_df["time"],
    fixed_df["payload_distance_change"],
    label="Fixed payload",
    linewidth=2,
)

add_braking_region()

plt.axhline(
    0.0,
    linestyle="--",
    linewidth=1,
)

plt.xlabel("Time [s]")
plt.ylabel("Change in Payload-Base Distance [m]")
plt.title("Payload-Base Distance Change")
save_plot("payload_distance_change_compare.png")


# ---------------------------------------------------
# Control Effort Magnitude
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    free_df["time"],
    free_df["control_effort"],
    label="Sliding payload",
    linewidth=2,
)

plt.plot(
    fixed_df["time"],
    fixed_df["control_effort"],
    label="Fixed payload",
    linewidth=2,
)

add_braking_region()

plt.xlabel("Time [s]")
plt.ylabel("Control Effort Magnitude")
plt.title("Control Effort During Braking")
save_plot("control_effort_compare.png")


# ---------------------------------------------------
# Longitudinal Control Input
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    free_df["time"],
    free_df["ctrl_x"],
    label="Sliding payload",
    linewidth=2,
)

plt.plot(
    fixed_df["time"],
    fixed_df["ctrl_x"],
    label="Fixed payload",
    linewidth=2,
)

add_braking_region()

plt.xlabel("Time [s]")
plt.ylabel("Longitudinal Control Input")
plt.title("Longitudinal Control Response")
save_plot("ctrl_x_compare.png")


print(f"Saved plots to: {PLOT_DIR}")