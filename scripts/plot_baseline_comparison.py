from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = ROOT / "logs" / "baseline_payload_comparison"
PLOT_DIR = ROOT / "plots" / "baseline_payload_comparison"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SLIDING_LOG = LOG_DIR / "baseline_sliding_payload.csv"
FIXED_LOG = LOG_DIR / "baseline_fixed_payload.csv"

BRAKE_START = 0.0
BRAKE_END = 2.0


def load_logs():
    if not SLIDING_LOG.exists():
        raise FileNotFoundError(f"Missing log file: {SLIDING_LOG}")

    if not FIXED_LOG.exists():
        raise FileNotFoundError(f"Missing log file: {FIXED_LOG}")

    sliding_df = pd.read_csv(SLIDING_LOG)
    fixed_df = pd.read_csv(FIXED_LOG)

    return sliding_df, fixed_df


def add_braking_region():
    plt.axvspan(
        BRAKE_START,
        BRAKE_END,
        alpha=0.15,
        label="Braking transient",
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


def plot_longitudinal_tracking(sliding_df, fixed_df):
    plt.figure(figsize=(8, 5))

    plt.plot(
        sliding_df["time"],
        sliding_df["x_des"],
        label="Desired x",
        linewidth=4,
    )

    plt.plot(
        sliding_df["time"],
        sliding_df["base_x"],
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
    plt.title("Baseline PD: Fixed vs Sliding Payload Tracking")
    save_plot("baseline_x_tracking_fixed_vs_sliding.png")


def plot_tracking_error(sliding_df, fixed_df):
    plt.figure(figsize=(8, 5))

    plt.plot(
        sliding_df["time"],
        sliding_df["tracking_error"],
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
    plt.title("Baseline PD: Tracking Error")
    save_plot("baseline_tracking_error_fixed_vs_sliding.png")


def plot_tracking_error_zoomed(sliding_df, fixed_df):
    plt.figure(figsize=(8, 5))

    plt.plot(
        sliding_df["time"],
        sliding_df["tracking_error"],
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
    plt.xlim(0.0, 5.0)

    plt.xlabel("Time [s]")
    plt.ylabel("Tracking Error [m]")
    plt.title("Baseline PD: Tracking Error Around Braking")
    save_plot("baseline_tracking_error_zoomed_fixed_vs_sliding.png")


def plot_x_error(sliding_df, fixed_df):
    plt.figure(figsize=(8, 5))

    plt.plot(
        sliding_df["time"],
        sliding_df["x_error"],
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
    plt.title("Baseline PD: Longitudinal Error")
    save_plot("baseline_x_error_fixed_vs_sliding.png")


def plot_payload_motion(sliding_df, fixed_df):
    plt.figure(figsize=(8, 5))

    plt.plot(
        sliding_df["time"],
        sliding_df["payload_disp_base_x"],
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
    plt.axhline(0.0, linestyle="--", linewidth=1)

    plt.xlabel("Time [s]")
    plt.ylabel("Payload Displacement in Base Frame [m]")
    plt.title("Internal Payload Motion Under Baseline PD")
    save_plot("baseline_payload_motion_fixed_vs_sliding.png")


def plot_payload_distance_change(sliding_df, fixed_df):
    plt.figure(figsize=(8, 5))

    plt.plot(
        sliding_df["time"],
        sliding_df["payload_distance_change"],
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
    plt.axhline(0.0, linestyle="--", linewidth=1)

    plt.xlabel("Time [s]")
    plt.ylabel("Change in Payload-Base Distance [m]")
    plt.title("Payload-Base Distance Change Under Baseline PD")
    save_plot("baseline_payload_distance_change_fixed_vs_sliding.png")


def plot_control_effort(sliding_df, fixed_df):
    plt.figure(figsize=(8, 5))

    plt.plot(
        sliding_df["time"],
        sliding_df["control_effort"],
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
    plt.title("Baseline PD: Control Effort")
    save_plot("baseline_control_effort_fixed_vs_sliding.png")


def plot_summary_bars(sliding_df, fixed_df):
    labels = ["Sliding payload", "Fixed payload"]

    rms_errors = [
        (sliding_df["tracking_error"] ** 2).mean() ** 0.5,
        (fixed_df["tracking_error"] ** 2).mean() ** 0.5,
    ]

    max_errors = [
        sliding_df["tracking_error"].max(),
        fixed_df["tracking_error"].max(),
    ]

    max_payload_disps = [
        sliding_df["payload_disp_base_x"].abs().max(),
        fixed_df["payload_disp_base_x"].abs().max(),
    ]

    mean_efforts = [
        sliding_df["control_effort"].mean(),
        fixed_df["control_effort"].mean(),
    ]

    metrics = {
        "RMS Tracking Error [m]": rms_errors,
        "Max Tracking Error [m]": max_errors,
        "Max Payload Displacement [m]": max_payload_disps,
        "Mean Control Effort": mean_efforts,
    }

    for metric_name, values in metrics.items():
        plt.figure(figsize=(7, 5))
        plt.bar(labels, values)
        plt.ylabel(metric_name)
        plt.title(f"Baseline PD: {metric_name}")
        plt.grid(True, axis="y")

        filename = (
            "baseline_"
            + metric_name.lower()
            .replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            .replace("/", "_")
            + "_fixed_vs_sliding.png"
        )

        plt.savefig(
            PLOT_DIR / filename,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()


def main():
    sliding_df, fixed_df = load_logs()

    plot_longitudinal_tracking(sliding_df, fixed_df)
    plot_tracking_error(sliding_df, fixed_df)
    plot_tracking_error_zoomed(sliding_df, fixed_df)
    plot_x_error(sliding_df, fixed_df)
    plot_payload_motion(sliding_df, fixed_df)
    plot_payload_distance_change(sliding_df, fixed_df)
    plot_control_effort(sliding_df, fixed_df)
    plot_summary_bars(sliding_df, fixed_df)

    print(f"Saved baseline payload comparison plots to: {PLOT_DIR}")


if __name__ == "__main__":
    main()