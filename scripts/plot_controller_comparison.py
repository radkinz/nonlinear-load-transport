from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = ROOT / "logs" / "controller_comparison"
PLOT_DIR = ROOT / "plots" / "controller_comparison"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILES = {
    "Baseline PD": LOG_DIR / "baseline_pd.csv",
    "Payload-Aware PD": LOG_DIR / "payload_aware_pd.csv",
    "Sliding Mode": LOG_DIR / "sliding_mode.csv",
}

BRAKE_START = 0.0
BRAKE_END = 2.0


def load_logs():
    data = {}

    for label, path in CSV_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing log file: {path}")
        data[label] = pd.read_csv(path)

    return data


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


def plot_longitudinal_tracking(data):
    plt.figure(figsize=(8, 5))

    # Desired trajectory is the same for all controllers.
    first_df = next(iter(data.values()))
    plt.plot(
        first_df["time"],
        first_df["x_des"],
        label="Desired x",
        linewidth=4,
    )

    for label, df in data.items():
        plt.plot(
            df["time"],
            df["base_x"],
            label=label,
            linewidth=2,
        )

    add_braking_region()

    plt.xlabel("Time [s]")
    plt.ylabel("x Position [m]")
    plt.title("Longitudinal Tracking: Controller Comparison")
    save_plot("x_tracking_controller_compare.png")


def plot_tracking_error(data):
    plt.figure(figsize=(8, 5))

    for label, df in data.items():
        plt.plot(
            df["time"],
            df["tracking_error"],
            label=label,
            linewidth=2,
        )

    add_braking_region()

    plt.xlabel("Time [s]")
    plt.ylabel("Tracking Error [m]")
    plt.title("Tracking Error: Controller Comparison")
    save_plot("tracking_error_controller_compare.png")


def plot_tracking_error_zoomed(data):
    plt.figure(figsize=(8, 5))

    for label, df in data.items():
        plt.plot(
            df["time"],
            df["tracking_error"],
            label=label,
            linewidth=2,
        )

    add_braking_region()

    plt.xlim(0.0, 4.0)

    plt.xlabel("Time [s]")
    plt.ylabel("Tracking Error [m]")
    plt.title("Tracking Error Around Braking")
    save_plot("tracking_error_zoomed_controller_compare.png")


def plot_x_error(data):
    plt.figure(figsize=(8, 5))

    for label, df in data.items():
        plt.plot(
            df["time"],
            df["x_error"],
            label=label,
            linewidth=2,
        )

    add_braking_region()

    plt.xlabel("Time [s]")
    plt.ylabel("x Error [m]")
    plt.title("Longitudinal Error: Controller Comparison")
    save_plot("x_error_controller_compare.png")


def plot_payload_motion(data):
    plt.figure(figsize=(8, 5))

    for label, df in data.items():
        plt.plot(
            df["time"],
            df["payload_disp_base_x"],
            label=label,
            linewidth=2,
        )

    add_braking_region()

    plt.axhline(0.0, linestyle="--", linewidth=1)

    plt.xlabel("Time [s]")
    plt.ylabel("Payload Displacement in Base Frame [m]")
    plt.title("Internal Payload Motion: Controller Comparison")
    save_plot("payload_disp_base_x_controller_compare.png")


def plot_payload_distance_change(data):
    plt.figure(figsize=(8, 5))

    for label, df in data.items():
        plt.plot(
            df["time"],
            df["payload_distance_change"],
            label=label,
            linewidth=2,
        )

    add_braking_region()

    plt.axhline(0.0, linestyle="--", linewidth=1)

    plt.xlabel("Time [s]")
    plt.ylabel("Change in Payload-Base Distance [m]")
    plt.title("Payload-Base Distance Change")
    save_plot("payload_distance_change_controller_compare.png")


def plot_control_effort(data):
    plt.figure(figsize=(8, 5))

    for label, df in data.items():
        plt.plot(
            df["time"],
            df["control_effort"],
            label=label,
            linewidth=2,
        )

    add_braking_region()

    plt.xlabel("Time [s]")
    plt.ylabel("Control Effort Magnitude")
    plt.title("Control Effort: Controller Comparison")
    save_plot("control_effort_controller_compare.png")


def plot_ctrl_x(data):
    plt.figure(figsize=(8, 5))

    for label, df in data.items():
        plt.plot(
            df["time"],
            df["ctrl_x"],
            label=label,
            linewidth=2,
        )

    add_braking_region()

    plt.xlabel("Time [s]")
    plt.ylabel("Longitudinal Control Input")
    plt.title("Longitudinal Control Response")
    save_plot("ctrl_x_controller_compare.png")


def plot_summary_bar_metrics(data):
    labels = []
    max_errors = []
    rms_errors = []
    max_payload_disps = []
    mean_control_efforts = []

    for label, df in data.items():
        labels.append(label)
        max_errors.append(df["tracking_error"].max())
        rms_errors.append((df["tracking_error"] ** 2).mean() ** 0.5)
        max_payload_disps.append(df["payload_disp_base_x"].abs().max())
        mean_control_efforts.append(df["control_effort"].mean())

    metrics = {
        "Max Tracking Error [m]": max_errors,
        "RMS Tracking Error [m]": rms_errors,
        "Max Payload Displacement [m]": max_payload_disps,
        "Mean Control Effort": mean_control_efforts,
    }

    for metric_name, values in metrics.items():
        plt.figure(figsize=(8, 5))
        plt.bar(labels, values)
        plt.ylabel(metric_name)
        plt.title(metric_name)
        plt.xticks(rotation=15)
        plt.grid(True, axis="y")

        filename = (
            metric_name.lower()
            .replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            .replace("/", "_")
        )

        plt.savefig(
            PLOT_DIR / f"{filename}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()


def main():
    data = load_logs()

    plot_longitudinal_tracking(data)
    plot_tracking_error(data)
    plot_tracking_error_zoomed(data)
    plot_x_error(data)
    plot_payload_motion(data)
    plot_payload_distance_change(data)
    plot_control_effort(data)
    plot_ctrl_x(data)
    plot_summary_bar_metrics(data)

    print(f"Saved controller comparison plots to: {PLOT_DIR}")


if __name__ == "__main__":
    main()