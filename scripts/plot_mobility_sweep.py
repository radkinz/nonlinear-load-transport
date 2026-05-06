from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = ROOT / "logs" / "mobility_sweep"
SUMMARY_PATH = LOG_DIR / "mobility_sweep_summary.csv"

PLOT_DIR = ROOT / "plots" / "mobility_sweep"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


CONTROLLER_LABELS = {
    "baseline_pd": "Baseline PD",
    "payload_aware_pd": "Payload-Aware PD",
    "sliding_mode": "Sliding Mode",
    "adaptive_pd": "Adaptive PD",
}


def load_summary():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_PATH}")

    df = pd.read_csv(SUMMARY_PATH)
    df["controller_label"] = df["controller"].map(CONTROLLER_LABELS)
    return df


def save_plot(filename):
    plt.legend()
    plt.grid(True)
    plt.savefig(
        PLOT_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def plot_metric_vs_mobility(df, metric, ylabel, title, filename):
    plt.figure(figsize=(8, 5))

    for controller, group in df.groupby("controller"):
        group = group.sort_values("payload_mobility")
        label = CONTROLLER_LABELS.get(controller, controller)

        plt.plot(
            group["payload_mobility"],
            group[metric],
            marker="o",
            linewidth=2,
            label=label,
        )

    plt.xlabel("Payload Mobility Range [m]")
    plt.ylabel(ylabel)
    plt.title(title)
    save_plot(filename)


def plot_rms_tracking_error(df):
    plot_metric_vs_mobility(
        df,
        metric="rms_tracking_error",
        ylabel="RMS Tracking Error [m]",
        title="Robustness to Payload Mobility: RMS Tracking Error",
        filename="mobility_vs_rms_tracking_error.png",
    )


def plot_max_tracking_error(df):
    plot_metric_vs_mobility(
        df,
        metric="max_tracking_error",
        ylabel="Max Tracking Error [m]",
        title="Robustness to Payload Mobility: Max Tracking Error",
        filename="mobility_vs_max_tracking_error.png",
    )


def plot_mean_control_effort(df):
    plot_metric_vs_mobility(
        df,
        metric="mean_control_effort",
        ylabel="Mean Control Effort",
        title="Control Effort Across Payload Mobility",
        filename="mobility_vs_mean_control_effort.png",
    )


def plot_max_control_effort(df):
    plot_metric_vs_mobility(
        df,
        metric="max_control_effort",
        ylabel="Max Control Effort",
        title="Peak Control Effort Across Payload Mobility",
        filename="mobility_vs_max_control_effort.png",
    )


def plot_max_payload_displacement(df):
    plot_metric_vs_mobility(
        df,
        metric="max_payload_disp_base_x",
        ylabel="Max Payload Displacement [m]",
        title="Internal Payload Motion Across Mobility Range",
        filename="mobility_vs_max_payload_displacement.png",
    )


def plot_settling_time(df):
    plot_metric_vs_mobility(
        df,
        metric="settling_time_0p05",
        ylabel="Settling Time [s]",
        title="Settling Time Across Payload Mobility",
        filename="mobility_vs_settling_time.png",
    )


def plot_bar_metric_at_max_mobility(df, metric, ylabel, title, filename):
    max_mobility = df["payload_mobility"].max()
    subset = df[df["payload_mobility"] == max_mobility].copy()

    subset["controller_label"] = subset["controller"].map(CONTROLLER_LABELS)
    subset = subset.sort_values(metric)

    plt.figure(figsize=(8, 5))

    plt.bar(
        subset["controller_label"],
        subset[metric],
    )

    plt.xlabel("Controller")
    plt.ylabel(ylabel)
    plt.title(f"{title} at ±{max_mobility:.2f} m Payload Mobility")
    plt.xticks(rotation=15)
    plt.grid(True, axis="y")

    plt.savefig(
        PLOT_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def plot_summary_bars(df):
    plot_bar_metric_at_max_mobility(
        df,
        metric="rms_tracking_error",
        ylabel="RMS Tracking Error [m]",
        title="RMS Tracking Error",
        filename="bar_rms_tracking_error_max_mobility.png",
    )

    plot_bar_metric_at_max_mobility(
        df,
        metric="mean_control_effort",
        ylabel="Mean Control Effort",
        title="Mean Control Effort",
        filename="bar_mean_control_effort_max_mobility.png",
    )

    plot_bar_metric_at_max_mobility(
        df,
        metric="max_payload_disp_base_x",
        ylabel="Max Payload Displacement [m]",
        title="Max Payload Displacement",
        filename="bar_max_payload_displacement_max_mobility.png",
    )


def main():
    df = load_summary()

    plot_rms_tracking_error(df)
    plot_max_tracking_error(df)
    plot_mean_control_effort(df)
    plot_max_control_effort(df)
    plot_max_payload_displacement(df)
    plot_settling_time(df)
    plot_summary_bars(df)

    print(f"Saved mobility sweep plots to: {PLOT_DIR}")


if __name__ == "__main__":
    main()