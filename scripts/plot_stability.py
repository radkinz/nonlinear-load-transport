from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = ROOT / "logs" / "stability_test"
SUMMARY_PATH = LOG_DIR / "initial_offset_stability_summary.csv"

PLOT_DIR = ROOT / "plots" / "stability_test"
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
    plt.savefig(PLOT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def plot_metric_vs_offset(df, metric, ylabel, title, filename):
    plt.figure(figsize=(8, 5))

    for controller, group in df.groupby("controller"):
        group = group.sort_values("initial_payload_offset")
        label = CONTROLLER_LABELS.get(controller, controller)

        plt.plot(
            group["initial_payload_offset"],
            group[metric],
            marker="o",
            linewidth=2,
            label=label,
        )

    plt.xlabel("Initial Payload Offset [m]")
    plt.ylabel(ylabel)
    plt.title(title)
    save_plot(filename)


def plot_summary_metrics(df):
    plot_metric_vs_offset(
        df,
        metric="rms_tracking_error",
        ylabel="RMS Tracking Error [m]",
        title="Stability Test: RMS Tracking Error",
        filename="offset_vs_rms_tracking_error.png",
    )

    plot_metric_vs_offset(
        df,
        metric="final_tracking_error",
        ylabel="Final Tracking Error [m]",
        title="Stability Test: Final Tracking Error",
        filename="offset_vs_final_tracking_error.png",
    )

    plot_metric_vs_offset(
        df,
        metric="settling_time_0p05",
        ylabel="Settling Time [s]",
        title="Stability Test: Settling Time",
        filename="offset_vs_settling_time.png",
    )

    plot_metric_vs_offset(
        df,
        metric="max_payload_disp_base_x",
        ylabel="Max Payload Displacement [m]",
        title="Stability Test: Max Payload Displacement",
        filename="offset_vs_max_payload_displacement.png",
    )

    plot_metric_vs_offset(
        df,
        metric="mean_control_effort",
        ylabel="Mean Control Effort",
        title="Stability Test: Mean Control Effort",
        filename="offset_vs_mean_control_effort.png",
    )


def plot_tracking_error_time_series():
    offsets_to_plot = [-0.15, 0.0, 0.15]

    for offset in offsets_to_plot:
        plt.figure(figsize=(8, 5))

        for controller, label in CONTROLLER_LABELS.items():
            safe_offset = str(offset).replace("-", "neg_").replace(".", "p")
            log_path = LOG_DIR / f"{controller}_offset_{safe_offset}.csv"

            if not log_path.exists():
                continue

            df = pd.read_csv(log_path)

            plt.plot(
                df["time"],
                df["tracking_error"],
                linewidth=2,
                label=label,
            )

        plt.xlabel("Time [s]")
        plt.ylabel("Tracking Error [m]")
        plt.title(f"Tracking Error from Initial Offset {offset:+.2f} m")
        save_plot(f"tracking_error_offset_{offset:+.2f}.png".replace("+", "pos_").replace("-", "neg_").replace(".", "p"))


def plot_payload_motion_time_series():
    offsets_to_plot = [-0.15, 0.0, 0.15]

    for offset in offsets_to_plot:
        plt.figure(figsize=(8, 5))

        for controller, label in CONTROLLER_LABELS.items():
            safe_offset = str(offset).replace("-", "neg_").replace(".", "p")
            log_path = LOG_DIR / f"{controller}_offset_{safe_offset}.csv"

            if not log_path.exists():
                continue

            df = pd.read_csv(log_path)

            plt.plot(
                df["time"],
                df["payload_disp_base_x"],
                linewidth=2,
                label=label,
            )

        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("Time [s]")
        plt.ylabel("Payload Displacement in Base Frame [m]")
        plt.title(f"Payload Motion from Initial Offset {offset:+.2f} m")
        save_plot(f"payload_motion_offset_{offset:+.2f}.png".replace("+", "pos_").replace("-", "neg_").replace(".", "p"))


def plot_boundedness_table(df):
    bounded_summary = (
        df.groupby("controller_label")[["bounded_tracking_error", "bounded_payload_motion"]]
        .mean()
        .reset_index()
    )

    bounded_summary["bounded_tracking_error"] *= 100.0
    bounded_summary["bounded_payload_motion"] *= 100.0

    plt.figure(figsize=(8, 5))

    x = range(len(bounded_summary))
    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        bounded_summary["bounded_tracking_error"],
        width=width,
        label="Tracking bounded",
    )

    plt.bar(
        [i + width / 2 for i in x],
        bounded_summary["bounded_payload_motion"],
        width=width,
        label="Payload bounded",
    )

    plt.xticks(x, bounded_summary["controller_label"], rotation=15)
    plt.ylabel("Trials Bounded [%]")
    plt.title("Boundedness Across Initial Payload Offsets")
    plt.ylim(0, 105)
    plt.grid(True, axis="y")

    plt.legend()
    plt.savefig(PLOT_DIR / "boundedness_summary.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    df = load_summary()

    plot_summary_metrics(df)
    plot_tracking_error_time_series()
    plot_payload_motion_time_series()
    plot_boundedness_table(df)

    print(f"Saved stability test plots to: {PLOT_DIR}")


if __name__ == "__main__":
    main()