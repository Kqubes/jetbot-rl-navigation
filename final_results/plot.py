import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, "plots")

NOMINAL_RESULTS = os.path.join(
    BASE_DIR,
    "final_nominal_results.csv",
)

ROBUSTNESS_RESULTS = os.path.join(
    BASE_DIR,
    "final_robustness_results.csv",
)

os.makedirs(PLOT_DIR, exist_ok=True)


MODEL_LABELS = {
    "Baseline": "Baseline",
    "ADR_Wheel_Camera": "ADR Wheel + Camera",
    "ADR_Wheel_Camera_Floor": "ADR Wheel + Camera + Floor",
}

TEST_LABELS = {
    "Camera_Noise": "Camera Noise",
    "Wheel_Friction": "Wheel Friction",
    "Floor_Friction": "Floor Friction",
    "Combined": "Combined",
}

MODELS = [
    "Baseline",
    "ADR_Wheel_Camera",
    "ADR_Wheel_Camera_Floor",
]

TESTS = [
    "Camera_Noise",
    "Wheel_Friction",
    "Floor_Friction",
    "Combined",
]


def save_plot(filename):
    """Save as PNG figure."""

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            f"{filename}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_nominal_performance(nominal):
    """Plot success, collision and timeout rates."""

    metrics = [
        "Success",
        "Collision",
        "Timeout",
    ]

    x = np.arange(len(nominal))
    width = 0.25

    plt.figure(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        plt.bar(
            x + (i - 1) * width,
            nominal[metric],
            width,
            label=metric,
        )

    plt.xticks(
        x,
        nominal["Display_Model"],
        rotation=10,
    )

    plt.xlabel("Model")
    plt.ylabel("Rate (%)")
    plt.title("Nominal Unseen Environment Performance")
    plt.ylim(0, 100)
    plt.legend()

    save_plot("01_nominal_performance")


def plot_robustness_metric(
    robustness,
    metric,
    filename,
    title,
    ylabel,
):
    """Plot one metric across all robustness conditions."""

    x = np.arange(len(TESTS))
    width = 0.25

    plt.figure(figsize=(10, 6))

    for i, model in enumerate(MODELS):
        values = []

        for test in TESTS:
            value = robustness[
                (robustness["Model"] == model)
                & (robustness["Test"] == test)
            ][metric].iloc[0]

            values.append(value)

        plt.bar(
            x + (i - 1) * width,
            values,
            width,
            label=MODEL_LABELS[model],
        )

    plt.xticks(
        x,
        [TEST_LABELS[test] for test in TESTS],
    )

    plt.xlabel("Perturbation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 100)
    plt.legend()

    save_plot(filename)


def main():
    nominal = pd.read_csv(NOMINAL_RESULTS)
    robustness = pd.read_csv(ROBUSTNESS_RESULTS)

    nominal["Display_Model"] = nominal["Model"].map(
        MODEL_LABELS
    )

    plot_nominal_performance(nominal)

    plot_robustness_metric(
        robustness,
        metric="Success",
        filename="02_robustness_success",
        title="Success Rate Under Robustness Perturbations",
        ylabel="Success Rate (%)",
    )

    plot_robustness_metric(
        robustness,
        metric="Collision",
        filename="03_robustness_collision",
        title="Collision Rate Under Robustness Perturbations",
        ylabel="Collision Rate (%)",
    )

    plot_robustness_metric(
        robustness,
        metric="Timeout",
        filename="04_robustness_timeout",
        title="Timeout Rate Under Robustness Perturbations",
        ylabel="Timeout Rate (%)",
    )

    print()
    print("Plots saved to:")
    print(PLOT_DIR)

    print()
    print("Generated plots:")

    for filename in sorted(os.listdir(PLOT_DIR)):
        if filename.endswith(".png"):
            print(filename)


if __name__ == "__main__":
    main()