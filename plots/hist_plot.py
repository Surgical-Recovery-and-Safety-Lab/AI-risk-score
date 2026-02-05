import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from pyrisk.data.preprocessing import extract_labels
from pyrisk.metrics.core import compute_all_CI, compute_score_metrics, print_metrics_CI
from pyrisk.models.core import get_full_proba, load_pipeline
from pyrisk.pipeline.Pipeline import Pipeline
from pyrisk.utils.config import get_configuration, get_file_path, split_version_number
from pyrisk.utils.io import load_data_from_csv, read_toml_configuration
from pyrisk.utils.logger import print_message


def plt_hist(operation_years):
    # 2. Create the plot
    plt.figure(dpi=300)

    # Define bins to align with calendar years
    bins = np.arange(min(operation_years), max(operation_years) + 2) - 0.5

    # Plotting the histogram
    colours = ["#99C7E0", "#2D90D8", "#1D6968", "#33367A", "#96690E"]
    plt.hist(
        operation_years,
        bins=bins,
        color=colours[0],
        edgecolor="black",
        linewidth=1.2,
        label="Patient count",
    )

    # 3. Add labels and styling for a research paper
    plt.xlabel("Operation year", fontweight="bold")
    plt.ylabel("Number of patients", fontweight="bold")

    # Ensure x-axis shows individual years or intervals
    plt.xticks(np.arange(min(operation_years), max(operation_years) + 1, 2))

    # Add a light grid to demonstrate the transparency effect
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    # Remove top and right spines for a cleaner look
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()

    # 4. Save the figure
    plt.savefig("patient_distribution_histogram.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and train a Pipeline of machine learning models"
    )
    parser.add_argument(
        "model_config_file",
        metavar="model-config-file",
        help="Path to the general configuration file",
    )

    args = parser.parse_args()

    print_message("Loading parameters from configuration file")
    # Read log and general configuration file
    log_config = read_toml_configuration("../config/log.toml")
    general_config = read_toml_configuration(args.model_config_file)
    logger = None
    script_name = ""
    i = 0  # 0: MORTALITY_90D, 1: ANY_COMP
    save_file = get_file_path(
        general_config,
        v_number="",
        path_type="fig",
        exists=False,
    )

    print_message(f"Version number: {general_config["version"]}", logger, script_name)
    data_version, _ = split_version_number(general_config["version"])

    # Get data configuration parameters
    data_config = get_configuration(
        general_config["data_parameters"],
        data_version,
    )
    pipeline = Pipeline(general_config, logger)
    print_message("Getting data", logger, script_name)
    data = load_data_from_csv(
        get_file_path(
            data_config, v_number=data_version[:4]  # Use only first 2 numbers
        )
    )
    data = data.dropna()

    plt_hist(data["OP_YEAR"])
