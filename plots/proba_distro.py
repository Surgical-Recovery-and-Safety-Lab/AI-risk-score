import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pyrisk.data.preprocessing import extract_labels
from pyrisk.models.core import get_positive_proba, load_pipeline
from pyrisk.pipeline.Pipeline import Pipeline
from pyrisk.utils.config import get_configuration, get_file_path, split_version_number
from pyrisk.utils.io import load_data_from_csv, read_toml_configuration
from pyrisk.utils.logger import print_message


def plot_prediction_distribution(
    y_pred_proba=[],
    title=[],
    label_list=[],
    colours=[],
    n_bins=10,
    save_path="",
    extension=".png",
    **kwargs,
):
    """
    Plots the prediction probabilities.

    Parameters
    ----------
    y_pred_proba : list[array]
        Predicted probabilities from the predictor.
    label_list : list[str], default: []
        List of predicted labels.
    colours : list[str]
        List of colours to plot with.
    n_bins : int, default: 10
        Number of bins for the histogram.
    save_path : str, default: []
        Path to the save file.
    extension : str, default: ".png"
        Extension to save figure in.
    show_fig : bool, default: True
        Flag to show the figure.
    **kwargs
        Extra arguments for the figure or axes objects.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    ValueError
        If y_pred_proba and y_pred_proba_calib are both empty.

    """

    # Split arguments based on where they should be sent
    fig_kwargs = {key: value for key, value in kwargs.items() if key in dir(Figure)}

    # Set figure and axes properties
    fig, ax = plt.subplots(**fig_kwargs)  # Create a new figure
    bins = np.linspace(0, 1, n_bins + 1)
    ax.set_xlabel("Predicted probabilities", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_yscale("log")

    ax.hist(
        y_pred_proba,
        color=colours,
        stacked=True,
        edgecolor="black",
        bins=bins,
        label=label_list,
    )
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 0.9), title=title[1])
    ax.set_ylim([1, 1e6])
    ax.set_xlim([-0.05, 1.05])
    ax.set_title(title[0], fontweight="bold")
    plt.tight_layout()
    fig.subplots_adjust(right=0.7, bottom=0.14)

    if save_path:
        save_file = save_path + extension
        plt.savefig(save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot probability distributions.")
    parser.add_argument(
        "model_config_file",
        metavar="model-config-file",
        help="Path to the general configuration file",
    )
    parser.add_argument("method", help="Method used")

    args = parser.parse_args()

    print_message("Loading parameters from configuration file")
    # Read log and general configuration file
    log_config = read_toml_configuration("../config/log.toml")
    general_config = read_toml_configuration(args.model_config_file)
    logger = None
    script_name = ""
    metric_dict = {}
    i = 0  # 0: MORTALITY_90D, 1: ANY_COMP
    models = ["predictor", "calibrator"]
    save_file = get_file_path(
        general_config,
        v_number="",
        path_type="fig",
        exists=False,
    )

    extension = general_config["fig_parameters"]["extension"]
    ext = ["MORTALITY_90D", "ANY_COMP"]
    colours = {
        "CSL": ["#2D90D8", "#33367A"],
        "ROS": ["#2D90D8", "#33367A", "#96690E", "#CDB4DB", "#F2CC8F"],
        "RUS": ["#2D90D8", "#33367A", "#96690E", "#CDB4DB", "#F2CC8F"],
        "SMOTE": ["#2D90D8", "#33367A", "#96690E", "#CDB4DB", "#F2CC8F"],
    }
    versions = {
        "CSL": [
            "v0.1.1.1-a.1.2.2",
            "v0.1.1.1-a.2.2.2",
        ],
        "SMOTE": [
            "v0.1.1.1-a.1.2.2",
            "v0.1.1.1-a.9.2.2",
            "v0.1.1.1-a.8.2.2",
            "v0.1.1.1-a.7.2.2",
            "v0.1.1.1-a.10.2.2",
        ],
        "ROS": [
            "v0.1.1.1-a.1.2.2",
            "v0.1.1.1-a.5.2.2",
            "v0.1.1.1-a.4.2.2",
            "v0.1.1.1-a.3.2.2",
            "v0.1.1.1-a.6.2.2",
        ],
        "RUS": [
            "v0.1.1.1-a.1.2.2",
            "v0.1.1.1-a.13.2.2",
            "v0.1.1.1-a.12.2.2",
            "v0.1.1.1-a.11.2.2",
            "v0.1.1.1-a.14.2.2",
        ],
    }

    label_list = {
        "CSL": (
            [
                "Baseline",
                "CSL",
                "Models",
            ],
            [
                "Baseline",
                "CSL",
                "Models",
            ],
        ),
        "ROS": (
            [
                "IR = 73.2",
                "IR = 54.9",
                "IR = 36.6",
                "IR = 18.3",
                "IR = 1.0",
                "Imbalance ratios",
            ],
            [
                "IR = 9.6",
                "IR = 7.2",
                "IR = 4.8",
                "IR = 2.4",
                "IR = 1.0",
                "Imbalance ratios",
            ],
        ),
        "RUS": (
            [
                "IR = 73.2",
                "IR = 54.9",
                "IR = 36.6",
                "IR = 18.3",
                "IR = 1.0",
                "Imbalance ratios",
            ],
            [
                "IR = 9.6",
                "IR = 7.2",
                "IR = 4.8",
                "IR = 2.4",
                "IR = 1.0",
                "Imbalance ratios",
            ],
        ),
        "SMOTE": (
            [
                "IR = 73.2",
                "IR = 54.9",
                "IR = 36.6",
                "IR = 18.3",
                "IR = 1.0",
                "Imbalance ratios",
            ],
            [
                "IR = 9.6",
                "IR = 7.2",
                "IR = 4.8",
                "IR = 2.4",
                "IR = 1.0",
                "Imbalance ratios",
            ],
        ),
    }

    for model in models:
        for i, outcome in enumerate(ext):
            proba_distro = []
            for version in versions[args.method]:
                # Swap version numbers if overloading
                general_config["version"] = version
                print_message(
                    f"Version number: {general_config["version"]}", logger, script_name
                )
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
                        data_config,
                        v_number=data_version[:4],  # Use only first 2 numbers
                    )
                )
                # Load model
                print_message("Loading model", logger, script_name)
                load_file = get_file_path(
                    general_config,
                    v_number=general_config["version"],
                )
                pipeline = load_pipeline(load_file)
                data = pipeline.transform(data)
                X_train, X_test_24 = pipeline.get_test_data(data)
                _, X_test_23 = pipeline.get_test_data(X_train)
                X_test = pd.concat((X_test_24, X_test_23))
                X_test, y_test = extract_labels(X_test, pipeline.label_list)
                proba_distro.append(
                    get_positive_proba(
                        pipeline.predict_proba(X_test, outcome, model)
                    ).squeeze()
                )

            plot_prediction_distribution(
                proba_distro,
                label_list=label_list[args.method][i],
                colours=colours[args.method],
                title=[args.method, label_list[args.method][i][-1]],
                save_path=save_file + f"{args.method}_proba_dist_{model}_{outcome}",
                extension=extension,
                n_bins=20,
                dpi=300,
                figsize=(5, 5),
            )
