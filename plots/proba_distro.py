import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from pyrisk.data.preprocessing import extract_labels
from pyrisk.metrics.core import compute_all_CI, compute_score_metrics, print_metrics_CI
from pyrisk.models.core import get_positive_proba, load_pipeline
from pyrisk.pipeline.Pipeline import Pipeline
from pyrisk.utils.config import get_configuration, get_file_path, split_version_number
from pyrisk.utils.io import load_data_from_csv, read_toml_configuration
from pyrisk.utils.logger import print_message


def plot_prediction_distribution(
    y_pred_proba=[],
    label_list=[],
    n_bins=10,
    save_path="",
    extension=".png",
    **kwargs,
):
    """
    Plots the prediction probabilities.

    Parameters
    ----------
    y_pred_proba : array-like of shape (n_samples, n_classes), default: []
        Predicted probabilities from the predictor.
    y_pred_proba_calib : array-like of shape (n_samples, n_classes), default: []
        Predicted probabilities from the calibrator.
    label_list : list[str], default: []
        List of predicted labels.
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
    colours = ["#99C7E0", "#2D90D8", "#1D6968", "#33367A", "#96690E"]

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
        color=(colours, 0.8),
        stacked=True,
        edgecolor="black",
        bins=bins,
        label=label_list,
    )
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 0.9), title="Model")
    plt.tight_layout()
    fig.subplots_adjust(right=0.7, bottom=0.14)
    ax.set_ylim([1, 1e6])
    ax.set_xlim([-0.05, 1.05])

    if save_path:
        save_file = save_path + extension
        plt.savefig(save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and train a Pipeline of machine learning models"
    )
    parser.add_argument(
        "model_config_file",
        metavar="model-config-file",
        help="Path to the general configuration file",
    )
    parser.add_argument(
        "version_numbers",
        metavar="version-numbers",
        help="Version numbe_rs to run",
        nargs="+",
    )

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
    ext = ["_MORTALITY_90D", "_ANY_COMP"]
    label_list = {
        "predictor": ["Original", "CSL", "SMOTE", "ROS", "RUS"],
        "calibrator": ["Original", "CSL", "SMOTE", "ROS", "RUS"],
    }

    """
    label_list = {
        "predictor": ["Original", "CSL", "SMOTE", "ROS", "RUS"],
        "calibrator": [
            "Re-calibrated original",
            "Re-calibrated CSL",
            "Re-calibrated SMOTE",
            "Re-calibrated ROS",
            "Re-calibrated RUS",
        ],
    }
    """
    for model in models:
        for i in range(2):
            proba_distro = []
            for version in args.version_numbers:
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
                _, X_test = pipeline.get_test_data(data)
                X_test, y_test = extract_labels(X_test, pipeline.label_list)
                proba_distro.append(
                    get_positive_proba(
                        pipeline.predict_proba(X_test, i, model)
                    ).squeeze()
                )

            plot_prediction_distribution(
                proba_distro,
                label_list=label_list[model],
                save_path=save_file + "_proba_dist_" + model + ext[i],
                extension=extension,
                n_bins=20,
                dpi=300,
                figsize=(5, 5),
            )
