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
from sklearn.calibration import calibration_curve


def plot_reliability_diagrams(
    y_test,
    y_pred_proba=[],
    y_pred_proba_calib=[],
    label_list=[],
    save_path="",
    extension=".png",
    display_kwargs={},
    **kwargs,
):
    """
    Plots the reliability diagrams using CalibrationDisplay from
    sklearn.calibration.

    Parameters
    ----------
    y_test : array-like of shape (n_samples, n_classes)
        Ground truth labels.
    y_pred_proba : array-like of shape (n_samples, n_classes), default: []
        Predicted probabilities from the predictor.
    y_pred_proba_calib : array-like of shape (n_samples, n_classes), default: []
        Predicted probabilities from the calibrator.
    label_list : list[str], default: []
    save_path : str, default: []
        Path to the save file.
    extension : str, default: ".png"
        Extension to save figure in.
    show_fig : bool, default: True
        Flag to show the figure.
    display_kwargs : dict[str, value], default: {}
        Extra arguments for the CalibrationDisplay.
    **kwargs
        Extra arguments for the figure or axes objects.

    Returns
    -------
    None
        Nothing is returned.

    """
    n_classes = len(y_pred_proba)  # Default number of classes
    colours = ["#99C7E0", "#2D90D8", "#1D6968", "#33367A", "#96690E"]
    labels = ["Original", "CSL", "SMOTE", "ROS", "RUS"]

    # Split arguments based on where they should be sent
    ax_kwargs = {key: value for key, value in kwargs.items() if key in dir(Axes)}
    fig_kwargs = {key: value for key, value in kwargs.items() if key in dir(Figure)}
    # Set figure and axes properties
    fig, ax = plt.subplots(**fig_kwargs)  # Create a new figure

    # Plot perfect calibration
    ax.plot(
        np.linspace(0, 1, 100),
        np.linspace(0, 1, 100),
        "k--",
        label="Perfectly calibrated",
    )

    for i in range(n_classes):
        prob_true, prob_pred = calibration_curve(
            y_test,
            y_pred_proba[i],
            **display_kwargs,
        )
        ax.plot(
            prob_pred,
            prob_true,
            marker=".",
            color=colours[i],
            label=labels[i],
        )

        prob_true, prob_pred = calibration_curve(
            y_test,
            y_pred_proba_calib[i],
            **display_kwargs,
        )

        ax.plot(
            prob_pred,
            prob_true,
            linestyle="--",
            marker=".",
            color=colours[i],
            label="Re-calibrated " + labels[i],
        )

    ax.set_xlabel("Predicted probabilities", fontweight="bold")
    ax.set_ylabel("Observed proportion", fontweight="bold")

    # Set ax_kwargs to override if needed
    for key, val in ax_kwargs.items():
        getattr(ax, key)(val)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    ax.legend(loc="upper right", bbox_to_anchor=(1.6, 0.9), title="Models")
    plt.tight_layout()
    fig.subplots_adjust(right=0.66, bottom=0.14)

    if save_path:
        save_file = save_path + f"_{label_list}" + extension
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

    extension = general_config["fig_parameters"]["extension"]
    ext = ["_MORTALITY_90D", "_ANY_COMP"]
    save_file = get_file_path(
        general_config,
        v_number="",
        path_type="fig",
        exists=False,
    )

    for i in range(2):
        y_pred_proba = []
        y_pred_proba_calib = []
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
                    data_config, v_number=data_version[:4]  # Use only first 2 numbers
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
            y_pred_proba.append(
                get_positive_proba(pipeline.predict_proba(X_test, i, "predictor"))
            )
            y_pred_proba_calib.append(
                get_positive_proba(pipeline.predict_proba(X_test, i, "calibrator"))
            )

        plot_reliability_diagrams(
            y_test[:, i],
            y_pred_proba,
            y_pred_proba_calib,
            label_list=pipeline.label_list[i],
            save_path=save_file + "_reliability_diagram",
            extension=extension,
            display_kwargs={"n_bins": 10, "strategy": "quantile"},
            dpi=300,
            figsize=(5, 5),
        )
