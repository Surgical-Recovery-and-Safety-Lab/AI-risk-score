import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from medpipe import (
    extract_labels,
    get_positive_proba,
    load_data_from_csv,
    load_pipeline,
    print_message,
    read_toml_configuration,
)
from medpipe.utils.config import get_configuration, get_file_path, split_version_number
from sklearn.calibration import calibration_curve


def plot_reliability_diagrams(
    y_test,
    y_pred_proba=[],
    label_list=[],
    title=[],
    save_path="",
    extension=".png",
    calibration_kwargs={},
    **kwargs,
):
    """
    Plots the reliability diagrams using the calibration curve function from
    sklearn.calibration.

    Parameters
    ----------
    y_test : array-like of shape (n_samples, n_classes)
        Ground truth labels.
    y_pred_proba : list[array]
        Predicted probabilities from the predictors.
    label_list : list[str], default: []
    save_path : str, default: []
        Path to the save file.
    extension : str, default: ".png"
        Extension to save figure in.
    show_fig : bool, default: True
        Flag to show the figure.
    calibration_kwargs : dict[str, value], default: {}
        Extra arguments for the calibration curve function.
    **kwargs
        Extra arguments for the figure or axes objects.

    Returns
    -------
    None
        Nothing is returned.

    """
    n_classes = len(y_pred_proba)  # Default number of classes
    colours = ["#2D90D8", "#33367A", "#96690E", "#CDB4DB", "#F2CC8F"]

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
            **calibration_kwargs,
        )

        boots = []
        for _ in range(1000):
            idx = np.random.choice(len(y_test), len(y_test), replace=True)
            prob_true_boot, _ = calibration_curve(
                y_test[idx],
                y_pred_proba[i][idx],
                **calibration_kwargs,
            )
            boots.append(prob_true_boot)

        lower = np.percentile(boots, 2.5, axis=0)
        upper = np.percentile(boots, 97.5, axis=0)

        ax.plot(
            prob_pred,
            prob_true,
            marker=".",
            color=colours[i],
            label=label_list[i],
        )
        ax.fill_between(
            prob_pred,
            lower,
            upper,
            color=colours[i],
            alpha=0.5,
            label=f"{label_list[i]} 95% CI",
        )

    ax.set_xlabel("Predicted probabilities", fontweight="bold")
    ax.set_ylabel("Observed proportion", fontweight="bold")
    ax.set_title(title[0], fontweight="bold")

    # Set ax_kwargs to override if needed
    for key, val in ax_kwargs.items():
        getattr(ax, key)(val)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.6, 0.9), title=title[1], frameon=False
    )
    plt.tight_layout()
    fig.subplots_adjust(right=0.66, bottom=0.14)

    if save_path:
        save_file = save_path + f"_{label_list[-1]}" + extension
        plt.savefig(save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plots the calibration curves for Figure 4."
    )
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

    extension = general_config["fig_parameters"]["extension"]
    outcomes = ["MORTALITY_90D", "ANY_COMP"]
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
    if args.method == "CSL":
        label_list = (
            ["Baseline", "CSL"],
            ["Baseline", "CSL"],
        )
        title = ["CSL", "Models"]
    else:
        label_list = (
            [
                "IR = 73.2",
                "IR = 54.9",
                "IR = 36.6",
                "IR = 18.3",
                "IR = 1.0",
                outcomes[0],
            ],
            [
                "IR = 9.6",
                "IR = 7.2",
                "IR = 4.8",
                "IR = 2.4",
                "IR = 1.0",
                outcomes[1],
            ],
        )
        title = [args.method, "Imbalance ratio"]
    save_file = get_file_path(
        general_config,
        v_number="",
        path_type="fig",
        exists=False,
    )

    for i in range(2):
        y_pred_proba = []
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
            X_train, X_test_24 = pipeline.get_test_data(data)
            _, X_test_23 = pipeline.get_test_data(X_train)
            X_test = pd.concat((X_test_24, X_test_23))
            X_test, y_test = extract_labels(X_test, pipeline.label_list)
            y_pred_proba.append(
                get_positive_proba(
                    pipeline.predict_proba(X_test, outcomes[i], "predictor")
                )
            )

        plot_reliability_diagrams(
            y_test[:, i],
            y_pred_proba,
            label_list=label_list[i],
            title=title,
            save_path=save_file + f"{args.method}_reliability_diagram_" + outcomes[i],
            extension=extension,
            calibration_kwargs={"n_bins": 10, "strategy": "quantile"},
            dpi=300,
            figsize=(5, 5),
        )
