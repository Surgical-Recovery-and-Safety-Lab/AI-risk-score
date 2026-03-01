import argparse

import matplotlib.pyplot as plt
import numpy as np
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.calibration import calibration_curve


def plot_reliability_diagrams(
    y_test,
    proba_list,
    label_list=[],
    colour="#2D90D8",
    n_bootstraps=200,
    save_path="",
    extension=".png",
    calibration_kwargs={},
    **kwargs,
):
    """
    Plots the reliability diagrams for the given probabilities.

    The 95% confidence interval is calculated using the bootstrap method for
    the calibration curve.
    The probability distribution can also be plotted under the calibration
    curve.

    Parameters
    ----------
    y_test : array-like of shape (n_samples, n_classes)
        Ground truth labels.
    proba_list : list[array]
        List of predicted probabilities.
    label_list : list[str], default: []
        List of labels for the legend.
    colour : str
        Colour for plotting.
    n_bootstraps : int, default: 200
        Number of iteration for the bootstrap.
    save_path : str, default: []
        Path to the save file.
    extension : str, default: ".png"
        Extension to save figure in.
    calibration_kwargs : dict[str, value], default: {}
        Extra arguments for the calibration function.
    **kwargs
        Extra arguments for the figure or axes objects.

    Returns
    -------
    None
        Nothing is returned.

    """

    # Split arguments based on where they should be sent
    ax_kwargs = {key: value for key, value in kwargs.items() if key in dir(Axes)}
    fig_kwargs = {key: value for key, value in kwargs.items() if key in dir(Figure)}

    # Set figure properties
    fig, ax = plt.subplots(**fig_kwargs)

    # Plot perfect calibration
    ax.plot(
        np.linspace(0, 1, 100),
        np.linspace(0, 1, 100),
        "k--",
        label="Perfectly calibrated",
    )

    for i in range(len(proba_list)):
        prob_true, prob_pred = calibration_curve(
            y_test,
            proba_list[i],
            **calibration_kwargs,
        )

        boots = []
        for _ in range(n_bootstraps):
            idx = np.random.choice(len(y_test), len(y_test), replace=True)
            prob_true_boot, _ = calibration_curve(
                y_test[idx],
                proba_list[i][idx],
                **calibration_kwargs,
            )
            boots.append(prob_true_boot)

        lower = np.percentile(boots, 2.5, axis=0)
        upper = np.percentile(boots, 97.5, axis=0)

        _plot_calibration(
            ax, prob_pred, prob_true, lower, upper, colour[i], label_list[i]
        )
    # Remove spines for aesthetics
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Create new plot for distribution
    divider = make_axes_locatable(ax)
    ax_dist = divider.append_axes("bottom", 0.5, pad=0.1, sharex=ax)

    if "n_bins" in calibration_kwargs.keys():
        bins = np.linspace(0, 1, calibration_kwargs["n_bins"] + 1)
    else:
        bins = np.linspace(0, 1, 21)

    ax_dist.hist(
        proba_list,
        stacked=True,
        color=colour,
        edgecolor="black",
        bins=bins,
        label=label_list,
    )
    ax_dist.set_yscale("log")
    ax_dist.set_xlabel("Predicted probabilities", fontweight="bold")

    # Set title and labels
    title = kwargs["set_title"] if "set_title" in kwargs.keys() else ""
    ax.set_title(title, fontweight="bold")
    if "set_title" in kwargs.keys():
        ax_kwargs.pop("set_title")
    ax.set_xlabel("Predicted probabilities", fontweight="bold")
    ax.set_ylabel("Observed proportion", fontweight="bold")

    # Set ax_kwargs to override if needed
    for key, val in ax_kwargs.items():
        getattr(ax, key)(val)

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.6, 0.9), title="Methods", frameon=False
    )
    plt.tight_layout()

    # Remove spines for aesthetics for distribution
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    fig.subplots_adjust(right=0.66, bottom=0.14)

    save_file = save_path + extension
    plt.savefig(save_file)

    plt.close()


def _plot_calibration(ax, prob_pred, prob_true, lower, upper, colour, label=None):
    """
    Helper function to plot the calibration and 95% CI on a Axes object.

    Parameters
    ----------
    ax : plt.Axes
        Axes on which to plot the data.
    prob_pred : np.array
        Predicted probabilities.
    prob_true : np.array
        True probabilities.
    lower : np.array
        Lower bounds of the confidence interval.
    upper : np.array
        Upper bounds of the confidence interval.
    colour : str
        Colour to plot in.
    label : str or None, default: None
        Label for the plotted curves.

    Returns
    -------
    None
        Nothing is returned.

    """
    ax.plot(
        prob_pred,
        prob_true,
        marker=".",
        color=colour,
        label=label,
    )
    ax.fill_between(
        prob_pred,
        lower,
        upper,
        color=colour,
        alpha=0.5,
        label=f"{label} 95% CI",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plots the calibration curves for supplementary calibration figure."
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
    metric_dict = {}

    extension = general_config["fig_parameters"]["extension"]
    outcomes = ["MORTALITY_90D", "ANY_COMP"]
    versions = [
        "v0.1.1.1-a.2.2.2",
        "v0.1.1.1-a.10.2.2",
        "v0.1.1.1-a.6.2.2",
        "v0.1.1.1-a.14.2.2",
    ]
    label_list = [
        "CSL",
        "CSL recalibrated",
        "SMOTE",
        "SMOTE recalibrated",
        "ROS",
        "ROS recalibrated",
        "RUS",
        "RUS recalibrated",
    ]
    save_file = get_file_path(
        general_config,
        v_number="",
        path_type="fig",
        exists=False,
    )

    colours = [
        "#2D90D8",
        "#33367A",
        "#2D90D8",
        "#96690E",
        "#2D90D8",
        "#CDB4DB",
        "#2D90D8",
        "#F2CC8F",
    ]

    for i in range(2):
        for j, version in enumerate(versions):
            # Swap version numbers if overloading
            y_pred_proba = []
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
            X_train, X_test = pipeline.get_test_data(data)
            X_test, y_test = extract_labels(X_test, pipeline.label_list)
            y_pred_proba.append(
                get_positive_proba(
                    pipeline.predict_proba(X_test, outcomes[i], "predictor")
                ).squeeze()
            )
            y_pred_proba.append(
                get_positive_proba(
                    pipeline.predict_proba(X_test, outcomes[i], "calibrator")
                ).squeeze()
            )

            plot_reliability_diagrams(
                y_test[:, i],
                y_pred_proba,
                label_list=label_list[j * 2 : j * 2 + 2],
                colour=colours[j * 2 : j * 2 + 2],
                save_path=save_file
                + f"_recalibrated_{label_list[j*2]}_reliability_diagram_{outcomes[i]}",
                extension=extension,
                calibration_kwargs={"n_bins": 10, "strategy": "quantile"},
                dpi=300,
                figsize=(5, 5),
            )
