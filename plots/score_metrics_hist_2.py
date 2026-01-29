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


def plot_metrics_CI(
    ci_dict,
    y,
    label_list,
    bar_width,
    save_path="",
    extension=".png",
    **kwargs,
):
    """
    Plots the metrics with confidence intrevals for each fold.

    Parameters
    ----------
    ci_dict : dict[str, tuple(float, float, float)]
        Dictionary containing the metric value and confidence intervals.
        The keys are the name of the metrics and the values are a tuple with
        first element the metric value, second the lower bound, and third the
        upper bound.
    label_list : list[str]
        List of predicted labels.
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

    """
    n_it = len(label_list)  # Number of iterations

    # Split arguments based on where they should be sent
    ax_kwargs = {key: value for key, value in kwargs.items() if key in dir(Axes)}
    fig_kwargs = {key: value for key, value in kwargs.items() if key in dir(Figure)}

    # Set up the figure and axis
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, **fig_kwargs)
    colours = ["#99C7E0", "#2D90D8", "#1D6968", "#33367A", "#96690E"]
    index = np.arange(len(ci_dict.keys()))
    y_labels = ["AUROC", "AUPRC", "Log loss"]

    # Loop through each metric
    for i, values in enumerate(ci_dict.values()):
        ax[i].bar(  # Plot original model
            y[0] + bar_width * 0.5,
            values[0][0],
            bar_width,
            color=colours[0],
            edgecolor=(0, 0, 0, 1),
            label=label_list[0],
        )
        # Error bars for confidence interval
        ax[i].errorbar(
            y[0] + bar_width * 0.5,
            values[0][0],
            yerr=values[0][0] - values[1][0],
            fmt="none",
            color="black",
            capsize=2,
        )
        ax[i].bar(  # Plot CSL model
            y[0] - bar_width * 0.5,
            values[0][1],
            bar_width,
            color=colours[1],
            edgecolor=(0, 0, 0, 1),
            label=label_list[1],
        )
        # Error bars for confidence interval
        ax[i].errorbar(
            y[0] - bar_width * 0.5,
            values[0][1],
            yerr=values[0][1] - values[1][1],
            fmt="none",
            color="black",
            capsize=2,
        )
        start = 2
        end = 6
        y_bar = y[1:] + bar_width
        for j in range(3):
            value = values[0][start:end]
            lower_b = values[1][start:end]
            ax[i].bar(
                y_bar,
                value,
                width=bar_width,
                color=colours[j + 2],
                edgecolor=(0, 0, 0, 1),
                label=label_list[j + 2],
            )

            ax[i].errorbar(
                y_bar,
                value,
                yerr=value - lower_b,
                fmt="none",
                color="black",
                capsize=2,
            )
            # Customize the chart
            ax[i].set_ylabel(y_labels[i], fontweight="bold")
            ax[i].set_ylim(ymin=0, ymax=1)
            if i == 2:
                ax[i].set_ylim(ymin=0, ymax=0.55)
            ax[i].spines["top"].set_visible(False)
            ax[i].spines["right"].set_visible(False)

            start = end
            end += 4
            y_bar -= bar_width
    # Set the x-ticks to be at the center of each group of bars
    ax[-1].set_xlabel("Imbalance ratio", fontweight="bold")
    ax[-1].set_xticks(y)
    ax[-1].set_xticklabels(y)

    # Place one legend for the whole figure
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center right", bbox_to_anchor=(0.98, 0.5), title="Methods"
    )

    # Set ax_kwargs to override if needed
    for key, val in ax_kwargs.items():
        getattr(ax, key)(val)

    plt.gca().invert_xaxis()
    plt.tight_layout()
    fig.subplots_adjust(right=0.75)

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

    args = parser.parse_args()
    versions = [
        "v0.1.1.1-a.1.2.2",
        "v0.1.1.1-a.2.2.2",
        "v0.1.1.1-a.9.2.2",
        "v0.1.1.1-a.8.2.2",
        "v0.1.1.1-a.7.2.2",
        "v0.1.1.1-a.10.2.2",
        "v0.1.1.1-a.5.2.2",
        "v0.1.1.1-a.4.2.2",
        "v0.1.1.1-a.3.2.2",
        "v0.1.1.1-a.6.2.2",
        "v0.1.1.1-a.13.2.2",
        "v0.1.1.1-a.12.2.2",
        "v0.1.1.1-a.11.2.2",
        "v0.1.1.1-a.14.2.2",
    ]

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
    extension = general_config["fig_parameters"]["extension"]
    ext = ["_MORTALITY_90D", "_ANY_COMP"]
    y = ([73.2, 54.9, 36.3, 18.3, 1.0], [9.6, 7.2, 4.8, 2.4, 1.0])
    bar_width = [4.5, 0.35]

    for i in range(2):
        metric_dict = {}
        metric_dict_cal = {}
        for version in versions:
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
            X_train, X_test = pipeline.get_test_data(data.dropna())
            X_train, y_train = extract_labels(X_train, pipeline.label_list)
            tmp_metric_dict = {}
            tmp_metric_dict_cal = {}

            for key in pipeline.calibrator_probabilities[i]:
                y_test = y_train[X_train["OP_YEAR"] == key]
                tmp_metric_dict[key] = compute_score_metrics(
                    ["auroc", "ap", "log_loss"],
                    y_test[:, i],
                    get_full_proba(pipeline.predictor_probabilities[i][key]),
                )
                tmp_metric_dict_cal[key] = compute_score_metrics(
                    ["auroc", "ap", "log_loss"],
                    y_test[:, i],
                    get_full_proba(pipeline.calibrator_probabilities[i][key]),
                )
                for k in tmp_metric_dict[key].keys():
                    if key not in metric_dict.keys():
                        metric_dict[key] = tmp_metric_dict[key]
                        metric_dict_cal[key] = tmp_metric_dict_cal[key]
                        break
                    else:
                        metric_dict[key][k] += tmp_metric_dict[key][k]
                        metric_dict_cal[key][k] += tmp_metric_dict_cal[key][k]

        ci_dict = compute_all_CI(metric_dict)
        ci_dict_cal = compute_all_CI(metric_dict_cal)
        plot_metrics_CI(
            ci_dict,
            np.array(y[i]),
            ["Original", "CSL", "SMOTE", "ROS", "RUS"],
            bar_width=bar_width[i],
            dpi=300,
            figsize=(5, 5),
            save_path=save_file + ext[i] + "_OG",
            extension=extension,
        )
        plot_metrics_CI(
            ci_dict_cal,
            np.array(y[i]),
            ["Original", "CSL", "SMOTE", "ROS", "RUS"],
            bar_width=bar_width[i],
            dpi=300,
            figsize=(5, 5),
            save_path=save_file + ext[i] + "_recal",
            extension=extension,
        )
