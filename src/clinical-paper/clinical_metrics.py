#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clinical_metrics.py

Plots the metrics for the clinical paper.
"""

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from medpipe import (
    compute_all_CI,
    compute_score_metrics,
    exception_handler,
    extract_labels,
    get_full_proba,
    load_data_from_csv,
    load_pipeline,
    print_message,
    read_toml_configuration,
    setup_logger,
)
from medpipe.utils.config import get_configuration, get_file_path, split_version_number

from constants import LABEL_MAP


def plot_metrics_CI(
    ci_dict,
    outcome,
    label_list,
    save_path="",
    no_plots=False,
    extension=".png",
    **kwargs,
):
    """
    Plots the metrics with confidence intrevals for each fold.

    Parameters
    ----------
    ci_dict : dict[str, tuple(float, float, float)]
        Dictionary containing the metric value and confidence intervals
        for the natural and recalibrated model.
    ci_dict_recal : dict[str, tuple(float, float, float)]
        Dictionary containing the metric value and confidence intervals
        for the recalibrated model.
    outcome : str
        Name of the outcome for the title.
    label_list : list[str]
        List of label for the legend.
    save_path : str, default: []
        Path to the save file.
    no_plots : bool, default: False
        Flag used to plot the figure or not.
    extension : str, default: ".png"
        Extension to save figure in.
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

    # Set up the figure and axis
    dims = (1, 3)
    fig, ax = plt.subplots(
        nrows=dims[0],
        ncols=dims[1],
        **fig_kwargs,
    )
    colours = [
        "#2D90D8",
        "#33367A",
        "#96690E",
        "#CDB4DB",
        "#F2CC8F",
    ]

    y_labels = {"auroc": "AUROC", "log_loss": "Log loss"}

    bar_width = 0.1
    x = np.arange(len(label_list)) * bar_width

    # Loop through each metric
    for i, (key, values) in enumerate(ci_dict.items()):
        for j in range(len(values[0])):
            value = values[0][j]
            lower_b = values[1][j]
            upper_b = values[2][j]
            ax[i].bar(
                x[j],
                value,
                width=bar_width,
                color=colours[j],
                edgecolor=(0, 0, 0, 1),
                label=label_list[j],
            )

            ax[i].errorbar(
                x[j],
                value,
                yerr=np.expand_dims([value - lower_b, upper_b - value], 1),
                fmt="none",
                color="black",
                capsize=5,
            )

        # Customize the chart
        ax[i].set_title(y_labels[key], fontweight="bold")
        if key != "log_loss":
            ax[i].set_ylim([0, 1.05])
        else:
            ax[i].set_ylim([0, 0.35])
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)

        # Set the x-ticks to be at the center of each group of bars
        ax[i].set_xticks([])
        ax[i].set_xticklabels([])

    # Place one legend for the whole figure
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles[0 : len(label_list)],
        labels[0 : len(label_list)],
        loc="center right",
        bbox_to_anchor=(1.0, 0.55),
        title="Models",
        frameon=False,
    )

    ax[2].axis("off")
    fig.suptitle(f"{LABEL_MAP[outcome]}", fontweight="bold", x=0.35)

    # Set ax_kwargs to override if needed
    for key, val in ax_kwargs.items():
        getattr(ax, key)(val)

    # Adjust the size of the plots
    plt.tight_layout()

    # Save and view
    if save_path:
        save_file = save_path + extension
        plt.savefig(save_file)
    if no_plots:
        plt.show()

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract metrics from a trained Pipeline"
    )
    parser.add_argument(
        "model_config_file",
        metavar="model-config-file",
        help="Path to the general configuration file",
    )
    parser.add_argument("--version", help="Version number overload")
    parser.add_argument(
        "--no-plots", action="store_false", help="Flag used to turn off plotting"
    )

    args = parser.parse_args()

    try:
        print_message("Loading parameters from configuration file")
        # Read log and general configuration file
        log_config = read_toml_configuration("../../config/log.toml")
        general_config = read_toml_configuration(args.model_config_file)

        if args.version:
            # Swap version numbers if overloading
            general_config["version"] = args.version

        print_message("Setting up logger")
        # Create logger
        script_name = str(pathlib.Path(__file__).stem)
        script_name += "_" + general_config["version"]
        log_dir = log_config["base_dir"] + log_config["log_dir"]
        log_dir = log_config["log_dir"]
        logger = setup_logger(script_name, log_dir)

        print_message(
            f"Version number: {general_config["version"]}", logger, script_name
        )

    except (TypeError, ValueError, FileNotFoundError, IsADirectoryError) as err:
        sys.stderr.write("An error occured when trying to create the logger\n")
        sys.stderr.write(repr(err))
        exit(1)

    try:
        data_version, _ = split_version_number(general_config["version"])

        # Get data configuration parameters
        data_config = get_configuration(
            general_config["data_parameters"],
            data_version,
        )

    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)

    try:
        print_message("Getting data", logger, script_name)
        data = load_data_from_csv(
            get_file_path(
                data_config, v_number=data_version[:4]  # Use only first 2 numbers
            )
        )
        data.fillna({"ASA": 0}, inplace=True)  # Fill ASA nan values to 0

        # Load model
        print_message("Loading model", logger, script_name)
        load_file = get_file_path(
            general_config,
            v_number=general_config["version"],
        )
        pipeline = load_pipeline(load_file)

        data = pipeline.preprocessor.transform(data)
        X_train, X_test = pipeline.get_test_data(data, test_group_vals=[2023, 2024])

    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)

    try:
        print_message("Preparing test set", logger, script_name)
        X_test, y_test = extract_labels(X_test, pipeline.label_list)
        X_train, y_train = extract_labels(X_train, pipeline.label_list)

    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)

    # Compute statistics and plots
    try:
        print_message("Computing model statistics", logger, script_name)
        group_name = data_config["cv_variables"]["group_name"]
        save_file = get_file_path(
            general_config,
            v_number=general_config["version"],
            path_type="fig",
            exists=False,
        )
        extension = general_config["fig_parameters"]["extension"]

        for i, outcome in enumerate(pipeline.label_list):
            # Iterate over each outcome to calculate the training metrics
            print_message(outcome, logger, script_name)
            metric_dict_natural = {}
            metric_dict_recal = {}

            for key in pipeline.predictor_probabilities[outcome]:
                # Get probabilities from each fold
                y_test = y_train[X_train[group_name] == key]
                metric_dict_natural[key] = compute_score_metrics(
                    ["auroc", "log_loss"],
                    y_test[:, i],
                    get_full_proba(pipeline.predictor_probabilities[outcome][key]),
                )
                metric_dict_recal[key] = compute_score_metrics(
                    ["auroc", "log_loss"],
                    y_test[:, i],
                    get_full_proba(pipeline.calibrator_probabilities[outcome][key]),
                )

            ci_dict_natural = compute_all_CI(metric_dict_natural)
            ci_dict_recal = compute_all_CI(metric_dict_recal)

            # Merge both dictionaries natural then recalibrated
            ci_dict = {}
            for key, value in ci_dict_natural.items():
                val = (
                    np.append(value[0], ci_dict_recal[key][0]),
                    np.append(value[1], ci_dict_recal[key][1]),
                    np.append(value[2], ci_dict_recal[key][2]),
                )
                ci_dict[key] = val

            plot_metrics_CI(
                ci_dict,
                outcome,
                ["Natural", "Recalibrated"],
                dpi=300,
                no_plots=args.no_plots,
                save_path=save_file + f"_{outcome}_metrics",
                extension=extension,
            )
    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)
