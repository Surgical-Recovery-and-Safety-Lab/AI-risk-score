#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clinical_calibration.py

Plots calibration curves for the clinical paper.
"""

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from constants import COLOUR_MAP, LABEL_MAP
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from medpipe import (
    exception_handler,
    extract_labels,
    get_positive_proba,
    load_data_from_csv,
    load_pipeline,
    print_message,
    read_toml_configuration,
    setup_logger,
)
from medpipe.utils.config import get_configuration, get_file_path, split_version_number
from medpipe.utils.exceptions import file_checks
from ml_insights import SplineCalib
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_clinical_calibration(
    y_test,
    proba_list,
    outcome,
    label,
    n_bootstraps=1000,
    save_path="",
    extension=".png",
    show_fig=True,
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
    outcome : str
        Predicted outcome.
    label : str
        Labels for the legend.
    distribution : bool, default: False
        Flag to plot the probability distribution as well.
    n_bootstraps : int, default: 200
        Number of iteration for the bootstrap.
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
    # Split arguments based on where they should be sent
    ax_kwargs = {key: value for key, value in kwargs.items() if key in dir(Axes)}
    fig_kwargs = {key: value for key, value in kwargs.items() if key in dir(Figure)}

    # Set figure properties
    fig, ax = plt.subplots(**fig_kwargs)
    grid_resolution = 100

    # Set title
    title = kwargs["set_title"] if "set_title" in kwargs.keys() else ""
    ax.set_title(title, fontweight="bold")
    ax_kwargs.pop("set_title")

    # Remove spines for aesthetics
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Plot perfect calibration
    ax.plot(
        np.linspace(0, 1, grid_resolution),
        np.linspace(0, 1, grid_resolution),
        "k--",
        label="Perfectly calibrated",
    )

    grid = np.linspace(0, 1, grid_resolution)
    main_spline = SplineCalib()
    main_spline.fit(proba_list, y_test)

    boots = np.zeros((n_bootstraps, grid_resolution))
    i = 0
    rng = np.random.default_rng()

    while i < n_bootstraps:
        idx = rng.choice(len(y_test), len(y_test), replace=True)
        try:
            spline_boot = SplineCalib()
            spline_boot.fit(proba_list[idx], y_test[idx])
            boots[i, :] = spline_boot.predict(grid)
            i += 1
        except Exception:
            continue

    lower = np.percentile(boots, 2.5, axis=0)
    upper = np.percentile(boots, 97.5, axis=0)

    ax.plot(
        grid,
        main_spline.predict(grid),
        color=COLOUR_MAP[outcome],
        label=label,
    )
    ax.fill_between(
        grid,
        lower,
        upper,
        color=COLOUR_MAP[outcome],
        alpha=0.5,
        label=f"{label} 95% CI",
    )

    # Create new plot for distribution
    divider = make_axes_locatable(ax)
    ax_dist = divider.append_axes("bottom", 0.5, pad=0.1)

    bins = np.linspace(0, 1, 21)

    ax_dist.hist(
        proba_list,
        stacked=True,
        color=COLOUR_MAP[outcome],
        edgecolor="black",
        bins=bins,
    )
    ax_dist.set_yscale("log")
    ax_dist.set_ylim(bottom=1, top=10e6)
    ax_dist.set_xlabel("Predicted probabilities", fontweight="bold")

    # Remove spines for aesthetics for distribution
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    ax.set_xlabel("Predicted probabilities", fontweight="bold")
    ax.set_ylabel("Observed proportion", fontweight="bold")

    # Set ax_kwargs to override if needed
    for key, val in ax_kwargs.items():
        getattr(ax, key)(val)

    ax.legend(loc="upper right", bbox_to_anchor=(1.6, 0.9), title="Key", frameon=False)

    fig.subplots_adjust(right=0.66, bottom=0.14)

    if save_path:
        save_file = save_path + extension
        file_checks(save_file, extension=extension, exists=False)
        plt.savefig(save_file)
    if show_fig:
        plt.show()

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the global calibration of each outcome"
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
        data = data.drop("DHB_NAME", axis=1)
        X_train, X_test = pipeline.get_test_data(data, test_group_vals=[2024])

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
        group_name = data_config["split_variables"]["group_name"]
        save_file = get_file_path(
            general_config,
            v_number=general_config["version"],
            path_type="fig",
            exists=False,
        )
        extension = general_config["fig_parameters"]["extension"]

        for i, outcome in enumerate(pipeline.label_list):
            print_message(outcome, logger, script_name)
            y_pred_proba = pipeline.predict_proba(
                X_test,
                label_list=outcome,
                model_type="calibrator",
            )

            y_pred_proba = np.squeeze(get_positive_proba(y_pred_proba))

            plot_clinical_calibration(
                y_test[:, i],
                y_pred_proba,
                outcome=outcome,
                label="Calibration",
                distribution=True,
                save_path=save_file + f"_{outcome}_reliability_diagram",
                extension=extension,
                n_bootstraps=4,
                show_fig=args.no_plots,
                dpi=300,
                figsize=(5, 5),
                set_title=f"{LABEL_MAP[outcome]}",
            )
            plt.close()
    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)
