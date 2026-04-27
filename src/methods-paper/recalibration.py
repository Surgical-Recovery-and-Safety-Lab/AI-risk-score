#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recalibration.py

Plots recalibration curves for the methods paper.
"""

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.calibration import calibration_curve


def plot_recalibration(
    y_test,
    proba_list,
    labels,
    n_bootstraps=1000,
    outcome="",
    save_path="",
    extension=".png",
    show_fig=True,
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
    labels : list[str]
        Labels for the legend.
    distribution : bool, default: False
        Flag to plot the probability distribution as well.
    n_bootstraps : int, default: 200
        Number of iteration for the bootstrap.
    outcome : str, default: ""
        Outcome that is being plotted.
    save_path : str, default: []
        Path to the save file.
    extension : str, default: ".png"
        Extension to save figure in.
    show_fig : bool, default: True
        Flag to show the figure.
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

    # Define colour map
    COLOUR_MAP = [
        "#2D90D8",
        "#33367A",
        "#96690E",
        "#CDB4DB",
        "#F2CC8F",
    ]

    # Set figure properties
    fig, ax = plt.subplots(**fig_kwargs)

    # Set title
    title = kwargs["set_title"] if "set_title" in kwargs.keys() else ""
    ax.set_title(title, fontweight="bold")
    ax_kwargs.pop("set_title")

    # Remove spines for aesthetics
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Plot perfect calibration
    ax.plot(
        np.linspace(0, 1, 100),
        np.linspace(0, 1, 100),
        "k--",
        label="Perfectly calibrated",
    )

    for j, proba in enumerate(proba_list):
        prob_true, prob_pred = calibration_curve(
            y_test,
            proba,
            **calibration_kwargs,
        )

        boots = []

        for _ in range(n_bootstraps):
            idx = np.random.choice(len(y_test), len(y_test), replace=True)
            prob_true_boot, prob_pred_boot = calibration_curve(
                y_test[idx],
                proba[idx],
                **calibration_kwargs,
            )
            boots.append(np.interp(prob_pred, prob_pred_boot, prob_true_boot))

        lower = np.percentile(boots, 2.5, axis=0)
        upper = np.percentile(boots, 97.5, axis=0)

        ax.plot(
            prob_pred,
            prob_true,
            marker=".",
            color=COLOUR_MAP[j],
            label=labels[j],
        )
        ax.fill_between(
            prob_pred,
            lower,
            upper,
            color=COLOUR_MAP[j],
            alpha=0.5,
            label=f"{labels[j]} 95% CI",
        )

        if outcome == "MORTALITY_90D":
            # Add inset if calibration curve does not cover enough of the graph
            # Remove spines for aesthetics
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)

            if j == 0:
                # Create inset only in first loop iteration
                ax_ins = ax.inset_axes(
                    [0.05, 0.5, 0.4, 0.4],
                    xlim=(-0.02, 0.2),
                    ylim=(-0.02, 0.2),
                    yticklabels=[],
                    xticklabels=[],
                )
                # Connect the inset to the zoomed area in the main plot
                ax.indicate_inset_zoom(ax_ins, edgecolor="black")

                ax_ins.plot(  # Reference line
                    np.linspace(0, max(prob_pred), 100),
                    np.linspace(0, max(prob_pred), 100),
                    "k--",
                )

            # Plot inset
            ax_ins.plot(
                prob_pred,
                prob_true,
                marker=".",
                color=COLOUR_MAP[j],
                label=labels[j],
            )
            ax_ins.fill_between(
                prob_pred,
                lower,
                upper,
                color=COLOUR_MAP[j],
                alpha=0.5,
                label=f"{labels[j]} 95% CI",
            )

    # Create new plot for distribution
    divider = make_axes_locatable(ax)
    ax_dist = divider.append_axes("bottom", 0.5, pad=0.1)

    bins = np.linspace(0, 1, 21)

    ax_dist.hist(
        proba_list,
        stacked=True,
        color=COLOUR_MAP[:3],
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

    ax.legend(loc="upper right", bbox_to_anchor=(1.75, 0.9), title="Key", frameon=False)

    fig.subplots_adjust(right=0.63, bottom=0.14)

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
    parser.add_argument(
        "--no-plots", action="store_false", help="Flag used to turn off plotting"
    )

    args = parser.parse_args()

    try:
        print_message("Loading parameters from configuration file")
        # Read log and general configuration file
        log_config = read_toml_configuration("../../config/log.toml")
        general_config = read_toml_configuration(args.model_config_file)

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

        load_file = get_file_path(general_config, v_number=general_config["version"])
        pipeline = load_pipeline(load_file)
        data = pipeline.preprocessor.transform(data)
        _, X_test = pipeline.get_test_data(data, test_group_vals=[2024])

        print_message("Preparing test set", logger, script_name)
        X_test, y_test = extract_labels(X_test, pipeline.label_list)

        # Define all versions to load as tuples
        versions = [
            ("v0.1.1.1-a.2.2.1", "v0.1.1.1-a.2.2.2"),
            ("v0.1.1.1-a.6.2.1", "v0.1.1.1-a.6.2.2"),
            ("v0.1.1.1-a.10.2.1", "v0.1.1.1-a.10.2.2"),
            ("v0.1.1.1-a.14.2.1", "v0.1.1.1-a.14.2.2"),
        ]
        labels = [
            ("CSL", "Platt recalibration", "Isotonic recalibration"),
            ("SMOTE", "Platt recalibration", "Isotonic recalibration"),
            ("ROS", "Platt recalibration", "Isotonic recalibration"),
            ("RUS", "Platt recalibration", "Isotonic recalibration"),
        ]

        for j, (platt_version, isotonic_version) in enumerate(versions):
            # Load models
            print_message("Loading models", logger, script_name)
            platt_load_file = get_file_path(general_config, v_number=platt_version)
            isotonic_load_file = get_file_path(
                general_config, v_number=isotonic_version
            )
            platt_pipeline = load_pipeline(platt_load_file)
            isotnic_pipeline = load_pipeline(isotonic_load_file)

            save_file = get_file_path(
                general_config,
                v_number="recalibration",
                path_type="fig",
                exists=False,
            )
            extension = general_config["fig_parameters"]["extension"]

            for i, outcome in enumerate(pipeline.label_list):
                print_message(outcome, logger, script_name)
                y_proba = platt_pipeline.predict_proba(
                    X_test,
                    label_list=outcome,
                    model_type="predictor",
                )
                y_platt = platt_pipeline.predict_proba(
                    X_test,
                    label_list=outcome,
                    model_type="calibrator",
                )
                y_iso = isotnic_pipeline.predict_proba(
                    X_test,
                    label_list=outcome,
                    model_type="calibrator",
                )
                plot_recalibration(
                    y_test[:, i],
                    [
                        get_positive_proba(y_proba).squeeze(),
                        get_positive_proba(y_platt).squeeze(),
                        get_positive_proba(y_iso).squeeze(),
                    ],
                    labels=labels[j],
                    distribution=True,
                    outcome=outcome,
                    save_path=save_file + f"_{outcome}_{labels[j][0]}",
                    extension=extension,
                    show_fig=args.no_plots,
                    calibration_kwargs={"n_bins": 10, "strategy": "quantile"},
                    dpi=300,
                    figsize=(5, 5),
                    set_title=f"{labels[j][0]}",
                )

    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)
