#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fairness.py

Computes fairness of models based on different strata.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from medpipe import (
    Pipeline,
    compute_score_metrics,
    extract_labels,
    get_positive_proba,
    load_data_from_csv,
    load_pipeline,
    print_message,
    read_toml_configuration,
    setup_logger,
)
from medpipe.utils.config import get_configuration, get_file_path
from ml_insights import SplineCalib
from pandas import DataFrame

from constants import MODEL_MAP

METRIC_MAP = {
    "auroc": "AUROC",
    "log_loss": "Log loss",
    "ici": "ICI",
}

COLOUR_MAP = [
    "#2D90D8",
    "#33367A",
    "#96690E",
    "#CDB4DB",
    "#F2CC8F",
]


MORT_MAP = {
    "MORTALITY_30D": "30-day mortality",
    "MORTALITY_90D": "90-day mortality",
    "MORTALITY_1Y": "1-year mortality",
    "READMIT_ACUTE_30D": "30-day acute readmission",
    "READMIT_ACUTE_90D": "90-day acute readmission",
    "ANY_COMP": "Any complications",
}
COMPLICATION_MAP = {
    "AKI": "AKI",
    "CARDIAC_ARRHYTHMIA": "Cardiac arrhythmia",
    "DELIRIUM": "Delirium",
    "GI_BLEEDING": "GI bleeding",
    "HAEMORRHAGE": "Haemorrhage",
    "IMPLANT_GRAFT": "IGC",
    "MYOCARDIAL_EVENT": "Myocardial event",
    "RESPIRATORY_FAILURE": "Respiratory failure",
    "PNEUMONIA": "Pneumonia",
    "SEPSIS": "Sepsis",
    "SHOCK": "Shock",
    "SSI": "SSI",
    "STROKE": "Stroke",
    "VTE": "VTE",
    "UTI": "UTI",
}

LABEL_MAP = MORT_MAP | COMPLICATION_MAP


def compute_metrics(
    pipeline: Pipeline, data: DataFrame, strata_name: str, strata: list[str] | int
):
    """
    Compute metrics for the different starta of data.

    Computed metrics are AUROC, log loss, calibration slope and intercept.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline used to compute metrics with.
    data : DataFrame
        Data to use for testing.
    strata_name : str
        Name of the column to use for stratified data.
    strata : list[str] | int
        Strata to use to extract the desired data.

    Returns
    -------
    stratified_scores : dict[str, dict[str, list[float]]]
        Dictionary of scores for each strata.

    """
    if type(strata) is int:
        strata = [strata, strata]  # Convert to list to get be iterable
    stratified_scores = {}

    for i, s in enumerate(strata):
        print_message(f"Preparing {s} test set", logger, script_name)
        if type(s) is int and i == 0:
            _data = data.loc[data[strata_name] < s]
            s = rf"< {s}"
        elif type(s) is int and i == 1:
            _data = data.loc[data[strata_name] >= s]
            s = rf"$\geq$ {s}"
        else:
            _data = data.loc[data[strata_name] == s]

        s_data = pipeline.preprocessor.transform(_data)

        X_train, X_test = pipeline.get_test_data(s_data, test_group_vals=[2024])
        X_train, y_train = extract_labels(X_train, pipeline.label_list)

        X_test, y_test = extract_labels(X_test, pipeline.label_list)

        metric_dict = {}

        for i, label in enumerate(pipeline.label_list):
            y_proba = pipeline.predict_proba(X_test, label, MODEL_MAP[label])
            y_proba_pos = np.squeeze(get_positive_proba(y_proba))

            # Train SplineCalib on calibration data
            y_train_proba = pipeline.predict_proba(X_train, label, MODEL_MAP[label])
            y_train_pos = np.squeeze(get_positive_proba(y_train_proba))
            spline = SplineCalib(logodds_scale=True)
            spline.fit(y_train_pos, y_train[:, i])
            smoothed_proba = spline.calibrate(y_proba_pos)

            scores = compute_score_metrics(["auroc", "log_loss"], y_test[:, i], y_proba)
            metric_dict[label] = scores | {
                "ici": np.mean(np.abs(smoothed_proba - y_proba_pos)),
            }
        stratified_scores[s] = metric_dict

    return stratified_scores


def strata_heatmap(
    outcome_list,
    metric_list,
    og_scores,
    strata_scores,
    save_path="",
    extension=".png",
    show_fig=True,
    **kwargs,
):
    """
    Plots the strata delta scores in a heatmap.

    Parameters
    ----------
    outcome_list : list[str]
        List of outcomes.
    metric_list : list[str]
        List of the metrics being plotted.
    og_scores : list[float]
        Scores for the original models.
    strata_scores : dict[str, dict[str, dict[str, list[float]]]]
        Dictionary of scores with the strata name as key.
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

    # Get title
    title = ""
    if "set_title" in kwargs.keys():
        title = kwargs["set_title"]
        ax_kwargs.pop("set_title")

    for metric in metric_list:  # Go through all metrics
        # Set figure properties
        fig, ax = plt.subplots(**fig_kwargs)

        # Set title
        if metric == "ici":
            ax.set_title(title + METRIC_MAP[metric] + " (%)", fontweight="bold")
        else:
            ax.set_title(title + METRIC_MAP[metric], fontweight="bold")

        strata_list = []  # List to contain the strata names for x-axis
        strata_plot_data = []  # List to contain the data to plot
        strata_plot_text = []  # List to contain text to display

        for strata_key, strata_data in strata_scores.items():
            tmp_strata_data = []  # Create an empty list to store strata data
            tmp_strata_text = []  # Create an empty list to store text to display
            strata_list.append(strata_key)

            for outcome in outcome_list:
                # Get strata data for all outcomes
                tmp_strata_data.append(
                    np.abs(
                        np.array(strata_data[outcome][metric])
                        - np.array(og_scores[outcome][metric])
                    )
                )
                tmp_strata_text.append(np.array(strata_data[outcome][metric]))

            strata_plot_data.append(np.array(tmp_strata_data))
            strata_plot_text.append(np.array(tmp_strata_text))

        strata_plot_arr = np.squeeze(np.array(strata_plot_data))
        strata_text_arr = np.squeeze(np.array(strata_plot_text))
        max_val = 0.1
        percent = ""
        if metric == "ici":
            strata_text_arr *= 100
            strata_plot_arr *= 100
            max_val = 0.5
            percent = " (%)"

        # Display heatmap
        im = ax.imshow(strata_plot_arr, cmap="cividis", aspect="equal", vmax=max_val)

        # Set colorbar and ticks
        fig.colorbar(
            im,
            ax=ax,
            cmap="cividis",
            shrink=0.8,
            extend="max",
            label=rf"|$\Delta$ {METRIC_MAP[metric]}|" + percent,
        )
        ax.set_yticks(
            np.arange(len(strata_list)),
            labels=strata_list,
        )

        outcome_label_list = [LABEL_MAP[ele] for ele in outcome_list]
        ax.set_xticks(
            np.arange(len(outcome_list)),
            labels=outcome_label_list,
            rotation=-30,
            rotation_mode="anchor",
            ha="left",
        )
        ax.set_ylabel("Strata", fontweight="bold")
        ax.set_xlabel("Outcomes", fontweight="bold")

        # Add text
        for i in range(len(outcome_label_list)):
            for j in range(len(strata_list)):
                colour = "w"
                if np.abs(strata_plot_arr[j, i]) >= 0.8 * max_val:
                    colour = "k"
                ax.text(
                    i,
                    j,
                    np.round(strata_text_arr[j, i], 2),
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=6,
                    fontweight="bold",
                )

        ax.spines[:].set_visible(False)

        # Set white space between squares
        ax.set_yticks(np.arange(len(strata_list) + 1) - 0.5, minor=True)
        ax.set_xticks(np.arange(len(outcome_list) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        plt.tight_layout()

        # Set ax_kwargs to override if needed
        for key, val in ax_kwargs.items():
            getattr(ax, key)(val)

        if save_path:
            save_file = save_path + metric + extension
            plt.savefig(save_file)
        if show_fig:
            plt.show()

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the fairness of the models in a heatmap."
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
    version = general_config["version"]

    # Get data configuration parameters
    data_config = get_configuration(
        general_config["data_parameters"],
        version[:8],
    )

    print_message("Getting data", logger, script_name)
    data = load_data_from_csv(
        get_file_path(data_config, v_number=version[:4])  # Use only first 2 numbers
    )
    # Load model
    print_message("Loading model", logger, script_name)
    load_file = get_file_path(general_config, v_number=version)
    pipeline: Pipeline = load_pipeline(load_file)

    if "ASA" in data.columns:
        data.fillna({"ASA": 0}, inplace=True)
    if "DHB_NAME" in data.columns:
        # Remove Overseas and undefined cases
        data.drop("DHB_NAME", axis=1, inplace=True)
    data.dropna(inplace=True)  # Remove Nan values
    _, strata_data = pipeline.get_test_data(data, test_group_vals=[2023, 2024])

    X_train, X_test = pipeline.get_test_data(strata_data, test_group_vals=[2024])
    X_train, y_train = extract_labels(X_train, pipeline.label_list)
    X_train = pipeline.preprocessor.transform(X_train)

    X_test, y_test = extract_labels(X_test, pipeline.label_list)
    X_test = pipeline.preprocessor.transform(X_test)
    og_metrics = {}  # Store the metrics for all labels with the original data

    for i, label in enumerate(pipeline.label_list):
        y_proba = pipeline.predict_proba(X_test, label, MODEL_MAP[label])
        y_pred_pos = np.squeeze(get_positive_proba(y_proba))

        # Train SplineCalib on calibration data
        y_train_proba = pipeline.predict_proba(X_train, label, MODEL_MAP[label])
        y_train_pos = np.squeeze(get_positive_proba(y_train_proba))
        spline = SplineCalib(logodds_scale=True)
        spline.fit(y_train_pos, y_train[:, i])
        smoothed_proba = spline.calibrate(y_pred_pos)
        scores = compute_score_metrics(["auroc", "log_loss"], y_test[:, i], y_proba)
        og_metrics[label] = scores | {
            "ici": np.mean(np.abs(smoothed_proba - y_pred_pos)),
        }

    sex_strata = data["SEX"].unique()
    eth_strata = data["ETHNICITY"].unique()
    age_strata = 50

    sex_scores = compute_metrics(pipeline, strata_data, "SEX", sex_strata)
    age_scores = compute_metrics(pipeline, strata_data, "AGE", age_strata)
    eth_scores = compute_metrics(pipeline, strata_data, "ETHNICITY", eth_strata)

    strata_metrics = {"All strata": og_metrics} | eth_scores | age_scores | sex_scores

    save_file = get_file_path(
        general_config,
        v_number=general_config["version"],
        path_type="fig",
        exists=False,
    )

    strata_heatmap(
        MORT_MAP.keys(),
        METRIC_MAP.keys(),
        og_metrics,
        strata_metrics,
        dpi=300,
        set_title=" ",
        show_fig=args.no_plots,
        save_path=save_file + "_mort_readmit_fairness_",
    )

    strata_heatmap(
        COMPLICATION_MAP.keys(),
        METRIC_MAP.keys(),
        og_metrics,
        strata_metrics,
        dpi=300,
        set_title=" ",
        show_fig=args.no_plots,
        save_path=save_file + "_fairness_",
    )
