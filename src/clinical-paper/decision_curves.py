#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decision_curves.py

Computes the DCA for mortality, readmission, and complications.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from constants import LABEL_MAP, MODEL_MAP
from medpipe import (
    extract_labels,
    get_positive_proba,
    load_data_from_csv,
    load_pipeline,
    print_message,
    read_toml_configuration,
)
from medpipe.utils.config import get_configuration, get_file_path, split_version_number
from sklearn.metrics import confusion_matrix


def net_benefit_fn(tp, fp, P, N):
    numerator = tp - (fp * P)
    return numerator / N


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot decision curves.")
    parser.add_argument(
        "model_config_file",
        metavar="model-config-file",
        help="Path to the general configuration file",
    )

    args = parser.parse_args()
    thresholds = np.linspace(0.0, 0.5, 100)

    print_message("Loading parameters from configuration file")
    # Read log and general configuration file
    log_config = read_toml_configuration("../../config/log.toml")
    general_config = read_toml_configuration(args.model_config_file)
    logger = None
    script_name = ""
    save_file = get_file_path(
        general_config,
        v_number="",
        path_type="fig",
        exists=False,
    )
    extension = general_config["fig_parameters"]["extension"]
    outcomes = [
        "MORTALITY_30D",
        "MORTALITY_90D",
        "MORTALITY_1Y",
        "READMIT_ACUTE_30D",
        "READMIT_ACUTE_90D",
        "ANY_COMP",
    ]
    colours = ["#2D90D8", "#33367A", "#96690E", "#CDB4DB", "#F2CC8F", "#1D6968"]

    # Set figure and axes properties
    fig, ax = plt.subplots(dpi=300, figsize=(8, 5))  # Create a new figure

    ax.hlines(
        0,
        xmin=thresholds[0],
        xmax=thresholds[-1],
        color="k",
        linestyles="-",
        label="Treat none",
    )
    P = thresholds / (1 - thresholds)

    print_message(f"Version number: {general_config["version"]}", logger, script_name)
    data_version, _ = split_version_number(general_config["version"])

    # Get data configuration parameters
    data_config = get_configuration(
        general_config["data_parameters"],
        data_version,
    )
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
    _, X_test = pipeline.get_test_data(pipeline.transform(data))
    X_test, y_test = extract_labels(X_test, pipeline.label_list)

    for i, outcome in enumerate(outcomes):
        net_benefit = np.zeros(len(thresholds))
        y_pred_proba = get_positive_proba(
            pipeline.predict_proba(X_test, outcome, MODEL_MAP[outcome])
        ).squeeze()

        for j, threshold in enumerate(thresholds):
            decisions = y_pred_proba > threshold
            _, fp, _, tp = confusion_matrix(y_test[:, i], decisions).ravel()
            net_benefit[j] = net_benefit_fn(tp, fp, P[j], len(y_test))

        if i == 0:
            prevalence = np.sum(y_test[:, i]) / len(y_test[:, i])
            ax.plot(
                thresholds,
                net_benefit_fn(
                    prevalence,
                    1 - prevalence,
                    P,
                    1,
                ),
                color="k",
                label="Treat all",
                linestyle="--",
            )

        ax.plot(
            thresholds,
            net_benefit,
            color=colours[i],
            label=LABEL_MAP[outcome],
        )

    ax.set_xlabel("Decision threshold", fontweight="bold")
    ax.set_ylabel("Net benefit", fontweight="bold")
    ax.set_ylim(bottom=-0.01, top=0.15)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
    fig.subplots_adjust(right=0.89, bottom=0.132, left=0.18, top=0.96)

    save_path = save_file + f"{general_config["version"]}_decision_curve" + extension
    plt.legend(frameon=False)
    plt.savefig(save_path)
    plt.show()
