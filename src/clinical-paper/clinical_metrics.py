#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clinical_metrics.py

Plots the metrics for the clinical paper.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
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
from ml_insights import SplineCalib

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
        df = pd.DataFrame()
        long_df = pd.DataFrame()
        long_dict = {}

        for i, outcome in enumerate(pipeline.label_list):
            # Iterate over each outcome to calculate the training metrics
            print_message(outcome, logger, script_name)
            metric_dict_natural = {}

            for key in pipeline.predictor_probabilities[outcome]:
                # Get probabilities from each fold
                y_test = y_train[X_train[group_name] == key]
                y_pred = pipeline.predictor_probabilities[outcome][key]
                metric_dict_natural[key] = compute_score_metrics(
                    ["auroc"],
                    y_test[:, i],
                    get_full_proba(y_pred),
                )

                # Train SplineCalib
                spline = SplineCalib(logodds_scale=True)
                spline.fit(np.squeeze(y_pred), y_test[:, i])
                smoothed_proba = spline.calibrate(np.squeeze(y_pred))
                metric_dict_natural[key] = metric_dict_natural[key] | {
                    "ici": [np.mean(np.abs(smoothed_proba - np.squeeze(y_pred)) * 100)]
                }

                long_dict[key] = {
                    f"{outcome}_auroc": metric_dict_natural[key]["auroc"][0],
                    f"{outcome}_ici": metric_dict_natural[key]["ici"][0],
                }
            ci_dict_natural = compute_all_CI(metric_dict_natural)

            df_dict = {}
            df_dict["outcome"] = outcome
            df_dict["auroc"] = (
                f"{ci_dict_natural["auroc"][0][0]:.2f} ({ci_dict_natural["auroc"][1][0]:.2f} -- {ci_dict_natural["auroc"][2][0]:.2f})"
            )
            df_dict["ici"] = (
                f"{ci_dict_natural["ici"][0][0]:.2f} ({ci_dict_natural["ici"][1][0]:.2f} -- {ci_dict_natural["ici"][2][0]:.2f})"
            )
            df = pd.concat([df, pd.DataFrame(df_dict, index=[i])], ignore_index=True)
            long_df = pd.concat(
                [long_df, pd.DataFrame(long_dict)],
            )

        df.to_csv(save_file + "_training_table.csv", index=False)
        long_df.to_csv(save_file + "_DHB_table.csv", index=True)
    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)
