#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_test_results.py

Computes the final results of the models using the test set.
"""

import argparse
import pathlib
import sys

from medpipe import (
    compute_score_metrics,
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
from scipy.special import logit
from numpy import clip
import numpy as np
import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression

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
        data.drop("DHB_NAME", axis=1, inplace=True)

        # Load model
        print_message("Loading model", logger, script_name)
        load_file = get_file_path(
            general_config,
            v_number=general_config["version"],
        )
        pipeline = load_pipeline(load_file)

        data = pipeline.preprocessor.transform(data)
        X_train, X_test = pipeline.get_test_data(data, test_group_vals=[2024, 2023])

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
        n_bootstraps = 1000

        for i, outcome in enumerate(pipeline.label_list):
            print_message(outcome, logger, script_name)
            y_pred_proba = pipeline.predict_proba(
                X_test, label_list=outcome, model_type="calibrator"
            )
            y_pred_pos = get_positive_proba(y_pred_proba)

            log_odds = logit(clip(y_pred_pos, 1e-15, 1-1e-15))
            X = sm.add_constant(log_odds)
            
            calib_model = sm.Logit(y_test[:, i], X).fit(disp=0)

            intercept, slope = calib_model.params

            metrics = compute_score_metrics(
                ["auroc", "log_loss"], y_test[:, i], y_pred_proba
            )

            rng = np.random.default_rng()
            boots = np.zeros((n_bootstraps, 4))  # Holds every metric bootstraps

            for j in range(n_bootstraps):
                idx = rng.choice(len(y_test), len(y_test), replace=True)
                y_boot = y_pred_pos[idx]
                log_odds = logit(clip(y_boot, 1e-15, 1-1e-15))
                X = sm.add_constant(log_odds)
                
                calib_model = sm.Logit(y_test[idx, i], X).fit(disp=0)

                boots[j, 0], boots[j, 1] = calib_model.params

                _metrics = compute_score_metrics(
                    ["auroc", "log_loss"], y_test[idx, i], y_pred_proba[idx]
                )
                boots[j, 2:4] = [_metrics['auroc'][0], _metrics['log_loss'][0]]

            lower = np.percentile(boots, 2.5, axis=0)
            upper = np.percentile(boots, 97.5, axis=0)

            print(f"{metrics['auroc'][0]:.2f} ({lower[2]:.2f} -- {upper[2]:.2f})")
            print(f"{metrics['log_loss'][0]:.2f} ({lower[3]:.2f} -- {upper[3]:.2f})")
            print(f"{slope:.2f} ({lower[1]:.2f} -- {upper[1]:.2f})")
            print(f"{intercept:.2f} ({lower[0]:.2f} -- {upper[0]:.2f})")

    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)
