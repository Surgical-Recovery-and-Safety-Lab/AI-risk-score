#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-models.py

Machine learning model creation and training script.
"""

import argparse
import pathlib
import sys

from pyrisk.data.preprocessing import extract_labels
from pyrisk.metrics.core import compute_all_CI, print_metrics_CI
from pyrisk.metrics.plots import (
    plot_metrics_CI,
    plot_prediction_distribution,
    plot_reliability_diagrams,
)
from pyrisk.models.core import get_positive_proba, load_pipeline, save_pipeline
from pyrisk.pipeline.Pipeline import Pipeline
from pyrisk.utils.config import get_configuration, get_file_path, split_version_number
from pyrisk.utils.io import load_data_from_csv, read_toml_configuration
from pyrisk.utils.logger import exception_handler, print_message, setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and train a Pipeline of machine learning models"
    )
    parser.add_argument(
        "model_config_file",
        metavar="model-config-file",
        help="Path to the general configuration file",
    )
    parser.add_argument("--load", "-l", action="store_true", help="Loading flag")
    parser.add_argument("--version", help="Version number overload for multiprocessing")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Flag to turn off printing"
    )
    parser.add_argument(
        "--no-plots", action="store_false", help="Flag used to turn off plotting"
    )

    args = parser.parse_args()

    try:
        print_message("Loading parameters from configuration file")
        # Read log and general configuration file
        log_config = read_toml_configuration("../config/log.toml")
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

        if args.quiet:
            # If quiet flag, turn off printing to screen
            logger.setLevel(-1)

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
        pipeline = Pipeline(general_config, logger)

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

        if not args.load:
            pipeline.preprocessor.fit(data)
            X_train, X_test = pipeline.get_test_data(data)
            pipeline.run(X_train)

            print_message("Saving pipeline", logger, script_name)
            save_file = get_file_path(
                general_config,
                v_number=general_config["version"],
                exists=False,
            )
            save_pipeline(pipeline, save_file)

        else:
            # Load model
            print_message("Loading model", logger, script_name)
            load_file = get_file_path(
                general_config,
                v_number=general_config["version"],
            )
            pipeline = load_pipeline(load_file)
            X_train, X_test = pipeline.get_test_data(data)

    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)

    try:
        print_message("Preparing test set", logger, script_name)
        X_test = pipeline.preprocessor.transform(X_test)
        X_test, y_test = extract_labels(X_test, pipeline.label_list)

    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)

    # Compute statistics and plots
    try:
        print_message("Computing model statistics", logger, script_name)
        ci_dict = compute_all_CI(pipeline.predictor_metrics)
        ci_calib_dict = compute_all_CI(pipeline.calibrator_metrics)

        print_message("Uncalibrated statistics", logger, script_name)
        print_metrics_CI(ci_dict, pipeline.label_list, logger)

        print_message("Calibrated statistics", logger, script_name)
        print_metrics_CI(ci_calib_dict, pipeline.label_list, logger)

        save_file = get_file_path(
            general_config,
            v_number=general_config["version"],
            path_type="fig",
            exists=False,
        )
        extension = general_config["fig_parameters"]["extension"]

        plot_metrics_CI(
            ci_dict,
            pipeline.label_list,
            dpi=300,
            figsize=(5, 5),
            show_fig=args.no_plots,
            save_path=save_file + "_metrics",
            extension=extension,
        )
        plot_metrics_CI(
            ci_calib_dict,
            pipeline.label_list,
            dpi=300,
            figsize=(5, 5),
            show_fig=args.no_plots,
            save_path=save_file + "_calibrated_metrics",
            extension=extension,
        )

        y_pred_proba = pipeline.predict_proba(X_test, "predictor")
        y_pred_proba_calib = pipeline.predict_proba(X_test, "calibrator")

        plot_prediction_distribution(
            get_positive_proba(y_pred_proba),
            get_positive_proba(y_pred_proba_calib),
            label_list=pipeline.label_list,
            save_path=save_file + "_proba_dist",
            extension=extension,
            show_fig=args.no_plots,
            n_bins=50,
            dpi=300,
            figsize=(5, 5),
        )
        plot_reliability_diagrams(
            y_test,
            get_positive_proba(y_pred_proba),
            get_positive_proba(y_pred_proba_calib),
            label_list=pipeline.label_list,
            save_path=save_file + "_reliability_diagram",
            extension=extension,
            show_fig=args.no_plots,
            display_kwargs={"n_bins": 10, "strategy": "quantile"},
            dpi=300,
            figsize=(5, 5),
        )

        # pyrisk.metrics.plots.plot_mean_ROC_curve(model_metrics, label_list)
        # pyrisk.metrics.plots.plot_mean_PR_curve(model_metrics, label_list)

    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)
