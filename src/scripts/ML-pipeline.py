#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-pipeline.py

Machine learning pipeline creation and training script.
"""

import argparse
import pathlib
import sys

from medpipe import (
    Pipeline,
    compute_all_CI,
    compute_score_metrics,
    exception_handler,
    extract_labels,
    get_full_proba,
    get_positive_proba,
    load_data_from_csv,
    load_pipeline,
    plot_metrics_CI,
    plot_prediction_distribution,
    plot_reliability_diagrams,
    print_message,
    read_toml_configuration,
    save_pipeline,
    setup_logger,
)
from medpipe.utils.config import get_configuration, get_file_path, split_version_number

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
            pipeline = Pipeline(general_config, logger)
            X_train, _ = pipeline.get_test_data(data)
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

        data = pipeline.preprocessor.transform(data)
        X_train, X_test = pipeline.get_test_data(data)

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
        label_list = ["Unadjusted", "Recalibrated"]

        for i, label in enumerate(pipeline.label_list):
            # Plot for each outcome individually
            metric_dict = {}  # Store unadjusted values
            metric_dict_cal = {}  # Store recalibrated values

            for key in pipeline.predictor_probabilities[label]:
                # Compute metric values for both unadjusted and recalibrated
                y_true = y_train[X_train[group_name] == key]
                metric_dict[key] = compute_score_metrics(
                    ["auroc", "ap", "log_loss"],
                    y_true[:, i],
                    get_full_proba(pipeline.predictor_probabilities[label][key]),
                )
                metric_dict_cal[key] = compute_score_metrics(
                    ["auroc", "ap", "log_loss"],
                    y_true[:, i],
                    get_full_proba(pipeline.calibrator_probabilities[label][key]),
                )

                for k in metric_dict[key].keys():
                    metric_dict[key][k] += metric_dict_cal[key][k]

            # Create one CI dict for both outcomes and plot results
            ci_dict = compute_all_CI(metric_dict)
            plot_metrics_CI(
                ci_dict,
                label_list=label_list,
                dpi=300,
                figsize=(5, 5),
                save_path=save_file + f"_{label}_",
                extension=extension,
                show_fig=args.no_plots,
            )

            y_pred_proba = pipeline.predict_proba(
                X_test, label_list=label, model_type="predictor"
            )
            y_pred_proba_calib = pipeline.predict_proba(
                X_test, label_list=label, model_type="calibrator"
            )

            plot_prediction_distribution(
                [
                    get_positive_proba(y_pred_proba).squeeze(),
                    get_positive_proba(y_pred_proba_calib).squeeze(),
                ],
                label_list=label_list,
                save_path=save_file + f"_{label}_proba_dist",
                extension=extension,
                show_fig=args.no_plots,
                n_bins=20,
                dpi=300,
                figsize=(5, 5),
            )
            plot_reliability_diagrams(
                y_test[:, i],
                [
                    get_positive_proba(y_pred_proba).squeeze(),
                    get_positive_proba(y_pred_proba_calib).squeeze(),
                ],
                label_list=label_list,
                save_path=save_file + f"_{label}_reliability_diagram",
                extension=extension,
                show_fig=args.no_plots,
                calibration_kwargs={"n_bins": 10, "strategy": "quantile"},
                dpi=300,
                figsize=(5, 5),
            )

    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)
