#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
op-averages.py

Calculate average risk for each operation.
"""

import argparse
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from constants import MODEL_MAP
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create operation averages from a pipeline"
    )
    parser.add_argument(
        "model_config_file",
        metavar="model-config-file",
        help="Path to the general configuration file",
    )
    parser.add_argument(
        "save_path", metavar="save-path", help="Path to save the operation averages"
    )
    parser.add_argument(
        "--extension",
        "-e",
        help="Save path extension",
        choices={".joblib", ".pkl"},
        default=".joblib",
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
        print_message("Loading model", logger, script_name)
        load_file = get_file_path(
            general_config,
            v_number=general_config["version"],
        )
        pipeline = load_pipeline(load_file)
        X_train, X_test = pipeline.get_test_data(pipeline.transform(data))
        X_test, y_test = extract_labels(X_test, pipeline.label_list)

    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)

    try:
        ops = X_test["CATEGORY_LEVEL_2"].unique()
        op_averages = {}

        for op in ops:
            # Iterate over all operation types
            print_message(f"Preparing {op} set", logger, script_name)
            op_idx = pd.Index(X_test["CATEGORY_LEVEL_2"] == op)  # Operation indices
            X_test_op = X_test[op_idx]
            y_test_op = y_test[op_idx]
            op_averages[op] = {}  # Create empty dictionary

            for complication in MODEL_MAP.keys():
                # Iterate on complications to get correct model to use
                # Bootstrap 2000 times to create 95% CI
                boots = []
                for _ in range(10):
                    predicted_proba_boot = pipeline.predict_proba(
                        X_test_op.sample(n=len(y_test_op), replace=True),
                        complication,
                        MODEL_MAP[complication],
                    )
                    boots.append(np.mean(get_positive_proba(predicted_proba_boot)))

                # Get 95% CI
                if len(X_test_op) < 30:
                    # If not enough data to bootstrap
                    lower_b = None
                    upper_b = None
                else:
                    lower_b = np.percentile(boots, 2.5)
                    upper_b = np.percentile(boots, 97.5)

                # Append average with 95% CI to list
                op_averages[op][complication] = (
                    np.mean(boots),
                    lower_b,
                    upper_b,
                )

        save_file = args.save_path + args.extension
        joblib.dump(op_averages, save_file, compress=3)

    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)
