#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HGB-model.py

Histogram Gradient Boosting model creation and training script.
"""

import argparse
import pathlib
import sys

import pandas as pd
import pyrisk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and train a Histogram Gradient Boosting model"
    )
    parser.add_argument(
        "model_config_file",
        metavar="model-config-file",
        help="Path to the model configuration file",
    )
    parser.add_argument("--load", "-l", action="store_true", help="Loading flag")

    args = parser.parse_args()

    try:
        print("[INFO] Setting up logger")
        # Read log configuration file
        log_config = pyrisk.utils.io.read_toml_configuration("../config/log.toml")

        # Create logger
        script_name = str(pathlib.Path(__file__).stem)
        log_path = log_config["log_path"]
        logger = pyrisk.utils.logger.setup_logger(
            script_name,
            log_path,
        )
    except (TypeError, ValueError, FileNotFoundError, IsADirectoryError) as err:
        sys.stderr.write("An error occured when trying to create the logger")
        sys.stderr.write(repr(err))
        exit(1)

    try:
        print("[INFO] Loading parameters from configuration file")
        general_config = pyrisk.utils.io.read_toml_configuration(args.model_config_file)

        data_version, model_version = pyrisk.utils.config.split_version_number(
            general_config["version"]
        )

        # Get model configuration parameters
        model_config = pyrisk.utils.config.get_configuration(
            general_config["model_parameters"],
            model_version,
            join_token="",
        )

        # Get data configuration parameters
        data_config = pyrisk.utils.config.get_configuration(
            general_config["data_parameters"],
            data_version,
        )
    except Exception:
        pyrisk.utils.exceptions.exception_handler(
            logger, log_path, log_config, script_name
        )
        exit(1)

    try:
        if not args.load:
            print("[INFO] Getting data")
            data = pyrisk.utils.io.load_data_from_csv(
                pyrisk.utils.config.get_file_path(data_config, v_number=data_version)
            )
            # Convert objects to categorical (not saved so needs to be here)
            data = pyrisk.data.preprocessing.convert_object_to_categorical(data)

            # Split data
            print("[INFO] Splitting data")
            split_vars = data_config["split_variables"]

            kfold_it = pyrisk.data.preprocessing.test_train_it(**split_vars)

            ## MODEL CREATION AND TRAINING
            print("[INFO] Creating model")
            model = pyrisk.models.core.create_model(
                "hgb", **model_config["config_parameters"]
            )

            print("[INFO] Training model")
            model_metrics = pyrisk.models.core.train_model(
                model,
                data,
                kfold_it,
                model_config["labels"]["label_list"],
                split_vars["group_name"],
            )

            print("[INFO] Saving model")
            save_file = pyrisk.utils.config.get_file_path(
                general_config,
                v_number=general_config["version"],
                exists=False,
            )
            pyrisk.models.core.save_model(model, save_file, model_metrics)

        else:
            # Load model
            print("[INFO] Loading model")
            load_file = pyrisk.utils.config.get_file_path(
                general_config,
                v_number=general_config["version"],
            )

            model, model_metrics = pyrisk.models.core.load_model(load_file)

    except Exception:
        pyrisk.utils.exceptions.exception_handler(
            logger, log_path, log_config, script_name
        )
        exit(1)

    # Compute statistics
    try:
        print("[INFO] Computing model statistics")
        ci_dict = pyrisk.metrics.core.compute_all_CI(model_metrics)
        pyrisk.metrics.core.print_metrics_CI(ci_dict)

    except Exception:
        pyrisk.utils.exceptions.exception_handler(
            logger, log_path, log_config, script_name
        )
        exit(1)
