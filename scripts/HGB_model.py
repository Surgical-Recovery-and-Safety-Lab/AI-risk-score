#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HGB-model.py

Histogram Gradient Boosting model creation and training script.
"""

import argparse
import pathlib
import sys

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

    args = parser.parse_args()

    try:
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
        model_config = pyrisk.utils.io.read_toml_configuration(args.model_config_file)
    except Exception:
        pyrisk.utils.exceptions.exception_handler(
            logger, log_path, log_config, script_name
        )
        exit(1)

    # Data loading and manipulation
    try:
        print("[INFO] Getting data")
        data_config_path = pyrisk.utils.config.get_file_path(
            model_config, path_type="data", version=False
        )
        data_config = pyrisk.utils.io.read_toml_configuration(data_config_path)
        data = pyrisk.utils.io.load_data_from_csv(
            pyrisk.utils.config.get_file_path(data_config)
        )

        # Convert objects to categorical (not saved so needs to be here)
        data = pyrisk.data.preprocessing.convert_object_to_categorical(data)

        print("[INFO] Splitting data")
        split_vars = data_config["split_variables"]
        train_data, test_data = pyrisk.data.preprocessing.split_test_train(
            data, split_vars["train_split"], split_vars["random_state"]
        )

        # Get prediction labels from data
        X_train, y_train = pyrisk.data.preprocessing.extract_labels(
            train_data, data_config["features"]["label_list"]
        )
        X_test, y_test = pyrisk.data.preprocessing.extract_labels(
            test_data, data_config["features"]["label_list"]
        )

    except Exception:
        pyrisk.utils.exceptions.exception_handler(
            logger, log_path, log_config, script_name
        )
        exit(1)

    # Model creation, training, and testing
    try:
        print("[INFO] Creating model")
        model = pyrisk.models.core.create_model(
            "hgb", **model_config["config_parameters"]
        )

        print("[INFO] Training model")
        pyrisk.models.core.train_model(model, X_train, y_train.ravel())

        print("[INFO] Testing model")
        score = pyrisk.models.core.test_model(model, X_test, y_test)

        print(f"\nModel score: {score}")

    except Exception:
        pyrisk.utils.exceptions.exception_handler(
            logger, log_path, log_config, script_name
        )
        exit(1)
