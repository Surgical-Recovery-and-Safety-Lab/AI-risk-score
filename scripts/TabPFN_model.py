#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TabPFN-model.py

TabP Foundational Model creation and training script.
"""

import argparse
import pathlib
import sys

import pandas as pd
import pyrisk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and train a TabP Foundational Model"
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
        data_config = pyrisk.utils.config.get_data_configuration(
            model_config["data_parameters"],
            model_config["version"],
        )
        data = pyrisk.utils.io.load_data_from_csv(
            pyrisk.utils.config.get_file_path(
                data_config, v_number=model_config["version"]
            )
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
            train_data, model_config["labels"]["label_list"]
        )
        X_test, y_test = pyrisk.data.preprocessing.extract_labels(
            test_data, model_config["labels"]["label_list"]
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
            "tabp", **model_config["config_parameters"]
        )

        print("[INFO] Training model")
        pyrisk.models.core.train_model(model, X_train[:49999], y_train[:49999].ravel())

        print("[INFO] Testing model")
        y_pred = model.predict(X_test)
        pyrisk.metrics.plots.plot_from_display(y_test, y_pred, "roc")
        pyrisk.metrics.plots.plot_from_display(y_test, y_pred, "confusion")
        pyrisk.metrics.plots.plot_from_display(
            pd.DataFrame(y_test), y_pred, "precision-recall"
        )

    except Exception:
        pyrisk.utils.exceptions.exception_handler(
            logger, log_path, log_config, script_name
        )
        exit(1)
