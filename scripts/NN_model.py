#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NN-model.py

Neural Network model creation and training script.
"""

import argparse
import pathlib
import sys

import pandas as pd
import pyrisk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and train a NN model")
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
        log_dir = log_config["base_dir"] + log_config["log_dir"]
        log_dir = log_config["log_dir"]
        logger = pyrisk.utils.logger.setup_logger(
            script_name,
            log_dir,
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
            logger, log_dir, log_config, script_name
        )
        exit(1)

    # Data loading and manipulation
    try:
        print("[INFO] Getting data")
        data = pyrisk.utils.io.load_data_from_csv(
            pyrisk.utils.config.get_file_path(data_config, v_number=data_version)
        )

        # Convert objects to categorical (not saved so needs to be here)
        data = pyrisk.data.preprocessing.convert_object_to_categorical(data)

        # Remove NaN values
        nb_nan_rows = data.isna().any(axis=1).sum()
        data = data.dropna()

        print(f"[INFO] Dropped {nb_nan_rows} rows with NaN values")

        data_true = data.loc[data[model_config["labels"]["label_list"][0]] == True]
        data_false = data.loc[data[model_config["labels"]["label_list"][0]] == False]
        data = pd.concat([data_true, data_false[: len(data_true)]])

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
            logger, log_dir, log_config, script_name
        )
        exit(1)

    # Model creation, training, and testing
    try:
        print("[INFO] Creating model")
        model = pyrisk.models.core.create_model(
            "nn", n_features=X_train.shape[1], **model_config["architecture"]
        )

        if not args.load:
            print("[INFO] Training model")
            pyrisk.models.core.train_model(
                model, X_train, y_train.ravel(), **model_config["hyperparameters"]
            )

            print("[INFO] Saving model")
            save_file = pyrisk.utils.config.get_file_path(
                general_config,
                v_number=general_config["version"],
                exists=False,
            )
            pyrisk.models.core.save_model(model, save_file, extension=".pt")

        else:
            print("[INFO] Loading model weights")
            model.load_model(
                pyrisk.utils.config.get_file_path(
                    general_config,
                    v_number=general_config["version"],
                )
            )

    except Exception:
        pyrisk.utils.exceptions.exception_handler(
            logger, log_dir, log_config, script_name
        )
        exit(1)

    try:
        print("[INFO] Testing model")
        y_pred = model.predict(X_test)
        pyrisk.metrics.plots.plot_from_display(y_test, y_pred, "roc")
        pyrisk.metrics.plots.plot_from_display(y_test, y_pred, "confusion")
        pyrisk.metrics.plots.plot_from_display(
            pd.DataFrame(y_test), y_pred, "precision-recall"
        )

    except Exception:
        pyrisk.utils.exceptions.exception_handler(
            logger, log_dir, log_config, script_name
        )
        exit(1)
