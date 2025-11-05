#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HGB-model.py

Histogram Gradient Boosting model creation and training script.
"""

import argparse
import pathlib
import sys
from tomllib import TOMLDecodeError

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
    except (
        TypeError,
        ValueError,
        IsADirectoryError,
        TOMLDecodeError,
        FileNotFoundError,
    ):
        logger.exception(log_config["log_message"] + f"{script_name}")
        sys.stderr.write(log_config["print_message"] + f"{log_path}/{script_name}.log")
        exit(1)

    try:
        print("[INFO] Creating model")
        model = pyrisk.models.core.create_model(
            "hgb", **model_config["config_parameters"]
        )
    except (TypeError, ValueError, KeyError):
        logger.exception(log_config["log_message"] + f"{script_name}")
        sys.stderr.write(log_config["print_message"] + f"{log_path}/{script_name}.log")
        exit(1)
    except Exception:
        logger.exception(log_config["log_message"] + f"{script_name}")
        sys.stderr.write(
            log_config["unexpected_message"] + f"{log_path}/{script_name}.log"
        )
        exit(1)

    try:
        print("[INFO] Getting training data")
        data_config_path = pyrisk.utils.config.get_file_path(
            model_config, path_type="data", version=False
        )
        data_config = pyrisk.utils.io.read_toml_configuration(data_config_path)
        data = pyrisk.utils.io.load_data_from_csv(
            pyrisk.utils.config.get_file_path(
                data_config, suffix=data_config["io_parameters"]["train_suffix"]
            )
        )

        # Convert objects to categoricals
        data = pyrisk.data.preprocessing.convert_object_to_categorical(data)

        # Get prediction labels from data
        X, y = pyrisk.data.preprocessing.extract_labels(
            data, data_config["features"]["label_list"]
        )

    except (
        TypeError,
        ValueError,
        KeyError,
        IsADirectoryError,
        TOMLDecodeError,
        FileNotFoundError,
    ):
        logger.exception(log_config["log_message"] + f"{script_name}")
        sys.stderr.write(log_config["print_message"] + f"{log_path}/{script_name}.log")
        exit(1)
    except Exception:
        logger.exception(log_config["log_message"] + f"{script_name}")
        sys.stderr.write(
            log_config["unexpected_message"] + f"{log_path}/{script_name}.log"
        )
        exit(1)

    try:
        print("[INFO] Training model")
        model.fit(X, y)

    except Exception:
        logger.exception(log_config["log_message"] + f"{script_name}")
        sys.stderr.write(
            log_config["unexpected_message"] + f"{log_path}/{script_name}.log"
        )
        exit(1)

    breakpoint()
