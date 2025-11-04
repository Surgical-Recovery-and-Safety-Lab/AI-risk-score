#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split-data.py

Splits the data into a train and test set.
"""

import argparse
import pathlib
import sys
from tomllib import TOMLDecodeError

import pyrisk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Splits data into a train and test set"
    )
    parser.add_argument(
        "data_config_file",
        metavar="data-config-file",
        help="Path to the data config file",
    )
    args = parser.parse_args()

    # Create logger
    try:
        # Read log configuration file
        log_config = pyrisk.utils.io.read_toml_configuration("../config/log.toml")

        script_name = str(pathlib.Path(__file__).stem)
        log_path = log_config["log_path"]
        logger = pyrisk.utils.logger.setup_logger(
            script_name,
            log_path,
        )
    except (
        TOMLDecodeError,
        TypeError,
        ValueError,
        FileNotFoundError,
        IsADirectoryError,
    ) as err:
        sys.stderr.write("An error occured when trying to create the logger")
        sys.stderr.write(repr(err))
        exit(1)

    try:
        logger.info("Loading data configuration")
        data_config = pyrisk.utils.io.read_toml_configuration(args.data_config_file)
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
        logger.info("Loading data")
        load_dict = data_config["load_parameters"]
        data = pyrisk.utils.io.load_data_from_csv(
            load_dict["load_dir"] + data_config["data_file"] + load_dict["extension"]
        )
    except (
        TypeError,
        ValueError,
        IsADirectoryError,
        FileNotFoundError,
    ):
        logger.exception(log_config["log_message"] + f"{script_name}")
        sys.stderr.write(log_config["print_message"] + f"{log_path}/{script_name}.log")
        exit(1)

    try:
        logger.info("Splitting data")
        split_vars = data_config["split_variables"]
        train_data, test_data = pyrisk.data.preprocessing.split_test_train(
            data, split_vars["train_split"], split_vars["random_state"]
        )
    except (TypeError, ValueError):
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
        logger.info("Saving train data")
        save_parameters = data_config["save_parameters"]
        pyrisk.utils.io.save_data_to_csv(
            train_data,
            save_parameters["save_dir"]
            + "_".join((data_config["data_file"], save_parameters["train_suffix"]))
            + save_parameters["extension"],
        )

        logger.info("Saving test data")
        pyrisk.utils.io.save_data_to_csv(
            train_data,
            save_parameters["save_dir"]
            + "_".join((data_config["data_file"], save_parameters["test_suffix"]))
            + save_parameters["extension"],
        )

    except (TypeError, ValueError):
        logger.exception(log_config["log_message"] + f"{script_name}")
        sys.stderr.write(log_config["print_message"] + f"{log_path}/{script_name}.log")
        exit(1)

    except Exception:
        logger.exception(log_config["log_message"] + f"{script_name}")
        sys.stderr.write(
            log_config["unexpected_message"] + f"{log_path}/{script_name}.log"
        )
        exit(1)
