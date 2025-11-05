#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract-data.py

Sends a query to a duckdb to extract and save data.
"""

import argparse
import pathlib
import sys
from tomllib import TOMLDecodeError

import pyrisk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Queries a duckdb to extract and save data"
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
        TypeError,
        ValueError,
        FileNotFoundError,
        TOMLDecodeError,
        IsADirectoryError,
    ) as err:
        sys.stderr.write("An error occured when trying to create the logger")
        sys.stderr.write(repr(err))
        exit(1)

    try:
        print("[INFO] Loading configuration")
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

    features = ",".join(data_config["features"]["feature_list"])
    db_parameters = data_config["db_parameters"]
    query = f"SELECT {features} FROM {db_parameters["table_name"]};"

    try:
        print("[INFO] Extracting data")
        df = pyrisk.data.db.extract_data_from_duckdb(
            pyrisk.utils.config.get_file_path(
                data_config, path_type="db", version=False
            ),
            query,
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
    except Exception:
        logger.exception(log_config["log_message"] + f"{script_name}")
        sys.stderr.write(
            log_config["unexpected_message"] + f"{log_path}/{script_name}.log"
        )
        exit(1)

    try:
        save_file = pyrisk.utils.config.get_file_path(data_config, path_type="io")
        print("[INFO] Saving data")
        df.to_csv(save_file, index=False)
    except Exception:
        logger.exception(log_config["log_message"] + f"{script_name}")
        sys.stderr.write(
            log_config["unexpected_message"] + f"{log_path}/{script_name}.log"
        )
        exit(1)
