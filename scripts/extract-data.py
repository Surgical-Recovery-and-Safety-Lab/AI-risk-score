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
        description="Queries a duckdb to extract, preprocess, and save data"
    )
    parser.add_argument(
        "data_config_file",
        metavar="data-config-file",
        help="Path to the data config file",
    )
    args = parser.parse_args()

    # Create logger
    try:
        print("[INFO] Setting up logger")
        # Read log configuration file
        log_config = pyrisk.utils.io.read_toml_configuration("../config/log.toml")

        script_name = str(pathlib.Path(__file__).stem)
        log_dir = log_config["log_dir"]
        logger = pyrisk.utils.logger.setup_logger(
            script_name,
            log_dir,
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
        pyrisk.utils.logger.print_message("Loading configuration", logger, script_name)
        data_config_top_level = pyrisk.utils.io.read_toml_configuration(
            args.data_config_file
        )
        data_config = pyrisk.utils.config.get_configuration(
            data_config_top_level["data_parameters"],
            data_config_top_level["version"],
        )
    except Exception:
        pyrisk.utils.logger.exception_handler(logger, log_dir, log_config, script_name)
        exit(1)

    features = ",".join(data_config["features"]["feature_list"])
    db_parameters = data_config["db_parameters"]
    query = f"SELECT {features} FROM {db_parameters["table_name"]};"

    try:
        pyrisk.utils.logger.print_message("Extracting data", logger, script_name)
        df = pyrisk.data.db.extract_data_from_duckdb(
            pyrisk.utils.config.get_file_path(data_config, path_type="db"),
            query,
        )
    except Exception:
        pyrisk.utils.logger.exception_handler(logger, log_dir, log_config, script_name)
        exit(1)

    try:
        pyrisk.utils.logger.print_message("Saving data", logger, script_name)
        save_file = pyrisk.utils.config.get_file_path(
            data_config, v_number=data_config_top_level["version"], exists=False
        )
        df.to_csv(save_file, index=False)
    except Exception:
        pyrisk.utils.logger.exception_handler(logger, log_dir, log_config, script_name)
        exit(1)
