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
    except Exception:
        pyrisk.utils.exceptions.exception_handler(
            logger, log_path, log_config, script_name
        )
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
    except Exception:
        pyrisk.utils.exceptions.exception_handler(
            logger, log_path, log_config, script_name
        )
        exit(1)

    preprocessing_config = data_config["preprocessing"]

    if preprocessing_config["preprocess"]:
        # If the preprocess flag is true
        print("[INFO] Preprocessing data")
        try:
            if preprocessing_config["label_encoder"]["feature_list"] != []:
                df = pyrisk.data.preprocessing.label_encode_data(
                    df, preprocessing_config["label_encoder"]["feature_list"]
                )

        except Exception:
            pyrisk.utils.exceptions.exception_handler(
                logger, log_path, log_config, script_name
            )
            exit(1)
        breakpoint()
    try:
        print("[INFO] Saving data")
        save_file = pyrisk.utils.config.get_file_path(data_config, path_type="io")
        df.to_csv(save_file, index=False)
    except Exception:
        pyrisk.utils.exceptions.exception_handler(
            logger, log_path, log_config, script_name
        )
        exit(1)
