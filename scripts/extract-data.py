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
        "duckdb_file",
        metavar="duckdb-file",
        help="Path to the .duckdb file",
    )
    parser.add_argument(
        "feature_list_file",
        metavar="feature-list-file",
        help="Path to the feature list file",
    )
    parser.add_argument(
        "-t",
        "--table-name",
        default="main",
        help="Name of the table to query in the database",
    )
    parser.add_argument(
        "-s",
        "--save-file",
        help="Name of the file to save the extracted data",
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
    except (TypeError, ValueError, FileNotFoundError, IsADirectoryError) as err:
        sys.stderr.write("An error occured when trying to create the logger")
        sys.stderr.write(repr(err))
        exit(1)

    try:
        logger.info("Loading features")
        feature_list = pyrisk.utils.io.read_toml_configuration(args.feature_list_file)
    except (
        TypeError,
        ValueError,
        IsADirectoryError,
        TOMLDecodeError,
        FileNotFoundError,
    ):
        logger.exception(log_config["log_message"] + f"{script_name}")
        sys.stderr.write(
            log_config["logger.info_message"] + f"{log_path}/{script_name}.log"
        )
        exit(1)

    features = ",".join(feature_list["features"])
    query = f"SELECT {features} FROM {args.table_name};"

    try:
        logger.info("Extracting data")
        df = pyrisk.data.db.extract_data_from_duckdb(args.duckdb_file, query)
    except (
        TypeError,
        ValueError,
        IsADirectoryError,
        FileNotFoundError,
    ):
        logger.exception(log_config["log_message"] + f"{script_name}")
        sys.stderr.write(
            log_config["logger.info_message"] + f"{log_path}/{script_name}.log"
        )
        exit(1)
    except Exception:
        logger.exception(log_config["log_message"] + f"{script_name}")
        sys.stderr.write(
            log_config["unexpected_message"] + f"{log_path}/{script_name}.log"
        )
        exit(1)

    if args.save_file:
        try:
            pyrisk.utils.exceptions.file_checks(args.save_file, ".csv")
        except (
            TypeError,
            ValueError,
            IsADirectoryError,
            FileNotFoundError,
        ):
            logger.exception(log_config["log_message"] + f"{script_name}")
            sys.stderr.write(
                log_config["logger.info_message"] + f"{log_path}/{script_name}.log"
            )
            exit(1)

        logger.info("Saving data")
        df.to_csv(args.save_file)
