#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert-parquet.py

Converts the data from a parquet file and creates a duckdb file.
"""

import argparse
import pathlib
import sys

import pyrisk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a parquet file to a duckdb file"
    )
    parser.add_argument(
        "parquet_file",
        metavar="parquet-file",
        help="Path to the parquet file",
    )
    parser.add_argument(
        "duckdb_file",
        metavar="duckdb-file",
        help="Path to the duckdb file",
    )
    parser.add_argument(
        "-t",
        "--table-name",
        default="main",
        help="Name of the table to create in the database",
    )

    args = parser.parse_args()

    try:
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
        pyrisk.data.db.parquet_to_duckdb(
            args.parquet_file, args.duckdb_file, args.table_name
        )
    except Exception:
        pyrisk.utils.exceptions.exception_handler(
            logger, log_dir, log_config, script_name
        )
        exit(1)
