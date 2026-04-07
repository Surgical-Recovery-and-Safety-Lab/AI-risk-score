#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_tables.py

Prints the outcome prevalence and the data distribution for the sets.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import TYPE_CHECKING

import numpy as np
from medpipe import (
    exception_handler,
    load_data_from_csv,
    load_pipeline,
    print_message,
    read_toml_configuration,
    setup_logger,
)
from medpipe.utils.config import get_configuration, get_file_path, split_version_number

if TYPE_CHECKING:
    import pandas as pd


def data_stats(dataset: pd.DataFrame, label_list: list[str]) -> None:
    """
    Compute statistics for a dataset.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset to analyse.
    label_list : list[str]
        List of the predicted labels.

    Returns
    -------
    Nothing is returned.

    """
    keys = dataset.columns

    inputs = [key for key in keys if key not in label_list]

    print_message("Inputs")

    total = len(dataset)
    print_message(f"N = {total:,}")

    for input in inputs:
        if input == "AGE":
            q75 = dataset[input].quantile(0.75)
            q25 = dataset[input].quantile(0.25)
            print_message(f"{input}: median {dataset[input].median()} IQR {q75-q25}")

        else:
            data = dataset[input]
            vals = np.sort(np.array(data.unique()))
            print_message(f"{input}")
            for val in vals:
                sum_val = np.sum(data == val)
                print_message(f"  {val}: {sum_val:,} ({100*(sum_val / total):.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print prevalence of outcomes and inputs to terminal."
    )
    parser.add_argument(
        "model_config_file",
        metavar="model-config-file",
        help="Path to the general configuration file",
    )
    parser.add_argument("--version", help="Version number overload")

    args = parser.parse_args()

    try:
        print_message("Loading parameters from configuration file")
        # Read log and general configuration file
        log_config = read_toml_configuration("../../config/log.toml")
        general_config = read_toml_configuration(args.model_config_file)

        if args.version:
            # Swap version numbers if overloading
            general_config["version"] = args.version

        print_message("Setting up logger")
        # Create logger
        script_name = str(pathlib.Path(__file__).stem)
        script_name += "_" + general_config["version"]
        log_dir = log_config["base_dir"] + log_config["log_dir"]
        log_dir = log_config["log_dir"]
        logger = setup_logger(script_name, log_dir)

        print_message(
            f"Version number: {general_config["version"]}", logger, script_name
        )

    except (TypeError, ValueError, FileNotFoundError, IsADirectoryError) as err:
        sys.stderr.write("An error occured when trying to create the logger\n")
        sys.stderr.write(repr(err))
        exit(1)

    try:
        data_version, _ = split_version_number(general_config["version"])

        # Get data configuration parameters
        data_config = get_configuration(
            general_config["data_parameters"],
            data_version,
        )

    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)

    try:
        print_message("Getting data", logger, script_name)
        data = load_data_from_csv(
            get_file_path(
                data_config, v_number=data_version[:4]  # Use only first 2 numbers
            )
        )
        data.fillna({"ASA": 0}, inplace=True)  # Replace Nan with 0 for ASA
        data.dropna(inplace=True)  # Remove Nan values

        # Load model
        print_message("Loading model", logger, script_name)
        load_file = get_file_path(
            general_config,
            v_number=general_config["version"],
        )
        pipeline = load_pipeline(load_file)

        X_train, X_test = pipeline.get_test_data(data, test_group_vals=[2023, 2024])

    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)

    try:
        print_message("Printing dataset stats", logger, script_name)
        data_stats(X_train, pipeline.label_list)
        data_stats(X_test, pipeline.label_list)

        print_message("Printing outcome prevalence", logger, script_name)
        for label in np.sort(pipeline.label_list):
            train_prev = X_train[label].sum()
            train_percent = 100 * train_prev / len(X_train)
            test_prev = X_test[label].sum()
            test_percent = 100 * test_prev / len(X_test)
            print_message(
                f"{label}: train | test \n"
                f"\t{train_prev:,} ({train_percent:.2f}) | "
                f"{test_prev:,} ({test_percent:.2f})",
                logger,
                script_name,
            )
    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)
