import argparse

import numpy as np
from medpipe import (
    load_data_from_csv,
    load_pipeline,
    print_message,
    read_toml_configuration,
)
from medpipe.utils.config import get_configuration, get_file_path, split_version_number


def data_stats(dataset, label_list):
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

    labels = [key for key in keys if key in label_list]
    inputs = [key for key in keys if key not in label_list]

    print_message("Inputs")

    total = len(dataset)
    print_message(f"N = {total}")

    for input in inputs:
        if input == "AGE":
            q75 = dataset[input].quantile(0.75)
            q25 = dataset[input].quantile(0.25)
            print_message(f"{input}: median {dataset[input].median()} IQR {q75-q25}")

        else:
            data = dataset[input]
            vals = np.sort(data.unique())
            print_message(f"{input}")
            for val in vals:
                sum_val = np.sum(data == val)
                print_message(f"  {val}: {sum_val} ({100*(sum_val / total):.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plots all metrics for balanced and unbalanced models."
    )
    parser.add_argument(
        "model_config_file",
        metavar="model-config-file",
        help="Path to the general configuration file",
    )

    args = parser.parse_args()

    print_message("Loading parameters from configuration file")
    # Read log and general configuration file
    log_config = read_toml_configuration("../config/log.toml")
    general_config = read_toml_configuration(args.model_config_file)
    logger = None
    script_name = ""
    print_message(f"Version number: {general_config["version"]}", logger, script_name)
    data_version, _ = split_version_number(general_config["version"])

    # Get data configuration parameters
    data_config = get_configuration(
        general_config["data_parameters"],
        data_version,
    )
    print_message("Getting data", logger, script_name)
    data = load_data_from_csv(
        get_file_path(
            data_config, v_number=data_version[:4]  # Use only first 2 numbers
        )
    )
    # Load model
    print_message("Loading model", logger, script_name)
    load_file = get_file_path(
        general_config,
        v_number=general_config["version"],
    )
    pipeline = load_pipeline(load_file)
    X_train, X_test = pipeline.get_test_data(pipeline.transform(data))
    X_train, X_cal = pipeline.get_test_data(X_train)
    data_stats(X_train, pipeline.label_list)
    data_stats(X_cal, pipeline.label_list)
    data_stats(X_test, pipeline.label_list)
