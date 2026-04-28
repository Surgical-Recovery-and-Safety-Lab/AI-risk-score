import argparse

import numpy as np
import pandas as pd
from medpipe import (
    extract_labels,
    get_positive_proba,
    load_data_from_csv,
    load_pipeline,
    print_message,
    read_toml_configuration,
)
from medpipe.utils.config import get_configuration, get_file_path, split_version_number

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the test scenario numbers")
    parser.add_argument(
        "model_config_file",
        metavar="model-config-file",
        help="Path to the general configuration file",
    )

    args = parser.parse_args()
    orig_version = "v0.1.1.1-a.1.2.2"  # Base model to compare to
    versions = [
        "v0.1.1.1-a.1.2.1",
        "v0.1.1.1-a.2.2.1",
        "v0.1.1.1-a.10.2.1",
        "v0.1.1.1-a.6.2.1",
        "v0.1.1.1-a.14.2.1",
    ]
    thresholds = [0.02, 0.1]

    print_message("Loading parameters from configuration file")
    # Read log and general configuration file
    log_config = read_toml_configuration("../../config/log.toml")
    general_config = read_toml_configuration(args.model_config_file)
    logger = None
    script_name = ""
    outcome = ["MORTALITY_90D", "ANY_COMP"]
    methods = ["Natural", "CSL", "SMOTE", "ROS", "RUS"]

    # Swap version numbers if overloading
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
            data_config,
            v_number=data_version[:4],  # Use only first 2 numbers
        )
    )
    print_message(f"Loading model {orig_version}", logger, script_name)
    load_file = get_file_path(
        general_config,
        v_number=orig_version,
    )
    pipeline = load_pipeline(load_file)
    _, X_test = pipeline.get_test_data(pipeline.transform(data), [2024])
    X_test, y_test = extract_labels(X_test, pipeline.label_list)

    orig_decisions = np.zeros(y_test.shape)
    for i in range(2):
        y_pred_proba = get_positive_proba(
            pipeline.predict_proba(X_test, outcome[i], "predictor")
        ).squeeze()
        orig_decisions[:, i] = np.array(y_pred_proba > thresholds[i])

    for k, version in enumerate(versions):
        # Load model
        print_message(f"Loading model {version}", logger, script_name)
        load_file = get_file_path(
            general_config,
            v_number=version,
        )
        pipeline = load_pipeline(load_file)

        decisions = np.zeros(y_test.shape)

        for i in range(2):
            y_pred_proba = get_positive_proba(
                pipeline.predict_proba(X_test, outcome[i], "calibrator")
            ).squeeze()
            decisions[:, i] = np.array(y_pred_proba > thresholds[i])

            print(
                f"{methods[k]} {outcome[i]} % risky surgeries: {100*np.sum(decisions[:, i])/len(decisions):.1f}"
            )
            print(
                f"{methods[k]} {outcome[i]} Delta% risky surgeries: {100*(np.sum(decisions[:, i])-np.sum(orig_decisions[:, i]))/len(decisions):.1f}"
            )
