import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyrisk.data.preprocessing import extract_labels
from pyrisk.metrics.core import compute_pred_metrics, compute_score_metrics
from pyrisk.models.core import get_positive_proba, load_pipeline
from pyrisk.pipeline.Pipeline import Pipeline
from pyrisk.utils.config import get_configuration, get_file_path, split_version_number
from pyrisk.utils.io import load_data_from_csv, read_toml_configuration
from pyrisk.utils.logger import print_message
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and train a Pipeline of machine learning models"
    )
    parser.add_argument(
        "model_config_file",
        metavar="model-config-file",
        help="Path to the general configuration file",
    )

    args = parser.parse_args()
    versions = [
        "v0.1.1.1-a.1.2.2",
        "v0.1.1.1-a.2.2.2",
        "v0.1.1.1-a.10.2.2",
        "v0.1.1.1-a.6.2.2",
        "v0.1.1.1-a.14.2.2",
    ]
    thresholds = [0.02, 0.1]

    print_message("Loading parameters from configuration file")
    # Read log and general configuration file
    log_config = read_toml_configuration("../config/log.toml")
    general_config = read_toml_configuration(args.model_config_file)
    logger = None
    script_name = ""
    i = 0  # 0: MORTALITY_90D, 1: ANY_COMP
    label = ["MORTALITY_90D", "ANY_COMP"]
    methods = ["Original", "CSL", "SMOTE", "ROS", "RUS"]

    for k, version in enumerate(versions):
        # Swap version numbers if overloading
        general_config["version"] = version
        print_message(
            f"Version number: {general_config["version"]}", logger, script_name
        )
        data_version, _ = split_version_number(general_config["version"])

        # Get data configuration parameters
        data_config = get_configuration(
            general_config["data_parameters"],
            data_version,
        )
        pipeline = Pipeline(general_config, logger)
        print_message("Getting data", logger, script_name)
        data = load_data_from_csv(
            get_file_path(
                data_config,
                v_number=data_version[:4],  # Use only first 2 numbers
            )
        )
        # Load model
        print_message("Loading model", logger, script_name)
        load_file = get_file_path(
            general_config,
            v_number=general_config["version"],
        )
        pipeline = load_pipeline(load_file)
        X_train, X_test_24 = pipeline.get_test_data(pipeline.transform(data))
        _, X_test_23 = pipeline.get_test_data(X_train)
        X_test = pd.concat((X_test_24, X_test_23))
        X_test, y_test = extract_labels(X_test, pipeline.label_list)

        y_pred_proba = np.zeros(y_test.shape)
        decisions = np.zeros(y_test.shape)

        for i in range(2):
            y_pred_proba[:, i] = get_positive_proba(
                pipeline.predict_proba(X_test, i, "predictor")
            ).squeeze()
            decisions[:, i] = np.array(y_pred_proba[:, i] > thresholds[i])

            if k == 0:
                orig_decisions = decisions
            print(
                f"{methods[k]} {label[i]} % risky surgeries: {100*np.sum(decisions[:, i])/len(decisions):.1f}"
            )
            print(
                rf"{methods[k]} {label[i]} $\Delta$% risky surgeries: {100*(np.sum(decisions[:, i])-np.sum(orig_decisions[:, i]))/len(decisions):.1f}"
            )
