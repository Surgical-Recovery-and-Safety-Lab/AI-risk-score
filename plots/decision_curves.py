import argparse

import matplotlib.pyplot as plt
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
from sklearn.metrics import confusion_matrix


def net_benefit_fn(tp, fp, P, N):
    numerator = tp - (fp * P)
    return numerator / N


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot decision curves.")
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
    thresholds = (np.linspace(0.0, 0.1, 20), np.linspace(0.0, 0.5, 50))

    print_message("Loading parameters from configuration file")
    # Read log and general configuration file
    log_config = read_toml_configuration("../config/log.toml")
    general_config = read_toml_configuration(args.model_config_file)
    logger = None
    script_name = ""
    save_file = get_file_path(
        general_config,
        v_number="",
        path_type="fig",
        exists=False,
    )
    extension = general_config["fig_parameters"]["extension"]
    outcomes = ["MORTALITY_90D", "ANY_COMP"]
    methods = ["Natural", "CSL", "SMOTE", "ROS", "RUS"]
    colours = ["#2D90D8", "#33367A", "#96690E", "#CDB4DB", "#F2CC8F", "#1D6968"]
    ylims = [(-0.0345, 0.0345), (-0.165, 0.165)]

    for i in range(2):
        # Set figure and axes properties
        fig, ax = plt.subplots(dpi=300, figsize=(5, 5))  # Create a new figure

        ax.hlines(
            0,
            xmin=thresholds[i][0],
            xmax=thresholds[i][-1],
            color="k",
            linestyles="--",
            label="Treat none",
        )
        P = thresholds[i] / (1 - thresholds[i])
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

            net_benefit = np.zeros(len(thresholds[i]))
            y_pred_proba = get_positive_proba(
                pipeline.predict_proba(X_test, outcomes[i], "predictor")
            ).squeeze()

            for j, threshold in enumerate(thresholds[i]):
                decisions = y_pred_proba > threshold
                _, fp, _, tp = confusion_matrix(y_test[:, i], decisions).ravel()
                net_benefit[j] = net_benefit_fn(tp, fp, P[j], len(y_test))

            if k == 0:
                prevalence = np.sum(y_test[:, i]) / len(y_test[:, i])
                ax.plot(
                    thresholds[i],
                    net_benefit_fn(
                        prevalence,
                        1 - prevalence,
                        P,
                        1,
                    ),
                    color=colours[-1],
                    label="Treat all",
                    linestyle="--",
                )

            ax.plot(
                thresholds[i],
                net_benefit,
                color=colours[k],
                label=methods[k],
            )

        ax.set_xlabel("Decision threshold", fontweight="bold")
        ax.set_ylabel("Net benefit", fontweight="bold")
        ax.set_ylim(ylims[i])

        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        plt.tight_layout()

        save_path = save_file + f"_decision_curve_{outcomes[i]}" + extension
        plt.legend(frameon=False)
        plt.savefig(save_path)
