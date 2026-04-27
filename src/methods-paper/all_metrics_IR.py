import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from medpipe import (
    compute_all_CI,
    compute_pred_metrics,
    compute_score_metrics,
    extract_labels,
    get_full_proba,
    load_data_from_csv,
    load_pipeline,
    print_message,
    read_toml_configuration,
)
from medpipe.utils.config import get_configuration, get_file_path, split_version_number


def plot_metrics_CI(
    ci_dict,
    x,
    label_list,
    save_path="",
    extension=".png",
    **kwargs,
):
    """
    Plots the metrics with confidence intrevals for each fold.

    Parameters
    ----------
    ci_dict : dict[str, tuple(float, float, float)]
        Dictionary containing the metric value and confidence intervals.
        The keys are the name of the metrics and the values are a tuple with
        first element the metric value, second the lower bound, and third the
        upper bound.
    x : array
        Values for x-axis.
    label_list : list[str]
        List of predicted labels.
    save_path : str, default: []
        Path to the save file.
    extension : str, default: ".png"
        Extension to save figure in.
    show_fig : bool, default: True
        Flag to show the figure.
    **kwargs
        Extra arguments for the figure or axes objects.

    Returns
    -------
    None
        Nothing is returned.

    """
    # Split arguments based on where they should be sent
    ax_kwargs = {key: value for key, value in kwargs.items() if key in dir(Axes)}
    fig_kwargs = {key: value for key, value in kwargs.items() if key in dir(Figure)}

    # Set up the figure and axis
    dims = (4, 2)
    fig, ax = plt.subplots(nrows=dims[0], ncols=dims[1], **fig_kwargs)
    colours = [
        "#2D90D8",
        "#33367A",
        "#96690E",
        "#CDB4DB",
        "#F2CC8F",
    ]
    markers = [".", "d", "v"]

    y_labels = {
        "auroc": "AUROC",
        "ap": "AUPRC",
        "log_loss": "Log loss",
        "accuracy": "Accuracy",
        "recall": "Recall",
        "precision": "Precision",
        "f1": "F1",
    }

    # Loop through each metric
    for i, (key, values) in enumerate(ci_dict.items()):
        indices = np.unravel_index(i, dims, "F")
        for j in range(len(label_list)):
            vals = values[0][j * 5 : (j + 1) * 5]
            lower_b = values[1][j * 5 : (j + 1) * 5]

            ax[indices[0], indices[1]].errorbar(
                x,
                vals,
                color=colours[j],
                linestyle="-",
                marker=markers[j],
                yerr=vals - lower_b,
                label=label_list[j],
                alpha=0.8,
            )
        # Customize the chart
        ax[indices[0], indices[1]].set_ylabel(y_labels[key], fontweight="bold")
        if key != "log_loss":
            ax[indices[0], indices[1]].set_ylim([-0.05, 1.05])
        ax[indices[0], indices[1]].spines["top"].set_visible(False)
        ax[indices[0], indices[1]].spines["right"].set_visible(False)
        ax[indices[0], indices[1]].xaxis.set_inverted(True)

        # Set the x-ticks to be at the center of each group of bars
        if i == 3 or i == 6:
            ax[indices[0], indices[1]].set_xticks(x)
            ax[indices[0], indices[1]].set_xlabel("Imbalance ratio", fontweight="bold")
            ax[indices[0], indices[1]].set_xticklabels(x)
        else:
            ax[indices[0], indices[1]].set_xticks([])
            ax[indices[0], indices[1]].set_xticklabels([])

    # Remove the last plot
    ax[dims[0] - 1, dims[1] - 1].axis("off")

    # Place one legend for the whole figure
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(
        handles[0 : len(label_list)],
        labels[0 : len(label_list)],
        loc="center right",
        bbox_to_anchor=(0.9, 0.15),
        title="Methods",
        frameon=False,
    )

    # Set ax_kwargs to override if needed
    for key, val in ax_kwargs.items():
        getattr(ax, key)(val)

    plt.tight_layout()
    if save_path:
        save_file = save_path + extension
        plt.savefig(save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plots the metrics based on imbalance ratios."
    )
    parser.add_argument(
        "model_config_file",
        metavar="model-config-file",
        help="Path to the general configuration file",
    )

    args = parser.parse_args()
    versions = [
        "v0.1.1.1-a.1.2.2",
        "v0.1.1.1-a.9.2.2",
        "v0.1.1.1-a.8.2.2",
        "v0.1.1.1-a.7.2.2",
        "v0.1.1.1-a.10.2.2",
        "v0.1.1.1-a.1.2.2",
        "v0.1.1.1-a.5.2.2",
        "v0.1.1.1-a.4.2.2",
        "v0.1.1.1-a.3.2.2",
        "v0.1.1.1-a.6.2.2",
        "v0.1.1.1-a.1.2.2",
        "v0.1.1.1-a.13.2.2",
        "v0.1.1.1-a.12.2.2",
        "v0.1.1.1-a.11.2.2",
        "v0.1.1.1-a.14.2.2",
    ]
    print_message("Loading parameters from configuration file")
    # Read log and general configuration file
    log_config = read_toml_configuration("../../config/log.toml")
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
    x = ([73.2, 54.9, 36.3, 18.3, 1.0], [9.6, 7.2, 4.8, 2.4, 1.0])

    for i in range(2):
        metric_dict = {}
        for version in versions:
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
            X_train, X_test = pipeline.get_test_data(data.dropna(), [2024])
            X_train, y_train = extract_labels(X_train, pipeline.label_list)
            tmp_metric_dict = {}
            tmp_metric_dict_pred = {}

            for key in pipeline.calibrator_probabilities[outcomes[i]].keys():
                y_test = y_train[X_train["OP_YEAR"] == key]
                tmp_metric_dict[key] = compute_score_metrics(
                    ["auroc", "ap", "log_loss"],
                    y_test[:, i],
                    get_full_proba(pipeline.predictor_probabilities[outcomes[i]][key]),
                )
                tmp_metric_dict_pred[key] = compute_pred_metrics(
                    ["accuracy", "recall", "precision", "f1"],
                    y_test[:, i],
                    pipeline.predictor_probabilities[outcomes[i]][key] > 0.5,
                )
                for k in tmp_metric_dict[key].keys():
                    if key not in metric_dict.keys():
                        metric_dict[key] = tmp_metric_dict[key]
                        break
                    else:
                        metric_dict[key][k] += tmp_metric_dict[key][k]
                for k in tmp_metric_dict_pred[key].keys():
                    if k not in metric_dict[key].keys():
                        metric_dict[key][k] = tmp_metric_dict_pred[key][k]
                    else:
                        metric_dict[key][k] += tmp_metric_dict_pred[key][k]

        ci_dict = compute_all_CI(metric_dict)

        plot_metrics_CI(
            ci_dict,
            np.array(x[i]),
            ["SMOTE", "ROS", "RUS"],
            dpi=300,
            figsize=(5, 5),
            save_path=save_file + f"_IR_{outcomes[i]}",
            extension=extension,
        )
