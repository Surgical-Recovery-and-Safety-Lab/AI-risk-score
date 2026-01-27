import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from pyrisk.data.preprocessing import extract_labels
from pyrisk.metrics.core import compute_CI, compute_score_metrics
from pyrisk.models.core import load_pipeline
from pyrisk.pipeline.Pipeline import Pipeline
from pyrisk.utils.config import get_configuration, get_file_path, split_version_number
from pyrisk.utils.io import load_data_from_csv, read_toml_configuration
from pyrisk.utils.logger import print_message


def plot_mean_ROC_curve(
    model_metrics,
    label_list,
    save_path="",
    extension=".png",
    nb_points=500,
    **kwargs,
):
    """
    Plots the ROC curve of each fold and the mean ROC curve.

    Parameters
    ----------
    model_metrics : dict[int, dict[str, float or tuple(array-like)]]
        Model metrics for different folds.
    label_list : list[str]
        List of predicted labels.
    save_path : str, default: []
        Path to the save file.
    extension : str, default: ".png"
        Extension to save figure in.
    show_fig : bool, default: True
        Flag to show the figure.
    nb_points : int, default: 500
        Number of points for the mean ROC curve.
    **kwargs
        Extra arguments for the figure or axes objects.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    TypeError
        If nb_points is not an int.

    """
    if type(nb_points) is not type(0):
        raise TypeError(f"nb_points should be an int, but got {type(nb_points)}")

    n_it = len(label_list)  # Number of print iterations

    mean_fpr = np.linspace(0, 1, nb_points)

    # Split arguments based on where they should be sent
    ax_kwargs = {key: value for key, value in kwargs.items() if key in dir(Axes)}
    fig_kwargs = {key: value for key, value in kwargs.items() if key in dir(Figure)}

    # Set up figure and colours
    _, ax = plt.subplots(**fig_kwargs)
    colours = ["#99C7E0", "#2D90D8", "#1D6968", "#33367A", "#96690E"]
    colour_val = 0

    for i in range(n_it):
        tprs = []
        aucs = []

        for fold in model_metrics.keys():
            fpr, tpr, _ = model_metrics[fold]["roc"][i]
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(model_metrics[fold]["auroc"][i])

        # Compute mean and std for all folds
        mean_tpr, lower_tpr, upper_tpr = compute_CI(tprs)
        mean_auroc, lower_auroc, upper_auroc = compute_CI(aucs)
        value = f" AUROC {mean_auroc[0]:.2f} "
        ci = f"(95% CI, [{lower_auroc[0]:.2f} - {upper_auroc[0]:.2f}])"

        # Plot mean curve with shaded std
        ax.plot(mean_fpr, mean_tpr, color=colours[i], label=label_list[i])
        ax.fill_between(
            mean_fpr,
            lower_tpr,
            upper_tpr,
            color=colours[i],
            alpha=0.5,
        )

        colour_val += 1
        ax.set_xlim(xmin=-0.05, xmax=1.05)
        ax.set_ylim(ymin=-0.05, ymax=1.05)
        ax.set_xlabel("False Positive Rate", fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.4, 0.9), title="Model")

        # Set ax_kwargs to override if needed
        for key, val in ax_kwargs.items():
            getattr(ax, key)(val)

        # Tight layout to avoid overlapping
        plt.tight_layout()
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
    if save_path:
        save_file = save_path + extension
        plt.savefig(save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and train a Pipeline of machine learning models"
    )
    parser.add_argument(
        "model_config_file",
        metavar="model-config-file",
        help="Path to the general configuration file",
    )
    parser.add_argument(
        "version_numbers",
        metavar="version-numbers",
        help="Version numbe_rs to run",
        nargs="+",
    )

    args = parser.parse_args()

    print_message("Loading parameters from configuration file")
    # Read log and general configuration file
    log_config = read_toml_configuration("../config/log.toml")
    general_config = read_toml_configuration(args.model_config_file)
    logger = None
    script_name = ""
    metric_dict = {}
    i = 1  # 0: MORTALITY_90D, 1: ANY_COMP
    models = ["predictor", "calibrator"]
    save_file = get_file_path(
        general_config,
        v_number="",
        path_type="fig",
        exists=False,
    )

    extension = general_config["fig_parameters"]["extension"]
    ext = ["_MORTALITY_90D", "_ANY_COMP"]
    label_list = {
        "predictor": ["Original", "CSL", "SMOTE", "ROS", "RUS"],
        "calibrator": ["Original", "CSL", "SMOTE", "ROS", "RUS"],
    }
    """
    label_list = {
        "predictor": ["Original", "CSL", "SMOTE", "ROS", "RUS"],
        "calibrator": [
            "Re-calibrated original",
            "Re-calibrated CSL",
            "Re-calibrated SMOTE",
            "Re-calibrated ROS",
            "Re-calibrated RUS",
        ],
    }
    """

    for model in models:
        for i in range(2):
            metric_dict = {}
            for version in args.version_numbers:
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
                data = pipeline.transform(data)
                X_train, X_test = pipeline.get_test_data(data)
                X_train, y_train = extract_labels(X_train, pipeline.label_list)
                tmp_metric_dict = {}

                for key in pipeline.calibrator_probabilities[i]:
                    y_test = y_train[X_train["OP_YEAR"] == key]
                    if model == "predictor":
                        y_prob = pipeline.predictor_probabilities[i][key]
                    else:
                        y_prob = pipeline.calibrator_probabilities[i][key]

                    tmp_metric_dict[key] = compute_score_metrics(
                        ["auroc", "roc"],
                        y_test[:, i],
                        y_prob,
                    )
                    for k in tmp_metric_dict[key].keys():
                        if key not in metric_dict.keys():
                            metric_dict[key] = tmp_metric_dict[key]
                            break
                        else:
                            metric_dict[key][k] += tmp_metric_dict[key][k]

            plot_mean_ROC_curve(
                metric_dict,
                label_list=label_list[model],
                save_path=save_file + "_ROC_" + model + ext[i],
                extension=extension,
                n_bins=20,
                dpi=300,
                figsize=(5, 5),
            )
