import argparse

import numpy as np
from medpipe import (
    load_data_from_csv,
    load_pipeline,
    print_message,
    read_toml_configuration,
)
from medpipe.data.preprocessing import bin_score
from medpipe.utils.config import get_configuration, get_file_path, split_version_number


def data_stats(X_train, X_cal, X_test, input):
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
    if input == "AGE":
        print_age(X_train)
        print_age(X_cal)
        print_age(X_test)

    else:
        print(input)
        print_feature(X_train, input)
        print_feature(X_cal, input)
        print_feature(X_test, input)


def print_age(dataset) -> None:
    """Print age for table"""
    q75 = dataset["AGE"].quantile(0.75)
    q25 = dataset["AGE"].quantile(0.25)
    print(
        "\\multicolumn{1}{l}{"
        + f"{round(dataset["AGE"].median())} ({round(q75-q25)})"
        + "} &"
    )


def print_feature(dataset, input) -> None:
    data = dataset[input]
    vals = np.sort(data.unique())
    print_str = "  \\multicolumn{1}{l}{\\begin{tabular}[c]{@{}l@{}}\\\\ "
    for val in vals:
        sum_val = np.sum(data == val)
        print_str += f"{sum_val:,} ({100*sum_val/len(data):.3f})\\\\ "
    print_str += "\\end{tabular}} &"
    print(print_str)


def outcome_table(X_train, X_cal, X_test, labels):
    """Prints to terminal the outcomes table"""
    train_len = len(X_train)
    cal_len = len(X_cal)
    test_len = len(X_test)
    total_len = train_len + cal_len + test_len
    print("\\begin{table}[hbp]")
    print("\\centering")
    print("\\caption{Clinical outcomes.}")
    print("\\label{tab:outcomes}")
    print("\\begin{tabular}{llll}")
    print("\\textbf{Outcome} & ")
    print(
        "  \\multicolumn{1}{c}{\\textbf{\\begin{tabular}[c]{@{}l@{}} Train set \\\\ (2010 -- 2022)\\\\ N="
        + f"{train_len:,}"
        + f" ({round(100*train_len/total_len)} \\%)"
        + "\\end{tabular}}} &"
    )
    print(
        "  \\multicolumn{1}{c}{\\textbf{\\begin{tabular}[c]{@{}l@{}} Calibration set \\\\ (2023)\\\\ N="
        + f"{cal_len:,}"
        + f" ({round(100*cal_len/total_len)} \\%)"
        + "\\end{tabular}}} &"
    )
    print(
        "  \\multicolumn{1}{c}{\\textbf{\\begin{tabular}[c]{@{}l@{}} Test set \\\\ (2024)\\\\ N="
        + f"{test_len:,}"
        + f" ({round(100*test_len/total_len)} \\%)"
        + "\\end{tabular}}} \\\\ \\hline\\hline"
    )
    print(
        "\\multicolumn{1}{l}{\\begin{tabular}[c]{@{}l@{}}\\textbf{90-day mortality}, count (\\%)\\\\ Yes\\\\ No\\end{tabular}} &"
    )
    yes_sum = np.sum(X_train[labels[0]])
    no_sum = np.sum(X_train[labels[0]] == 0)
    print(
        "  \\multicolumn{1}{l}{\\begin{tabular}[c]{@{}l@{}}\\\\ "
        + f"{yes_sum:,} ({100*yes_sum/train_len:.1f})"
        + f"\\\\ {no_sum:,} ({100*no_sum/train_len:.1f})"
        + "\\end{tabular}} &"
    )
    yes_sum = np.sum(X_cal[labels[0]])
    no_sum = np.sum(X_cal[labels[0]] == 0)
    print(
        "  \\multicolumn{1}{l}{\\begin{tabular}[c]{@{}l@{}}\\\\ "
        + f"{yes_sum:,} ({100*yes_sum/cal_len:.1f})"
        + f"\\\\ {no_sum:,} ({100*no_sum/cal_len:.1f})"
        + "\\end{tabular}} &"
    )
    yes_sum = np.sum(X_test[labels[0]])
    no_sum = np.sum(X_test[labels[0]] == 0)
    print(
        "  \\multicolumn{1}{l}{\\begin{tabular}[c]{@{}l@{}}\\\\ "
        + f"{yes_sum:,} ({100*yes_sum/test_len:.1f})"
        + f"\\\\ {no_sum:,} ({100*no_sum/test_len:.1f})"
        + "\\end{tabular}} \\\\ \\hline"
    )
    print(
        "\\multicolumn{1}{l}{\\begin{tabular}[c]{@{}l@{}}\\textbf{Any complications}, count (\\%)\\\\ Yes\\\\ No\\end{tabular}} &"
    )
    yes_sum = np.sum(X_train[labels[1]])
    no_sum = np.sum(X_train[labels[1]] == 0)
    print(
        "  \\multicolumn{1}{l}{\\begin{tabular}[c]{@{}l@{}}\\\\ "
        + f"{yes_sum:,} ({100*yes_sum/train_len:.1f})"
        + f"\\\\ {no_sum:,} ({100*no_sum/train_len:.1f})"
        + "\\end{tabular}} &"
    )
    yes_sum = np.sum(X_cal[labels[1]])
    no_sum = np.sum(X_cal[labels[1]] == 0)
    print(
        "  \\multicolumn{1}{l}{\\begin{tabular}[c]{@{}l@{}}\\\\ "
        + f"{yes_sum:,} ({100*yes_sum/cal_len:.1f})"
        + f"\\\\ {no_sum:,} ({100*no_sum/cal_len:.1f})"
        + "\\end{tabular}} &"
    )
    yes_sum = np.sum(X_test[labels[1]])
    no_sum = np.sum(X_test[labels[1]] == 0)
    print(
        "  \\multicolumn{1}{l}{\\begin{tabular}[c]{@{}l@{}}\\\\ "
        + f"{yes_sum:,} ({100*yes_sum/test_len:.1f})"
        + f"\\\\ {no_sum:,} ({100*no_sum/test_len:.1f})"
        + "\\end{tabular}} \\\\ \\hline"
    )
    print("\\end{tabular}")
    print("\\end{table}")


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
    log_config = read_toml_configuration("../../config/log.toml")
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
    data["M3_SCORE"] = bin_score(data["M3_SCORE"].to_numpy())
    X_train, X_test = pipeline.get_test_data(data.dropna(), [2024])
    X_train, X_cal = pipeline.get_test_data(X_train, [2023])
    outcome_table(X_train, X_cal, X_test, pipeline.label_list)
    for feature in X_train.columns:
        data_stats(X_train, X_cal, X_test, feature)
