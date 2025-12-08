#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-models-parallel.py

Parallel call to ML-models.py
"""

import argparse
import multiprocessing
import subprocess


def run_parallel(model_config_file, version_number):
    """
    Runs the ML-models.py script for parallel call.

    Parameters
    ----------
    model_config_file : str
        Path to the model configuration file.
    version_number : str
        Version number to use.

    Returns
    -------
    None
        Nothing is returned.

    """
    subprocess.run(
        [
            "python3",
            "ML-models.py",
            model_config_file,
            "--version",
            version_number,
            "--no-plots",
            "-q",
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and train a machine learning model"
    )
    parser.add_argument(
        "model_config_file",
        metavar="model-config-file",
        help="Path to the model configuration file",
    )
    parser.add_argument(
        "n_processes",
        metavar="n-processes",
        type=int,
        help="Number of processes to use",
    )

    args = parser.parse_args()
    version_numbers = ["v0.2.1.3-b.3.2", "v0.2.1.3-b.3.1"]
    input_args = [(args.model_config_file, version) for version in version_numbers]

    with multiprocessing.Pool(processes=args.n_processes) as pool:
        pool.starmap(run_parallel, input_args)
