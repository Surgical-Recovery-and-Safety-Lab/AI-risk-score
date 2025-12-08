#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-models-sequential.py

Sequential call to ML-models.py
"""

import argparse
import subprocess


def run_sequential(version_number):
    """
    Runs the ML_models.py script for sequential call.

    Parameters
    ----------
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
            args.model_config_file,
            "--version",
            version_number,
            "--no-plots",
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

    args = parser.parse_args()
    version_numbers = [
        "v0.2.1.3-a.1.1",
        "v0.2.1.3-a.2.1",
        "v0.2.1.3-a.3.1",
        "v0.2.1.3-a.4.1",
    ]

    for version in version_numbers:
        run_sequential(version)
