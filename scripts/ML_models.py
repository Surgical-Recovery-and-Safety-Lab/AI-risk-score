#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-models.py

Machine learning model creation and training script.
"""

import argparse
import pathlib
import sys

import pyrisk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and train a machine learning model"
    )
    parser.add_argument(
        "model_config_file",
        metavar="model-config-file",
        help="Path to the model configuration file",
    )
    parser.add_argument("--load", "-l", action="store_true", help="Loading flag")

    args = parser.parse_args()

    try:
        print("[INFO] Setting up logger")
        # Read log configuration file
        log_config = pyrisk.utils.io.read_toml_configuration("../config/log.toml")

        # Create logger
        script_name = str(pathlib.Path(__file__).stem)
        log_dir = log_config["base_dir"] + log_config["log_dir"]
        log_dir = log_config["log_dir"]
        logger = pyrisk.utils.logger.setup_logger(
            script_name,
            log_dir,
        )
    except (TypeError, ValueError, FileNotFoundError, IsADirectoryError) as err:
        sys.stderr.write("An error occured when trying to create the logger")
        sys.stderr.write(repr(err))
        exit(1)

    try:
        pyrisk.utils.logger.info_handler(
            logger, "Loading parameters from configuration file", script_name
        )
        general_config = pyrisk.utils.io.read_toml_configuration(args.model_config_file)

        model_type = general_config["model_type"]
        data_version, model_version = pyrisk.utils.config.split_version_number(
            general_config["version"]
        )

        # Get model configuration parameters
        model_config = pyrisk.utils.config.get_configuration(
            general_config["model_parameters"],
            model_version,
            join_token="",
        )

        # Get data configuration parameters
        data_config = pyrisk.utils.config.get_configuration(
            general_config["data_parameters"],
            data_version,
        )
    except Exception:
        pyrisk.utils.logger.exception_handler(logger, log_dir, log_config, script_name)
        exit(1)

    try:
        if not args.load:
            pyrisk.utils.logger.info_handler(logger, "Getting data", script_name)
            data = pyrisk.utils.io.load_data_from_csv(
                pyrisk.utils.config.get_file_path(data_config, v_number=data_version)
            )
            # Convert objects to categorical (not saved so needs to be here)
            data = pyrisk.data.preprocessing.convert_object_to_categorical(data)

            # Remove NaN values
            nb_nan_rows = data.isna().any(axis=1).sum()
            data = data.dropna()

            pyrisk.utils.logger.info_handler(
                logger, f"Dropped {nb_nan_rows} rows with NaN values", script_name
            )

            # Split data
            pyrisk.utils.logger.info_handler(logger, "Splitting data", script_name)
            split_vars = data_config["split_variables"]

            kfold_it = pyrisk.data.preprocessing.test_train_it(**split_vars)

            ## MODEL CREATION AND TRAINING
            pyrisk.utils.logger.info_handler(logger, "Creating model", script_name)
            if model_type == "nn":
                # Get number of features for NN model
                n_features = len(data_config["features"]["feature_list"]) - len(
                    model_config["labels"]["label_list"]
                )

                if split_vars["group_name"]:
                    # Remove group name if using GroupKFold
                    n_features -= 1
                breakpoint()
                model = pyrisk.models.core.create_model(
                    model_type,
                    n_features=n_features,
                    **model_config["architecture"],
                )
            else:
                model = pyrisk.models.core.create_model(
                    model_type, **model_config["config_parameters"]
                )

            pyrisk.utils.logger.info_handler(logger, "Training model", script_name)
            if model_type == "nn":
                model_metrics = pyrisk.models.core.train_model(
                    model,
                    data,
                    kfold_it,
                    model_config["labels"]["label_list"],
                    split_vars["group_name"],
                    **model_config["hyperparameters"],
                )

            else:
                model_metrics = pyrisk.models.core.train_model(
                    model,
                    data,
                    kfold_it,
                    model_config["labels"]["label_list"],
                    split_vars["group_name"],
                )

            pyrisk.utils.logger.info_handler(logger, "Saving model", script_name)
            save_file = pyrisk.utils.config.get_file_path(
                general_config,
                v_number=general_config["version"],
                exists=False,
            )

            if model_type == "nn":
                pyrisk.models.core.save_model(
                    model.state_dict(), save_file, model_metrics
                )
            else:
                pyrisk.models.core.save_model(model, save_file, model_metrics)

        else:
            # Load model
            pyrisk.utils.logger.info_handler(logger, "Loading model", script_name)
            load_file = pyrisk.utils.config.get_file_path(
                general_config,
                v_number=general_config["version"],
            )

            if model_type == "nn":
                state_dict, model_metrics = pyrisk.models.core.load_model(load_file)

                # Get the number of input features from the first layer
                first_layer_dict = next(iter(state_dict.values()))
                n_features = first_layer_dict.shape[1]

                # Create model and load state_dict
                model = pyrisk.models.core.create_model(
                    model_type,
                    n_features=n_features,
                    **model_config["architecture"],
                )
                model.load_state_dict(state_dict)
            else:
                model, model_metrics = pyrisk.models.core.load_model(load_file)

    except Exception:
        pyrisk.utils.logger.exception_handler(logger, log_dir, log_config, script_name)
        exit(1)

    # Compute statistics
    try:
        pyrisk.utils.logger.info_handler(
            logger, "Computing model statistics", script_name
        )
        ci_dict = pyrisk.metrics.core.compute_all_CI(model_metrics)
        pyrisk.metrics.core.print_metrics_CI(ci_dict)

        pyrisk.metrics.plots.plot_mean_ROC_curve(model_metrics)
        pyrisk.metrics.plots.plot_mean_PR_curve(model_metrics)

    except Exception:
        pyrisk.utils.logger.exception_handler(logger, log_dir, log_config, script_name)
        exit(1)
