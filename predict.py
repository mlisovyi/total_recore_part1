"""Unearthed Prediction Template (Used for Local Testing Only)"""
import logging
import argparse
from os import getenv
from os.path import join

import pandas as pd

from preprocess import preprocess, target_columns
from train import model_fn

# needed to load the model from pickle
from mdl import CustomRidgeCV

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    """Prediction.

    The main function is only used by the Unearthed CLI.

    When a submission is made online AWS SageMaker Processing Jobs are used to perform
    preprocessing and Batch Transform Jobs are used to pass the result of preprocessing
    to the trained model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, default=getenv("SM_MODEL_DIR", "/opt/ml/models")
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
    )
    parser.add_argument(
        "--output_dir", type=str, default=getenv("SM_OUTPUT_DIR", "/opt/ml/output/")
    )
    args, _ = parser.parse_known_args()

    # call preprocessing on the data
    df = preprocess(join(args.data_dir, "public.csv.gz"))

    # load the model
    model = model_fn(args.model_dir)

    logger.info("creating predictions")

    inputs = df.drop(columns=target_columns)

    predictions = model.predict(inputs)
    logger.info(f"predictions have shape of {predictions.shape}")

    # save the predictions
    pd.DataFrame(predictions).to_csv(
        join(args.output_dir, "public.csv.out"), index=False, header=False
    )
