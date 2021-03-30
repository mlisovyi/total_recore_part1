"""Unearthed Training Template"""
import argparse
import logging
import pickle
import sys
from io import StringIO
from os import getenv
from os.path import abspath, join

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import make_scorer, r2_score
from sklearn.linear_model import Ridge, RidgeCV

from preprocess import preprocess, target_columns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Work around for a SageMaker path issue
# (see https://github.com/aws/sagemaker-python-sdk/issues/648)
# WARNING - removing this may cause the submission process to fail
if abspath("/opt/ml/code") not in sys.path:
    sys.path.append(abspath("/opt/ml/code"))


class CustomRidgeCV(RidgeCV):
    _v_min = [
        -0.0,
        -0.0,
        -0.0,
        30.0,
        -0.0,
        -0.0,
        2.0,
        3.0,
        1.0,
        76.0,
        0.0,
        3.0,
        0.0,
        -0.0,
        100.0,
        -0.0,
        -1.0,
        0.0,
        6.0,
        -50.0,
        90.0,
        -100.0,
        0.0,
        0.0,
        -2.0,
        1.0,
        -0.0,
        3100.0,
        500.0,
        1.0,
        5.0,
        0.0,
        2.0,
        -0.0,
        0.0,
        -0.0,
        1000.0,
        1.0,
        1.0,
        1100.0,
        400.0,
        0.0,
        3.0,
        -0.0,
        1.0,
        -0.0,
        110.0,
        0.0,
        1.0,
    ]

    def predict(self, X):
        y_preds = super().predict(X)
        y_preds = np.clip(y_preds, self._v_min, None)
        return y_preds


def train(args):
    """Train

    Your model code goes here.
    """
    logger.info("calling training function")

    # preprocess
    # if you require any particular preprocessing to create features then this
    # *must* be contained in the preprocessing function for the Unearthed pipeline
    # apply it to the private data
    df = preprocess(join(args.data_dir, "public.csv.gz"))

    y_train = df[target_columns]
    logger.info(f"training target shape is {y_train.shape}")
    X_train = df.drop(columns=target_columns)
    logger.info(f"training input shape is {X_train.shape}")

    # the example model
    # model = Ridge()
    model = CustomRidgeCV(
        alphas=np.logspace(-5, 5, 50),
        scoring=make_scorer(r2_score),
        alpha_per_target=True,
    )
    model.fit(X_train, y_train)

    # save the model to disk
    save_model(model, args.model_dir)


def save_model(model, model_dir):
    """Save model to a binary file.

    This function must write the model to disk in a format that can
    be loaded from the model_fn.

    WARNING - modifying this function may cause the submission process to fail.
    """
    logger.info(f"saving model to {model_dir}")
    with open(join(model_dir, "model.pkl"), "wb") as model_file:
        pickle.dump(model, model_file)


def model_fn(model_dir):
    """Load model from binary file.

    This function loads the model from disk. It is called by SageMaker.

    WARNING - modifying this function may case the submission process to fail.
    """
    logger.info("loading model")
    with open(join(model_dir, "model.pkl"), "rb") as file:
        return pickle.load(file)


def input_fn(input_data, content_type):
    """Take request data and de-serialize the data into an object for prediction.

    In the Unearthed submission pipeline the data is passed as "text/csv". This
    function reads the CSV into a Pandas dataframe ready to be passed to the model.

    WARNING - modifying this function may cause the submission process to fail.
    """
    return pd.read_csv(StringIO(input_data), index_col=0)


if __name__ == "__main__":
    """Training Main

    The main function is called by both Unearthed's SageMaker pipeline and the
    Unearthed CLI's "unearthed train" command.
    
    WARNING - modifying this function may cause the submission process to fail.

    The main function must call preprocess, arrange th
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
    train(parser.parse_args())
