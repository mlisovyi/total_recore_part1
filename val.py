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
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import (
    Lasso,
    LassoCV,
    MultiTaskElasticNetCV,
    MultiTaskLassoCV,
    Ridge,
    RidgeCV,
)
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from preprocess import preprocess, target_columns
from score import scoring_fn
from train import CustomRidgeCV, save_model, model_fn, input_fn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Work around for a SageMaker path issue
# (see https://github.com/aws/sagemaker-python-sdk/issues/648)
# WARNING - removing this may cause the submission process to fail
if abspath("/opt/ml/code") not in sys.path:
    sys.path.append(abspath("/opt/ml/code"))


def train_evaluate_model(X_trn, X_tst, y_trn, y_tst, verbose=True):
    # if verbose:
    print(f"TRN: {len(X_trn)}, TST: {len(X_tst)}")

    # the example model
    # model = RidgeCV(alphas=(1e-2, 1e-1, 1, 1e1, 1e2), scoring=make_scorer(r2_score))
    # model = MultiTaskLassoCV()
    model = CustomRidgeCV(
        alphas=np.logspace(-5, 5, 50),
        scoring=make_scorer(r2_score),
        alpha_per_target=True,
    )
    # model = Pipeline([("scaler", MinMaxScaler()), ("model", model)])
    # model = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42)

    model.fit(X_trn, y_trn)

    preds_trn = pd.DataFrame(model.predict(X_trn), columns=target_columns)
    preds_tst = pd.DataFrame(model.predict(X_tst), columns=target_columns)

    score_trn = scoring_fn(y_trn, preds_trn)
    score_tst = scoring_fn(y_tst, preds_tst)

    if verbose:
        # scores_series = pd.Series(
        #     scoring_fn(y_tst, preds_tst, True), index=target_columns
        # )
        # print(scores_series.sort_values().tail())
        print(f"SCORE: TRN = {score_trn:.3f}, TST = {score_tst:.3f}")
    return model, score_trn, score_tst, preds_tst


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

    if not args.cv:
        X_trn, X_tst, y_trn, y_tst = train_test_split(
            X_train, y_train, test_size=0.25, random_state=314
        )
        _ = train_evaluate_model(X_trn, X_tst, y_trn, y_tst)
    else:
        scores = []
        y_oof = np.zeros_like(y_train.values)
        cv = KFold(n_splits=4, random_state=42, shuffle=True)
        for i, (idx_trn, idx_tst) in enumerate(cv.split(X_train, y_train)):
            logger.info(f"------ {i} -------------")
            X_trn, y_trn = X_train.iloc[idx_trn, :], y_train.iloc[idx_trn]
            X_tst, y_tst = X_train.iloc[idx_tst, :], y_train.iloc[idx_tst]
            # allow usage a subsampling to check dependence of performace on data sample
            n = int(len(y_trn) * 1)
            _, _, score_tst, preds_tst = train_evaluate_model(
                X_trn.head(n), X_tst, y_trn.head(n), y_tst, verbose=True
            )
            scores.append(score_tst)
            y_oof[idx_tst, :] = preds_tst.values

        print(f"Performance in CV: {np.mean(scores):.3f} +- {np.std(scores):.3f}")
        y_oof = pd.DataFrame(y_oof, index=y_train.index, columns=y_train.columns)
        y_oof.to_csv(join(args.data_dir, "oof.csv"))

    # save the model to disk)  #
    # save_model(model, args.model_dir)


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
    parser.add_argument(
        "--cv", default=False, action="store_true",
    )
    train(parser.parse_args())
