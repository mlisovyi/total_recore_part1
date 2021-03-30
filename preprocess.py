"""Preprocess data for the challenge.

This script will be invoked in two ways during the Unearthed scoring pipeline:
 - first during model training on the 'public' dataset
 - secondly during generation of predictions on the 'private' dataset
"""
import argparse
import logging

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

target_columns = [
    "AgPPM",
    "AsPPM",
    "AuPPM",
    "BaPPM",
    "BiPPM",
    "CdPPM",
    "CoPPM",
    "CuPPM",
    "FePCT",
    "MnPPM",
    "MoPPM",
    "NiPPM",
    "PPCT",
    "PbPPM",
    "SPPM",
    "SbPPM",
    "SePPM",
    "SnPPM",
    "SrPPM",
    "TePPB",
    "ThPPB",
    "UPPB",
    "VPCT",
    "WPPM",
    "ZnPPM",
    "ZrPPM",
    "BePPM",
    "AlPPM",
    "CaPPM",
    "CePPM",
    "CrPPM",
    "CsPPM",
    "GaPPM",
    "GePPM",
    "HfPPM",
    "InPPM",
    "KPPM",
    "LaPPM",
    "LiPPM",
    "MgPPM",
    "NaPPM",
    "NbPPM",
    "RbPPM",
    "RePPM",
    "ScPPM",
    "TaPPM",
    "TiPPM",
    "TlPPM",
    "YPPM",
]


def preprocess(data_file):
    """Apply preprocessing and featurization steps to each file in the data directory.

    Your preprocessing and feature generation goes here.
    """
    logger.info(f"running preprocess on {data_file}")

    # read the data file
    df = pd.read_csv(data_file, index_col=0)
    logger.info(f"data read from {data_file} has shape of {df.shape}")

    # split target and the features
    try:
        df_y = df[target_columns]
        df_x = df[[c for c in df if c not in target_columns]]
    except KeyError:
        df_y = None
        df_x = df

    # keep only the spectra
    cols_id = ["DepthTo", "SiteID", "DepthFrom"]
    df_sp = df_x.drop(columns=cols_id, axis=1)

    i_min = 100
    i_max = -300
    v_max = df_sp.iloc[:, i_min:i_max].max(axis=1)
    v_min = df_sp.iloc[:, i_min:i_max].min(axis=1)
    df_sp = df_sp.clip(v_min, v_max, axis=0).div(v_max, axis=0)

    n = 5
    df_sp = df_sp.rolling(n, axis=1, min_periods=1).sum().loc[:, ::n]

    # df_diff = df_sp.diff(axis=1).dropna(axis=1).rename(lambda s: f"diff_{s}", axis=1)

    df_x = pd.concat([df_x[cols_id], df_sp], axis=1)
    # rejoin the columns
    if df_y is not None:
        df = pd.concat([df_x, df_y], axis=1)
    else:
        df = df_x

    logger.info(f"data after preprocessing has shape of {df.shape}")
    return df


if __name__ == "__main__":
    """Preprocess Main

    The main function is called by both Unearthed's SageMaker pipeline and the
    Unearthed CLI's "unearthed preprocess" command.

    WARNING - modifying this file may cause the submission process to fail.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="/opt/ml/processing/input/public/public.csv.gz"
    )
    parser.add_argument(
        "--output", type=str, default="/opt/ml/processing/output/preprocess/public.csv"
    )
    args, _ = parser.parse_known_args()

    # call preprocessing on private data
    df = preprocess(args.input)

    # write to the output location
    df.to_csv(args.output)
