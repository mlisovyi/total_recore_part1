import logging
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def scoring_fn(y, y_pred, individual_scores=False):
    """Scoring Function

    The score is determined by calculating the r2 score for each target columns, and then taking an average of r2 scores.
    """
    logger.info("scoring_fn")
    r2_scores = np.zeros(0)
    weight_array = np.array(
        [
            0.025,
            0.01,
            0.04,
            0.01,
            0.01,
            0.01,
            0.01,
            0.35,
            0.025,
            0.01,
            0.01,
            0.025,
            0.01,
            0.025,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.025,
            0.01,
            0.01,
            0.025,
            0.025,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.025,
            0.01,
            0.01,
            0.025,
            0.025,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
        ]
    )

    for i in range(0, y.shape[1]):
        r2_scores = np.append(
            r2_scores,
            r2_score(y.iloc[:, i], y_pred.iloc[:, i], multioutput="uniform_average"),
        )

    if individual_scores:
        return r2_scores * weight_array
    else:
        return np.dot(r2_scores, weight_array)


if __name__ == "__main__":
    """Scoring Function

    This function is called by Unearthed's SageMaker pipeline. It must be left intact.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actual", type=str, default="/opt/ml/processing/input/public/public.csv.gz"
    )
    parser.add_argument(
        "--predicted",
        type=str,
        default="/opt/ml/processing/input/predictions/public.csv.out",
    )
    parser.add_argument(
        "--output", type=str, default="/opt/ml/processing/output/scores/public.txt"
    )
    args = parser.parse_args()

    # read the data file
    df_actual = pd.read_csv(args.actual, index_col=0)

    # recreate the targets
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
    targets = df_actual[target_columns]
    logger.info(f"true targets have shape of {targets.shape}")

    # read the predictions
    df_pred = pd.read_csv(args.predicted, header=None)
    logger.info(f"predictions have shape of {df_pred.shape}")

    score = scoring_fn(targets, df_pred)

    # write to the output location
    with open(args.output, "w") as f:
        f.write(str(score))

