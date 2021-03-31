from sklearn.linear_model import Ridge, RidgeCV
import numpy as np


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


class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict(self, inputs):
        # Do something here with self.models to calculate your predictions.
        # then return them.
        return self.models.predict(inputs)

