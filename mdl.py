from sklearn.linear_model import Ridge, RidgeCV
import numpy as np


class CustomRidgeCV(RidgeCV):
    _v_min = 0

    def predict(self, X):
        y_preds = super().predict(X)
        y_preds = np.clip(y_preds, self._v_min, None)
        return y_preds

