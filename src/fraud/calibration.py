import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(s: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(s / (1 - s)).reshape(-1, 1)


class Calibrator:
    def __init__(self, method: str):
        if method not in ("isotonic", "platt"):
            raise ValueError(method)
        self.method = method

    def fit(self, scores, y) -> "Calibrator":
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores, y)
        else:
            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(scores), y)
        return self

    def transform(self, scores) -> np.ndarray:
        if self.method == "isotonic":
            return np.asarray(self.model.predict(scores), dtype=float)
        return self.model.predict_proba(_logit(scores))[:, 1]
