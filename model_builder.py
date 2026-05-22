import numpy as np

class Attr:
    def __init__(self, name):
        self.name = name

class Domain:
    def __init__(self, names):
        self.attributes = [Attr(n) for n in names]

class DummyModel:
    def __init__(self, feature_names):
        self.domain = Domain(feature_names)

    def predict(self, X):
        return (X[:, 0] > 15).astype(int)

    def predict_proba(self, X):
        p = np.clip(X[:, 0] / 100.0, 0, 1)
        probs = np.vstack([1 - p, p]).T
        return probs
