"""rpboost: Residual Prototype Boosting, a scikit-learn compatible classifier.

    from rpboost import RPBClassifier
    clf = RPBClassifier().fit(X_train, y_train)
    clf.predict(X_test)
"""
from ._rpb import RPBClassifier

__all__ = ["RPBClassifier"]
__version__ = "1.2.0"
