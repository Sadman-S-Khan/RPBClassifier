import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_iris, make_moons
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils.estimator_checks import parametrize_with_checks

from rpboosting import RPBClassifier


# scikit-learn's official API-compliance suite (small n_rounds keeps it fast)
@parametrize_with_checks([RPBClassifier(n_rounds=10, random_state=0)])
def test_sklearn_compatible(estimator, check):
    check(estimator)


def test_binary_accuracy():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=0)
    clf = RPBClassifier(n_rounds=100, random_state=0).fit(Xtr, ytr)
    assert clf.score(Xte, yte) > 0.90
    proba = clf.predict_proba(Xte)
    assert proba.shape == (len(yte), 2)
    np.testing.assert_allclose(proba.sum(1), 1.0)
    assert clf.decision_function(Xte).shape == (len(yte),)


def test_multiclass_and_string_labels():
    X, y = load_iris(return_X_y=True)
    names = np.array(["setosa", "versicolor", "virginica"])[y]
    clf = RPBClassifier(n_rounds=50, random_state=0).fit(X, names)
    assert set(clf.classes_) == set(names)
    assert clf.predict_proba(X).shape == (len(y), 3)
    assert clf.score(X, names) > 0.9


def test_nonlinear_boundary():
    X, y = make_moons(400, noise=0.2, random_state=0)
    assert cross_val_score(RPBClassifier(n_rounds=100, random_state=0), X, y, cv=3).mean() > 0.9


def test_reproducible_with_random_state():
    X, y = make_moons(200, noise=0.3, random_state=1)
    a = RPBClassifier(n_rounds=30, random_state=7).fit(X, y).predict_proba(X)
    b = RPBClassifier(n_rounds=30, random_state=7).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)


def test_feature_importances_and_pandas_names():
    pd = pytest.importorskip("pandas")
    data = load_breast_cancer(as_frame=True)
    clf = RPBClassifier(n_rounds=30, random_state=0).fit(data.data, data.target)
    assert clf.feature_importances_.shape == (data.data.shape[1],)
    assert np.isclose(clf.feature_importances_.sum(), 1.0)
    assert list(clf.feature_names_in_) == list(data.data.columns)


def test_pipeline_gridsearch_early_stopping():
    X, y = load_breast_cancer(return_X_y=True)
    X = X.copy()
    X[::17, 3] = np.nan                                   # missing values -> imputer
    pipe = make_pipeline(SimpleImputer(), RPBClassifier(n_rounds=20, random_state=0))
    grid = GridSearchCV(pipe, {"rpbclassifier__learning_rate": [0.1, 0.3]}, cv=2).fit(X, y)
    assert grid.best_score_ > 0.9
    es = RPBClassifier(n_rounds=500, early_stopping=True, random_state=0)
    es.fit(np.nan_to_num(X), y)
    assert es.model_.n_rounds_ <= 500
    clone(es)                                              # clonable


def test_init_logistic_and_metrics():
    X, y = load_breast_cancer(return_X_y=True)
    for kw in ({"init": "logistic"}, {"metric": "euclidean"}, {"metric": "contrast"},
               {"feature_sampling": "uniform"}, {"subspace": 0.5}):
        assert RPBClassifier(n_rounds=20, random_state=0, **kw).fit(X, y).score(X, y) > 0.9


def test_errors():
    X, y = make_moons(50, random_state=0)
    with pytest.raises(ValueError):
        RPBClassifier().fit(X, np.zeros(50))               # single class
    clf = RPBClassifier(n_rounds=5, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        clf.predict(np.zeros((3, 5)))                      # wrong number of features


def test_categorical_rules_and_scale():
    # one-hot style data whose label is a 2-way AND of categories: rule learners can express it
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 3, 300), rng.integers(0, 3, 300)
    X = np.column_stack([np.eye(3)[a], np.eye(3)[b], rng.normal(size=300)])
    y = ((a == 0) & (b == 1)).astype(int)
    for kw in ({}, {"rules": 3}, {"rules": 3, "rule_min_support": 10}, {"categorical": "none", "rules": 2}):
        clf = RPBClassifier(n_rounds=60, random_state=0, **kw).fit(X, y)
        assert clf.score(X, y) > 0.9
    clf = RPBClassifier(n_rounds=60, rules=3, random_state=0).fit(X, y)
    assert any(isinstance(l[0], str) for l in clf.model_.learners_)   # a rule learner was used
    assert np.isclose(clf.feature_importances_.sum(), 1.0)
    with pytest.raises(ValueError):
        RPBClassifier(categorical="bad").fit(X, y)


def test_prior_init_and_v12_defaults():
    p = RPBClassifier().get_params()
    assert (p["n_rounds"], p["learning_rate"], p["init"], p["categorical"], p["rules"]) == \
        (400, 0.15, "logistic", "scale", 0)
    X, y = load_breast_cancer(return_X_y=True)
    assert RPBClassifier(n_rounds=20, init="prior", random_state=0).fit(X, y).score(X, y) > 0.9
