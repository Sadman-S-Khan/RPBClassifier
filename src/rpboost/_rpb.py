"""
Residual Prototype Boosting (RPB), v1.2.

Gradient boosting (log-loss) whose weak learner is a smooth, closed-form
"residual-contrast prototype" function instead of a tree:

    s(x) = exp(-gamma * d_w(x_S, c_plus) / tau_plus) - exp(-gamma * d_w(x_S, c_minus) / tau_minus)
    d_w(x, c) = sum_j w_j (x_j - c_j)^2

  * c_plus / c_minus : |residual|-weighted centroids of the samples the model
                       under-predicts / over-predicts, computed inside a local
                       neighbourhood around a hardness-sampled anchor
  * w (contrast metric): w_j proportional to ((c_plus_j - c_minus_j) / spread_j)^2, so
                       features that separate the two residual groups count and
                       noise features are ignored; neighbourhood + prototypes are
                       recomputed once in this metric ("adaptive": the plain and the
                       contrast metric both compete, the higher gain wins)
  * tau             : median distance to the prototype (scale-adaptive width,
                       fixes distance concentration in high dimensions)
  * S               : feature subspace, sampled half uniformly, half in proportion
                       to how useful each feature has been in earlier rounds
  * leaf weight     : Newton step  beta = -sum(g*s) / (sum(h*s^2) + lambda)
  * candidate choice: gain = sum(g*s)^2 / (sum(h*s^2) + lambda)
  * if one side of the neighbourhood is empty -> single bump
  * F0(x)           : cross-fitted logistic model (default) or the class prior

New in v1.2 (categorical data):
  * categorical="scale": 0/1 indicator columns (e.g. one-hot categories) stay 0/1 instead
    of being standardized, so a rare category no longer dominates distances
  * rules=k: "rule" weak learners s(x) = AND of up to k category indicators (one path of
    a small tree, found by beam search) compete with the prototype pairs each round by
    the same gain, so categorical interactions can be modelled directly. Off by default
    (it helps rule-structured data such as tic-tac-toe/car and can overfit small noisy
    data); a good hyperparameter to tune over {0, 3}
  * new defaults (300 -> 400 rounds, lr 0.1 -> 0.15, init "prior" -> "logistic"),
    chosen on the benchmark datasets: better on both numeric and categorical data
  * faster fit (same results): one exp for all gammas, vectorized gains, partition median

F(x) = F0(x) + nu * sum_m beta_m * s_m(x),   p(x) = sigmoid(F(x))

Usage (same as any scikit-learn model):
    from rpboost import RPBClassifier
    model = RPBClassifier()
    model.fit(X_train, y_train)
    model.predict(X_test); model.predict_proba(X_test); model.score(X_test, y_test)
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def _rbf_stack(z, gammas):
    """Rows exp(-gamma * z) for each gamma. For the default gammas (0.5, 1, 2, 4) one exp
    and three squarings replace four exps."""
    if gammas == (0.5, 1.0, 2.0, 4.0):
        e1 = np.exp(-0.5 * z)
        e2 = e1 * e1
        e3 = e2 * e2
        return np.stack([e1, e2, e3, e3 * e3])
    return np.stack([np.exp(-g * z) for g in gammas])


def _sqdist(Xs, c, w):
    return ((Xs - c) ** 2) @ w


def _wmean(Xs, idx, wts):
    """np.average(Xs[idx], axis=0, weights=wts) without the overhead."""
    return (wts @ Xs[idx]) / wts.sum()


def _median(v):
    """np.median of a 1-D array, by partition instead of a full sort."""
    n = len(v)
    h = n // 2
    if n % 2:
        return np.partition(v, h)[h]
    part = np.partition(v, (h - 1, h))
    return 0.5 * (part[h - 1] + part[h])


def _contrast_weights(Xs, pos, neg, cp, cm):
    """Per-feature weight = squared standardized gap between the two residual prototypes.
    Features that separate under- from over-predicted points get large weight,
    noise features get weight ~ 0.  Normalized to sum to 1."""
    k = Xs.shape[1]
    if cp is None or cm is None:
        return np.full(k, 1.0 / k)
    Z = Xs[np.r_[pos, neg]]
    spread = np.sqrt(((Z - Z.mean(0)) ** 2).mean(0)) + 1e-3
    w = ((cp - cm) / spread) ** 2
    tot = w.sum()
    return w / tot if tot > 1e-12 else np.full(k, 1.0 / k)


def _indicator_columns(X):
    """Columns whose only values are 0 and 1 (one-hot categories, binary flags)."""
    return np.array([j for j in range(X.shape[1])
                     if len(u := np.unique(X[:, j])) == 2 and np.isin(u, (0.0, 1.0)).all()],
                    dtype=int)


class _BinaryRPB(ClassifierMixin, BaseEstimator):
    def __init__(self, n_rounds=400, learning_rate=0.15, n_candidates=12,
                 subspace="auto", gammas=(0.5, 1.0, 2.0, 4.0),
                 neighbourhoods=(0.05, 0.15, 0.4, 1.0), reg_lambda=1.0,
                 subsample=0.8, init="logistic", init_C=1.0, metric="adaptive",
                 feature_sampling="adaptive", categorical="scale", rules=0, rule_beam=5,
                 rule_min_support=0.05, early_stopping=False, validation_fraction=0.15,
                 n_iter_no_change=20, random_state=None):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.n_candidates = n_candidates
        self.subspace = subspace
        self.gammas = gammas
        self.neighbourhoods = neighbourhoods
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.init = init
        self.init_C = init_C
        self.metric = metric
        self.feature_sampling = feature_sampling
        self.categorical = categorical
        self.rules = rules
        self.rule_beam = rule_beam
        self.rule_min_support = rule_min_support
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    # ---- weak learners ------------------------------------------------
    @staticmethod
    def _eval(Xs, c, w, tau, gamma):
        return 0.0 if c is None else np.exp(-gamma * _sqdist(Xs, c, w) / tau)

    def _s(self, X, learner):
        if isinstance(learner[0], str):            # rule: AND of category indicators
            cols = learner[1]
            return np.all(X[:, cols] > self.ind_cut_[cols], axis=1).astype(float)
        feats, w, cp, cm, tp, tm, gamma, _ = learner
        Xs = X[:, feats]
        return self._eval(Xs, cp, w, tp, gamma) - self._eval(Xs, cm, w, tm, gamma)

    def _candidate(self, Xs_all, r, p_anchor, rng):
        n, k = Xs_all.shape
        frac = rng.choice(self.neighbourhoods)
        anchor = Xs_all[rng.choice(n, p=p_anchor)] if frac < 1.0 else None  # hard points likelier
        m = min(n, max(8, int(frac * n)))
        w = np.full(k, 1.0 / k)
        variants = []
        # pass 1: neighbourhood + prototypes in the plain metric
        # pass 2: re-weight features by prototype contrast, redo neighbourhood + prototypes
        n_pass = 1 if self.metric == "euclidean" else 2
        for step in range(n_pass):
            idx = np.arange(n) if anchor is None else \
                np.argpartition(_sqdist(Xs_all, anchor, w), m - 1)[:m]
            ri = r[idx]
            pos, neg = idx[ri > 0], idx[ri < 0]
            cp = _wmean(Xs_all, pos, r[pos]) if len(pos) else None
            cm = _wmean(Xs_all, neg, -r[neg]) if len(neg) else None
            if step == 0 and self.metric != "contrast":
                variants.append((w, cp, cm))          # plain-metric learner
            if n_pass == 2:
                w = _contrast_weights(Xs_all, pos, neg, cp, cm)
                if step == 1:
                    variants.append((w, cp, cm))      # contrast-metric learner
        return variants

    def _best_rule(self, B, gs, hs):
        """Beam search for the rule (AND of up to `rules` indicators) with the highest gain.
        B: boolean (rows x indicator columns). Returns (gain, column positions in B)."""
        Bf = B.astype(float)
        lam = self.reg_lambda
        ms = self.rule_min_support                 # int = rows, float < 1 = fraction of rows
        ms = max(2, int(np.ceil(ms * len(B)))) if isinstance(ms, float) and ms < 1 \
            else max(2, int(ms))
        G, H = gs @ Bf, hs @ Bf + lam
        gain = G * G / H
        gain[Bf.sum(0) < ms] = -1
        top = int(np.argmax(gain))
        best = (gain[top], (top,))
        beam = [((j,), B[:, j]) for j in np.argsort(-gain)[:self.rule_beam]]
        for _ in range(1, self.rules):
            cand = []
            for cols, mask in beam:
                mf = mask.astype(float)            # (g*mask) @ B == g @ (B*mask), no n x |B| copy
                G, H = (gs * mf) @ Bf, (hs * mf) @ Bf + lam
                gn = G * G / H
                gn[list(cols)] = -1
                gn[mf @ Bf < ms] = -1              # too little support
                for j in np.argsort(-gn)[:self.rule_beam]:
                    if gn[j] > 0:
                        cand.append((gn[j], tuple(sorted(cols + (j,))), mask & B[:, j]))
            if not cand:
                break
            cand.sort(key=lambda t: -t[0])
            seen, beam = set(), []
            for gn, cols, mask in cand:
                if cols not in seen:
                    seen.add(cols)
                    beam.append((cols, mask))
                    if gn > best[0]:
                        best = (gn, cols)
                if len(beam) >= self.rule_beam:
                    break
        return best

    # ---- initial score F0(x) -------------------------------------------
    def _F0(self, X):
        if self.init_model_ is None:
            return np.full(len(X), self.F0_)
        return self.init_model_.decision_function(X)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = np.random.default_rng(self.random_state)
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("_BinaryRPB needs exactly two classes")
        if self.categorical not in ("scale", "none"):
            raise ValueError('categorical must be "scale" or "none"')
        self.scaler_ = StandardScaler().fit(X)
        ind_cols = _indicator_columns(X)
        if self.categorical == "scale":            # indicators stay 0/1
            self.scaler_.mean_[ind_cols] = 0.0
            self.scaler_.scale_[ind_cols] = 1.0
        # midpoint between 0 and 1 in scaled units: recovers each indicator after scaling
        self.ind_cut_ = (0.5 - self.scaler_.mean_) / self.scaler_.scale_
        X = self.scaler_.transform(X)
        y = (y == self.classes_[1]).astype(float)

        if self.early_stopping:
            perm = rng.permutation(len(X))
            nv = max(1, int(self.validation_fraction * len(X)))
            Xv, yv, X, y = X[perm[:nv]], y[perm[:nv]], X[perm[nv:]], y[perm[nv:]]
        n, d = X.shape
        if self.subspace == "auto":             # ~d^0.75 features: all when d is small
            k = min(d, max(2, int(round(d ** 0.75))))
        else:
            k = max(1, int(round(self.subspace * d)))

        p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.F0_ = np.log(p0 / (1 - p0))
        self.init_model_ = None
        F = np.full(n, self.F0_)
        if self.init == "logistic" and min(y.sum(), n - y.sum()) >= 3:
            lr = LogisticRegression(C=self.init_C, max_iter=2000)
            self.init_model_ = clone(lr).fit(X, y)
            # cross-fitted start: each training row gets the score of a logistic model that
            # never saw it, so the residuals the prototypes chase are honest (not overfit)
            folds = min(5, int(min(y.sum(), n - y.sum())))
            F = cross_val_predict(lr, X, y, cv=StratifiedKFold(folds, shuffle=True,
                                  random_state=int(rng.integers(1 << 31))),
                                  method="decision_function")
        self.learners_ = []
        if self.early_stopping:
            Fv, best_loss, best_m, wait = self._F0(Xv), np.inf, 0, 0

        n_sub = max(16, int(self.subsample * n)) if self.subsample < 1 else n
        imp = np.zeros(d)                      # running feature usefulness
        gammas = tuple(float(g) for g in self.gammas)
        use_rules = bool(self.rules) and len(ind_cols) > 0
        for _ in range(self.n_rounds):
            p = _sigmoid(F)
            g, h = p - y, p * (1 - p)          # gradient, hessian of log-loss
            r = -g                             # pseudo-residual
            sub = rng.choice(n, min(n, n_sub), replace=False) if n_sub < n else np.arange(n)
            gs, hs, rs = g[sub], h[sub], r[sub]
            p_anchor = np.abs(rs) + 1e-12
            p_anchor /= p_anchor.sum()
            best = None
            if self.feature_sampling == "adaptive" and imp.sum() > 0:
                # half uniform, half proportional to how useful each feature has been so far
                p_feat = 0.5 / d + 0.5 * imp / imp.sum()
            else:
                p_feat = None
            feat_sets = [np.sort(rng.choice(d, k, replace=False, p=p_feat))
                         for _ in range(self.n_candidates)]
            for feats in feat_sets:
                Xs = X[np.ix_(sub, feats)]
                for w, cp, cm in self._candidate(Xs, rs, p_anchor, rng):
                    if cp is None and cm is None:
                        continue
                    # distances once, reused for every gamma; tau = median distance (scale-free)
                    dp = _sqdist(Xs, cp, w) if cp is not None else None
                    dm = _sqdist(Xs, cm, w) if cm is not None else None
                    tp = _median(dp) + 1e-12 if dp is not None else 1.0
                    tm = _median(dm) + 1e-12 if dm is not None else 1.0
                    S = (_rbf_stack(dp / tp, gammas) if dp is not None else 0.0) \
                        - (_rbf_stack(dm / tm, gammas) if dm is not None else 0.0)
                    G, H = S @ gs, (S * S) @ hs + self.reg_lambda
                    gain = G * G / H
                    a = int(np.argmax(gain))
                    if best is None or gain[a] > best[0]:
                        best = (gain[a], feats, w, cp, cm, tp, tm, gammas[a])
            rule = None
            if use_rules:
                B = X[np.ix_(sub, ind_cols)] > self.ind_cut_[ind_cols]
                rule = self._best_rule(B, gs, hs)
            if rule is not None and (best is None or rule[0] > best[0]):
                learner = ["rule", ind_cols[list(rule[1])], 0.0]
                beta_i, w_imp = 2, None
            elif best is not None:
                _, feats, w, cp, cm, tp, tm, gamma = best
                learner = [feats, w, cp, cm, tp, tm, gamma, 0.0]
                beta_i, w_imp = 7, w
            else:
                break
            s = self._s(X, learner)
            # Newton step on the full training set
            learner[beta_i] = -(g * s).sum() / ((h * s * s).sum() + self.reg_lambda)
            F += self.learning_rate * learner[beta_i] * s
            self.learners_.append(tuple(learner))
            cols = learner[1] if w_imp is None else learner[0]
            np.add.at(imp, cols, abs(learner[beta_i]) * (w_imp if w_imp is not None
                                                          else 1.0 / len(cols)))

            if self.early_stopping:
                Fv += self.learning_rate * learner[beta_i] * self._s(Xv, learner)
                pv = np.clip(_sigmoid(Fv), 1e-12, 1 - 1e-12)
                loss = -np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))
                if loss < best_loss - 1e-6:
                    best_loss, best_m, wait = loss, len(self.learners_), 0
                else:
                    wait += 1
                    if wait >= self.n_iter_no_change:
                        break
        if self.early_stopping:
            self.learners_ = self.learners_[:best_m]
        self.n_rounds_ = len(self.learners_)
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = self.scaler_.transform(check_array(X))
        F = self._F0(X)
        for learner in self.learners_:
            F += self.learning_rate * learner[-1] * self._s(X, learner)
        return F

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def feature_importances(self, n_features):
        """|weight|-weighted usage of each feature (like tree feature importance)."""
        imp = np.zeros(n_features)
        for learner in self.learners_:
            if isinstance(learner[0], str):
                imp[learner[1]] += abs(learner[2]) / len(learner[1])
            else:
                feats, w, *_, beta = learner
                imp[feats] += abs(beta) * w
        return imp / imp.sum() if imp.sum() else imp


class RPBClassifier(ClassifierMixin, BaseEstimator):
    """Residual Prototype Boosting classifier.

    Gradient boosting with smooth, closed-form "residual-contrast prototype" weak
    learners instead of trees. Binary problems are solved directly; multi-class
    problems use one-vs-rest. Numeric features are standardized internally (0/1 indicator
    columns such as one-hot categories are kept as 0/1), so no scaler is needed. Missing values are not supported (use an imputer in a Pipeline).

    Parameters
    ----------
    n_rounds : int, default=400              number of boosting rounds (weak learners)
    learning_rate : float, default=0.15      shrinkage nu
    n_candidates : int, default=12           prototype pairs tried per round
    subspace : "auto" or float               fraction of features per learner ("auto" = d**0.75 features)
    gammas : tuple, default=(0.5, 1, 2, 4)   RBF sharpness, relative to the median distance
    neighbourhoods : tuple                   local-neighbourhood sizes (fraction of data; 1.0 = global)
    reg_lambda : float, default=1.0          L2 penalty on learner weight (as in XGBoost)
    subsample : float, default=0.8           row fraction used to search candidates each round
    init : {"logistic", "prior"}             starting score: a cross-fitted logistic model
                                             (default) or the class prior
    init_C : float, default=1.0              inverse L2 strength of that logistic model
    metric : {"adaptive", "contrast", "euclidean"}
                                             feature weights of each learner's distance:
                                             "contrast" = weights from prototype gap,
                                             "adaptive" = try both, keep the higher gain
    feature_sampling : {"adaptive", "uniform"}
                                             "adaptive" = sample features that helped before more often
    categorical : {"scale", "none"}          "scale" = 0/1 indicator columns (one-hot categories)
                                             are kept 0/1 instead of standardized
    rules : int, default=0                   max length of categorical rule learners (AND of
                                             indicators); 0 = off. Try 3 for categorical data
    rule_beam : int, default=5               beam width of the rule search
    rule_min_support : int or float          minimum rows a rule must cover (float < 1 = fraction)
    early_stopping : bool, default=False     hold out validation data and stop when log-loss stalls
    validation_fraction : float              size of that hold-out set
    n_iter_no_change : int                   patience for early stopping
    random_state : int or None               seed

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during fit.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Column names, only when X was a pandas DataFrame with string column names.
    feature_importances_ : ndarray of shape (n_features_in_,)
        Normalized |weight|-weighted usage of each feature across all weak learners.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> from rpboost import RPBClassifier
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    >>> clf = RPBClassifier(random_state=0).fit(X_train, y_train)
    >>> clf.score(X_test, y_test)  # doctest: +SKIP
    0.96...
    """

    def __init__(self, n_rounds=400, learning_rate=0.15, n_candidates=12,
                 subspace="auto", gammas=(0.5, 1.0, 2.0, 4.0),
                 neighbourhoods=(0.05, 0.15, 0.4, 1.0), reg_lambda=1.0,
                 subsample=0.8, init="logistic", init_C=1.0, metric="adaptive",
                 feature_sampling="adaptive", categorical="scale", rules=0, rule_beam=5,
                 rule_min_support=0.05, early_stopping=False, validation_fraction=0.15,
                 n_iter_no_change=20, random_state=None):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.n_candidates = n_candidates
        self.subspace = subspace
        self.gammas = gammas
        self.neighbourhoods = neighbourhoods
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.init = init
        self.init_C = init_C
        self.metric = metric
        self.feature_sampling = feature_sampling
        self.categorical = categorical
        self.rules = rules
        self.rule_beam = rule_beam
        self.rule_min_support = rule_min_support
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the model to training data X (n_samples, n_features) and labels y."""
        cols = getattr(X, "columns", None)
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        if len(self.classes_) < 2:
            raise ValueError("RPBClassifier needs at least 2 classes in y; got "
                             f"{len(self.classes_)} class: {self.classes_}")
        self.n_features_in_ = X.shape[1]
        if cols is not None and all(isinstance(c, str) for c in cols):
            self.feature_names_in_ = np.asarray(cols, dtype=object)
        base = _BinaryRPB(**self.get_params())
        self.model_ = base.fit(X, y) if len(self.classes_) == 2 \
            else OneVsRestClassifier(base).fit(X, y)
        return self

    def _check_X(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but RPBClassifier is expecting "
                             f"{self.n_features_in_} features as input.")
        return X

    def decision_function(self, X):
        """Raw scores: log-odds (binary, shape (n,)) or one column per class."""
        X = self._check_X(X)
        return self.model_.decision_function(X)

    def predict_proba(self, X):
        """Class probabilities, shape (n_samples, n_classes)."""
        X = self._check_X(X)
        return self.model_.predict_proba(X)

    def predict(self, X):
        """Predicted class labels."""
        X = self._check_X(X)
        return self.model_.predict(X)

    @property
    def feature_importances_(self):
        check_is_fitted(self)
        ests = [self.model_] if len(self.classes_) == 2 else self.model_.estimators_
        imp = np.mean([e.feature_importances(self.n_features_in_) for e in ests], 0)
        return imp / imp.sum() if imp.sum() else imp
