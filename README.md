# rpboosting: RPBoosting (Residual Prototype Boosting)

**RPBoosting** is a gradient-boosting classifier whose weak learners are smooth, closed-form **prototype functions** instead of decision trees. It is fully **scikit-learn compatible**: call it, `fit` it and `predict` with it exactly like `DecisionTreeClassifier`, `RandomForestClassifier` or `SVC`.

```bash
pip install rpboosting
```

```python
from rpboosting import RPBClassifier

model = RPBClassifier()              # 1. call
model.fit(X_train, y_train)          # 2. fit
model.predict(X_test)                # 3. predict
model.predict_proba(X_test)          #    probabilities
model.score(X_test, y_test)          # 4. evaluate (accuracy)
```

It works with everything in scikit-learn: `cross_val_score`, `GridSearchCV`, `Pipeline`, `clone`, metrics, and so on. Binary and multi-class labels (including string labels) and pandas DataFrames are supported.

---

## How it works

Like Gradient Boosting and XGBoost, RPBoosting builds an additive model

$$F(\mathbf{x}) = F_0 + \nu \sum_{m=1}^{M} \beta_m\, s_m(\mathbf{x}), \qquad p(y=1\mid\mathbf{x}) = \sigma(F(\mathbf{x}))$$

that is fit round by round to the gradient of the log-loss. The difference is the weak learner. Instead of a tree, each round adds a **residual-contrast prototype** function:

$$s(\mathbf{x}) = \exp\!\Big(-\gamma\,\tfrac{D_\mathbf{w}(\mathbf{x},\,\mathbf{c}^+)}{\tau^+}\Big) - \exp\!\Big(-\gamma\,\tfrac{D_\mathbf{w}(\mathbf{x},\,\mathbf{c}^-)}{\tau^-}\Big)$$

- $\mathbf{c}^+$ and $\mathbf{c}^-$ are the residual-weighted centroids of the points the model currently **under-predicts** and **over-predicts**. They are computed in closed form, inside a local neighbourhood around a hard-to-classify point.
- $D_\mathbf{w}$ is a weighted distance. The **contrast metric** weights each feature by how well it separates the two groups, so irrelevant features are ignored.
- $\tau^\pm$ is the median distance, so the bump width adapts to the data's scale and dimension.
- $\beta_m$ is the Newton step $-\sum g_i s_i / (\sum h_i s_i^2 + \lambda)$, and candidates are ranked by the XGBoost-style gain $(\sum g_i s_i)^2 / (\sum h_i s_i^2 + \lambda)$.

For categorical data (one-hot columns), v1.2 adds **rule learners**: $s(\mathbf{x})$ = AND of up to 3 category indicators (one path of a small tree, found by beam search). They compete with the prototype pairs each round by the same gain, so categorical interactions can be modelled directly (`rules=3`). One-hot columns are kept as 0/1 instead of standardized.

The result is a boosted sum of smooth bumps. Its decision boundaries are **smooth, curved and oblique** (like an SVM's), but it is trained greedily in linear time, with no kernel matrix.

## When to use it

- Tabular data with **continuous** features, where class boundaries may be curved or diagonal, and mixed numeric + categorical data (try `rules=3` when categories interact).
- When some features may be irrelevant or noisy.
- As a diverse member of a voting/stacking ensemble: its errors differ from those of tree ensembles.

**Less suited to:** purely axis-aligned or rule-like problems, where trees are still somewhat better, and very large datasets. The current implementation is pure NumPy and trains several times slower than XGBoost.

## Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `n_rounds` | 400 | number of boosting rounds |
| `learning_rate` | 0.15 | shrinkage |
| `reg_lambda` | 1.0 | L2 regularization of each learner's weight |
| `subsample` | 0.8 | row fraction per round |
| `init` | `"logistic"` | start score: cross-fitted logistic regression, or `"prior"` (class prior) |
| `categorical` | `"scale"` | keep 0/1 indicator columns (one-hot categories) as 0/1; `"none"` standardizes them |
| `rules` | 0 | max length of categorical rule learners (0 = off; try 3 for categorical data) |
| `metric` | `"adaptive"` | `"euclidean"`, `"contrast"`, or both competing by gain |
| `early_stopping` | False | stop when validation log-loss stops improving |
| `random_state` | None | seed for reproducibility |

The full list is in `help(RPBClassifier)`. Numeric features are standardized internally, so no scaler is needed. Missing values are not supported, so put a `SimpleImputer` in a `Pipeline`.

## Examples

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from rpboosting import RPBClassifier

cross_val_score(RPBClassifier(random_state=0), X, y, cv=5)

GridSearchCV(RPBClassifier(random_state=0),
             {"learning_rate": [0.05, 0.15, 0.3], "rules": [0, 3]}, cv=3).fit(X, y)

make_pipeline(SimpleImputer(), RPBClassifier()).fit(X, y)

RPBClassifier(n_rounds=2000, early_stopping=True).fit(X, y)

model.feature_importances_       # like tree ensembles
```

See `examples/quickstart.py`. Source code and issues: <https://github.com/Sadman-S-Khan/RPBClassifier>.

## Benchmark

31 real OpenML-CC18 datasets, 10 classifiers. Nested cross-validation: 5 outer folds; inside each, Optuna tuning with 20 trials × 3-fold CV and the same budget for every model.

| # | Model | Mean accuracy | Average rank (lower is better) |
|---|---|---|---|
| 1 | **RPBoosting (rpboosting 1.2)** | 0.8313 | **3.89** |
| 2 | SVM (RBF) | 0.8330 | 4.52 |
| 3 | CatBoost | 0.8296 | 5.05 |
| 4 | LightGBM | 0.8292 | 5.21 |
| 5 | XGBoost | 0.8293 | 5.26 |
| 6 | Random Forest | 0.8253 | 5.31 |
| 7 | Extra Trees | 0.8254 | 5.69 |
| 8 | Logistic Regression | 0.8214 | 5.95 |
| 9 | HistGradientBoosting | 0.8269 | 6.32 |
| 10 | KNN | 0.8032 | 7.81 |

RPBoosting has the best average rank, and also the best rank with untuned default parameters. After Holm correction it is significantly better than KNN and Logistic Regression and statistically tied with the other top models. Full results are in the accompanying paper.

## Citation

If you use rpboosting in research, please cite it (see `CITATION.cff`):

```
Khan, S. S. (2026). rpboosting: RPBoosting (Residual Prototype Boosting) (Version 1.2.0) [Computer software].
```

## License

BSD 3-Clause. See `LICENSE`.
