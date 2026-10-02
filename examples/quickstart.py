"""rpboost quickstart: call -> fit -> predict -> evaluate, exactly like any sklearn model."""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from rpboost import RPBClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                    random_state=42)

# 1. call   2. fit   3. predict   4. evaluate
model = RPBClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("ROC-AUC :", round(roc_auc_score(y_test, y_prob), 4))
print(classification_report(y_test, y_pred))

# Drop-in replacement: same code for any classifier
for name, clf in [("DecisionTree", DecisionTreeClassifier(random_state=0)),
                  ("RandomForest", RandomForestClassifier(random_state=0)),
                  ("SVM", SVC()),
                  ("RPB", RPBClassifier(random_state=0))]:
    print(f"{name:13s} 5-fold CV accuracy: {cross_val_score(clf, X, y, cv=5).mean():.4f}")
