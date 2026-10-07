from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

from features import build_dataset, check_build_dataset
from betting import payout, choose_bet

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "data" / "hands.csv")
check_build_dataset(df)  # vectorised features == website's features
X, y, groups = build_dataset(df)

# 9.1 Split BY SHOE (never by random hand -> leakage)
shoes = sorted(groups.unique())
cut = int(len(shoes) * 0.8)
train_mask = groups.isin(shoes[:cut])
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]
print(f"train hands: {len(X_train)}  test hands: {len(X_test)}")

# 9.2 Models
models = {
    "logistic": LogisticRegression(max_iter=2000),
    "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               num_leaves=15, min_child_samples=200, verbose=-1),
}

def betting_result(bets, outcomes):
    profits = np.array([payout(b, o) for b, o in zip(bets, outcomes)])
    mean = profits.mean()
    ci = 1.96 * profits.std(ddof=1) / np.sqrt(len(profits))
    return mean, ci

# 9.3 Baselines
freq = y_train.value_counts(normalize=True)
base_probs = np.tile([freq.get(c, 0) for c in sorted(freq.index)], (len(y_test), 1))
print(f"\nBaseline log loss (class frequencies): {log_loss(y_test, base_probs, labels=sorted(freq.index)):.5f}")
m, ci = betting_result(["B"] * len(y_test), y_test)
print(f"Always-Banker profit per bet: {m*100:+.3f}% ± {ci*100:.3f}%")

# 9.4 Evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    ll = log_loss(y_test, proba, labels=classes)
    bets = [choose_bet(dict(zip(classes, p)))[0] for p in proba]
    m, ci = betting_result(bets, y_test)
    share = pd.Series(bets).value_counts(normalize=True).round(3).to_dict()
    results[name] = (ll, m)
    print(f"\n[{name}] log loss: {ll:.5f}")
    print(f"[{name}] profit per bet: {m*100:+.3f}% ± {ci*100:.3f}%")
    print(f"[{name}] bet mix: {share}")

# 9.5 Save the model with the best log loss
best = min(results, key=lambda k: results[k][0])
joblib.dump({"model": models[best], "columns": list(X.columns), "name": best},
            ROOT / "models" / "model.joblib")
print(f"\nSaved {best} to models/model.joblib")
