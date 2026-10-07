# Baccarat ML Project — Step-by-Step Build Guide

A machine learning project that tests whether any model can beat the simplest baccarat strategy ("always bet Banker"), plus a simple website for entering live hands and tracking the model's picks with paper money.

---

## 0. The goal (read this first)

The question this project answers is:

> **Can a model trained on past hands beat "always bet Banker" on hands it has never seen?**

Expected results with a correct dataset:

| Bet | Win rate | House edge |
|---|---|---|
| Banker (pays 0.95:1) | ~45.9% | −1.06% |
| Player (pays 1:1) | ~44.6% | −1.24% |
| Tie (pays 8:1) | ~9.5% | −14.4% |

Each hand is close to independent of previous hands, so the most likely outcome is that **no model beats the baseline**. That's a valid, publishable result. The website is a **paper-trading tracker**: it never touches real money, and it shows whether the model's picks actually do better than "always Banker" over time.

---

## 1. Tech stack

- **Python 3.10+**
- **pandas / numpy**: data handling
- **scipy / statsmodels**: statistical tests
- **scikit-learn**: logistic regression, splitting, metrics
- **LightGBM**: gradient boosting
- **joblib**: saving the model
- **Flask**: tiny web server
- **Plain HTML + CSS + JavaScript**: the UI (no framework)

---

## 2. Project structure

```
baccarat-ml/
├── data/
│   ├── raw/                 # the Kaggle CSV goes here
│   └── hands.csv            # cleaned, standardized data (created in Step 4)
├── models/
│   └── model.joblib         # saved model (created in Step 9)
├── notebooks/
│   └── exploration.ipynb    # optional, for Steps 4–6
├── src/
│   ├── prepare_data.py      # Step 4
│   ├── stats_tests.py       # Step 6
│   ├── features.py          # Step 7 (shared by training AND the website)
│   ├── betting.py           # Step 8 (payouts + bet choice)
│   └── train.py             # Steps 8–9
├── static/
│   └── index.html           # Step 10 (the UI)
├── app.py                   # Step 10 (the web server)
└── requirements.txt
```

---

## 3. Set up the environment

```bash
mkdir baccarat-ml && cd baccarat-ml
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
```

`requirements.txt`:

```
pandas
numpy
scipy
statsmodels
scikit-learn
lightgbm
joblib
flask
```

```bash
pip install -r requirements.txt
mkdir -p data/raw models src static notebooks
```

---

## 4. Get and standardize the data

### 4.1 Download

Dataset: https://www.kaggle.com/datasets/victornascimento/baccarat-dataset (a **simulation** of baccarat shoes).

Download it from the page (Download button), or with the Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d victornascimento/baccarat-dataset -p data/raw --unzip
```

### 4.2 Inspect the columns

```python
import pandas as pd
df = pd.read_csv("data/raw/<file>.csv")
print(df.head(20))
print(df.columns.tolist())
print(df.shape)
```

Answer these before going further:

1. **Which column identifies the shoe?** (If none, can you infer shoe boundaries?)
2. **Which column holds the result?** How are Banker / Player / Tie written?
3. **Are individual cards recorded?** If yes, you can add card-counting features later (Step 7.3).

### 4.3 Convert to a standard format

Everything downstream expects `data/hands.csv` with exactly these columns:

| column | meaning |
|---|---|
| `shoe_id` | which shoe the hand belongs to |
| `hand_no` | order of the hand within the shoe (0, 1, 2…) |
| `outcome` | `B`, `P`, or `T` |

`src/prepare_data.py`. **Edit the three names at the top** to match your dataset:

```python
import pandas as pd

RAW_FILE = "data/raw/<file>.csv"
SHOE_COL = "shoe"        # <-- change to the real shoe column
RESULT_COL = "winner"    # <-- change to the real result column
RESULT_MAP = {           # <-- map the dataset's values to B / P / T
    "banker": "B", "player": "P", "tie": "T",
}

df = pd.read_csv(RAW_FILE)
out = pd.DataFrame({
    "shoe_id": df[SHOE_COL],
    "outcome": df[RESULT_COL].astype(str).str.strip().str.lower().map(RESULT_MAP),
})
missing = out["outcome"].isna().sum()
assert missing == 0, f"{missing} results didn't map — check RESULT_MAP"

out["hand_no"] = out.groupby("shoe_id").cumcount()
out.to_csv("data/hands.csv", index=False)
print(out["outcome"].value_counts(normalize=True))
print("shoes:", out["shoe_id"].nunique(), "hands:", len(out))
```

---

## 5. Sanity-check the outcome rates

The printout from Step 4.3 should be close to **B ≈ 0.459, P ≈ 0.446, T ≈ 0.095**.

- **Close:** the simulator is behaving like real baccarat. Continue.
- **Far off:** the simulator may be flawed. That's the most interesting thing you could find, so note it and investigate before modeling.

---

## 6. Statistical tests (before any ML)

If these find no dependence between hands, ML models almost certainly won't either.

`src/stats_tests.py`:

```python
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.sandbox.stats.runs import runstest_1samp

df = pd.read_csv("data/hands.csv").sort_values(["shoe_id", "hand_no"])

# 6.1 Transition test: does the next outcome depend on the previous one?
df["prev"] = df.groupby("shoe_id")["outcome"].shift(1)
pairs = df.dropna(subset=["prev"])
table = pd.crosstab(pairs["prev"], pairs["outcome"])
print("Transition counts (rows = previous, cols = next):\n", table)
print("\nRow percentages:\n", (table.div(table.sum(axis=1), axis=0) * 100).round(2))
chi2, p, dof, _ = chi2_contingency(table)
print(f"\nChi-square p-value: {p:.4f}  (p > 0.05 = no evidence of dependence)")

# 6.2 Runs test on Banker/Player (ties removed): are streaks longer/shorter than random?
bp = df[df["outcome"] != "T"]
seq = (bp["outcome"] == "B").astype(int).values
z, p_runs = runstest_1samp(seq, cutoff=0.5)
print(f"Runs test p-value: {p_runs:.4f}  (p > 0.05 = streaks look random)")
```

Note: the runs test here treats the whole dataset as one sequence for simplicity. To be stricter, run it per shoe and look at the distribution of p-values.

**Record these results.** They are the core evidence in your final write-up.

---

## 7. Feature engineering

### 7.1 Key rule: one feature function for training AND the website

The website must compute features exactly the same way as training. Otherwise the model sees different inputs live than it was trained on. So the features live in one file used by both.

### 7.2 Results-only features

`src/features.py`:

```python
N_LAGS = 5

def features_from_history(history):
    """history: list of 'B'/'P'/'T' for the CURRENT shoe, oldest first.
    Returns a dict of features describing the state before the next hand."""
    f = {}

    # Last N outcomes, one-hot (all zeros if not enough hands yet)
    for i in range(1, N_LAGS + 1):
        o = history[-i] if len(history) >= i else None
        for k in "BPT":
            f[f"lag{i}_{k}"] = int(o == k)

    # Position in shoe and counts so far
    f["hand_index"] = len(history)
    for k in "BPT":
        f[f"count_{k}"] = history.count(k)

    # Current streak, ignoring ties
    non_tie = [h for h in history if h != "T"]
    streak, side = 0, 0
    if non_tie:
        last = non_tie[-1]
        for h in reversed(non_tie):
            if h == last:
                streak += 1
            else:
                break
        side = 1 if last == "B" else -1
    f["streak_len"] = streak
    f["streak_side"] = side
    return f


def build_dataset(df):
    """Turns hands.csv into (X, y, groups): one row per hand,
    features from the hands BEFORE it in the same shoe."""
    import pandas as pd
    rows, labels, groups = [], [], []
    for shoe_id, shoe in df.sort_values(["shoe_id", "hand_no"]).groupby("shoe_id"):
        outcomes = shoe["outcome"].tolist()
        for i, outcome in enumerate(outcomes):
            rows.append(features_from_history(outcomes[:i]))
            labels.append(outcome)
            groups.append(shoe_id)
    return pd.DataFrame(rows), pd.Series(labels), pd.Series(groups)
```

### 7.3 Optional: card-composition features (only if the dataset has cards)

If the dataset records each card dealt, add features for **how many of each card value (A, 2–9, 10/J/Q/K) remain in the shoe**. This is the only input with real predictive information in baccarat, though the effect is small and mostly appears late in the shoe. If you do this, the website must also let you enter cards, not just results. Start with results-only and add this as Version 2.

---

## 8. Payouts and bet choice

`src/betting.py`:

```python
def payout(bet, outcome):
    """Profit in units for a 1-unit bet."""
    if bet == "B":
        return 0.95 if outcome == "B" else (0.0 if outcome == "T" else -1.0)
    if bet == "P":
        return 1.0 if outcome == "P" else (0.0 if outcome == "T" else -1.0)
    if bet == "T":
        return 8.0 if outcome == "T" else -1.0
    raise ValueError(bet)


def choose_bet(probs):
    """probs: {'B': p, 'P': p, 'T': p}. Picks the bet with the highest
    expected value. EV accounts for real payouts and ties pushing."""
    ev = {
        "B": 0.95 * probs["B"] - probs["P"],
        "P": probs["P"] - probs["B"],
        "T": 8 * probs["T"] - (1 - probs["T"]),
    }
    best = max(ev, key=ev.get)
    return best, ev
```

Why EV and not "most likely outcome": a model can be "right" more often and still lose money because Banker pays 0.95 and Tie pays 8. Money is what matters.

---

## 9. Train and evaluate

`src/train.py`:

```python
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

from features import build_dataset
from betting import payout, choose_bet

df = pd.read_csv("data/hands.csv")
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
            "models/model.joblib")
print(f"\nSaved {best} to models/model.joblib")
```

Run from the project root:

```bash
cd src && python train.py && cd ..
```

### 9.6 How to read the results

- **Log loss:** lower is better. If a model's log loss is not clearly below the baseline's, it learned nothing useful.
- **Profit per bet ± range:** if the model's range overlaps the Always-Banker range, there's no evidence it's better.
- **Bet mix:** a sensible model will bet Banker nearly 100% of the time. If it often picks Player or Tie, it's probably overfitting.
- **Test set size matters.** With a few thousand test hands, the ± range is several percent, which is far too wide to detect a 1% difference. Use as much data as possible.

---

## 10. Build the website (paper-trading tracker)

### What it does

1. Shows the model's probabilities and suggested bet for the **next** hand.
2. You click **Banker / Player / Tie** after each real hand finishes.
3. It scores the suggestion against the actual result and keeps two running totals (paper units):
   - **Model:** following the model's suggestion each hand
   - **Always Banker:** the baseline
4. **Undo** fixes a mis-click. **New shoe** clears the shoe history (features reset) but keeps the running totals.

The model never retrains. Your entered hands are only its **input**.

### 10.1 The server: `app.py`

```python
import sys
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

sys.path.append("src")
from features import features_from_history
from betting import choose_bet

bundle = joblib.load("models/model.joblib")
model, columns = bundle["model"], bundle["columns"]

app = Flask(__name__, static_folder="static")


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/predict")
def predict():
    history = request.get_json().get("history", [])
    history = [h for h in history if h in ("B", "P", "T")]
    X = pd.DataFrame([features_from_history(history)])[columns]
    proba = model.predict_proba(X)[0]
    probs = {c: float(p) for c, p in zip(model.classes_, proba)}
    bet, ev = choose_bet(probs)
    return jsonify({"probs": probs, "bet": bet,
                    "ev": {k: float(v) for k, v in ev.items()},
                    "model": bundle["name"]})


if __name__ == "__main__":
    app.run(debug=True)
```

### 10.2 The UI: `static/index.html`

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Baccarat Model Tracker</title>
<style>
  :root { --b:#c0392b; --p:#2463eb; --t:#1e9e5a; --bg:#f6f7f9; --card:#fff; --text:#1d2330; --muted:#6b7280; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, sans-serif; background:var(--bg); color:var(--text); }
  main { max-width: 560px; margin: 0 auto; padding: 16px; }
  h1 { font-size: 1.3rem; margin: 8px 0 4px; }
  .note { color: var(--muted); font-size: .85rem; margin: 0 0 16px; }
  .card { background: var(--card); border-radius: 12px; padding: 16px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .label { font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); margin-bottom: 8px; }
  .bet { font-size: 1.6rem; font-weight: 700; }
  .probs { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin-top: 12px; text-align: center; }
  .probs div { background: var(--bg); border-radius: 8px; padding: 8px; font-size: .9rem; }
  .buttons { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; }
  button { border: 0; border-radius: 10px; padding: 14px 8px; font-size: 1rem; font-weight: 600; cursor: pointer; color: #fff; }
  .btn-B { background: var(--b); } .btn-P { background: var(--p); } .btn-T { background: var(--t); }
  .row { display: flex; gap: 8px; margin-top: 8px; }
  .secondary { background: #e5e7eb; color: var(--text); flex: 1; }
  .history { display: flex; flex-wrap: wrap; gap: 4px; min-height: 24px; }
  .dot { width: 22px; height: 22px; border-radius: 50%; color: #fff; font-size: .7rem; display: grid; place-items: center; font-weight: 700; }
  .dot.B { background: var(--b); } .dot.P { background: var(--p); } .dot.T { background: var(--t); }
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .stat { background: var(--bg); border-radius: 8px; padding: 10px; }
  .stat .v { font-size: 1.3rem; font-weight: 700; }
  .pos { color: var(--t); } .neg { color: var(--b); }
</style>
</head>
<body>
<main>
  <h1>Baccarat Model Tracker</h1>
  <p class="note">Paper trading only. Compares the model's picks against "always Banker."</p>

  <div class="card">
    <div class="label">Model suggests for next hand</div>
    <div class="bet" id="bet">…</div>
    <div class="probs" id="probs"></div>
  </div>

  <div class="card">
    <div class="label">Enter result of the hand that just finished</div>
    <div class="buttons">
      <button class="btn-B" onclick="record('B')">Banker</button>
      <button class="btn-P" onclick="record('P')">Player</button>
      <button class="btn-T" onclick="record('T')">Tie</button>
    </div>
    <div class="row">
      <button class="secondary" onclick="undo()">Undo</button>
      <button class="secondary" onclick="newShoe()">New shoe</button>
    </div>
  </div>

  <div class="card">
    <div class="label">Current shoe (<span id="shoeCount">0</span> hands)</div>
    <div class="history" id="history"></div>
  </div>

  <div class="card">
    <div class="label">Paper results (<span id="handsTotal">0</span> hands scored)</div>
    <div class="stats">
      <div class="stat"><div class="label">Model</div><div class="v" id="modelTotal">0.00</div><div id="modelPer" class="note"></div></div>
      <div class="stat"><div class="label">Always Banker</div><div class="v" id="bankerTotal">0.00</div><div id="bankerPer" class="note"></div></div>
    </div>
  </div>
</main>

<script>
  const NAMES = { B: "Banker", P: "Player", T: "Tie" };
  let shoe = [];        // outcomes in the current shoe (model input)
  let ledger = [];      // every scored hand: {outcome, bet, modelProfit, bankerProfit, newShoeBefore}
  let current = null;   // latest prediction
  let pendingNewShoe = false;

  function payout(bet, o) {
    if (bet === "B") return o === "B" ? 0.95 : (o === "T" ? 0 : -1);
    if (bet === "P") return o === "P" ? 1 : (o === "T" ? 0 : -1);
    return o === "T" ? 8 : -1;
  }

  async function predict() {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ history: shoe })
    });
    current = await res.json();
    render();
  }

  function record(outcome) {
    if (!current) return;
    ledger.push({
      outcome, bet: current.bet,
      modelProfit: payout(current.bet, outcome),
      bankerProfit: payout("B", outcome),
      shoeBefore: [...shoe]
    });
    shoe.push(outcome);
    predict();
  }

  function undo() {
    const last = ledger.pop();
    if (!last) return;
    shoe = last.shoeBefore;
    predict();
  }

  function newShoe() {
    if (shoe.length && !confirm("Start a new shoe? Totals are kept.")) return;
    shoe = [];
    predict();
  }

  function fmt(x) { return (x >= 0 ? "+" : "") + x.toFixed(2); }

  function render() {
    document.getElementById("bet").textContent = NAMES[current.bet];
    document.getElementById("probs").innerHTML = ["B","P","T"].map(k =>
      `<div><strong>${NAMES[k]}</strong><br>${(current.probs[k]*100).toFixed(1)}%</div>`).join("");

    document.getElementById("shoeCount").textContent = shoe.length;
    document.getElementById("history").innerHTML =
      shoe.map(o => `<div class="dot ${o}">${o}</div>`).join("");

    const n = ledger.length;
    const m = ledger.reduce((s, r) => s + r.modelProfit, 0);
    const b = ledger.reduce((s, r) => s + r.bankerProfit, 0);
    document.getElementById("handsTotal").textContent = n;
    for (const [id, val] of [["modelTotal", m], ["bankerTotal", b]]) {
      const el = document.getElementById(id);
      el.textContent = fmt(val) + " u";
      el.className = "v " + (val >= 0 ? "pos" : "neg");
    }
    document.getElementById("modelPer").textContent = n ? `${fmt(m / n * 100)}% per bet` : "";
    document.getElementById("bankerPer").textContent = n ? `${fmt(b / n * 100)}% per bet` : "";
  }

  predict();
</script>
</body>
</html>
```

### 10.3 Run it

```bash
python app.py
```

Open http://127.0.0.1:5000. Click the result after each hand. Click **New shoe** when the dealer starts a new shoe.

Note: results are kept in the browser tab's memory, so refreshing the page resets them. For a long test, add an "Export CSV" button or save each record on the server.

---

## 11. Test the website before using it

1. **No-input check:** on load, the suggestion should appear (usually Banker).
2. **Payout check:** click Banker once. Always Banker should show +0.95. Click Tie. Always Banker should not change.
3. **Undo check:** click Player, then Undo. Totals and shoe history should return to their previous values.
4. **New shoe check:** the shoe history clears, totals stay.
5. **Consistency check:** pick a shoe from your test set, enter its hands, and confirm the website's probabilities match what `model.predict_proba` gives in Python for the same history.

---

## 12. Using the tracker and judging the result

- Track **at least several hundred hands** before drawing any conclusion. Over one session, either line can be ahead by luck.
- The comparison that matters is **Model vs. Always Banker per bet**, not whether the model is positive on a given day.
- If the two lines stay close and both trend negative, the model has no edge. That matches the statistical tests in Step 6.
- Keep it paper-only. Nothing in this project justifies betting real money on the model's picks.

---

## 13. Final write-up (portfolio)

Include:

1. The question and why "always Banker" is the baseline
2. Outcome rates and whether the simulator matches real baccarat (Step 5)
3. Statistical test results (Step 6)
4. Features used and why the split is by shoe (Steps 7, 9.1)
5. Log loss and profit per bet for each model vs. baseline, with ± ranges (Step 9)
6. Paper-trading results from the tracker (Step 12)
7. Conclusion and what you learned

---

## 14. Possible Version 2 upgrades

- Card-composition features (Step 7.3) with card entry in the UI
- Export / save the paper ledger to CSV
- A chart of both running totals over time
- Cross-validation across several shoe splits instead of one 80/20 split
