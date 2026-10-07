# Baccarat AI

Can a model trained on past hands beat the simplest baccarat strategy, "always bet Banker", on hands it has never seen?

This project trains logistic regression and LightGBM models on 7.3 million simulated hands and compares them with that baseline. It also includes a small website for paper-trading the model's picks during live play. No real money is involved.

The full step-by-step reasoning is in [`baccarat-ml-build-guide.md`](baccarat-ml-build-guide.md).

## Results

| | Log loss | Profit per bet (test set, 1.46M hands) |
|---|---|---|
| Always Banker (baseline) | 0.94144 | **−0.905% ± 0.150%** |
| Logistic regression | 0.94144 | −0.982% ± 0.152% |
| LightGBM | 0.94146 | −0.987% ± 0.152% |

Neither model beats the baseline. Their log loss is no better than predicting the overall Banker/Player/Tie rates for every hand, so past results carry no usable information. The models also bet Player about a third of the time, which costs a little because Player has the higher house edge.

The statistical tests agree. The previous outcome doesn't affect the next one (chi-square p = 0.21). Streaks are 0.04 percentage points longer than chance, which is far too small to profit from.

## Quick start: run the website with the saved model

The trained model is saved in the repo as `models/model.joblib`, so you don't need the dataset or any training to use the website. You'll need Python 3.10 or newer and git.

1. Get the code:

   ```bash
   git clone https://github.com/yakootakiii/Baccarat-AI-.git
   cd Baccarat-AI-
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   `requirements.txt` pins `scikit-learn==1.9.1`, the version that saved the model. A different version may refuse to load it.

4. Start the website:

   ```bash
   flask --app app run --port 5001
   ```

5. Open http://127.0.0.1:5001 in your browser. On macOS, port 5000 (Flask's default) is taken by AirPlay Receiver, which is why this uses 5001.

6. Stop the server with Ctrl+C when you're done.

See [Using the website](#using-the-website) below for what to do on the page.

## Reproduce the results from scratch

These steps download the dataset, rerun the tests and retrain the model, which overwrites `models/model.joblib`. On macOS, LightGBM needs the OpenMP library first: `brew install libomp`.

### 1. Set up

Do steps 1–3 of the quick start above.

### 2. Download the dataset

The dataset is [victornascimento/baccarat-dataset](https://www.kaggle.com/datasets/victornascimento/baccarat-dataset) on Kaggle: 100,000 simulated shoes, one JSON file of dealt cards per shoe (about 780 MB). It's public, so you don't need a Kaggle account:

```bash
pip install kagglehub
python -c "import kagglehub, shutil; shutil.copytree(kagglehub.dataset_download('victornascimento/baccarat-dataset'), 'data/raw', dirs_exist_ok=True)"
```

The files end up in `data/raw/data/`. `kagglehub` also leaves a copy in `~/.cache/kagglehub`, which you can delete afterwards.

### 3. Prepare the data

```bash
python src/prepare_data.py
```

This works out each hand's winner from the card totals and writes `data/hands.csv` (about 30 seconds). The printed rates should be close to Banker 45.9%, Player 44.6%, Tie 9.5%.

### 4. Run the statistical tests

```bash
python src/stats_tests.py
```

### 5. Train the models

```bash
python src/train.py
```

This trains both models on 80% of the shoes, tests them on the other 20%, and saves the one with the better log loss to `models/model.joblib`. It takes several minutes and prints nothing while each model is training.

### 6. Start the website

Do steps 4–5 of the quick start. The website now uses your newly trained model.

## Using the website

1. Paper-bet 1 unit on the bet shown.
2. When the hand is over, tap who won. The shortcuts are **B**, **P** and **T**, and **U** undoes the last entry.
3. Tap **Dealer started a new shoe** when the dealer does.

The page compares the model's running total with always betting Banker. Track at least a few hundred hands before drawing any conclusion. Refreshing the page resets the totals.

## Project layout

```
data/raw/          dataset (git-ignored)
data/hands.csv     one row per hand: shoe_id, hand_no, outcome (git-ignored)
models/            saved model (model.joblib), used by the website
src/
  prepare_data.py  raw JSON -> hands.csv
  stats_tests.py   transition (chi-square) and runs tests
  features.py      features, shared by training and the website
  betting.py       payouts and expected-value bet choice
  train.py         train, evaluate against the baseline, save the model
app.py             Flask server
static/index.html  paper-trading page
```
