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
