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
    features from the hands BEFORE it in the same shoe.

    Vectorised equivalent of calling features_from_history(outcomes[:i])
    for every hand (millions of rows); check_build_dataset() verifies
    the two agree."""
    import pandas as pd
    df = df.sort_values(["shoe_id", "hand_no"]).reset_index(drop=True)
    shoe, outcome = df["shoe_id"], df["outcome"]
    g = outcome.groupby(shoe)
    f = {}

    for i in range(1, N_LAGS + 1):
        lag = g.shift(i)
        for k in "BPT":
            f[f"lag{i}_{k}"] = (lag == k).astype("int8")

    f["hand_index"] = g.cumcount().astype("int16")
    for k in "BPT":
        is_k = (outcome == k).astype("int16")
        f[f"count_{k}"] = (is_k.groupby(shoe).cumsum() - is_k).astype("int16")

    # Streak (ties ignored) ending at each non-tie hand, inclusive...
    nt = df[outcome != "T"]
    new_run = nt["outcome"] != nt.groupby("shoe_id")["outcome"].shift(1)
    run_id = new_run.astype(int).groupby(nt["shoe_id"]).cumsum()
    streak = pd.Series(float("nan"), index=df.index)
    side = pd.Series(float("nan"), index=df.index)
    streak[nt.index] = nt.groupby([nt["shoe_id"], run_id]).cumcount() + 1
    side[nt.index] = (nt["outcome"] == "B").map({True: 1, False: -1})
    # ...then the state BEFORE each hand = latest non-tie strictly earlier.
    for name, s in (("streak_len", streak), ("streak_side", side)):
        f[name] = (s.groupby(shoe).shift(1).groupby(shoe).ffill()
                   .fillna(0).astype("int8"))

    return pd.DataFrame(f), outcome.rename(None), shoe.rename(None)


def check_build_dataset(df, n_shoes=500):
    """Asserts build_dataset matches features_from_history on a sample."""
    import pandas as pd
    sample = df[df["shoe_id"].isin(df["shoe_id"].drop_duplicates()[:n_shoes])]
    X, _, _ = build_dataset(sample)
    rows = []
    for _, shoe in sample.sort_values(["shoe_id", "hand_no"]).groupby("shoe_id"):
        outcomes = shoe["outcome"].tolist()
        rows.extend(features_from_history(outcomes[:i]) for i in range(len(outcomes)))
    expected = pd.DataFrame(rows)
    assert list(X.columns) == list(expected.columns)
    assert (X.to_numpy() == expected.to_numpy()).all(), "build_dataset mismatch"
