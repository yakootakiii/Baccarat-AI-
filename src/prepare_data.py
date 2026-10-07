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
