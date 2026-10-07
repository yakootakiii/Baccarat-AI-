import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "data"   # one JSON file per shoe
OUT_FILE = ROOT / "data" / "hands.csv"

# Each file: {"burn": [card], "hands": [[player_cards, banker_cards], ...]}
# Cards are rank + suit, e.g. "8S", "AC", "0D" ("0" = ten).
CARD_VALUES = {"A": 1, "0": 0, "J": 0, "Q": 0, "K": 0,
               **{str(n): n for n in range(2, 10)}}


def hand_total(cards):
    return sum(CARD_VALUES[c[0]] for c in cards) % 10


def outcome(player_cards, banker_cards):
    p, b = hand_total(player_cards), hand_total(banker_cards)
    return "P" if p > b else "B" if b > p else "T"


rows = []
files = sorted(RAW_DIR.glob("*.json"))
assert files, f"no JSON files in {RAW_DIR}"
for path in files:
    shoe = json.loads(path.read_text())
    for hand_no, (player, banker) in enumerate(shoe["hands"]):
        rows.append((path.stem, hand_no, outcome(player, banker)))

out = pd.DataFrame(rows, columns=["shoe_id", "hand_no", "outcome"])
out.to_csv(OUT_FILE, index=False)
print(out["outcome"].value_counts(normalize=True))
print("shoes:", out["shoe_id"].nunique(), "hands:", len(out))
