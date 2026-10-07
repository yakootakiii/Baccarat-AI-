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
