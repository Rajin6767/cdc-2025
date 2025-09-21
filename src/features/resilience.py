import pandas as pd
from pathlib import Path

CSV = Path("data/processed/clean_va_price.csv")
assert CSV.exists(), f"Missing {CSV}. Pull latest and make sure the file is there."

df = pd.read_csv(CSV)

print("✅ loaded:", CSV)
print("rows:", len(df))
print("columns:", list(df.columns))

# quick peek
print("\nfirst 8 rows:")
print(df.head(8).to_string(index=False))

# basic checks (these help us know what we can compute next)
if "Year" in df.columns:
    try:
        years = pd.to_numeric(df["Year"], errors="coerce")
        print("\nYear range:", int(years.min()), "→", int(years.max()))
    except Exception:
        pass


# -----------------------------
# Drawdown 2020 calculation
# -----------------------------
drawdown_metrics = []
for industry, g in df.groupby("Industry"):
    g = g.sort_values("Year")

    try:
        val2019 = g.loc[g["Year"] == 2019, "Real_Value"].values[0]
        val2020 = g.loc[g["Year"] == 2020, "Real_Value"].values[0]
    except IndexError:
        continue

    drawdown = (val2020 - val2019) / val2019
    drawdown_metrics.append({"Industry": industry, "Drawdown_2020": drawdown})

drawdown_df = pd.DataFrame(drawdown_metrics)
print("\n--- Drawdown 2020 (top 10 worst hit) ---")
print(drawdown_df.sort_values("Drawdown_2020").head(10))


# -----------------------------
# Recovery Years calculation
# -----------------------------
recovery_metrics = []
for industry, g in df.groupby("Industry"):
    g = g.sort_values("Year")

    try:
        val2019 = g.loc[g["Year"] == 2019, "Real_Value"].values[0]
    except IndexError:
        continue

    recovery = None
    for y in [2020, 2021, 2022, 2023]:
        vals = g.loc[g["Year"] == y, "Real_Value"].values
        if len(vals) > 0 and vals[0] >= val2019:
            recovery = y - 2019
            break

    recovery_metrics.append({"Industry": industry, "Recovered_Years": recovery})

recovery_df = pd.DataFrame(recovery_metrics)
print("\n--- Recovery Years (sample) ---")
print(recovery_df.head(15))


# -----------------------------
# Merge both metrics
# -----------------------------
merged = pd.merge(drawdown_df, recovery_df, on="Industry", how="outer")
print("\n--- Combined metrics (first 15 industries) ---")
print(merged.head(15))

# save to processed folder
out_path = Path("data/processed/resilience_metrics.csv")
merged.to_csv(out_path, index=False)
print(f"\n✅ Saved resilience metrics → {out_path}")
