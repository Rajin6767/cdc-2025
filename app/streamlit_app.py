import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Resilience Explorer", layout="wide")
st.title("🛰️ Industry Resilience Explorer")

metrics_path = Path("data/processed/resilience_metrics.csv")

if metrics_path.exists():
    df = pd.read_csv(metrics_path)

    st.subheader("Resilience Metrics (full dataset)")
    st.dataframe(df)

    # Top 10 worst hit industries
    st.subheader("📉 Top 10 Worst Hit in 2020")
    st.dataframe(df.sort_values("Drawdown_2020").head(10))

    # Fastest to recover
    st.subheader("⚡ Fastest to Recover")
    st.dataframe(
        df.dropna(subset=["Recovered_Years"])
        .sort_values("Recovered_Years")
        .head(10)
    )

else:
    st.warning("⚠️ No metrics yet. Run resilience.py first to generate them.")
