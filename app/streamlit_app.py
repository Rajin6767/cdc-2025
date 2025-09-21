import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Resilience Explorer", layout="wide")
st.title("🛰️ Industry Resilience Explorer")

p = Path("data/processed/resilience_metrics.csv")
if p.exists():
    st.subheader("Resilience Metrics (preview)")
    st.dataframe(pd.read_csv(p))
else:
    st.warning("No metrics yet — we’ll generate them next.")
