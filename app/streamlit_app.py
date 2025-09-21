import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Resilience Explorer", layout="wide")
st.title("🛰️ Industry Resilience Explorer")

metrics_path = Path("data/processed/resilience_metrics.csv")

if metrics_path.exists():
    df = pd.read_csv(metrics_path)

    if "Resilience_Score" not in df.columns:
        df["Resilience_Score"] = df.apply(
            lambda row: (-row["Drawdown_2020"]) / row["Recovered_Years"]
            if pd.notna(row["Recovered_Years"]) and row["Recovered_Years"] > 0
            else None,
            axis=1,
        )

    st.subheader("📊 Resilience Metrics (Full Dataset)")
    st.dataframe(df)

    st.subheader("📉 Worst-Hit Industries (2020)")
    worst = df.sort_values("Drawdown_2020").head(15)

    bar_chart = alt.Chart(worst).mark_bar().encode(
        x=alt.X("Drawdown_2020:Q", title="Drawdown Severity"),
        y=alt.Y("Industry:N", sort='-x'),
        color=alt.condition(
            alt.datum.Drawdown_2020 < 0,
            alt.value("red"),
            alt.value("green")
        ),
        tooltip=["Industry", "Drawdown_2020", "Recovered_Years"]
    )
    st.altair_chart(bar_chart, use_container_width=True)

    st.subheader("⏳ Recovery Speed by Industry")
    heatmap = alt.Chart(df.dropna(subset=["Recovered_Years"])).mark_rect().encode(
        x=alt.X("Recovered_Years:O", title="Years to Recover"),
        y=alt.Y("Industry:N", sort="-x"),
        color="Recovered_Years:Q",
        tooltip=["Industry", "Drawdown_2020", "Recovered_Years"]
    )
    st.altair_chart(heatmap, use_container_width=True)

    st.subheader("⚖️ Shock vs Recovery Trade-off")
    bubble = alt.Chart(df.dropna(subset=["Recovered_Years", "Resilience_Score"])).mark_circle(size=200).encode(
        x=alt.X("Drawdown_2020:Q", title="Shock (Drawdown 2020)"),
        y=alt.Y("Recovered_Years:Q", title="Years to Recover"),
        size=alt.Size("Resilience_Score:Q", scale=alt.Scale(range=[50, 1000])),
        color="Resilience_Score:Q",
        tooltip=["Industry", "Drawdown_2020", "Recovered_Years", "Resilience_Score"]
    )
    st.altair_chart(bubble, use_container_width=True)

    st.subheader("🏆 Top 10 Resilient Industries")
    top_resilient = (
        df.dropna(subset=["Resilience_Score"])
        .sort_values("Resilience_Score", ascending=False)
        .head(10)
    )
    st.table(top_resilient[["Industry", "Drawdown_2020", "Recovered_Years", "Resilience_Score"]])

else:
    st.warning("⚠️ No metrics yet. Run resilience.py first to generate them.")

if "Resilience_Score" not in df.columns:
    df["Resilience_Score"] = df.apply(
        lambda row: (-row["Drawdown_2020"]) / row["Recovered_Years"]
        if pd.notna(row["Recovered_Years"]) and row["Recovered_Years"] > 0
        else None,
        axis=1,
    )

df["Resilience_Score"].replace([float("inf"), float("-inf")], None, inplace=True)

if df["Resilience_Score"].notna().any():
    df["Resilience_Score"].replace([float("inf"), float("-inf")], None, inplace=True)

    min_val = float(df["Resilience_Score"].min())
    max_val = float(df["Resilience_Score"].max())

    st.subheader("🎯 Resilience Threshold")
    score_threshold = st.slider(
        "Select minimum resilience score",
        min_value=min_val,
        max_value=max_val,
        value=min_val,
        step=0.01,
    )

    filtered_df = df[df["Resilience_Score"] >= score_threshold]

    st.subheader(f"📌 Industries with Resilience Score ≥ {score_threshold}")
    st.dataframe(filtered_df[["Industry", "Drawdown_2020", "Recovered_Years", "Resilience_Score"]])

