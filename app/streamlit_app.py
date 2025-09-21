import streamlit as st
import pandas as pd
import altair as alt
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="Resilience Explorer", layout="wide")
st.title("🛰️ Industry Resilience Explorer")

METRICS_CSV = Path("data/processed/resilience_metrics.csv")
FEATURES_CSV = Path("data/processed/clean_va_price.csv")
MODEL_PKL   = Path("models/ridge_model.pkl")
SCALER_PKL  = Path("models/scaler.pkl")

def build_ml_frame() -> pd.DataFrame:
    if not (FEATURES_CSV.exists() and METRICS_CSV.exists()):
        return pd.DataFrame()
    feats = pd.read_csv(FEATURES_CSV)
    labs  = pd.read_csv(METRICS_CSV)
    df = pd.merge(feats, labs[["Industry", "Recovered_Years"]].drop_duplicates(),
                  on="Industry", how="inner")
    value_col = None
    for cand in ["Real_Value", "Nominal_Value"]:
        if cand in df.columns:
            value_col = cand
            break
    if value_col is None or "Year" not in df.columns:
        return pd.DataFrame()
    df = df.sort_values(["Industry", "Year"])
    df["GrowthRate"] = df.groupby("Industry")[value_col].pct_change()
    vol_roll = df.groupby("Industry")[value_col].apply(
        lambda s: s.pct_change().rolling(window=3, min_periods=2).std()
    ).reset_index(level=0, drop=True)
    df["Volatility"] = vol_roll
    grp_std = df.groupby("Industry")["GrowthRate"].transform("std")
    df["Volatility"] = df["Volatility"].fillna(grp_std)
    df["Volatility"] = df["Volatility"].fillna(0.0)
    pre = df[df["Year"].between(2015, 2019)]
    base_map = pre.groupby("Industry")[value_col].mean()
    df["Baseline"] = df["Industry"].map(base_map)
    val_2019 = df[df["Year"] == 2019].groupby("Industry")[value_col].mean()
    df["Baseline"] = df["Baseline"].fillna(df["Industry"].map(val_2019))
    ind_mean = df.groupby("Industry")[value_col].transform("mean")
    df["Baseline"] = df["Baseline"].fillna(ind_mean)
    df["Baseline"] = df["Baseline"].fillna(df[value_col].mean())
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["GrowthRate", "Volatility", "Baseline", "Recovered_Years"])
    return df

def safe_predict_matrix(ridge, scaler, X: pd.DataFrame) -> np.ndarray:
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    if X.empty:
        return np.array([])
    X_scaled = scaler.transform(X.values)
    return ridge.predict(X_scaled)

if METRICS_CSV.exists():
    df_metrics = pd.read_csv(METRICS_CSV)
    if "Resilience_Score" not in df_metrics.columns:
        df_metrics["Resilience_Score"] = df_metrics.apply(
            lambda row: (-row["Drawdown_2020"]) / row["Recovered_Years"]
            if pd.notna(row.get("Recovered_Years")) and row.get("Recovered_Years", 0) > 0
            else None,
            axis=1,
        )
    tab1, tab2, tab3 = st.tabs([
        "📊 Resilience Analysis",
        "🤖 Model Predictions",
        "🔮 Shock Simulator"
    ])

    with tab1:
        st.subheader("📊 Resilience Metrics (Full Dataset)")
        st.dataframe(df_metrics)
        st.subheader("📉 Worst-Hit Industries (2020)")
        worst = df_metrics.sort_values("Drawdown_2020").head(15)
        bar_chart = alt.Chart(worst).mark_bar().encode(
            x=alt.X("Drawdown_2020:Q", title="Drawdown Severity"),
            y=alt.Y("Industry:N", sort='-x'),
            color=alt.condition(alt.datum.Drawdown_2020 < 0, alt.value("red"), alt.value("green")),
            tooltip=["Industry", "Drawdown_2020", "Recovered_Years"]
        )
        st.altair_chart(bar_chart, use_container_width=True)
        st.subheader("⏳ Recovery Speed by Industry")
        heatmap = alt.Chart(df_metrics.dropna(subset=["Recovered_Years"])).mark_rect().encode(
            x=alt.X("Recovered_Years:O", title="Years to Recover"),
            y=alt.Y("Industry:N", sort="-x"),
            color="Recovered_Years:Q",
            tooltip=["Industry", "Drawdown_2020", "Recovered_Years"]
        )
        st.altair_chart(heatmap, use_container_width=True)
        st.subheader("🏆 Top 10 Resilient Industries")
        top_resilient = (
            df_metrics.dropna(subset=["Resilience_Score"])
            .sort_values("Resilience_Score", ascending=False)
            .head(10)
        )
        st.table(top_resilient[["Industry", "Drawdown_2020", "Recovered_Years", "Resilience_Score"]])
        if df_metrics["Resilience_Score"].notna().any():
            df_metrics["Resilience_Score"] = df_metrics["Resilience_Score"].replace([float("inf"), float("-inf")], np.nan)
            min_val = float(df_metrics["Resilience_Score"].min())
            max_val = float(df_metrics["Resilience_Score"].max())
            score_threshold = st.slider(
                "🎯 Minimum Resilience Score",
                min_value=min_val,
                max_value=max_val,
                value=min_val,
                step=0.05,
            )
            filtered_df = df_metrics[df_metrics["Resilience_Score"] >= score_threshold]
            st.subheader(f"📌 Industries with Resilience Score ≥ {score_threshold}")
            st.dataframe(filtered_df[["Industry", "Drawdown_2020", "Recovered_Years", "Resilience_Score"]])

    with tab2:
        st.subheader("🤖 Ridge Regression Model Predictions")
        if MODEL_PKL.exists() and SCALER_PKL.exists():
            ridge  = joblib.load(MODEL_PKL)
            scaler = joblib.load(SCALER_PKL)
            st.write("🎛️ Test a new scenario:")
            g = st.slider("Growth Rate", -0.5, 0.5, 0.0, step=0.01)
            v = st.slider("Volatility", 0.0, 1.0, 0.1, step=0.01)
            b = st.slider("Baseline (normalized/level)", 0.0, 1.0, 0.5, step=0.01)
            X_input = np.array([[g, v, b, g * v]], dtype=np.float64)
            X_input = np.nan_to_num(X_input, nan=0.0, posinf=0.0, neginf=0.0)
            X_input = scaler.transform(X_input)
            pred = ridge.predict(X_input)[0]
            st.success(f"📌 Predicted recovery years: {pred:.2f}")
            ml_df = build_ml_frame()
            if not ml_df.empty:
                X_all = ml_df[["GrowthRate", "Volatility", "Baseline"]].copy()
                X_all["Growth_x_Volatility"] = X_all["GrowthRate"] * X_all["Volatility"]
                y_all = ml_df["Recovered_Years"]
                mask = np.isfinite(X_all).all(axis=1) & np.isfinite(y_all.values)
                X_all = X_all[mask]
                y_all = y_all[mask]
                X_scaled = scaler.transform(X_all.values)
                y_pred_all = ridge.predict(X_scaled)
                mse = mean_squared_error(y_all, y_pred_all)
                r2  = r2_score(y_all, y_pred_all)
                st.subheader("📊 Model Evaluation")
                st.write(f"**MSE:** {mse:.4f}")
                st.write(f"**R² Score:** {r2:.4f}")
                chart_data = pd.DataFrame({"Actual": y_all, "Predicted": y_pred_all})
                scatter = alt.Chart(chart_data).mark_circle(size=80).encode(
                    x="Actual:Q", y="Predicted:Q",
                    tooltip=["Actual", "Predicted"]
                ).properties(width=600, height=400)
                line = alt.Chart(chart_data).mark_line(color="red").encode(
                    x="Actual:Q", y="Actual:Q"
                )
                st.altair_chart(scatter + line, use_container_width=True)
            else:
                st.warning("Could not build ML frame (missing Year/values).")
        else:
            st.warning("⚠️ No trained model yet. Run notebooks/model_dev.ipynb and save ridge_model.pkl + scaler.pkl.")

    with tab3:
        st.subheader("🔮 Shock Simulator (table view)")
        if MODEL_PKL.exists() and SCALER_PKL.exists():
            ridge  = joblib.load(MODEL_PKL)
            scaler = joblib.load(SCALER_PKL)
            ml_df  = build_ml_frame()
            if ml_df.empty:
                st.warning("Could not build ML frame (ensure data/processed/*.csv has Industry + Year + values).")
            else:
                years_full = list(range(2012, 2041))
                sel_year = st.selectbox("Select Year (2012–2040)", years_full, index=years_full.index(2040))
                shock = st.slider("Apply shock to GrowthRate (e.g., -0.30 = -30%)", -1.0, 1.0, -0.20, 0.01)

                rows = []
                for _, g in ml_df.groupby("Industry"):
                    g = g.sort_values("Year")
                    if sel_year in g["Year"].values:
                        row = g[g["Year"] == sel_year].iloc[-1]
                    else:
                        g2 = g[g["Year"] < sel_year]
                        if not g2.empty:
                            row = g2.iloc[-1]
                        else:
                            row = g.iloc[0]
                    rows.append(row)
                picked = pd.DataFrame(rows).reset_index(drop=True)

                picked["GrowthRate_sim"] = picked["GrowthRate"] + shock
                picked["Volatility_sim"] = picked["Volatility"]
                picked["Baseline_sim"]   = picked["Baseline"]
                picked["GXV_sim"]        = picked["GrowthRate_sim"] * picked["Volatility_sim"]

                X_sim = picked[["GrowthRate_sim","Volatility_sim","Baseline_sim","GXV_sim"]].replace([np.inf,-np.inf], np.nan).dropna()
                if X_sim.empty:
                    st.warning("No valid rows after cleaning.")
                else:
                    valid_idx = X_sim.index
                    X_scaled = scaler.transform(X_sim.values)
                    y_pred   = ridge.predict(X_scaled)
                    out = picked.loc[valid_idx, ["Industry", "Year"]].copy()
                    out.rename(columns={"Year": "Base_Year"}, inplace=True)
                    out["Scenario_Year"]       = sel_year
                    out["Shock_on_Growth"]     = shock
                    out["Pred_Recovery_Years"] = y_pred
                    st.subheader(f"📌 Predicted recovery years under shock {shock:+.2f} in {sel_year}")
                    st.dataframe(out.sort_values("Pred_Recovery_Years"))
        else:
            st.warning("⚠️ No trained model yet. Run notebooks/model_dev.ipynb and save ridge_model.pkl + scaler.pkl.")
else:
    st.warning("⚠️ No metrics yet. Run resilience.py first to generate them.")
