import json
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
SPLITS_DIR = BASE_DIR / "splits"
METRICS_PATH = BASE_DIR / "metrics" / "metrics.json"
PRED_PATH = SPLITS_DIR / "predictions_test.csv"

st.set_page_config(page_title="NBA PTS — Dashboard", layout="wide")
st.title("Dashboard — Predicción de PTS (NBA Player Stats 2)")

if METRICS_PATH.exists():
    with METRICS_PATH.open(encoding="utf-8") as f:
        metrics = json.load(f)
    st.subheader("Métricas del modelo (test)")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['mae']:.2f} pts")
    c2.metric("RMSE", f"{metrics['rmse']:.2f} pts")
    c3.metric("R²", f"{metrics['r2']:.3f}")
else:
    st.warning("No se encontró metrics.json")

if PRED_PATH.exists():
    df_pred = pd.read_csv(PRED_PATH)
    st.subheader("Muestra de predicciones")
    st.dataframe(df_pred.head())

    if "Pos" in df_pred.columns:
        st.subheader("PTS real promedio por posición")
        st.bar_chart(df_pred.groupby("Pos")["PTS_real"].mean())

    if "MP" in df_pred.columns:
        st.subheader("Relación MP vs PTS real")
        st.scatter_chart(df_pred, x="MP", y="PTS_real")
else:
    st.warning("No se encontró predictions_test.csv")
