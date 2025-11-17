from pathlib import Path
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "NBA_Player_Stats_2.csv"
METRICS_DIR = BASE_DIR / "metrics"
MODEL_DIR = BASE_DIR / "model"
SPLITS_DIR = BASE_DIR / "splits"

METRICS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
SPLITS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df[df["PTS"].notna()].copy()

if "Season" in df.columns:
    def era_from_year(y):
        try:
            y = int(y)
        except Exception:
            return "Unknown"
        if y <= 1980:
            return "Old-school (<=1980)"
        if y <= 1995:
            return "Pre-pace&space (1981-1995)"
        if y <= 2010:
            return "Hand-check era (1996-2010)"
        return "Modern era (2011+)"
    df["Era"] = df["Season"].apply(era_from_year)
else:
    df["Era"] = "Unknown"

for col in ["FGA", "3PA", "FTA"]:
    if col not in df.columns:
        df[col] = 0.0

df["ShotVolume"] = df["FGA"] + df["3PA"] + 0.44 * df["FTA"]
den_ts = 2 * (df["FGA"] + 0.44 * df["FTA"])
df["TS%"] = np.where(den_ts > 0, df["PTS"] / den_ts, np.nan)

candidate_num = ['Age', 'G', 'GS', 'MP', 'FGA', '3PA', 'FTA', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'FG%', '3P%', 'FT%', 'ShotVolume', 'TS%']
candidate_cat = ['Pos', 'Tm', 'Era']

num_features = [c for c in candidate_num if c in df.columns]
cat_features = [c for c in candidate_cat if c in df.columns]

X = df[num_features + cat_features].copy()
y = df["PTS"].copy()

mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask]
y = y.loc[mask]
df_model = df.loc[mask].copy()

X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df_model, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

model = RandomForestRegressor(
    n_estimators=80,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
)

pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

mae = float(mean_absolute_error(y_test, y_pred))
rmse = float(mean_squared_error(y_test, y_pred, squared=False))
r2 = float(r2_score(y_test, y_pred))

metrics = {
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "num_features": num_features,
    "cat_features": cat_features,
}
with (METRICS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

joblib.dump(pipe, MODEL_DIR / "nba_pts_pipeline.joblib")

df_train.to_csv(SPLITS_DIR / "train.csv", index=False)
df_test.to_csv(SPLITS_DIR / "test.csv", index=False)

pred_df = df_test.copy()
pred_df["PTS_real"] = y_test.values
pred_df["PTS_pred"] = y_pred
pred_df["Error"] = pred_df["PTS_pred"] - pred_df["PTS_real"]
pred_df["AbsError"] = pred_df["Error"].abs()
bins = [-0.01, 2, 5, 8, float("inf")]
labels = ["<=2 pts", "2-5 pts", "5-8 pts", ">8 pts"]
pred_df["ErrorBin"] = pd.cut(pred_df["AbsError"], bins=bins, labels=labels)

cols = []
for c in ["Player", "Season", "Pos", "Age", "Tm", "G", "MP", "FGA", "3PA", "FTA",
          "TRB", "AST", "PTS_real", "PTS_pred", "Error", "AbsError", "ErrorBin"]:
    if c in pred_df.columns and c not in cols:
        cols.append(c)

pred_df[cols].to_csv(SPLITS_DIR / "predictions_test.csv", index=False)

print("Entrenamiento completo:")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)
