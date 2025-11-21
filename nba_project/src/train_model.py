from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .utils import project_root_from, localizar_dataset, pick_column, ensure_dir
from .feature_select import ranking_abs_corr
FEATURES_8=['Pos','Age','MP','FT%','3P%','2P%','FGA','3PA']
def _interactive_predict(model, feats, df, PLAYER, SEASON):
    print('\n--- Predicción interactiva ---')
    try: mode=input('¿Usar registro del dataset? (S/N) [S]: ').strip().lower()
    except EOFError: mode='s'
    if mode=='' or mode.startswith('s'):
        sdf=df.copy()
        if PLAYER and PLAYER in df.columns:
            name=input('Nombre jugador (parte): ').strip()
            if name: sdf=sdf[sdf[PLAYER].astype(str).str.contains(name, case=False, na=False)]
        if SEASON and SEASON in df.columns:
            season=input('Season (opcional): ').strip()
            if season: sdf=sdf[sdf[SEASON].astype(str)==season]
        if not sdf.empty:
            print(sdf[[c for c in [PLAYER,SEASON] if c in sdf.columns]].head(10).reset_index(drop=True))
            idx=int(input('Índice [0]: ') or 0); row=sdf.head(10).iloc[idx]; x={f:row.get(f,0) for f in feats}
            import pandas as pd; yhat=float(model.predict(pd.DataFrame([x]))[0]); print(f'Predicción PTS: {yhat:.3f}'); return yhat
    vals={}; 
    for f in feats: vals[f]=float(input(f'Ingrese {f}: ') or 0)
    import pandas as pd; yhat=float(model.predict(pd.DataFrame([vals]))[0]); print(f'Predicción PTS: {yhat:.3f}'); return yhat
def main(k_top=8, ask_input=False, random_state=42):
    root=project_root_from(__file__); data_path=(root/'data/processed/clean.csv') if (root/'data/processed/clean.csv').exists() else localizar_dataset(root); df=pd.read_csv(data_path)
    PTS=pick_column(['PTS'], df.columns); PLAYER=pick_column(['Player','Player Name','Name'], df.columns); SEASON=pick_column(['Season','Year'], df.columns)
    if PLAYER and PLAYER in df.columns:
        tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=random_state).split(df, groups=df[PLAYER]))
    else:
        n=len(df); idx=np.arange(n); rs=np.random.RandomState(random_state); rs.shuffle(idx); cut=int(n*0.8); tr,te=idx[:cut],idx[cut:]
    train_df, test_df=df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    feats=[f for f in FEATURES_8 if f in df.columns]; missing=[f for f in FEATURES_8 if f not in feats]
    if missing: (root/'reports'/'notes.txt').write_text('Faltan columnas: '+str(missing), encoding='utf-8')
    Xtr, Xte = train_df[feats].copy(), test_df[feats].copy(); ytr, yte = train_df[PTS].values, test_df[PTS].values
    num_cols=[c for c in feats if c!='Pos' and pd.api.types.is_numeric_dtype(df[c])]; cat_cols=[c for c in feats if c=='Pos' and c in df.columns]
    pre=ColumnTransformer([('num', Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]), num_cols),
                           ('cat', Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]), cat_cols)], remainder='drop')
    model=Pipeline([('pre',pre),('lin',LinearRegression())]); model.fit(Xtr, ytr)
    rank=ranking_abs_corr(Xtr.select_dtypes(include=[np.number]), pd.Series(ytr)); ensure_dir(root/'reports'); (root/'reports'/'feature_ranking.csv').write_text(rank.to_csv(index=False), encoding='utf-8')
    ypred=model.predict(Xte); overall={'RMSE':float(np.sqrt(mean_squared_error(yte, ypred))), 'MAE':float(mean_absolute_error(yte, ypred)), 'R2':float(r2_score(yte, ypred))}
    (root/'reports'/'metrics.json').write_text(json.dumps({'overall':overall}, indent=2), encoding='utf-8')
    ensure_dir(root/'reports/figuras')
    import matplotlib.pyplot as plt
    plt.figure(); plt.scatter(yte, ypred, alpha=0.6); plt.xlabel('PTS real'); plt.ylabel('PTS predicho'); plt.title('Real vs Predicho (PTS)'); plt.tight_layout(); plt.savefig(root/'reports/figuras/y_vs_pred.png'); plt.close()
    res=yte-ypred; plt.figure(); plt.scatter(ypred, res, alpha=0.6); plt.axhline(0,color='gray'); plt.xlabel('Predicho'); plt.ylabel('Residuo'); plt.title('Residuales'); plt.tight_layout(); plt.savefig(root/'reports/figuras/residuals.png'); plt.close()
    ensure_dir(root/'models'); import joblib as _j; _j.dump(model, root/'models/best_model.joblib')
    cols=[]; 
    if PLAYER and PLAYER in test_df.columns: cols.append(PLAYER)
    if SEASON and SEASON in test_df.columns: cols.append(SEASON)
    pred_df = test_df[cols].copy() if cols else pd.DataFrame(index=test_df.index)
    pred_df['PTS_real']=test_df[PTS].values; pred_df['PTS_pred']=ypred; ensure_dir(root/'data/final'); pred_df.to_csv(root/'data/final/predicciones_test.csv', index=False)
    print('Métricas:', overall)
    if ask_input: _interactive_predict(model, feats, df, PLAYER, SEASON)
if __name__=='__main__': main()
