from pathlib import Path
import pandas as pd, numpy as np
from .utils import ensure_dir, era_from_season
from .viz import hist, bar, boxplot, heatmap_corr, scatter
def perfil_dataset(df:pd.DataFrame):
    info={'filas':len(df),'columnas':len(df.columns),'tipos':df.dtypes.astype(str).to_dict(),'nulos':df.isna().sum().to_dict()}
    n=len(df); info['pct_nulos']={k:(v/n)*100 if n>0 else 0 for k,v in info['nulos'].items()}
    return pd.DataFrame({'tipo':info['tipos'],'nulos':info['nulos'],'%nulos':info['pct_nulos']})
def estadisticos_basicos(df:pd.DataFrame, cols):
    use=[c for c in cols if c in df.columns]; d=df[use].describe(percentiles=[.25,.5,.75]).T
    d=d.rename(columns={'25%':'Q1','50%':'Mediana','75%':'Q3'}); return d[['mean','Mediana','std','min','Q1','Q3','max']]
def graficos_descriptivos(df:pd.DataFrame, out_dir:Path):
    out_dir=Path(out_dir); ensure_dir(out_dir)
    cols=[c for c in ['PTS','MP','FGA','3PA','FTA','FG%','3P%','2P%','FT%'] if c in df.columns]
    for c in cols: hist(df[c], f'Distribución {c}', out_dir/f'hist_{c}.png')
    d2=df.copy()
    if 'Season' in d2.columns: d2['era']=d2['Season'].map(era_from_season)
    if 'Pos' in d2.columns and 'PTS' in d2.columns:
        agg=d2.groupby('Pos',dropna=False)['PTS'].mean().reset_index(); bar(agg['Pos'], agg['PTS'], 'PTS medio por Pos', out_dir/'bar_pts_por_pos.png')
        boxplot(d2.dropna(subset=['Pos']), 'PTS', 'Pos', 'PTS por Pos', out_dir/'box_pts_por_pos.png')
    if 'era' in d2.columns and 'PTS' in d2.columns:
        agg2=d2.groupby('era',dropna=False)['PTS'].mean().reset_index(); bar(agg2['era'], agg2['PTS'], 'PTS medio por era', out_dir/'bar_pts_por_era.png')
        boxplot(d2.dropna(subset=['era']), 'PTS', 'era', 'PTS por era', out_dir/'box_pts_por_era.png')
    heatmap_corr(d2.select_dtypes(include=[np.number]), out_dir/'corr_matrix.png')
    for c in ['MP','FGA','3PA','FTA','Age']:
        if c in d2.columns and 'PTS' in d2.columns: scatter(d2[c], d2['PTS'], f'PTS vs {c}', out_dir/f'scatter_pts_vs_{c}.png')
    return {'figdir': str(out_dir)}
