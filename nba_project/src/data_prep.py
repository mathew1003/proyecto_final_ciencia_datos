from pathlib import Path
import pandas as pd, numpy as np
from .utils import project_root_from, localizar_dataset, ensure_dir, pick_column, era_from_season
REQUIRED_8=['Pos','Age','MP','FT%','3P%','2P%','FGA','3PA']
def _to_numeric(df, cols):
    for c in cols:
        if c in df.columns: df[c]=pd.to_numeric(df[c], errors='coerce')
    return df
def main():
    root=project_root_from(__file__); data_path=localizar_dataset(root); df=pd.read_csv(data_path)
    PTS=pick_column(['PTS'], df.columns); PLAYER=pick_column(['Player','Player Name','Name'], df.columns); SEASON=pick_column(['Season','Year'], df.columns)
    if PTS is None: raise ValueError('No se encontró PTS')
    num_like=['Age','MP','FT%','3P%','2P%','FGA','3PA','PTS','FTA','FG%']; df=_to_numeric(df, [c for c in num_like if c in df.columns])
    if SEASON: df['era']=df[SEASON].map(era_from_season)
    if PLAYER and SEASON and PLAYER in df.columns and SEASON in df.columns: df=df.drop_duplicates(subset=[PLAYER,SEASON])
    needed=[c for c in REQUIRED_8 if c in df.columns]+[PTS]; df=df.dropna(subset=needed)
    out=root/'data/processed/clean.csv'; ensure_dir(out.parent); df.to_csv(out, index=False); print('Limpieza completa ->', out); print('Filas:', len(df)); return df
if __name__=='__main__': main()
