from pathlib import Path
import joblib, pandas as pd
FEATURES_8=['Pos','Age','MP','FT%','3P%','2P%','FGA','3PA']
def main():
    root=Path(__file__).resolve().parents[1]; model=joblib.load(root/'models/best_model.joblib')
    vals={}; print('Características:', FEATURES_8)
    for f in FEATURES_8: vals[f]=float(input(f'{f}: ') or 0)
    y=float(model.predict(pd.DataFrame([vals]))[0]); print(f'PTS predicho: {y:.3f}')
if __name__=='__main__': main()
