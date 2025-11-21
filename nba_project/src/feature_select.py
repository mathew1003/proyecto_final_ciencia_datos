import pandas as pd
def ranking_abs_corr(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    vals={}
    y=pd.to_numeric(y_train, errors='coerce')
    for c in X_train.columns:
        x=pd.to_numeric(X_train[c], errors='coerce')
        if x.notna().sum()>3:
            corr=x.corr(y)
            if pd.notna(corr): vals[c]=abs(float(corr))
    return pd.DataFrame([{'feature':k,'abs_corr':v} for k,v in vals.items()]).sort_values('abs_corr', ascending=False).reset_index(drop=True)
