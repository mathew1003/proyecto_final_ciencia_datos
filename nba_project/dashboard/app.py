import streamlit as st, pandas as pd, json, joblib
from pathlib import Path
import plotly.express as px
ROOT=Path(__file__).resolve().parents[1]; CLEAN=ROOT/'data/processed/clean.csv'; MET=ROOT/'reports/metrics.json'; MODEL=ROOT/'models/best_model.joblib'
FEATURES_8=['Pos','Age','MP','FT%','3P%','2P%','FGA','3PA']
st.set_page_config(page_title='NBA Stats Dashboard', layout='wide'); st.title('NBA — Descriptivos y Modelo PTS')
def load_model():
    if MODEL.exists():
        try: return joblib.load(MODEL)
        except Exception as e: st.error(f'No pude cargar el modelo: {e}')
    else: st.info('No encuentro models/best_model.joblib. Ejecuta el entrenamiento primero.')
    return None
def ensure_features(df: pd.DataFrame):
    df=df.copy(); keep=[c for c in FEATURES_8 if c in df.columns]; missing=[c for c in FEATURES_8 if c not in df.columns]
    for c in keep:
        if c!='Pos': df[c]=pd.to_numeric(df[c], errors='coerce')
    df=df.reindex(columns=keep); return df, missing
if not CLEAN.exists():
    st.warning('No encuentro data/processed/clean.csv. Corre: python -m src.data_prep')
else:
    df=pd.read_csv(CLEAN)
    c1,c2,c3,c4=st.columns(4); c1.metric('Registros', len(df))
    if 'PTS' in df.columns: c2.metric('PTS medio', f"{df['PTS'].mean():.2f}")
    if 'FGA' in df.columns: c3.metric('FGA medio', f"{df['FGA'].mean():.2f}")
    if '3PA' in df.columns: c4.metric('3PA medio', f"{df['3PA'].mean():.2f}")
    with st.sidebar:
        season=st.selectbox('Season', sorted(df['Season'].dropna().unique()) if 'Season' in df.columns else [], index=None, placeholder='Todas')
        pos=st.multiselect('Pos', sorted(df['Pos'].dropna().unique()) if 'Pos' in df.columns else [])
        team=st.multiselect('Team', sorted(df['Tm'].dropna().unique()) if 'Tm' in df.columns else [])
        if 'Age' in df.columns: age=st.slider('Age', int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
        else: age=None
    flt=df.copy()
    if age and 'Age' in flt.columns: flt=flt[(flt['Age']>=age[0])&(flt['Age']<=age[1])]
    if season and 'Season' in flt.columns: flt=flt[flt['Season']==season]
    if pos and 'Pos' in flt.columns: flt=flt[flt['Pos'].isin(pos)]
    if team and 'Tm' in flt.columns: flt=flt[flt['Tm'].isin(team)]
    tab1,tab2,tab3,tab4=st.tabs(['Distribuciones','Relaciones','Modelo','Predicción CSV'])
    with tab1:
        for col in ['PTS','MP','FGA','3PA','FTA','FG%','3P%','2P%','FT%']:
            if col in flt.columns: st.plotly_chart(px.histogram(flt, x=col, nbins=30), use_container_width=True)
    with tab2:
        for x,y in [('MP','PTS'),('FGA','PTS'),('3PA','PTS'),('FTA','PTS'),('Age','PTS')]:
            if x in flt.columns and y in flt.columns: st.plotly_chart(px.scatter(flt, x=x, y=y, opacity=.6, trendline='ols'), use_container_width=True)
    with tab3:
        if MET.exists():
            try: met=json.loads(MET.read_text())['overall']; st.json(met)
            except Exception: st.info('No pude leer reports/metrics.json.')
            st.image(str(ROOT/'reports/figuras/y_vs_pred.png')); st.image(str(ROOT/'reports/figuras/residuals.png'))
            pred_path=ROOT/'data/final/predicciones_test.csv'
            if pred_path.exists(): st.download_button('Descargar predicciones (test)', pred_path.read_bytes(), 'predicciones_test.csv')
        else: st.info('No hay métricas aún. Ejecuta entrenamiento.')
        st.markdown('---'); st.subheader('Predicción manual (1 registro)'); model=load_model()
        if model:
            c1,c2,c3,c4=st.columns(4)
            pos_val=c1.selectbox('Pos', options=['PG','SG','SF','PF','C'], index=0)
            age_val=c2.number_input('Age', 15, 50, 25, 1)
            mp_val =c3.number_input('MP', 0.0, 48.0, 30.0, 0.5)
            ft_val =c4.number_input('FT%', 0.0, 1.0, 0.78, 0.01)
            c5,c6,c7,c8=st.columns(4)
            p3_val=c5.number_input('3P%', 0.0, 1.0, 0.36, 0.01)
            p2_val=c6.number_input('2P%', 0.0, 1.0, 0.50, 0.01)
            fga_val=c7.number_input('FGA', 0.0, 50.0, 15.0, 0.5)
            p3a_val=c8.number_input('3PA', 0.0, 30.0, 6.0, 0.5)
            if st.button('Predecir PTS (manual)'):
                row=pd.DataFrame([{'Pos':pos_val,'Age':age_val,'MP':mp_val,'FT%':ft_val,'3P%':p3_val,'2P%':p2_val,'FGA':fga_val,'3PA':p3a_val}])[FEATURES_8]
                try: yhat=float(model.predict(row)[0]); st.success(f'PTS predicho: **{yhat:.2f}**')
                except Exception as e: st.error(f'Error al predecir: {e}')
    with tab4:
        st.subheader('Sube un CSV para predecir PTS en lote')
        st.caption('El CSV debe incluir, al menos: ' + ', '.join(FEATURES_8))
        up=st.file_uploader('CSV (una fila por jugador-temporada)', type=['csv']); model=load_model()
        if up and model:
            import pandas as pd
            try: raw=pd.read_csv(up)
            except Exception as e: st.error(f'No pude leer el CSV: {e}'); st.stop()
            X, missing = ensure_features(raw)
            if missing: st.warning(f'Faltan columnas: {missing}. Solo se usarán las disponibles.')
            if len(X.columns)==0: st.error('No hay columnas requeridas. Revisa los nombres.')
            else:
                st.write('Vista previa de entrada:', X.head())
                try:
                    yhat=model.predict(X); out=raw.copy(); out['PTS_pred']=yhat
                    st.success(f'Predicciones generadas: {len(out)} filas'); st.dataframe(out.head(20))
                    st.download_button('Descargar CSV con PTS_pred', out.to_csv(index=False).encode('utf-8'), 'predicciones_upload.csv')
                except Exception as e: st.error(f'No pude predecir con el CSV subido: {e}')
