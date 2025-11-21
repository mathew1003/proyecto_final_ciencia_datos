(proyecto realizado Haider Acosta y Carlos Cruz 
este proyecto fue relizado con la ayuda de chat gpt para organizar la estructura e instruciones de comandos para ejecutarlo en pycharm
en base al avance y primera revision del docente se realizaron las correcciones pertinentes
- src/: código real (limpieza, EDA, entrenamiento, utilidades y gráficos).
- notebooks/: cuadernillos *livianos*; solo llaman funciones de src/.
- dashboard/app.py: app interactiva en *Streamlit*.
- reports/: métricas y *figuras exportadas* (incluye carpeta descriptiva/ para el EDA previo).
- data/: raw/ (base), processed/ (clean.csv) y final/ (predicciones).
- Dataset base encontrado: *NBA_Player_Stats_2.csv*.
Se agrega un cuadernillo de Power BI para replicar los indicadores y visuales del proyecto sin depender del notebook.

---

## 2) ¿Qué es Streamlit y por qué lo usamos?
*Streamlit* es una librería de Python que convierte scripts en *aplicaciones web*.  
Nos permite mostrar *KPI, gráficos y predicciones* con controles (filtros, carga de CSV) sin depender del cuaderno.  
Ventajas para evaluación: *claridad, **trazabilidad* y *cero fricción* para reproducir.

Para abrir la app: streamlit run dashboard/app.py (pasos completos abajo).

---

## 3) Pasos para ver todo (desde nba_project/)

### 3.1 Entorno e instalación
*Windows (PowerShell)*
powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt



### 3.2 EDA *antes* de limpiar (opcional pero recomendado)
Abra Jupyter y ejecute: notebooks/00_estadistica_descriptiva.ipynb.  
Se generan histogramas, barras por posición/era, boxplots, heatmap y dispersión.  
Figuras en: reports/figuras/descriptiva/.

### 3.3 Limpieza
bash
python -m src.data_prep

Produce data/processed/clean.csv.  
Hace casting de tipos, elimina vacíos en variables clave y evita duplicados Player+Season.

### 3.4 Entrenamiento (*8 features sin fuga*)
bash
python -c "from src.train_model import main; main(k_top=8, ask_input=False)"

Artefactos:
- reports/metrics.json (MAE, RMSE, R²)
- reports/figuras/y_vs_pred.png, reports/figuras/residuals.png
- data/final/predicciones_test.csv
- models/best_model.joblib
- reports/feature_ranking.csv

### 3.5 Dashboard (Streamlit)
bash
streamlit run dashboard/app.py

Pestañas principales:
- *Distribuciones*: histogramas de PTS, MP, FGA, 3PA, FTA, FG%, 3P%, 2P%, FT%.
- *Relaciones*: PTS vs MP/FGA/3PA/FTA/Age pip install statsmodels).
- *Modelo*: métricas + figuras + descarga de predicciones_test.csv.
nba_project/

│

├── data/

│   ├── raw/                 → datos originales (CSV, Excel, JSON)

│   ├── processed/           → datos limpios

│   └── external/            → datos descargados de APIs

│

├── notebooks/

│   ├── 01_exploracion.ipynb

│   ├── 02_limpieza.ipynb

│   ├── 03_modelos.ipynb

│   └── 04_conclusiones.ipynb

│

├── src/

│   ├── data_prep.py         → funciones de limpieza

│   ├── train_model.py       → script de entrenamiento

│   ├── utils.py             → funciones auxiliares

│   └── viz.py               → funciones de gráficos

│

├── reports/

│   ├── informe_final.pdf    → documento del proyecto

│   └── figuras/             → gráficos exportados

│

├── dashboard/

│   └── app.py               → dashboard Streamlit

│

├── requirements.txt         → librerías necesarias

└── README.md                → explicación del proyecto
- *Predicción CSV*: suba un CSV con Pos, Age, MP, FT%, 3P%, 2P%, FGA, 3PA y descargue el resultado con PTS_pred.  
  Además, *predicción manual* para un solo registro.

---

## 4) Cómo están organizados los cuadernillos
- *00_estadistica_descriptiva.ipynb* → EDA previo (gráficos y tablas).  
- *01_exploracion.ipynb* → head(), tipos, nulos y perfil de columnas.  
- *02_limpieza.ipynb* → llama a src.data_prep y muestra clean.csv.  
- *03_modelos.ipynb* → llama a src.train_model y presenta métricas/figuras.  
- *04_conclusiones.ipynb* → cierra con R²/RMSE/MAE y mejoras sugeridas.

> Nota de rutas: si ejecuta desde /notebooks, use ROOT = Path.cwd().parent para acceder a la raíz del proyecto.


## 6) Problemas típicos y solución
- ModuleNotFoundError: src... → corra los comandos desde la *raíz* y use python -m ....  
- PowerShell bloquea el venv → Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned.  
- Puerto ocupado (Streamlit) → streamlit run dashboard/app.py --server.port 8502.  
- Línea OLS en scatter → pip install statsmodels o quite trendline="ols" en el código.  
- Rutas en notebooks → ROOT = Path.cwd().parent y use ROOT/'data'/....).

## Pasos rápidos
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.data_prep
python -c "from src.train_model import main; main(k_top=8, ask_input=False)"
streamlit run dashboard/app.py
```
