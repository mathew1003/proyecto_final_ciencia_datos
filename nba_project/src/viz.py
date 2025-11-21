from pathlib import Path
import matplotlib.pyplot as plt, seaborn as sns, pandas as pd
from .utils import ensure_dir
def hist(series, title, out_path=None, bins=30):
    plt.figure(); pd.to_numeric(series, errors='coerce').dropna().hist(bins=bins)
    plt.title(title); plt.xlabel(series.name); plt.ylabel('Frecuencia')
    if out_path: ensure_dir(Path(out_path).parent); plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
def bar(x,y,title,out_path=None):
    plt.figure(); sns.barplot(x=x,y=y, errorbar=None); plt.title(title); plt.xticks(rotation=45)
    if out_path: ensure_dir(Path(out_path).parent); plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
def boxplot(df,column,by,title,out_path=None):
    plt.figure(); df.boxplot(column=column, by=by); plt.title(title); plt.suptitle('')
    if out_path: ensure_dir(Path(out_path).parent); plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
def heatmap_corr(df,out_path):
    plt.figure(figsize=(8,6)); sns.heatmap(df.corr(numeric_only=True), cmap='viridis'); plt.title('Matriz de correlación')
    ensure_dir(Path(out_path).parent); plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
def scatter(x,y,title,out_path=None):
    plt.figure(); plt.scatter(x,y,alpha=0.6); plt.title(title); plt.xlabel(x.name); plt.ylabel(y.name)
    if out_path: ensure_dir(Path(out_path).parent); plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
