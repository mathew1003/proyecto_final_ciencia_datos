from pathlib import Path
import re
def project_root_from(file:str)->Path: return Path(file).resolve().parents[1]
def ensure_dir(p:Path)->Path: p=Path(p); p.mkdir(parents=True, exist_ok=True); return p
def localizar_dataset(root:Path)->Path:
    raw=root/'data'/'raw'
    cand=raw/'NBA_Player_Stats_2.csv'
    if cand.exists(): return cand
    for p in raw.glob('*.csv'): return p
    raise FileNotFoundError('No dataset en data/raw')
def pick_column(cands, cols):
    cols=[str(c) for c in cols]; low={c.lower():c for c in cols}
    for c in cands:
        if c in cols: return c
        if c.lower() in low: return low[c.lower()]
    return None
def era_from_season(season):
    if season is None: return None
    m=re.search(r'(19|20)\d{2}', str(season))
    if not m: return None
    y=int(m.group(0)); return 'pre-2010' if y<2010 else '2010s+'
def savefig(path):
    from matplotlib import pyplot as plt
    p=Path(path); ensure_dir(p.parent); plt.tight_layout(); plt.savefig(p, dpi=150, bbox_inches='tight')
