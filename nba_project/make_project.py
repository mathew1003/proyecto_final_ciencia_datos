from pathlib import Path, sys, argparse, subprocess
from src.data_prep import main as clean
from src.train_model import main as train
ROOT=Path(__file__).resolve().parent
def create_structure():
    for p in ['data/raw','data/processed','data/external','data/final','reports/figuras/descriptiva','dashboard','models','notebooks','src']:
        (ROOT/p).mkdir(parents=True, exist_ok=True)
    (ROOT/'reports'/'informe_final.pdf').touch(exist_ok=True)
def create_venv():
    import venv
    vdir=ROOT/'.venv'
    if not vdir.exists(): venv.EnvBuilder(with_pip=True).create(vdir)
    return str(vdir/('Scripts/python.exe' if sys.platform.startswith('win') else 'bin/python'))
def pip_install(python_exec):
    req=ROOT/'requirements.txt'
    return subprocess.call([python_exec,'-m','pip','install','-r',str(req)])
def run_all():
    clean(); train(k_top=8, ask_input=False)
    ok=(ROOT/'reports/metrics.json').exists() and (ROOT/'reports/figuras/y_vs_pred.png').exists() and (ROOT/'data/final/predicciones_test.csv').exists()
    print('Artefactos OK' if ok else 'Faltan artefactos'); return 0 if ok else 1
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--init',action='store_true'); ap.add_argument('--venv',action='store_true'); ap.add_argument('--install',action='store_true'); ap.add_argument('--run-all',action='store_true'); a=ap.parse_args()
    py=sys.executable
    if a.init: create_structure(); print('Estructura creada.')
    if a.venv: py=create_venv(); print('Venv:', py)
    if a.install: sys.exit(pip_install(py))
    if a.run_all: sys.exit(run_all())
if __name__=='__main__': main()
