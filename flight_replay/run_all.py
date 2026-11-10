#!/usr/bin/env python3
"""
Driver for the complete flight re-evaluation (resumable; safe to re-run after an interruption).

Steps
  1. structural reference rasters for both study areas and both models (transform_map.py)
  2. per-frame replay for every route x model x mode (replay_flights.py)
  3. sensitivity runs (heading, altitude, initialisation) on the causal replay

Outputs land in flight_replay/runs/<run>.csv, one row per frame. Tables are built later from these.
Usage: python run_all.py [--jobs N] [--only main|sens|maps] [--routes a,b]
"""
import os, sys, subprocess, argparse, itertools, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable

ROUTES = {  # folder -> (label, area)
    'guzergah_2_tezde_ucus_1': ('Urban1', 'urgup'),
    'guzergah_3_tezde_ucus_2': ('Urban2', 'urgup'),
    'guzergah_4_tezde_ucus_3': ('Urban3', 'urgup'),
    '6_tezde_ucus_4': ('Urban4', 'urgup'),
    '1_tezde_ucus_5': ('Urban5', 'urgup'),
    '2_tezde_ucus_6': ('Urban6', 'urgup'),
    '3_tezde_ucus_7': ('Urban7', 'urgup'),
    '4_tezde_ucus_8': ('Urban8', 'urgup'),
    'guzergah_7_to_gurlevek': ('Rural1', 'karlik'),
    'guzergah_6_to_tagar': ('Rural2', 'karlik'),
    'guzergah_5_to_karain': ('Rural3', 'karlik'),
    # flights never used for model selection or in the thesis tables (held-out)
    '5_tezde_ucus_9': ('Urban9_heldout', 'urgup'),
    'guzergah_1': ('Urban10_heldout', 'urgup'),
    'guzergah_1_donus': ('Urban11_heldout', 'urgup'),
    'guzergah_8_koy_vadi': ('Rural4_heldout', 'karlik'),
    'guzergah_9_koy_dag': ('Rural5_heldout', 'karlik'),
}
AREAS = {  # Copernicus GLO-30 DEM resampled to the mosaic grids (tiled GeoTIFFs; sampled by lat/lon)
    'urgup': dict(rgb='urgup_30_cm_yeni_gmaps_utm.tif', dem='urgup_gmap_30_cm_elevations_560.tif'),
    'karlik': dict(rgb='karlik_30_cm_bingmap_utm.tif', dem='ana_harita_karlik_30_cm_bingmap_elevations_576.tif'),
}
sys.path.insert(0, HERE)
import models_registry as MR
from gpu_fault import GPU_FAULT_RC, is_gpu_fault
MODELS = {  # label -> model file (.h5 with TensorFlow or .onnx, see models_registry.py); '' = grayscale baseline
    'm4e300': MR.model_path('m4e300'),            # thesis lineage f48-k5-s2, archived epoch 300
    'f48k4s1': MR.model_path('f48k4s1'),          # benchmark-selected checkpoint 1850 (Table 6)
    'f48k4s1c1550': MR.model_path('f48k4s1c1550'),  # checkpoint 1550 (best on the core mosaics)
    'gray': '',
    'sobel': 'sobel',   # classical structural baseline (gradient magnitude), same matcher
}
MODES = ['track_aided', 'track_causal', 'oracle']
# Input preprocessing of the CAE (and of the gradient-magnitude baseline):
#   VN_EQ=0 (default): grey -> (x - 127.5) / 127.5, no equalisation - the preprocessing of the thesis flight code
#                      and of the cross-source benchmark (main protocol of the article);
#   VN_EQ=1: PIL-style histogram equalisation first (contract written in the *.pre.json sidecars), kept as a
#            sensitivity analysis; its results live in maps/ and runs/.
EQ = int(os.environ.get('VN_EQ', '0'))
MAPS_DIR = 'flight_replay/maps' if EQ else 'flight_replay/maps_ne'
RUNS_DIR = 'flight_replay/runs' if EQ else 'flight_replay/runs_ne'
# priority order: what the article needs first
MAIN = [('m4e300', 'track_aided'), ('m4e300', 'track_causal'), ('m4e300', 'oracle'), ('gray', 'oracle'),
        ('f48k4s1', 'track_aided'), ('f48k4s1', 'track_causal'), ('f48k4s1', 'oracle'), ('sobel', 'oracle'), ('sobel', 'track_causal'), ('gray', 'track_causal'), ('gray', 'track_aided'),
        ('f48k4s1c1550', 'oracle'), ('f48k4s1c1550', 'track_causal')]
if not EQ:  # grayscale NCC uses no preprocessing, so its runs in runs/ are valid for both settings
    MAIN = [m for m in MAIN if m[0] != 'gray']
SENS = [  # (suffix, extra args) applied to m4e300 + f48k4s1 track_causal
    ('yaw+5', ['--yaw-offset', '5']), ('yaw-5', ['--yaw-offset', '-5']),
    ('yaw+10', ['--yaw-offset', '10']), ('yaw-10', ['--yaw-offset', '-10']),
    ('alt+5pct', ['--alt-scale', '1.05']), ('alt-5pct', ['--alt-scale', '0.95']),
    ('alt+10pct', ['--alt-scale', '1.10']), ('alt-10pct', ['--alt-scale', '0.90']),
    ('init+200m', ['--init-offset-m', '200']), ('init+500m', ['--init-offset-m', '500']),
]


import threading
_LOG_LOCK = threading.Lock()
_LOG_PATH = os.path.join(ROOT, 'flight_replay', 'runs', 'run_all.log')
_GPU_FAULT = threading.Event()  # set once any job reports a sticky CUDA fault: stop everything (gpu_fault.py)


def _emit(line):
    """Write a line to the console and to flight_replay/runs/run_all.log."""
    with _LOG_LOCK:
        try:
            sys.stdout.write(line)
        except UnicodeEncodeError:
            enc = sys.stdout.encoding or 'ascii'
            sys.stdout.write(line.encode(enc, 'replace').decode(enc))
        sys.stdout.flush()
        try:
            with open(_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(time.strftime('%H:%M:%S ') + line)
        except OSError:
            pass


def _kill_tree(p):
    """Kill a process with its children. p.kill() alone would leave the child of a replay supervisor running
    (on the GPU, and writing the CSV the next job of its group also writes)."""
    if os.name == 'nt':
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(p.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        p.kill()


def _abort_on_gpu_fault():
    """Exit with GPU_FAULT_RC if a job reported a sticky CUDA fault (run_replay.bat stops on a non-zero code)."""
    if _GPU_FAULT.is_set():
        _emit('\n*** ABORT: GPU hatasi (CUDA illegal memory access vb.). Isler durduruldu, yeniden denenmedi.\n'
              '*** Bilgisayari yeniden baslatin, sonra run_replay.bat ile kaldigi yerden devam edin.\n'
              '*** ABORT: sticky CUDA fault on the GPU; all jobs stopped, nothing retried. Reboot, then re-run to resume.\n')
        sys.exit(GPU_FAULT_RC)


def sh(cmd, stall_s=None):
    """Run a command, stream its output to console + log. If stall_s is given and the command prints
    nothing for that many seconds, it is killed (the replay is resumable, a re-run fills the gap).
    Once any job reports a sticky CUDA fault, no new command starts and the running ones are killed: the job
    that reported it gets 90 s to exit on its own, the others are stopped at once."""
    if _GPU_FAULT.is_set():
        return GPU_FAULT_RC
    _emit('>> ' + ' '.join(cmd) + '\n')
    bench_cache = os.path.normpath(os.path.join(ROOT, '..', 'benchmark', 'outputs', 'cuda_cache'))
    env = dict(os.environ, PYTHONUNBUFFERED='1', CUDA_CACHE_MAXSIZE=str(4 * 1024 ** 3),
               CUDA_CACHE_PATH=os.environ.get('CUDA_CACHE_PATH', bench_cache if os.path.isdir(bench_cache) else os.path.join(HERE, 'cuda_cache')))
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                         text=True, encoding='utf-8', errors='replace', bufsize=1)
    last = [time.time()]
    own_fault = [None]  # time this command first reported a sticky CUDA fault

    def reader():
        for line in p.stdout:
            last[0] = time.time()
            _emit(line)
            if own_fault[0] is None and is_gpu_fault(line):
                own_fault[0] = time.time(); _GPU_FAULT.set()
                _emit(f'!! CUDA fault on the GPU; stopping all jobs, no retries: {" ".join(cmd)}\n')
    t = threading.Thread(target=reader, daemon=True); t.start()
    killed = False
    while p.poll() is None:
        time.sleep(5)
        if _GPU_FAULT.is_set() and (own_fault[0] is None or time.time() - own_fault[0] > 90):
            _kill_tree(p); killed = True
            _emit(f'!! stopped after a CUDA fault (re-run after a reboot to resume): {" ".join(cmd)}\n')
            break
        if stall_s and time.time() - last[0] > stall_s:
            _kill_tree(p); killed = True
            _emit(f'!! no output for {stall_s} s; killed (re-run to resume): {" ".join(cmd)}\n')
            break
    try:
        rc = p.wait(timeout=120 if killed else None)
    except subprocess.TimeoutExpired:  # e.g. stuck in a hung GPU driver; do not block run_all on it
        _emit(f'!! process {p.pid} did not exit 120 s after being killed: {" ".join(cmd)}\n')
        rc = -1
    t.join(timeout=10)
    if rc == GPU_FAULT_RC:  # the replay supervisor saw the fault (also covers output lost before the reader)
        _GPU_FAULT.set()
    if rc != 0 and not killed:
        _emit(f'!! exit code {rc}: {" ".join(cmd)}\n')
    return rc


def map_path(area, model):
    return f'{MAPS_DIR}/{area}_{model}.tif'


def acquire_lock():
    """Refuse to start if another run_all.py is already running (two copies would write the same CSVs)."""
    os.makedirs(os.path.join(ROOT, 'flight_replay/runs'), exist_ok=True)
    f = open(os.path.join(ROOT, 'flight_replay/runs/.run_all.lock'), 'a+')
    try:
        if os.name == 'nt':
            import msvcrt
            f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print('\n*** Baska bir run_all.py zaten calisiyor. Ikinci kopya baslatilmadi. ***\n'
              '*** Another run_all.py is already running; this copy will not start. ***\n', flush=True)
        sys.exit(2)
    return f


def main():
    _lock = acquire_lock()
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--only', default='all')
    ap.add_argument('--routes', default='')
    a = ap.parse_args()
    _emit(f'[run_all] --only {a.only} | backend: {MR.backend()} | python: {sys.executable}\n')
    if a.only == 'sens':  # run_replay.bat calls '--only sens' last: complete everything that is still missing
        a.only = 'all'
    _emit(f'[run_all] preprocessing eq={EQ} | maps: {MAPS_DIR} | runs: {RUNS_DIR}\n')
    os.makedirs(os.path.join(ROOT, MAPS_DIR), exist_ok=True)
    os.makedirs(os.path.join(ROOT, RUNS_DIR), exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'flight_replay/runs'), exist_ok=True)

    if True:  # reference rasters are always checked first (fast when complete); two at a time
        mjobs = [[PY, 'flight_replay/transform_map.py', A['rgb'], map_path(area, mlabel), onnx, '--budget', '1e9', '--eq', str(EQ)]
                 for area, A in AREAS.items() for mlabel, onnx in MODELS.items() if onnx]
        with ThreadPoolExecutor(max_workers=2) as ex:
            list(ex.map(sh, mjobs))
        _abort_on_gpu_fault()
        import rasterio
        for area, A in AREAS.items():
            for mlabel, onnx in MODELS.items():
                if not onnx: continue
                out = os.path.join(ROOT, map_path(area, mlabel)); rd = out + '.rows_done'
                with rasterio.open(out) as ds: nby = -(-ds.height // 512)
                n = len(set(open(rd).read().split())) if os.path.exists(rd) else 0
                if n < nby:
                    print(f'\n*** ABORT: harita tamamlanmadi / map incomplete: {out} ({n}/{nby} rows).\n'
                          f'*** Hata kaydi / error log: {out}.error.log\n', flush=True); sys.exit(1)
        if a.only == 'maps': return

    routes = [r for r in ROUTES if (not a.routes or r in a.routes.split(','))]
    main_jobs, sens_jobs = [], []
    if a.only in ('all', 'main'):
        for (mlabel, mode) in MAIN:
            for r in routes:
                label, area = ROUTES[r]
                mp = AREAS[area]['rgb'] if mlabel == 'gray' else map_path(area, mlabel)
                run = f'{mlabel}_{mode}'
                main_jobs.append([PY, 'flight_replay/replay_flights.py', '--frames', f'guzergahlar/{r}', '--route', label,
                                  '--map', mp, '--dem', AREAS[area]['dem'], '--model', MODELS[mlabel], '--mode', mode,
                                  '--out', f'{RUNS_DIR}/{run}.csv', '--run', run, '--eq', str(EQ)])
    if a.only in ('all', 'sens'):
        for mlabel in ('m4e300',):  # sensitivity for the thesis lineage only (Table 5)
            for suffix, extra in SENS:
                for r in routes:
                    label, area = ROUTES[r]
                    if label.endswith('heldout'): continue
                    run = f'{mlabel}_track_causal_{suffix}'
                    sens_jobs.append([PY, 'flight_replay/replay_flights.py', '--frames', f'guzergahlar/{r}', '--route', label,
                                      '--map', map_path(area, mlabel), '--dem', AREAS[area]['dem'], '--model', MODELS[mlabel],
                                      '--mode', 'track_causal', '--out', f'{RUNS_DIR}/{run}.csv', '--run', run, '--eq', str(EQ)] + extra)
    # skip route jobs that are already complete (avoids starting TensorFlow just to find nothing to do)
    import csv as _csv

    def _todo(jobs):
        cache = {}

        def done(out, run, route):
            key = os.path.join(ROOT, out)
            if key not in cache:
                c = {}
                if os.path.exists(key):
                    with open(key, newline='') as f:
                        for row in _csv.DictReader(f):
                            c.setdefault((row['run'], row['route']), set()).add(row['file'])
                cache[key] = c
            return len(cache[key].get((run, route), ()))
        todo = []
        for j in jobs:
            out = j[j.index('--out') + 1]; run = j[j.index('--run') + 1]; route = j[j.index('--route') + 1]
            folder = os.path.join(ROOT, j[j.index('--frames') + 1])
            n = len({f.lower() for f in os.listdir(folder) if f.lower().endswith('.jpg')})
            if done(out, run, route) < n: todo.append(j)
        return todo

    def _pool(jobs, name):
        jobs = _todo(jobs)
        groups = {}  # one CSV is written by at most one process at a time
        for j in jobs:
            groups.setdefault(j[j.index('--out') + 1], []).append(j)
        _emit(f'[{name}] {len(jobs)} route jobs to run in {len(groups)} runs; jobs={a.jobs}\n')
        t0 = time.time()

        def run_group(items):
            for cmd in items:
                sh(cmd, stall_s=600)
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            list(ex.map(run_group, groups.values()))
        _abort_on_gpu_fault()
        _emit(f'[{name}] done in {(time.time() - t0) / 3600:.1f} h\n')

    if main_jobs:
        _pool(main_jobs, 'main')
    if a.only in ('all', 'sens'):
        # low-texture probe on the district benchmark mosaics; resumable
        sh([PY, 'flight_replay/texture_probe.py', '--eq', str(EQ), '--out', 'flight_replay/texture_probe' + ('' if EQ else '_ne')])
        _abort_on_gpu_fault()
    if sens_jobs:
        _pool(sens_jobs, 'sensitivity')
    if not EQ and a.only == 'all':
        # complete the equalised (eq=1) sensitivity runs of Model 4: Urban 1 was re-run after the duplicate-file
        # fix (the earlier copy processed every frame twice) and the Rural 1 gap of the GNSS-aided run is filled
        fix = []
        for mode in ('track_aided', 'track_causal'):
            for r in routes:
                label, area = ROUTES[r]
                run = f'm4e300_{mode}'
                fix.append([PY, 'flight_replay/replay_flights.py', '--frames', f'guzergahlar/{r}', '--route', label,
                            '--map', f'flight_replay/maps/{area}_m4e300.tif', '--dem', AREAS[area]['dem'], '--model', MODELS['m4e300'],
                            '--mode', mode, '--out', f'flight_replay/runs/{run}.csv', '--run', run, '--eq', '1'])
        _pool(fix, 'eq1-completion')
    _emit('[run_all] finished\n')

if __name__ == '__main__':
    main()
