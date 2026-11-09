"""VM chunk runner: processes jobs (priority order) for up to BUDGET seconds per call; resumable."""
import os, sys, subprocess, time, csv
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 160
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
R = {  # folder: (label, area)
 'guzergah_2_tezde_ucus_1': ('Urban1','urgup'), 'guzergah_3_tezde_ucus_2': ('Urban2','urgup'), 'guzergah_4_tezde_ucus_3': ('Urban3','urgup'),
 '6_tezde_ucus_4': ('Urban4','urgup'), '1_tezde_ucus_5': ('Urban5','urgup'), '2_tezde_ucus_6': ('Urban6','urgup'), '3_tezde_ucus_7': ('Urban7','urgup'),
 '4_tezde_ucus_8': ('Urban8','urgup'), 'guzergah_7_to_gurlevek': ('Rural1','karlik'), 'guzergah_6_to_tagar': ('Rural2','karlik'), 'guzergah_5_to_karain': ('Rural3','karlik'),
 '5_tezde_ucus_9': ('Urban9_heldout','urgup'), 'guzergah_1': ('Urban10_heldout','urgup'), 'guzergah_1_donus': ('Urban11_heldout','urgup'),
 'guzergah_8_koy_vadi': ('Rural4_heldout','karlik'), 'guzergah_9_koy_dag': ('Rural5_heldout','karlik')}
AREA = {'urgup': dict(rgb='urgup_30_cm_yeni_gmaps_utm.tif', dem='urgup_gmap_30_cm_elevations_560.tif', m4='haritalar_replay/urgup_gmaps_f48k5s2_e300.tif'),
        'karlik': dict(rgb='karlik_30_cm_bingmap_utm.tif', dem='ana_harita_karlik_30_cm_bingmap_elevations_576.tif', m4='haritalar_replay/karlik_bing_f48k5s2_e300.tif')}
JOBS = [('m4e300','track_causal'), ('m4e300','track_aided'), ('m4e300','oracle'), ('gray','oracle'), ('gray','track_causal')]
def n_done(out, run, label):
    if not os.path.exists(out): return 0
    with open(out, newline='') as f:
        return sum(1 for r in csv.DictReader(f) if r['run'] == run and r['route'] == label)
t0 = time.time()
for mlabel, mode in JOBS:
    for folder, (label, area) in R.items():
        if time.time() - t0 > BUDGET: sys.exit(0)
        run = f'{mlabel}_{mode}'; out = f'flight_replay/runs_vm/{run}.csv'
        nframes = len([f for f in os.listdir(os.path.join(ROOT, 'guzergahlar', folder)) if f.lower().endswith('.jpg')])
        if n_done(os.path.join(ROOT, out), run, label) >= nframes: continue
        model = '' if mlabel == 'gray' else 'flight_replay/onnx/f48k5s2_e300.onnx'
        mp = AREA[area]['rgb'] if mlabel == 'gray' else AREA[area]['m4']
        left = BUDGET - (time.time() - t0)
        cmd = [sys.executable, 'flight_replay/replay_flights.py', '--frames', f'guzergahlar/{folder}', '--route', label, '--map', mp,
               '--dem', AREA[area]['dem'], '--model', model, '--mode', mode, '--out', out, '--run', run, '--budget', str(max(5, left - 8))]
        print('>>', run, label, flush=True)
        subprocess.call(cmd, cwd=ROOT, stdout=subprocess.DEVNULL)
print('all VM jobs complete')
