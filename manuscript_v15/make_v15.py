#!/usr/bin/env python3
"""Build the revised manuscript end to end from the replay records (v15 chain; v16 adds the gradient-magnitude
benchmark result and the comparative framing; v18 adds paired block-bootstrap intervals of the flight differences
(flight_ci.py) and the coverage of the captured frames).
Usage: python make_v15.py OUT.docx --runs SPEC [SPEC ...] [--eq1 SPEC ...] [--texture CSV] [--recon JSON] [--fig6 PNG]
                                   [--sobel-bench sobel_summary.json]
"""
import os, sys, json, argparse, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import narrative, text_final

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--runs', nargs='+', required=True); ap.add_argument('--eq1', nargs='*', default=[])
ap.add_argument('--texture', default=''); ap.add_argument('--recon', default=''); ap.add_argument('--fig6', default='')
ap.add_argument('--sobel-bench', default='', help='sobel_summary.json (gradient magnitude on the district benchmark)')
ap.add_argument('--work', default=os.path.join(HERE, 'build'))
ap.add_argument('--src', default='', help='submitted v14 docx (default: V15_SRC or the OneDrive path in build_v15.py)')
a = ap.parse_args()
if a.src: os.environ['V15_SRC'] = a.src
os.makedirs(a.work, exist_ok=True)
T = os.path.join(a.work, 'tables'); NJ = os.path.join(a.work, 'numbers.json'); EJ = os.path.join(a.work, 'extra.json')
py = sys.executable
subprocess.check_call([py, os.path.join(HERE, 'make_tables.py'), T] + a.runs)
# paired block-bootstrap intervals of the flight differences (flight_ci.py; written next to the tables)
subprocess.check_call([py, os.path.join(HERE, 'flight_ci.py'), os.path.join(T, 'flight_ci')] + a.runs, stdout=subprocess.DEVNULL)
cmd = [py, os.path.join(HERE, 'fill_extra.py'), NJ] + a.runs + ['--flight-ci', os.path.join(T, 'flight_ci.json')]
if a.texture: cmd += ['--texture', a.texture]
if a.recon: cmd += ['--recon', a.recon]
if a.sobel_bench: cmd += ['--sobel-bench', a.sobel_bench]
if a.eq1: cmd += ['--eq1'] + a.eq1
subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
N = json.load(open(NJ))
E = narrative.build(N)
for k in ('aided_minus_causal_m4', 'aided_minus_causal_f48'):
    if k in N: E[k] = N[k]
E['data_avail'] = text_final.DATA_AVAIL
if a.fig6:
    E['fig6_png'] = a.fig6
json.dump(E, open(EJ, 'w'), indent=1, ensure_ascii=False)
json.dump(N, open(NJ, 'w'), indent=1, ensure_ascii=False)
stage1 = os.path.join(a.work, 'v15_stage1.docx')
subprocess.check_call([py, os.path.join(HERE, 'build_v15.py'), stage1])
subprocess.check_call([py, os.path.join(HERE, 'build_results.py'), T, stage1, a.out, EJ])

# wording fixes in paragraphs inherited from v14 (run-level, each must occur exactly once)
import docx as _docx
FIXES = [
    ("mini-batches of 32 tiles and 250 epochs for the thesis-selected model, with a checkpoint saved every 10 epochs (Table 2).",
     "mini-batches of 32 tiles and 250 epochs for the thesis-selected model, with a checkpoint saved every 10 epochs "
     "(Table 2); the same run was continued to epoch 500, and because the epoch-250 file was not archived its later "
     "checkpoints (epochs 300, 450 and 500) are the ones evaluated in Section 4."),
    ("and the same scaling is used at inference (Section 4.5)",
     "and the same scaling is used at inference (Section 4.5); the training scripts also equalized the histogram of "
     "each tile, which the inference protocol of this article does not repeat (Section 4.2)"),
    ("Input normalization at inference must match training exactly, and the benchmark enforces this",
     "The input scaling at inference must match the scaling used in training, and the benchmark enforces this"),
]
D = _docx.Document(a.out)
for old, new in FIXES:
    hits = [r for p_ in D.paragraphs for r in p_.runs if old in r.text]
    if len(hits) != 1:
        print('FIX NOT APPLIED (%d hits):' % len(hits), old[:60]); continue
    hits[0].text = hits[0].text.replace(old, new)
D.save(a.out)
print('saved', a.out)
