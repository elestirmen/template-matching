#!/usr/bin/env python3
"""One-page cover letter for the IJEG submission (docx). Usage: python make_cover_letter.py OUT.docx
Keyboard hyphens only. Numbers are the ones of the v16/v17 manuscript (numbers.json of the build)."""
import sys, json, os
from docx import Document
from docx.shared import Pt, Cm

HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, '..', 'out', 'numbers.json'), encoding='utf-8'))
g = lambda k: N[k]
TITLE = ('Learned versus Classical Structural Representations for GNSS-Denied UAV Localization by Template Matching: '
         'A Cross-Source Map Benchmark and Flight Replay')

d = Document()
st = d.styles['Normal']; st.font.name = 'Cambria'; st.font.size = Pt(10.5)
st.paragraph_format.space_after = Pt(6)
for s in d.sections:
    s.top_margin = s.bottom_margin = Cm(2.0); s.left_margin = s.right_margin = Cm(2.2)


def par(text, bold=False, space_after=6):
    p = d.add_paragraph(); r = p.add_run(text); r.bold = bold
    p.paragraph_format.space_after = Pt(space_after); return p


par('To the Editor-in-Chief\nInternational Journal of Engineering and Geosciences (IJEG)', space_after=2)
import datetime as _dt
_d = _dt.date.today()  # the letter carries the day it is generated (generate it on the day of submission)
par(f"{_d.day} {['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][_d.month - 1]} {_d.year}", space_after=8)
par(f'Subject: Submission of a research article, "{TITLE}"', bold=True)
par('Dear Editor,')
par('On behalf of my co-author, I submit the above manuscript for consideration in IJEG. The work asks a question of '
    'direct interest to engineering geodesy and UAV photogrammetry: when a nadir UAV frame is localized against a '
    'georeferenced satellite map by normalized cross-correlation, does a learned structural representation - a '
    'compact cross-autoencoder trained to render satellite and UAV images as map-like images - improve the fix beyond '
    'what a classical structural representation, the gradient magnitude of the smoothed image, already provides?')
par('Both representations are evaluated with the identical matcher, on the same data, in two settings: a cross-source '
    f'benchmark of Google and Bing mosaics over about 500 km² with {g("sb_n")} spatially blocked queries per direction, '
    'and sixteen DJI flights over Ürgüp (2,951 frames) replayed frame by frame, including a causal replay that reads '
    'GNSS only at the first frame. The main finding is that structure matters more than learning. On the benchmark, '
    f'global success at 25 m rises from {g("sb_gray_c_0")}% (grayscale) to {g("sb_cae_c_0")}% with the best learned '
    f'checkpoint, but to {g("sb_sob_c_0")}% with untrained gradient magnitude, a paired difference of {g("sb_d_cae_c")} '
    f'points (95% block-bootstrap interval {g("sb_ci_cae_c")}) that holds in both matching directions, on both halves '
    'of a spatial split and under photometric degradation. The learned representation does improve on raw intensity, '
    'and the paper explains why (false-peak suppression, cartographic target), but it does not improve on the classical '
    'filter; on the flights the differences between representations are smaller than the spread between training '
    'checkpoints. We regard this negative result as useful: any learned representation for this task should be measured '
    'against gradient magnitude, and the paper provides the means to do so.')
par('Every number in the paper is produced by released scripts from per-query and per-frame records: the benchmark, '
    'the flight replay program, the frame manifest, the result records and the scripts that build the tables are in '
    'three public GitHub repositories (Data availability section), so the comparison can be repeated and extended.')
par('An earlier version of this work was submitted to Unmanned Systems (World Scientific) and was rejected after '
    'review. The reviewers found the flight statistics of that version internally inconsistent and the baselines weak. '
    'We agreed. The present manuscript was rebuilt from scratch: all flight results were regenerated from per-frame '
    'records, the outcome definitions were fixed, GNSS-independent operation was demonstrated by a causal replay, and '
    'the classical baseline whose result now defines the paper was added. The manuscript is not under consideration '
    'elsewhere. It is derived from the first author\'s doctoral thesis (Çukurova University, 2024), which is cited and '
    'acknowledged in the text; the benchmark, the flight replay and the comparative result are new.')
par('Thank you for your consideration.')
par('Sincerely,', space_after=2)
par('Dr. Ahmet Ertuğrul Arık (corresponding author)\nDepartment of Information Systems and Technologies, Kapadokya '
    'University, Nevşehir, Türkiye\nertugrul.arik@kapadokya.edu.tr | ORCID 0000-0002-7952-4311', space_after=2)
par('Co-author: Assoc. Prof. Dr. Nuri Emrahoğlu, Department of Remote Sensing and Geographic Information Systems, '
    'Çukurova University, Adana, Türkiye (nemrahoglu@cu.edu.tr | ORCID 0000-0003-4347-5279)')
txt = '\n'.join(p.text for p in d.paragraphs)
bad = [c for c in '–—−' if c in txt]
assert not bad, f'forbidden dashes {bad}'
print('words:', len(txt.split()))
d.save(sys.argv[1]); print('saved', sys.argv[1])
