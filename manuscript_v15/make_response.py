#!/usr/bin/env python3
"""Point-by-point response to the reviews (docx), filled from build/numbers.json.
Usage: python make_response.py NUMBERS.json OUT.docx"""
import sys, json, re
from docx import Document
from docx.shared import Pt, Cm
import narrative, response_text as RT

N = json.load(open(sys.argv[1]))
narrative.build(N)  # adds the derived ranges (ca_range, ...)
g = lambda k: N.get(k, '?')
N['texture_answer'] = (
    f"on regular tiles grayscale NCC, gradient magnitude and the cross-autoencoder reach {g('reg_gray')}%, "
    f"{g('reg_sobel')}% and {g('reg_cae')}%, and on low-texture tiles {g('low_gray')}%, {g('low_sobel')}% and "
    f"{g('low_cae')}%. Within a ±1 km window the low-texture tiles are therefore not harder than regular ones, and the "
    "gate does not remove queries that would change the comparison. The probe also shows that the gradient-magnitude "
    "baseline is more accurate than the cross-autoencoder in this local search, as it is in the global benchmark "
    "(Section 4.3); the revised manuscript states this openly (Section 4.4).")


class SafeDict(dict):
    def __missing__(self, k):
        return '?'


def fill(t):
    return t.format_map(SafeDict(N))


d = Document()
st = d.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(11)
for s in d.sections:
    s.left_margin = s.right_margin = Cm(2.2)
d.add_heading('Response to the reviews of US-D-26-00277', level=1)
d.add_paragraph(fill(RT.INTRO))
d.add_heading('Summary of the main changes', level=2)
for item in [
    "All flight tables of the previous version were removed. Every flight result was regenerated from per-frame "
    "records with one released replay program (flight_replay); Tables 3-5 are computed from these records by a script "
    "that also checks their internal consistency.",
    "One set of eligible frames for every representation and replay mode; success = correct / eligible, with unresolved "
    "frames counted as failures; availability and precision reported separately; F1 and the true-negative class removed.",
    "Three replay modes: an oracle search (representation only), the GNSS-aided replay of the doctoral study, and a "
    "causal replay that reads GNSS only at the first frame; a sensitivity analysis for heading, height and initialization.",
    "Controlled baselines on identical frames and code: grayscale NCC and gradient magnitude; the thesis model and two "
    "checkpoints of the benchmark-selected network; an input-reconstruction control trained with the same recipe.",
    "Benchmark: checkpoint selection separated from evaluation by a spatial split with a 1 km buffer; the low-texture "
    "gate quantified and probed.",
    "Schleiss (2019) added and compared; the abstract, discussion and conclusions now state that the flight gains are "
    "smaller than the benchmark gains and depend on the checkpoint.",
    f"The gradient-magnitude baseline was run through the district benchmark on the identical queries: it reaches "
    f"{g('sb_sob_c')}% (clean) and {g('sb_sob_h')}% (degraded) against {g('sb_cae_c')}% and {g('sb_cae_h')}% for the "
    "benchmark-selected cross-autoencoder (Section 4.3). The title, abstract, introduction, discussion and conclusions "
    "were reframed as a controlled comparison between a learned and a classical structural representation.",
]:
    d.add_paragraph(item, style='List Bullet')
heads1 = {1: 'Internally incompatible flight statistics.', 2: 'Outcome definitions and comparison populations.',
          3: 'Model selection is not separated from final evaluation.',
          4: 'GNSS-independent operation is not demonstrated by the replay.', 5: 'Relation to prior work and baselines.'}
heads2 = {1: 'Unresolved frames as true negatives.', 2: 'True negatives in a GPS-centred search; recomputation from per-frame records.',
          3: 'Current-frame GNSS inputs; causal replay; sensitivity.', 4: 'Flown model versus benchmark model on the same frames.',
          5: 'Low-texture queries.'}
for title, heads, R, fmt in (('Reviewer 1', heads1, RT.R1, '{}.'), ('Reviewer 2', heads2, RT.R2, '({})')):
    d.add_heading(title, level=2)
    for k in sorted(R):
        p = d.add_paragraph(); r = p.add_run(f'{fmt.format(k)} {heads[k]} '); r.bold = True
        p.add_run(fill(R[k]))
txt = '\n'.join(p.text for p in d.paragraphs)
bad = [c for c in '—–−' if c in txt]
if bad: print('WARNING: non-keyboard dashes', bad)
if '?' in re.sub(r'\?\s*$', '', txt): print('note: unresolved placeholders present:', txt.count('?%'), "'?%'")
d.save(sys.argv[2]); print('saved', sys.argv[2])
