#!/usr/bin/env python3
"""Checks on the built manuscript: forbidden dashes, placeholders, abstract length, table consistency, eligible count.
Usage: python check_v16.py OUT.docx TABLES_DIR [--show]   (--show prints the paragraphs that v16 changed)"""
import sys, re, docx, pandas as pd
d = docx.Document(sys.argv[1]); TD = sys.argv[2]; show = '--show' in sys.argv
txt = '\n'.join(p.text for p in d.paragraphs)
for t in d.tables:
    for r in t.rows:
        for c in r.cells: txt += '\n' + c.text
bad = sorted({ch for ch in txt if ch in '—–−'})
print('forbidden dashes:', [hex(ord(c)) for c in bad] or 'none')
print('"?" outside equations:', len(re.findall(r'\?', txt)), '| PLACEHOLDER:', txt.count('PLACEHOLDER'))
ab = [p for p in d.paragraphs if p.text.startswith('Unmanned aerial vehicles (UAVs) rely')][0].text
print('abstract words (split):', len(ab.split()), '| words without lone hyphens:', len([w for w in ab.split() if w != '-']))
G = pd.read_csv(f'{TD}/by_group.csv'); R = pd.read_csv(f'{TD}/by_route.csv')
print('consistent (by_group / by_route):', G.consistent.astype(bool).all(), R.consistent.astype(bool).all())
doc = G[G.group == 'doctoral']
print('doctoral eligible values:', sorted(doc.eligible.unique()))
print('title:', d.paragraphs[0].text)
if show:
    keys = ['Keywords', 'This article is derived from, and extends', 'Controlled comparisons', 'Gradient-magnitude baseline',
            'The benchmark also exposes', 'Fig. 7. Success', 'Table 7 lists', 'Table 7. Best', 'Low-texture queries',
            'Interpretation.', 'Comparison with the literature', 'Validity boundaries', 'Future work', 'We asked whether',
            'Unmanned aerial vehicles (UAVs) rely']
    for i, p in enumerate(d.paragraphs):
        if any(p.text.startswith(k) for k in keys):
            print(f'\n[{i}] {p.text}')
    t7 = d.tables[6]
    print('\nTABLE 7:')
    for r in t7.rows: print(' | '.join(c.text for c in r.cells))
