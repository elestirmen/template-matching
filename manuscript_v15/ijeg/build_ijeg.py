#!/usr/bin/env python3
"""Pour the v16 manuscript (World Scientific layout) into the IJEG two-column template.
Usage: python build_ijeg.py V16.docx IJEG_Template.docx refs_apa.txt OUT.docx
Rules applied (IJEG template): Cambria 10 body, two columns, first-line indent, numeric bracket citations [n],
APA 6 numbered reference list with hanging indent, "Figure n." captions, tables 9 pt without vertical lines,
keywords 3-5, abstract <= 300 words. Keyboard hyphens only."""
import sys, re, copy
import docx
from docx.shared import Pt, Cm, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC, TPL, REFS, OUT = sys.argv[1:5]
src = docx.Document(SRC)
tpl = docx.Document(TPL)
refs = [re.sub(r'^\d+\.\s*', '', l) for l in open(REFS, encoding='utf-8').read().splitlines() if l.strip()]

COL_W = Cm(8.5); PAGE_W = Cm(18.0)
KEYWORDS = ['GNSS-denied navigation', 'UAV localization', 'Template matching', 'Cross-autoencoder', 'Cross-source benchmark']
TITLE = src.paragraphs[0].text.strip()
ABSTRACT = next(p.text for p in src.paragraphs if p.text.startswith('Unmanned aerial vehicles'))

# ------------------------------------------------------------------ section renumbering
SEC_MAP = {'2': '1.1', '3': '2', '4': '3', '5': '4', '6': '5'}
HEAD_MAP = {'1. Introduction': '1. Introduction', '2. Related Work': '1.1. Related work',
            '3. Materials and Methods': '2. Method', '4. Results': '3. Results', '5. Discussion': '4. Discussion',
            '6. Conclusions': '5. Conclusion'}


def renum(num):  # '3.4' -> '2.4', '2' -> '1.1', '4' -> '3'
    a, _, b = num.partition('.')
    if a == '1': return num
    if a == '2' and not b: return '1.1'
    return SEC_MAP.get(a, a) + ('.' + b if b else '')


TEXT_FIXES = [
    ("It was added after the first round of review, at the reviewers' request, and run in global search only,",
     "It serves as a control and was run in global search only,"),
    ("Thresholded F1 scores are no longer used:", "The thresholded F1 scores of the doctoral study are not used:"),
    ("(released with the code; Data Availability)", "(released with the code; Data availability)"),
    ("run through the district benchmark after the first round of review in global search on the identical 4,000 queries per direction,",
     "run through the district benchmark in global search on the identical 4,000 queries per direction,"),
    ("The gradient-magnitude baseline added after review (last row of Table 7)", "The gradient-magnitude baseline (last row of Table 7)"),
    ("and on the additional flights gradient magnitude matches or exceeds every checkpoint.",
     "and on the additional flights gradient magnitude is the best representation in independent search and within two points of the best checkpoint in the causal replay."),
    ("and on the additional flights gradient magnitude matched or exceeded all checkpoints.",
     "and on the additional flights gradient magnitude was the best representation in independent search and within two points of the best checkpoint in the causal replay."),
    ("the same run was continued to epoch 500, and because the epoch-250 file was not archived its later checkpoints (epochs 300, 450 and 500) are the ones evaluated in Section 4.",
     "because the epoch-250 file was not archived, later archived checkpoints of the same lineage (epochs 300, 450 and 500) are the ones evaluated in Section 4."),
]


def fix_text(t):
    for a, b in TEXT_FIXES:
        if a in t: t = t.replace(a, b)
    # "Section 3.4", "Sections 3.1 and 3.4", "Sections 4.3 and 4.4", "Section 5, Validity boundaries"
    def repl(m):
        head, body = m.group(1), m.group(2)
        body2 = re.sub(r'\d(?:\.\d+)?', lambda n: renum(n.group()), body)
        return head + body2
    t = re.sub(r'(Sections? )((?:\d(?:\.\d+)?)(?:(?:, | and |-)\d(?:\.\d+)?)*)', repl, t)
    t = t.replace('Figs. ', 'Figures ').replace('Fig. ', 'Figure ')
    t = t.replace('pre-process', 'preprocess').replace('Pre-process', 'Preprocess')
    for ch in '–—−': t = t.replace(ch, '-')
    return t


# ------------------------------------------------------------------ template skeleton
body = tpl.element.body
elems = list(body)
# locate anchors: title paragraph (index 3 of paragraphs), affiliation paragraphs 7-9, the sectPr paragraph that ends
# section 0 (first paragraph carrying a sectPr), and the paragraph carrying the second sectPr (end of 2-column part)
paras = tpl.paragraphs
sect_paras = [p for p in paras if p._p.find('.//' + qn('w:sectPr')) is not None]
p_sect0, p_sect1 = sect_paras[0], sect_paras[1]


def set_text(p, text, size=None, bold=None, italic=None):
    for r in list(p.runs)[1:]: r._r.getparent().remove(r._r)
    if not p.runs: p.add_run()
    p.runs[0].text = text
    f = p.runs[0].font
    if size: f.size = Pt(size)
    if bold is not None: f.bold = bold
    if italic is not None: f.italic = italic
    return p


set_text(paras[3], TITLE)
# authors line with superscript affiliation marks
pa = paras[5]
# the template puts an ORCID icon (inline picture linked to the ORCID page) after every author; keep one as a model
_icon = copy.deepcopy(next(r for r in pa._p.findall(qn('w:r')) if r.find('.//' + qn('w:drawing')) is not None))
for r in list(pa.runs): r._r.getparent().remove(r._r)
ORCID = ['0000-0002-7952-4311', '0000-0003-4347-5279']  # Arık, Emrahoğlu
def orcid_icon(p, k):
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    r = copy.deepcopy(_icon)
    rid = tpl.part.relate_to('https://orcid.org/' + ORCID[k], RT.HYPERLINK, is_external=True)
    for h in r.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}hlinkClick'): h.set(qn('r:id'), rid)
    for dp in r.iter(qn('wp:docPr')): dp.set('id', str(9001 + k)); dp.set('name', f'ORCID {k + 1}')
    p._p.append(r)
def run(p, text, sup=False, bold=None, size=None):
    r = p.add_run(text); r.font.superscript = sup
    if bold is not None: r.font.bold = bold
    if size: r.font.size = Pt(size)
    r.font.name = 'Cambria'
    return r
run(pa, 'Ahmet Ertuğrul Arık '); run(pa, '*1', sup=True); orcid_icon(pa, 0)
run(pa, ', Nuri Emrahoğlu '); run(pa, '2', sup=True); orcid_icon(pa, 1)
set_text(paras[7], '1 Kapadokya University, Department of Information Systems and Technologies, Türkiye, ertugrul.arik@kapadokya.edu.tr', size=9)
set_text(paras[8], '2 Çukurova University, Department of Remote Sensing and Geographic Information Systems, Türkiye, nemrahoglu@cu.edu.tr', size=9)
paras[9]._p.getparent().remove(paras[9]._p)
# cite box
cite = tpl.tables[1].rows[0].cells[1].paragraphs[0]
set_text(cite, f'Arık, A. E., & Emrahoğlu, N. (2026). {TITLE}. International Journal of Engineering and Geosciences, 11 (x), page numbers.')
# keywords / abstract box
kw_cell = tpl.tables[2].rows[1].cells[0]
kp = kw_cell.paragraphs
set_text(kp[0], KEYWORDS[0], size=9)
for extra in kp[1:]: extra._p.getparent().remove(extra._p)
for k in KEYWORDS[1:]:
    np_ = copy.deepcopy(kp[0]._p); kw_cell._tc.append(np_)
    from docx.text.paragraph import Paragraph
    set_text(Paragraph(np_, kw_cell), k, size=9)
ab_cell = tpl.tables[2].rows[1].cells[2]
ap = ab_cell.paragraphs
set_text(ap[0], ABSTRACT, size=9)
for extra in ap[1:]: extra._p.getparent().remove(extra._p)
meta_cell = tpl.tables[2].rows[2].cells[0]
for p in meta_cell.paragraphs:
    if p.text.startswith('Research/Review'): set_text(p, 'Research Article', size=9)

# ------------------------------------------------------------------ remove template body between the two sectPr paragraphs
i0 = elems.index(p_sect0._p); i1 = elems.index(p_sect1._p)
for el in elems[i0 + 1:i1]:
    body.remove(el)
anchor = p_sect1._p  # new content is inserted before this paragraph


NEG = re.compile(r'(?:^|(?<=[\s(\[/]))-(?=\d)')  # the sign of a negative number: -0.8, (-1.1 to 3.4), -6.3/-6.9


def add_text(r, t):
    """Fill run r with text t. The sign of a negative number is written as Word's non-breaking hyphen
    (w:noBreakHyphen, Ctrl+Shift+- in Word: the same glyph as the keyboard hyphen), so that a line can never
    end with the sign and leave the digits on the next line. Every other hyphen stays a plain '-'."""
    def put(s):
        if s:
            e = OxmlElement('w:t'); e.text = s; e.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve'); r._r.append(e)
    pos = 0
    for m in NEG.finditer(t):
        put(t[pos:m.start()]); r._r.append(OxmlElement('w:noBreakHyphen')); pos = m.end()
    put(t[pos:])


def new_par(text_runs=None, style=None, align=None, indent=True, size=10, bold=False, italic=False, space_after=0, keep_next=False):
    p = OxmlElement('w:p'); anchor.addprevious(p)
    from docx.text.paragraph import Paragraph
    P = Paragraph(p, tpl._body)
    if style: P.style = tpl.styles[style]
    pf = P.paragraph_format
    pf.space_before = Pt(0); pf.space_after = Pt(space_after); pf.line_spacing = 1.0
    pf.first_line_indent = Cm(0.5) if indent else Cm(0)
    if align is not None: pf.alignment = align
    if keep_next: pf.keep_with_next = True
    for t, fmt in (text_runs or []):
        r = P.add_run(); r.font.name = 'Cambria'; r.font.size = Pt(fmt.get('size', size)); add_text(r, t)
        r.font.bold = fmt.get('bold', bold); r.font.italic = fmt.get('italic', italic)
        if fmt.get('sup'): r.font.superscript = True
        if fmt.get('sub'): r.font.subscript = True
    return P


def blank(size=8, keep_next=False):
    return new_par([(' ', {'size': size})], indent=False, size=size, keep_next=keep_next)


CIT = re.compile(r'^[0-9][0-9,\- ]*$')
ABBR_END = re.compile(r'\b(?:et al|e\.g|i\.e|etc|cf|vs)$')  # abbreviations whose final period stays with the word


def convert_runs(p):
    """Return list of (text, fmt) for paragraph p with superscript citations converted to [n] placed before
    trailing punctuation of the preceding run."""
    out = []
    runs = list(p.runs)
    for r in runs:
        t = r.text
        if t is None: continue
        f = r.font
        fmt = {'bold': bool(f.bold), 'italic': bool(f.italic), 'sup': bool(f.superscript), 'sub': bool(f.subscript)}
        if fmt['sup'] and CIT.match(t.strip()) and t.strip() and not is_equation:
            cit = '[' + t.strip().replace(' ', '') + ']'
            # move before trailing punctuation of previous text
            if out and out[-1][0] and out[-1][0][-1] in '.,;:)':
                prev, pf = out[-1]
                k = len(prev) - 1
                while k >= 0 and prev[k] in '.,;:)': k -= 1
                head, tail = prev[:k + 1], prev[k + 1:]
                # the period of an abbreviation belongs to the word: "Fragoso et al.^17" -> "Fragoso et al. [17]"
                if tail.startswith('.') and ABBR_END.search(head):
                    head, tail = head + '.', tail[1:]
                out[-1] = (head + ' ' + cit, pf); out.append((tail, pf))
            else:
                out.append(((' ' if out and out[-1][0] and not out[-1][0].endswith((' ', '(', '[')) else '') + cit, {'bold': False, 'italic': False}))
            continue
        out.append((fix_text(t), fmt))
    # merge adjacent same-format runs; drop empties
    merged = []
    for t, f in out:
        if not t: continue
        if merged and merged[-1][1] == f: merged[-1] = (merged[-1][0] + t, f)
        else: merged.append((t, f))
    # a citation that follows a citation: "[1,2] [3]" -> "[1,2,3]" is not attempted; "] [" cleanup only
    return merged


def add_table(src_tbl, wide):
    t = copy.deepcopy(src_tbl._tbl)
    anchor.addprevious(t)
    # fonts 9 pt, Cambria; widths scaled to target
    for r in t.iter(qn('w:r')):
        rpr = r.find(qn('w:rPr'))
        if rpr is None: rpr = OxmlElement('w:rPr'); r.insert(0, rpr)
        for tag in ('w:rFonts',):
            e = rpr.find(qn(tag))
            if e is None: e = OxmlElement(tag); rpr.insert(0, e)
            e.set(qn('w:ascii'), 'Cambria'); e.set(qn('w:hAnsi'), 'Cambria'); e.set(qn('w:cs'), 'Cambria')
        for tag in ('w:sz', 'w:szCs'):
            e = rpr.find(qn(tag))
            if e is None: e = OxmlElement(tag); rpr.append(e)
            e.set(qn('w:val'), '18')
    for p in t.iter(qn('w:p')):
        ppr = p.find(qn('w:pPr'))
        if ppr is None: ppr = OxmlElement('w:pPr'); p.insert(0, ppr)
        for tag in ('w:spacing',):
            e = ppr.find(qn(tag))
            if e is None: e = OxmlElement(tag); ppr.append(e)
            e.set(qn('w:before'), '0'); e.set(qn('w:after'), '0'); e.set(qn('w:line'), '240'); e.set(qn('w:lineRule'), 'auto')
        ind = ppr.find(qn('w:ind'))
        if ind is None: ind = OxmlElement('w:ind'); ppr.append(ind)
        ind.set(qn('w:firstLine'), '0'); ind.set(qn('w:left'), '0'); ind.set(qn('w:hanging'), '0')
    grid = t.find(qn('w:tblGrid'))
    cols = [int(g.get(qn('w:w'))) for g in grid.findall(qn('w:gridCol'))]
    target = int((PAGE_W if wide else COL_W) / 635)  # EMU -> twips
    scale = target / sum(cols)
    for g, w in zip(grid.findall(qn('w:gridCol')), cols): g.set(qn('w:w'), str(int(w * scale)))
    tblpr = t.find(qn('w:tblPr')); tw = tblpr.find(qn('w:tblW'))
    if tw is not None: tw.set(qn('w:w'), str(target)); tw.set(qn('w:type'), 'dxa')
    lay = tblpr.find(qn('w:tblLayout'))
    if lay is None: lay = OxmlElement('w:tblLayout'); tblpr.append(lay)
    lay.set(qn('w:type'), 'fixed')
    ind = tblpr.find(qn('w:tblInd'))
    if ind is not None: tblpr.remove(ind)
    # cell widths from the grid (handles gridSpan)
    new_cols = [int(g.get(qn('w:w'))) for g in grid.findall(qn('w:gridCol'))]
    for tr in t.findall(qn('w:tr')):
        col = 0
        for tc in tr.findall(qn('w:tc')):
            tcpr = tc.find(qn('w:tcPr'))
            gs = tcpr.find(qn('w:gridSpan')); span = int(gs.get(qn('w:val'))) if gs is not None else 1
            tcw = tcpr.find(qn('w:tcW'))
            if tcw is None: tcw = OxmlElement('w:tcW'); tcpr.insert(0, tcw)
            tcw.set(qn('w:w'), str(sum(new_cols[col:col + span]))); tcw.set(qn('w:type'), 'dxa')
            col += span
    jc = tblpr.find(qn('w:jc'))
    if jc is None: jc = OxmlElement('w:jc'); tblpr.append(jc)
    jc.set(qn('w:val'), 'center')
    # keep the table on one page: every row is kept with the next one (rows already cannot split), so Word moves a
    # table that does not fit to the next page instead of leaving its caption and first rows behind (Table 8)
    trs = t.findall(qn('w:tr'))
    for tr in trs[:-1]:
        for p in tr.iter(qn('w:p')):
            ppr = p.find(qn('w:pPr'))
            if ppr is None: ppr = OxmlElement('w:pPr'); p.insert(0, ppr)
            if ppr.find(qn('w:keepNext')) is None: ppr.insert(0, OxmlElement('w:keepNext'))
    return t


def section_break(cols):
    """Continuous section break ending the current section with `cols` columns."""
    P = new_par([], indent=False, size=8)
    sect = copy.deepcopy(tpl.sections[1]._sectPr)
    for c in sect.findall(qn('w:cols')): sect.remove(c)
    c = OxmlElement('w:cols'); c.set(qn('w:num'), str(cols)); c.set(qn('w:space'), '567'); sect.append(c)
    ty = sect.find(qn('w:type'))
    if ty is None: ty = OxmlElement('w:type'); sect.insert(0, ty)
    ty.set(qn('w:val'), 'continuous')
    for hd in sect.findall(qn('w:headerReference')) + sect.findall(qn('w:footerReference')): pass
    ppr = P._p.get_or_add_pPr(); ppr.append(sect)
    return P


import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
# higher-resolution replacements (label -> file); the figs/ files come from make_figs.py (>= 400 dpi at 18 cm)
FIG_OVERRIDE = {'Figure 6.': _os.path.join(_HERE, 'fig6_hires.jpg'),
                'Figure 1.': _os.path.join(_HERE, 'figs', 'fig1_pipeline.png'),
                'Figure 7.': _os.path.join(_HERE, 'figs', 'fig7_search_radius.png'),
                'Figure 8.': _os.path.join(_HERE, 'figs', 'fig8_screening_separation.png'),
                'Figure 9.': _os.path.join(_HERE, 'figs', 'fig9_checkpoint_sweep.png')}


def add_picture(src_par, wide, label=None):
    import io, os
    width = PAGE_W if wide else COL_W
    P = new_par([], indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, keep_next=True)
    if label in FIG_OVERRIDE and os.path.exists(FIG_OVERRIDE[label]):
        P.add_run().add_picture(FIG_OVERRIDE[label], width=width); return P
    blip = src_par._p.find('.//' + qn('a:blip')); rid = blip.get(qn('r:embed'))
    part = src.part.related_parts[rid]
    P.add_run().add_picture(io.BytesIO(part.blob), width=width)
    return P


# ------------------------------------------------------------------ walk the source body
WIDE_TABLES = {0, 2, 3, 4, 5, 6, 7}  # source table indices that need the full page width (0-based: T2 Table 3, T3 Table 4, T5 Table 6, T6 Table 7, T7 Table 8)
WIDE_FIGS = {'Figure 1.', 'Figure 6.', 'Figure 7.', 'Figure 8.', 'Figure 9.'}
src_body = list(src.element.body)
from docx.text.paragraph import Paragraph
from docx.table import Table
pi = -1; ti = -1; in_body = False; open_wide = False
pending_pic = None
is_equation = False
i = 0
while i < len(src_body):
    el = src_body[i]
    tag = el.tag.split('}')[1]
    if tag == 'p':
        pi += 1; P = Paragraph(el, src._body); text = P.text.strip()
        if text == '1. Introduction': in_body = True
        if not in_body: i += 1; continue
        if text == 'References':
            break
        if el.find('.//' + qn('a:blip')) is not None:
            pending_pic = P; i += 1; continue
        if not text:
            i += 1; continue
        is_equation = bool(re.search(r'\t\(\d\)$', P.text)) or text.endswith(('(1)', '(2)', '(3)', '(4)')) and '=' in text
        if text in HEAD_MAP:  # level-1 / related-work heading
            if open_wide: section_break(1); open_wide = False  # (should not happen)
            blank()
            lvl2 = text.startswith('2. ')
            new_par([(HEAD_MAP[text], {'bold': True, 'size': 10})], indent=False, keep_next=True)
            blank(keep_next=True)
        elif re.match(r'^\d\.\d\. ', text):  # level-2 heading
            num, _, rest = text.partition(' ')
            new_num = renum(num.rstrip('.')) + '.'
            blank()
            new_par([(f'{new_num} {rest}', {'bold': True, 'size': 10})], indent=False, keep_next=True)
            blank(keep_next=True)
        elif text.startswith(('Fig. ', 'Figure ')) and re.match(r'^Fig\. \d+\.', text):
            cap = convert_runs(P)
            label = fix_text(text[:text.index('.', 5) + 1])
            wide = label in WIDE_FIGS
            if pending_pic is not None:
                if wide: section_break(2); open_wide = True
                blank()
                add_picture(pending_pic, wide, label); pending_pic = None
                new_par(cap, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER if not wide else WD_ALIGN_PARAGRAPH.JUSTIFY)
                blank()
                if wide: section_break(1); open_wide = False
        elif re.match(r'^Table \d+\. ', text):
            # caption, then the table follows; look ahead for the table element
            j = i + 1
            while j < len(src_body) and src_body[j].tag.split('}')[1] != 'tbl': j += 1
            ti_here = sum(1 for e in src_body[:j] if e.tag.split('}')[1] == 'tbl')
            wide = ti_here in WIDE_TABLES
            if wide: section_break(2); open_wide = True
            blank()
            cap_p = new_par(convert_runs(P), indent=False, keep_next=True)
            cap_p.paragraph_format.keep_together = True   # a multi-line caption must not split from its table
            add_table(src.tables[ti_here], wide)
            # footnote paragraphs directly after the table (start with a superscript letter or 'a ', 'b ', 'h ')
            k = j + 1
            while k < len(src_body) and src_body[k].tag.split('}')[1] == 'p':
                Q = Paragraph(src_body[k], src._body)
                if Q.text.strip() and (Q.runs and Q.runs[0].font.superscript) or Q.text.strip().startswith(('Each row is a complete causal replay',)):
                    new_par([('Note: ', {'size': 9})] + [(t, dict(f, size=9)) for t, f in convert_runs(Q)], indent=False, size=9)
                    k += 1
                else:
                    break
            blank()
            if wide: section_break(1); open_wide = False
            i = k; continue
        elif text in ('Acknowledgments', 'Author Contributions', 'Data Availability', 'Conflicts of Interest'):
            blank()
            new_par([({'Acknowledgments': 'Acknowledgement', 'Author Contributions': 'Author contributions',
                       'Data Availability': 'Data availability', 'Conflicts of Interest': 'Conflicts of interest'}[text], {'bold': True})], indent=False, keep_next=True)
            blank(keep_next=True)
        elif text.startswith('Authors’ Biographies'):
            break
        elif is_equation:
            new_par(convert_runs(P), indent=False, align=WD_ALIGN_PARAGRAPH.LEFT)
        else:
            new_par(convert_runs(P), align=WD_ALIGN_PARAGRAPH.JUSTIFY)
        i += 1
    elif tag == 'tbl':
        ti += 1; i += 1  # tables are emitted with their captions
    else:
        i += 1

# ------------------------------------------------------------------ references
blank()
new_par([('References', {'bold': True})], indent=False, keep_next=True)
blank(keep_next=True)
# reference runs with italics: taken from the v09 (APA) manuscript where available, hand-written otherwise
V09 = sys.argv[5] if len(sys.argv) > 5 else None
ref_runs = {}
if V09:
    v9 = docx.Document(V09); ps9 = v9.paragraphs
    k0 = [k for k, p in enumerate(ps9) if p.text.strip() == 'References'][-1]
    lst = [p for p in ps9[k0 + 1:] if p.text.strip()]
    for j, p in enumerate(lst, 1):
        rr = [(r.text, bool(r.font.italic)) for r in p.runs if r.text]
        rr[0] = (re.sub(r'^\d+\.\s*', '', rr[0][0]), rr[0][1])
        ref_runs[j] = rr
CUSTOM = {
    20: [('Schleiss, M. (2019). Translating aerial images into street-map-like representations for visual self-localization of UAVs. ', False),
         ('The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, XLII-2/W13', True),
         (', 575-580. https://doi.org/10.5194/isprs-archives-XLII-2-W13-575-2019', False)],
    45: [('European Space Agency, & Airbus. (2022). ', False), ('Copernicus DEM: Copernicus Digital Elevation Model (GLO-30)', True),
         (' [Data set]. European Space Agency. https://doi.org/10.5270/ESA-c5d3d65', False)],
    51: [('Arık, A. E. (2026). Vision-based UAV altitude estimation using deep learning: A ResNet50 approach on nadir images. ', False),
         ('Academic Platform Journal of Engineering and Smart Systems, 14', True), ('(1), 46-54.', False)],
}
for n, r in enumerate(refs, 1):
    if n in CUSTOM: rr = CUSTOM[n]
    elif ref_runs: rr = ref_runs[n if n < 20 else n - 1]
    else: rr = [(r, False)]
    P = new_par([], indent=False, align=WD_ALIGN_PARAGRAPH.LEFT)
    pf = P.paragraph_format; pf.left_indent = Cm(0.5); pf.first_line_indent = Cm(-0.5)
    run(P, f'{n}. ', size=10)
    for t, it in rr:
        for ch in '\u2013\u2014\u2212': t = t.replace(ch, '-')
        x = run(P, t, size=10); x.font.italic = it
# running header: the template carries "2026, 11(3)" on the first page and "2025, 10(1)" on the others; both are
# set to the issue-neutral form of the cite box (the journal fills volume, issue and pages)
HEADER = ', 2026, 11(x), 00-0X'
for sec in tpl.sections:
    for hdr in (sec.header, sec.first_page_header, sec.even_page_header):
        for p in hdr.paragraphs:
            if 'International Journal of Engineering and Geosciences' in p.text and p.runs:
                rs = p.runs
                k = next(j for j, r in enumerate(rs) if 'Geosciences' in r.text)
                if k + 1 < len(rs):
                    rs[k + 1].text = HEADER
                    for r in rs[k + 2:]: r.text = ''
tpl.save(OUT)
print('saved', OUT)
