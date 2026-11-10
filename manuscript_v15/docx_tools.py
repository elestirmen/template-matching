"""Small helpers for editing the manuscript (python-docx)."""
import copy, re
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH

SUP = re.compile(r'\^\{([^}]*)\}')


def find_par(doc, prefix, start=0):
    for i, p in enumerate(doc.paragraphs):
        if i >= start and p.text.strip().startswith(prefix):
            return i, p
    raise KeyError(prefix)


def _add_run(p, text, base_rpr=None, sup=False, italic=False, bold=False, sub=False):
    r = p.add_run(text)
    if base_rpr is not None:
        r._r.insert(0, copy.deepcopy(base_rpr))
    if sup: r.font.superscript = True
    if sub: r.font.subscript = True
    if italic: r.font.italic = True
    if bold: r.font.bold = True
    return r


def set_par(p, markup, keep_rpr=True):
    """Replace the text of paragraph p. Markup: ^{17} superscript, _{w} subscript, *text* italic, **text** bold."""
    base = None
    if keep_rpr and p.runs:
        rpr = p.runs[0]._r.find(qn('w:rPr'))
        if rpr is not None:
            base = copy.deepcopy(rpr)
            for tag in ('w:vertAlign', 'w:i', 'w:iCs', 'w:b', 'w:bCs'):
                for e in base.findall(qn(tag)): base.remove(e)
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    # tokenise
    pos = 0
    tokens = []
    pattern = re.compile(r'\^\{([^}]*)\}|_\{([^}]*)\}|\*\*([^*]+)\*\*|\*([^*]+)\*')
    for m in pattern.finditer(markup):
        if m.start() > pos: tokens.append(('t', markup[pos:m.start()]))
        if m.group(1) is not None: tokens.append(('s', m.group(1)))
        elif m.group(2) is not None: tokens.append(('u', m.group(2)))
        elif m.group(3) is not None: tokens.append(('b', m.group(3)))
        else: tokens.append(('i', m.group(4)))
        pos = m.end()
    if pos < len(markup): tokens.append(('t', markup[pos:]))
    for kind, text in tokens:
        _add_run(p, text, base, sup=(kind == 's'), italic=(kind == 'i'), bold=(kind == 'b'), sub=(kind == 'u'))
    return p


def insert_par_after(p, markup, style=None):
    """Insert a new paragraph after p (same style unless given)."""
    new_p = copy.deepcopy(p._p)
    for child in list(new_p):
        if child.tag != qn('w:pPr'): new_p.remove(child)
    p._p.addnext(new_p)
    from docx.text.paragraph import Paragraph
    np_ = Paragraph(new_p, p._parent)
    if style: np_.style = style
    # base run formatting from p
    base = None
    if p.runs:
        rpr = p.runs[0]._r.find(qn('w:rPr'))
        if rpr is not None: base = copy.deepcopy(rpr)
    set_par(np_, markup, keep_rpr=False)
    if base is not None:
        for tag in ('w:vertAlign', 'w:i', 'w:iCs', 'w:b', 'w:bCs'):
            for e in base.findall(qn(tag)): base.remove(e)
        for r in np_.runs:
            existing = r._r.find(qn('w:rPr'))
            b = copy.deepcopy(base)
            if existing is not None:
                for e in existing: b.append(copy.deepcopy(e))
                r._r.remove(existing)
            r._r.insert(0, b)
    return np_


def delete_par(p):
    p._p.getparent().remove(p._p)


def _tc_pr(width, header, first_col):
    tcPr = OxmlElement('w:tcPr')
    w = OxmlElement('w:tcW'); w.set(qn('w:w'), str(width)); w.set(qn('w:type'), 'dxa'); tcPr.append(w)
    b = OxmlElement('w:tcBorders')
    for side in ('top', 'left', 'bottom', 'right'):
        e = OxmlElement(f'w:{side}')
        if header and side in ('top', 'bottom'):
            e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '6'); e.set(qn('w:space'), '0'); e.set(qn('w:color'), '000000')
        else:
            e.set(qn('w:val'), 'none'); e.set(qn('w:sz'), '0'); e.set(qn('w:space'), '0'); e.set(qn('w:color'), 'FFFFFF')
        b.append(e)
    tcPr.append(b)
    if header:
        shd = OxmlElement('w:shd'); shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), 'EDEDED'); tcPr.append(shd)
    mar = OxmlElement('w:tcMar')
    for side, val in (('top', 40), ('left', 60), ('bottom', 40), ('right', 60)):
        e = OxmlElement(f'w:{side}'); e.set(qn('w:w'), str(val)); e.set(qn('w:type'), 'dxa'); mar.append(e)
    tcPr.append(mar)
    va = OxmlElement('w:vAlign'); va.set(qn('w:val'), 'center'); tcPr.append(va)
    return tcPr


def build_table(doc, template_tbl, header, rows, widths, last_row_border=True, align_first_left=True, group_header=None, left_cols=1):
    """Build a table in the manuscript style. header: list of str; rows: list of lists of str (markup allowed).
    group_header: optional list of (label, span) placed above the header (spans must sum to len(header)).
    Returns the new table object (not yet placed)."""
    from docx.table import Table
    tbl = OxmlElement('w:tbl')
    tblPr = copy.deepcopy(template_tbl._tbl.find(qn('w:tblPr')))
    tw = tblPr.find(qn('w:tblW'))
    if tw is not None: tw.set(qn('w:w'), str(sum(widths)))
    tbl.append(tblPr)
    grid = OxmlElement('w:tblGrid')
    for w in widths:
        g = OxmlElement('w:gridCol'); g.set(qn('w:w'), str(w)); grid.append(g)
    tbl.append(grid)
    all_rows = [header] + rows
    for ri, cells in enumerate(all_rows):
        tr = OxmlElement('w:tr')
        trPr = OxmlElement('w:trPr'); cs = OxmlElement('w:cantSplit'); trPr.append(cs)
        if ri == 0:
            th = OxmlElement('w:tblHeader'); trPr.append(th)
        tr.append(trPr)
        for ci, text in enumerate(cells):
            tc = OxmlElement('w:tc')
            tc.append(_tc_pr(widths[ci], ri == 0, ci == 0))
            p = OxmlElement('w:p')
            if not (align_first_left and ci < left_cols):
                pPr = OxmlElement('w:pPr'); jc = OxmlElement('w:jc'); jc.set(qn('w:val'), 'center'); pPr.append(jc); p.append(pPr)
            tc.append(p)
            tr.append(tc)
        if ri == len(all_rows) - 1 and last_row_border:
            for tc in tr.findall(qn('w:tc')):
                bot = tc.find(qn('w:tcPr')).find(qn('w:tcBorders')).find(qn('w:bottom'))
                bot.set(qn('w:val'), 'single'); bot.set(qn('w:sz'), '6'); bot.set(qn('w:color'), '000000')
        tbl.append(tr)
    t = Table(tbl, doc._body)
    if group_header:
        gtr = OxmlElement('w:tr')
        gpr = OxmlElement('w:trPr'); gpr.append(OxmlElement('w:cantSplit')); gpr.append(OxmlElement('w:tblHeader')); gtr.append(gpr)
        col = 0
        for label, span in group_header:
            tc = OxmlElement('w:tc')
            w = sum(widths[col:col + span])
            tcPr = _tc_pr(w, True, col == 0)
            if span > 1:
                gs = OxmlElement('w:gridSpan'); gs.set(qn('w:val'), str(span)); tcPr.insert(1, gs)
            # only a bottom rule under labelled groups, no rule above the header row below
            tc.append(tcPr)
            pp = OxmlElement('w:p'); pPr = OxmlElement('w:pPr'); jc = OxmlElement('w:jc'); jc.set(qn('w:val'), 'center'); pPr.append(jc); pp.append(pPr)
            tc.append(pp); gtr.append(tc); col += span
        tbl.insert(list(tbl).index(tbl.findall(qn('w:tr'))[0]), gtr)
        from docx.table import _Row
        grow = t.rows[0]
        for ci, (label, span) in enumerate(group_header):
            _fill_cell(grow.cells[sum(s_ for _, s_ in group_header[:ci])].paragraphs[0], label, bold=True)
        off = 1
    else:
        off = 0
    for ri, cells in enumerate(all_rows):
        for ci, text in enumerate(cells):
            par = t.rows[ri + off].cells[ci].paragraphs[0]
            _fill_cell(par, str(text), bold=(ri == 0))
    return t


def _fill_cell(par, markup, bold=False):
    pattern = re.compile(r'\^\{([^}]*)\}|\*([^*]+)\*')
    pos = 0; parts = []
    for m in pattern.finditer(markup):
        if m.start() > pos: parts.append(('t', markup[pos:m.start()]))
        parts.append(('s', m.group(1)) if m.group(1) is not None else ('i', m.group(2)))
        pos = m.end()
    if pos < len(markup): parts.append(('t', markup[pos:]))
    for kind, text in parts:
        r = par.add_run(text)
        r.font.size = __import__('docx').shared.Pt(9)
        if bold: r.font.bold = True
        if kind == 's': r.font.superscript = True
        if kind == 'i': r.font.italic = True


def replace_table(doc, old_tbl, new_tbl):
    old_tbl._tbl.addnext(new_tbl._tbl)
    old_tbl._tbl.getparent().remove(old_tbl._tbl)


def insert_table_after_par(p, new_tbl):
    p._p.addnext(new_tbl._tbl)


def renumber_citations(doc, insert_at):
    """A new reference is inserted at number insert_at: every citation number >= insert_at becomes +1
    (superscript runs made only of digits, commas and hyphens)."""
    pat = re.compile(r'^[0-9][0-9,\- ]*$')
    def bump(txt):
        return re.sub(r'\d+', lambda m: str(int(m.group()) + 1) if int(m.group()) >= insert_at else m.group(), txt)
    n = 0
    for p in doc.paragraphs:
        for r in p.runs:
            if r.font.superscript and pat.match(r.text or '') and r.text.strip() not in ('1', '2'):
                new = bump(r.text)
                if new != r.text: r.text = new; n += 1
    return n


def replace_picture_before(doc, caption_prefix, png_path, max_width_emu=None):
    """Replace the image in the paragraph just before the caption that starts with caption_prefix.
    Keeps the drawing, swaps the image bytes and adapts the height to the new aspect ratio."""
    from PIL import Image
    idx, cap = find_par(doc, caption_prefix)
    for j in range(idx - 1, max(idx - 4, -1), -1):
        par = doc.paragraphs[j]
        blips = par._p.findall('.//' + qn('a:blip'))
        if blips: break
    else:
        raise KeyError('no picture before ' + caption_prefix)
    rid = blips[0].get(qn('r:embed'))
    part = par.part.related_parts[rid]
    data = open(png_path, 'rb').read()
    part._blob = data
    w, h = Image.open(png_path).size
    for ext in par._p.findall('.//' + qn('wp:extent')) + par._p.findall('.//' + qn('a:ext')):
        cx = int(ext.get('cx'))
        if max_width_emu: cx = min(cx, max_width_emu)
        ext.set('cx', str(cx)); ext.set('cy', str(int(cx * h / w)))
    return part
