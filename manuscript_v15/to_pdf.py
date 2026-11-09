#!/usr/bin/env python3
"""Export a docx to PDF with Word (COM) and report the page count. Usage: python to_pdf.py IN.docx [OUT.pdf]"""
import os, sys
import win32com.client as w

src = os.path.abspath(sys.argv[1])
out = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.splitext(src)[0] + '.pdf'
word = w.gencache.EnsureDispatch('Word.Application') if False else w.Dispatch('Word.Application')
word.Visible = False
doc = word.Documents.Open(src, ReadOnly=True)
try:
    doc.Repaginate()
    pages = doc.ComputeStatistics(2)        # wdStatisticPages
    words = doc.ComputeStatistics(0)        # wdStatisticWords
    doc.ExportAsFixedFormat(out, 17)        # wdExportFormatPDF
finally:
    doc.Close(False)
    word.Quit()
print(f'saved {out} | pages {pages} | words {words}')
