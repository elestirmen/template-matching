#!/usr/bin/env python3
"""Compose Fig. 6 from the two panels made by flight_replay/make_fig6.py.
Usage: python compose_fig6.py A_URBAN5.png B_RURAL1.png OUT.png"""
import sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
a, b = Image.open(sys.argv[1]).convert('RGB'), Image.open(sys.argv[2]).convert('RGB')
H = a.height
b = b.resize((round(b.width * H / b.height), H), Image.LANCZOS)
gap = 40
c = Image.new('RGB', (a.width + gap + b.width, H), 'white')
c.paste(a, (0, 0)); c.paste(b, (a.width + gap, 0))
c = c.resize((c.width // 2, c.height // 2), Image.LANCZOS)
c.save(sys.argv[3], optimize=True)
print('saved', sys.argv[3], c.size)
