import os

svg1 = open('outputs/1.svg', 'r', encoding='utf-8').read()
svg2 = open('outputs/1_clean.svg', 'r', encoding='utf-8').read()

def resize_svg(s):
    return s.replace('<svg xmlns="http://www.w3.org/2000/svg" version="1.1"',
                     '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" style="width:300px;height:300px"')

html_parts = [
    '<!DOCTYPE html>',
    '<html><head><meta charset="utf-8"><title>SVG Comparison</title>',
    '<style>body{background:#888;display:flex;gap:20px;padding:20px;font-family:sans-serif}',
    '.box{background:#fff;padding:10px;border-radius:8px}',
    'h3{margin:0 0 8px;font-size:13px;text-align:center}',
    '.box img{display:block;width:300px;height:300px}</style>',
    '</head><body>',
    '<div class="box"><h3>Original JPG</h3>',
    '<img src="file:///d:/Ai%20Projects/GitHub/First-Stage-SVG/inputs/1.jpg"></div>',
    '<div class="box"><h3>Basic trace (threshold=128)</h3>',
    resize_svg(svg1),
    '</div>',
    '<div class="box"><h3>Improved (OTSU + morph-close)</h3>',
    resize_svg(svg2),
    '</div>',
    '</body></html>',
]

with open('outputs/preview.html', 'w', encoding='utf-8') as f:
    f.write('\n'.join(html_parts))

print('Written outputs/preview.html')
