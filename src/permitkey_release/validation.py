"""Reproduce human-review point metrics from per-category confusion counts."""
import json
from .common import SELECTED

def summarize(path, all25=False):
    data=json.load(open(path, encoding='utf-8'))
    rows=[r for r in data['rows'] if r['category'] not in {'MICRO','MACRO'} and (all25 or r['category'] in SELECTED)]
    if len(rows)!=(25 if all25 else 14):
        raise ValueError('Unexpected category membership')
    tp,fp,fn=[sum(r[k] for r in rows) for k in ['tp','fp','fn']]
    return {'categories':len(rows),'precision':tp/(tp+fp),'recall':tp/(tp+fn),
        'micro_f1':2*tp/(2*tp+fp+fn),'macro_f1':sum(2*r['tp']/(2*r['tp']+r['fp']+r['fn']) for r in rows)/len(rows),
        'tp':tp,'fp':fp,'fn':fn,'endpoint':'Blind human review of offered categories'}

def figure(path, destination):
    from html import escape
    from .common import fresh
    if not str(destination).lower().endswith('.svg'):
        raise ValueError('Figure output must be an SVG file')
    rows=[r for r in json.load(open(path,encoding='utf-8'))['rows'] if r['category'] in SELECTED]
    rows.sort(key=lambda r:-r['support'])
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="660" viewBox="0 0 920 660">',
         '<rect width="920" height="660" fill="white"/>',
         '<g font-family="Arial,sans-serif" font-size="14" fill="#222">']
    for pct in [0,25,50,75,100]:
        x=250+5.5*pct
        svg.append(f'<path d="M{x} 30 V540" stroke="#ddd"/><text x="{x}" y="567" text-anchor="middle">{pct}</text>')
    for i,r in enumerate(rows):
        y=45+i*36
        svg.append(f'<text x="230" y="{y+5}" text-anchor="end">{escape(r["category"].replace("_"," ").title())}</text>')
        ci=r['f1_ci']
        if ci['low'] is not None:
            svg.append(f'<path d="M{250+550*ci["low"]} {y} H{250+550*ci["high"]}" stroke="#de8c85" stroke-width="2"/>')
        for metric,color in [('precision','#2770ad'),('recall','#26814a'),('f1','#bb3629')]:
            svg.append(f'<circle cx="{250+550*r[metric]}" cy="{y}" r="4" fill="{color}"/>')
        svg.append(f'<text x="825" y="{y+5}">n={r["support"]}</text>')
    svg+=['<text x="440" y="590">Performance (%)</text>',
          '<text x="250" y="616" fill="#2770ad">Precision</text><text x="350" y="616" fill="#26814a">Recall</text><text x="430" y="616" fill="#bb3629">F1</text><text x="480" y="616">Line: F1 95% CI</text>',
          '<text x="250" y="645" font-size="12">Blind human review; 95% bootstrap confidence intervals for F1.</text></g></svg>']
    fresh(destination).write_text('\n'.join(svg),encoding='utf-8')
